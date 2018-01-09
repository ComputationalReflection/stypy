
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Boundary value problem solver.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: from warnings import warn
5: 
6: import numpy as np
7: from numpy.linalg import norm, pinv
8: 
9: from scipy.sparse import coo_matrix, csc_matrix
10: from scipy.sparse.linalg import splu
11: from scipy.optimize import OptimizeResult
12: 
13: 
14: EPS = np.finfo(float).eps
15: 
16: 
17: def estimate_fun_jac(fun, x, y, p, f0=None):
18:     '''Estimate derivatives of an ODE system rhs with forward differences.
19: 
20:     Returns
21:     -------
22:     df_dy : ndarray, shape (n, n, m)
23:         Derivatives with respect to y. An element (i, j, q) corresponds to
24:         d f_i(x_q, y_q) / d (y_q)_j.
25:     df_dp : ndarray with shape (n, k, m) or None
26:         Derivatives with respect to p. An element (i, j, q) corresponds to
27:         d f_i(x_q, y_q, p) / d p_j. If `p` is empty, None is returned.
28:     '''
29:     n, m = y.shape
30:     if f0 is None:
31:         f0 = fun(x, y, p)
32: 
33:     dtype = y.dtype
34: 
35:     df_dy = np.empty((n, n, m), dtype=dtype)
36:     h = EPS**0.5 * (1 + np.abs(y))
37:     for i in range(n):
38:         y_new = y.copy()
39:         y_new[i] += h[i]
40:         hi = y_new[i] - y[i]
41:         f_new = fun(x, y_new, p)
42:         df_dy[:, i, :] = (f_new - f0) / hi
43: 
44:     k = p.shape[0]
45:     if k == 0:
46:         df_dp = None
47:     else:
48:         df_dp = np.empty((n, k, m), dtype=dtype)
49:         h = EPS**0.5 * (1 + np.abs(p))
50:         for i in range(k):
51:             p_new = p.copy()
52:             p_new[i] += h[i]
53:             hi = p_new[i] - p[i]
54:             f_new = fun(x, y, p_new)
55:             df_dp[:, i, :] = (f_new - f0) / hi
56: 
57:     return df_dy, df_dp
58: 
59: 
60: def estimate_bc_jac(bc, ya, yb, p, bc0=None):
61:     '''Estimate derivatives of boundary conditions with forward differences.
62: 
63:     Returns
64:     -------
65:     dbc_dya : ndarray, shape (n + k, n)
66:         Derivatives with respect to ya. An element (i, j) corresponds to
67:         d bc_i / d ya_j.
68:     dbc_dyb : ndarray, shape (n + k, n)
69:         Derivatives with respect to yb. An element (i, j) corresponds to
70:         d bc_i / d ya_j.
71:     dbc_dp : ndarray with shape (n + k, k) or None
72:         Derivatives with respect to p. An element (i, j) corresponds to
73:         d bc_i / d p_j. If `p` is empty, None is returned.
74:     '''
75:     n = ya.shape[0]
76:     k = p.shape[0]
77: 
78:     if bc0 is None:
79:         bc0 = bc(ya, yb, p)
80: 
81:     dtype = ya.dtype
82: 
83:     dbc_dya = np.empty((n, n + k), dtype=dtype)
84:     h = EPS**0.5 * (1 + np.abs(ya))
85:     for i in range(n):
86:         ya_new = ya.copy()
87:         ya_new[i] += h[i]
88:         hi = ya_new[i] - ya[i]
89:         bc_new = bc(ya_new, yb, p)
90:         dbc_dya[i] = (bc_new - bc0) / hi
91:     dbc_dya = dbc_dya.T
92: 
93:     h = EPS**0.5 * (1 + np.abs(yb))
94:     dbc_dyb = np.empty((n, n + k), dtype=dtype)
95:     for i in range(n):
96:         yb_new = yb.copy()
97:         yb_new[i] += h[i]
98:         hi = yb_new[i] - yb[i]
99:         bc_new = bc(ya, yb_new, p)
100:         dbc_dyb[i] = (bc_new - bc0) / hi
101:     dbc_dyb = dbc_dyb.T
102: 
103:     if k == 0:
104:         dbc_dp = None
105:     else:
106:         h = EPS**0.5 * (1 + np.abs(p))
107:         dbc_dp = np.empty((k, n + k), dtype=dtype)
108:         for i in range(k):
109:             p_new = p.copy()
110:             p_new[i] += h[i]
111:             hi = p_new[i] - p[i]
112:             bc_new = bc(ya, yb, p_new)
113:             dbc_dp[i] = (bc_new - bc0) / hi
114:         dbc_dp = dbc_dp.T
115: 
116:     return dbc_dya, dbc_dyb, dbc_dp
117: 
118: 
119: def compute_jac_indices(n, m, k):
120:     '''Compute indices for the collocation system Jacobian construction.
121: 
122:     See `construct_global_jac` for the explanation.
123:     '''
124:     i_col = np.repeat(np.arange((m - 1) * n), n)
125:     j_col = (np.tile(np.arange(n), n * (m - 1)) +
126:              np.repeat(np.arange(m - 1) * n, n**2))
127: 
128:     i_bc = np.repeat(np.arange((m - 1) * n, m * n + k), n)
129:     j_bc = np.tile(np.arange(n), n + k)
130: 
131:     i_p_col = np.repeat(np.arange((m - 1) * n), k)
132:     j_p_col = np.tile(np.arange(m * n, m * n + k), (m - 1) * n)
133: 
134:     i_p_bc = np.repeat(np.arange((m - 1) * n, m * n + k), k)
135:     j_p_bc = np.tile(np.arange(m * n, m * n + k), n + k)
136: 
137:     i = np.hstack((i_col, i_col, i_bc, i_bc, i_p_col, i_p_bc))
138:     j = np.hstack((j_col, j_col + n,
139:                    j_bc, j_bc + (m - 1) * n,
140:                    j_p_col, j_p_bc))
141: 
142:     return i, j
143: 
144: 
145: def stacked_matmul(a, b):
146:     '''Stacked matrix multiply: out[i,:,:] = np.dot(a[i,:,:], b[i,:,:]).
147: 
148:     In our case a[i, :, :] and b[i, :, :] are always square.
149:     '''
150:     # Empirical optimization. Use outer Python loop and BLAS for large
151:     # matrices, otherwise use a single einsum call.
152:     if a.shape[1] > 50:
153:         out = np.empty_like(a)
154:         for i in range(a.shape[0]):
155:             out[i] = np.dot(a[i], b[i])
156:         return out
157:     else:
158:         return np.einsum('...ij,...jk->...ik', a, b)
159: 
160: 
161: def construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle, df_dp,
162:                          df_dp_middle, dbc_dya, dbc_dyb, dbc_dp):
163:     '''Construct the Jacobian of the collocation system.
164: 
165:     There are n * m + k functions: m - 1 collocations residuals, each
166:     containing n components, followed by n + k boundary condition residuals.
167: 
168:     There are n * m + k variables: m vectors of y, each containing n
169:     components, followed by k values of vector p.
170: 
171:     For example, let m = 4, n = 2 and k = 1, then the Jacobian will have
172:     the following sparsity structure:
173: 
174:         1 1 2 2 0 0 0 0  5
175:         1 1 2 2 0 0 0 0  5
176:         0 0 1 1 2 2 0 0  5
177:         0 0 1 1 2 2 0 0  5
178:         0 0 0 0 1 1 2 2  5
179:         0 0 0 0 1 1 2 2  5
180: 
181:         3 3 0 0 0 0 4 4  6
182:         3 3 0 0 0 0 4 4  6
183:         3 3 0 0 0 0 4 4  6
184: 
185:     Zeros denote identically zero values, other values denote different kinds
186:     of blocks in the matrix (see below). The blank row indicates the separation
187:     of collocation residuals from boundary conditions. And the blank column
188:     indicates the separation of y values from p values.
189: 
190:     Refer to [1]_  (p. 306) for the formula of n x n blocks for derivatives
191:     of collocation residuals with respect to y.
192: 
193:     Parameters
194:     ----------
195:     n : int
196:         Number of equations in the ODE system.
197:     m : int
198:         Number of nodes in the mesh.
199:     k : int
200:         Number of the unknown parameters.
201:     i_jac, j_jac : ndarray
202:         Row and column indices returned by `compute_jac_indices`. They
203:         represent different blocks in the Jacobian matrix in the following
204:         order (see the scheme above):
205: 
206:             * 1: m - 1 diagonal n x n blocks for the collocation residuals.
207:             * 2: m - 1 off-diagonal n x n blocks for the collocation residuals.
208:             * 3 : (n + k) x n block for the dependency of the boundary
209:               conditions on ya.
210:             * 4: (n + k) x n block for the dependency of the boundary
211:               conditions on yb.
212:             * 5: (m - 1) * n x k block for the dependency of the collocation
213:               residuals on p.
214:             * 6: (n + k) x k block for the dependency of the boundary
215:               conditions on p.
216: 
217:     df_dy : ndarray, shape (n, n, m)
218:         Jacobian of f with respect to y computed at the mesh nodes.
219:     df_dy_middle : ndarray, shape (n, n, m - 1)
220:         Jacobian of f with respect to y computed at the middle between the
221:         mesh nodes.
222:     df_dp : ndarray with shape (n, k, m) or None
223:         Jacobian of f with respect to p computed at the mesh nodes.
224:     df_dp_middle: ndarray with shape (n, k, m - 1) or None
225:         Jacobian of f with respect to p computed at the middle between the
226:         mesh nodes.
227:     dbc_dya, dbc_dyb : ndarray, shape (n, n)
228:         Jacobian of bc with respect to ya and yb.
229:     dbc_dp: ndarray with shape (n, k) or None
230:         Jacobian of bc with respect to p.
231: 
232:     Returns
233:     -------
234:     J : csc_matrix, shape (n * m + k, n * m + k)
235:         Jacobian of the collocation system in a sparse form.
236: 
237:     References
238:     ----------
239:     .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
240:        Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
241:        Number 3, pp. 299-316, 2001.
242:     '''
243:     df_dy = np.transpose(df_dy, (2, 0, 1))
244:     df_dy_middle = np.transpose(df_dy_middle, (2, 0, 1))
245: 
246:     h = h[:, np.newaxis, np.newaxis]
247: 
248:     dtype = df_dy.dtype
249: 
250:     # Computing diagonal n x n blocks.
251:     dPhi_dy_0 = np.empty((m - 1, n, n), dtype=dtype)
252:     dPhi_dy_0[:] = -np.identity(n)
253:     dPhi_dy_0 -= h / 6 * (df_dy[:-1] + 2 * df_dy_middle)
254:     T = stacked_matmul(df_dy_middle, df_dy[:-1])
255:     dPhi_dy_0 -= h**2 / 12 * T
256: 
257:     # Computing off-diagonal n x n blocks.
258:     dPhi_dy_1 = np.empty((m - 1, n, n), dtype=dtype)
259:     dPhi_dy_1[:] = np.identity(n)
260:     dPhi_dy_1 -= h / 6 * (df_dy[1:] + 2 * df_dy_middle)
261:     T = stacked_matmul(df_dy_middle, df_dy[1:])
262:     dPhi_dy_1 += h**2 / 12 * T
263: 
264:     values = np.hstack((dPhi_dy_0.ravel(), dPhi_dy_1.ravel(), dbc_dya.ravel(),
265:                         dbc_dyb.ravel()))
266: 
267:     if k > 0:
268:         df_dp = np.transpose(df_dp, (2, 0, 1))
269:         df_dp_middle = np.transpose(df_dp_middle, (2, 0, 1))
270:         T = stacked_matmul(df_dy_middle, df_dp[:-1] - df_dp[1:])
271:         df_dp_middle += 0.125 * h * T
272:         dPhi_dp = -h/6 * (df_dp[:-1] + df_dp[1:] + 4 * df_dp_middle)
273:         values = np.hstack((values, dPhi_dp.ravel(), dbc_dp.ravel()))
274: 
275:     J = coo_matrix((values, (i_jac, j_jac)))
276:     return csc_matrix(J)
277: 
278: 
279: def collocation_fun(fun, y, p, x, h):
280:     '''Evaluate collocation residuals.
281: 
282:     This function lies in the core of the method. The solution is sought
283:     as a cubic C1 continuous spline with derivatives matching the ODE rhs
284:     at given nodes `x`. Collocation conditions are formed from the equality
285:     of the spline derivatives and rhs of the ODE system in the middle points
286:     between nodes.
287: 
288:     Such method is classified to Lobbato IIIA family in ODE literature.
289:     Refer to [1]_ for the formula and some discussion.
290: 
291:     Returns
292:     -------
293:     col_res : ndarray, shape (n, m - 1)
294:         Collocation residuals at the middle points of the mesh intervals.
295:     y_middle : ndarray, shape (n, m - 1)
296:         Values of the cubic spline evaluated at the middle points of the mesh
297:         intervals.
298:     f : ndarray, shape (n, m)
299:         RHS of the ODE system evaluated at the mesh nodes.
300:     f_middle : ndarray, shape (n, m - 1)
301:         RHS of the ODE system evaluated at the middle points of the mesh
302:         intervals (and using `y_middle`).
303: 
304:     References
305:     ----------
306:     .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
307:            Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
308:            Number 3, pp. 299-316, 2001.
309:     '''
310:     f = fun(x, y, p)
311:     y_middle = (0.5 * (y[:, 1:] + y[:, :-1]) -
312:                 0.125 * h * (f[:, 1:] - f[:, :-1]))
313:     f_middle = fun(x[:-1] + 0.5 * h, y_middle, p)
314:     col_res = y[:, 1:] - y[:, :-1] - h / 6 * (f[:, :-1] + f[:, 1:] +
315:                                               4 * f_middle)
316: 
317:     return col_res, y_middle, f, f_middle
318: 
319: 
320: def prepare_sys(n, m, k, fun, bc, fun_jac, bc_jac, x, h):
321:     '''Create the function and the Jacobian for the collocation system.'''
322:     x_middle = x[:-1] + 0.5 * h
323:     i_jac, j_jac = compute_jac_indices(n, m, k)
324: 
325:     def col_fun(y, p):
326:         return collocation_fun(fun, y, p, x, h)
327: 
328:     def sys_jac(y, p, y_middle, f, f_middle, bc0):
329:         if fun_jac is None:
330:             df_dy, df_dp = estimate_fun_jac(fun, x, y, p, f)
331:             df_dy_middle, df_dp_middle = estimate_fun_jac(
332:                 fun, x_middle, y_middle, p, f_middle)
333:         else:
334:             df_dy, df_dp = fun_jac(x, y, p)
335:             df_dy_middle, df_dp_middle = fun_jac(x_middle, y_middle, p)
336: 
337:         if bc_jac is None:
338:             dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(bc, y[:, 0], y[:, -1],
339:                                                        p, bc0)
340:         else:
341:             dbc_dya, dbc_dyb, dbc_dp = bc_jac(y[:, 0], y[:, -1], p)
342: 
343:         return construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy,
344:                                     df_dy_middle, df_dp, df_dp_middle, dbc_dya,
345:                                     dbc_dyb, dbc_dp)
346: 
347:     return col_fun, sys_jac
348: 
349: 
350: def solve_newton(n, m, h, col_fun, bc, jac, y, p, B, bvp_tol):
351:     '''Solve the nonlinear collocation system by a Newton method.
352: 
353:     This is a simple Newton method with a backtracking line search. As
354:     advised in [1]_, an affine-invariant criterion function F = ||J^-1 r||^2
355:     is used, where J is the Jacobian matrix at the current iteration and r is
356:     the vector or collocation residuals (values of the system lhs).
357: 
358:     The method alters between full Newton iterations and the fixed-Jacobian
359:     iterations based
360: 
361:     There are other tricks proposed in [1]_, but they are not used as they
362:     don't seem to improve anything significantly, and even break the
363:     convergence on some test problems I tried.
364: 
365:     All important parameters of the algorithm are defined inside the function.
366: 
367:     Parameters
368:     ----------
369:     n : int
370:         Number of equations in the ODE system.
371:     m : int
372:         Number of nodes in the mesh.
373:     h : ndarray, shape (m-1,)
374:         Mesh intervals.
375:     col_fun : callable
376:         Function computing collocation residuals.
377:     bc : callable
378:         Function computing boundary condition residuals.
379:     jac : callable
380:         Function computing the Jacobian of the whole system (including
381:         collocation and boundary condition residuals). It is supposed to
382:         return csc_matrix.
383:     y : ndarray, shape (n, m)
384:         Initial guess for the function values at the mesh nodes.
385:     p : ndarray, shape (k,)
386:         Initial guess for the unknown parameters.
387:     B : ndarray with shape (n, n) or None
388:         Matrix to force the S y(a) = 0 condition for a problems with the
389:         singular term. If None, the singular term is assumed to be absent.
390:     bvp_tol : float
391:         Tolerance to which we want to solve a BVP.
392: 
393:     Returns
394:     -------
395:     y : ndarray, shape (n, m)
396:         Final iterate for the function values at the mesh nodes.
397:     p : ndarray, shape (k,)
398:         Final iterate for the unknown parameters.
399:     singular : bool
400:         True, if the LU decomposition failed because Jacobian turned out
401:         to be singular.
402: 
403:     References
404:     ----------
405:     .. [1]  U. Ascher, R. Mattheij and R. Russell "Numerical Solution of
406:        Boundary Value Problems for Ordinary Differential Equations"
407:     '''
408:     # We know that the solution residuals at the middle points of the mesh
409:     # are connected with collocation residuals  r_middle = 1.5 * col_res / h.
410:     # As our BVP solver tries to decrease relative residuals below a certain
411:     # tolerance it seems reasonable to terminated Newton iterations by
412:     # comparison of r_middle / (1 + np.abs(f_middle)) with a certain threshold,
413:     # which we choose to be 1.5 orders lower than the BVP tolerance. We rewrite
414:     # the condition as col_res < tol_r * (1 + np.abs(f_middle)), then tol_r
415:     # should be computed as follows:
416:     tol_r = 2/3 * h * 5e-2 * bvp_tol
417: 
418:     # We also need to control residuals of the boundary conditions. But it
419:     # seems that they become very small eventually as the solver progresses,
420:     # i. e. the tolerance for BC are not very important. We set it 1.5 orders
421:     # lower than the BVP tolerance as well.
422:     tol_bc = 5e-2 * bvp_tol
423: 
424:     # Maximum allowed number of Jacobian evaluation and factorization, in
425:     # other words the maximum number of full Newton iterations. A small value
426:     # is recommended in the literature.
427:     max_njev = 4
428: 
429:     # Maximum number of iterations, considering that some of them can be
430:     # performed with the fixed Jacobian. In theory such iterations are cheap,
431:     # but it's not that simple in Python.
432:     max_iter = 8
433: 
434:     # Minimum relative improvement of the criterion function to accept the
435:     # step (Armijo constant).
436:     sigma = 0.2
437: 
438:     # Step size decrease factor for backtracking.
439:     tau = 0.5
440: 
441:     # Maximum number of backtracking steps, the minimum step is then
442:     # tau ** n_trial.
443:     n_trial = 4
444: 
445:     col_res, y_middle, f, f_middle = col_fun(y, p)
446:     bc_res = bc(y[:, 0], y[:, -1], p)
447:     res = np.hstack((col_res.ravel(order='F'), bc_res))
448: 
449:     njev = 0
450:     singular = False
451:     recompute_jac = True
452:     for iteration in range(max_iter):
453:         if recompute_jac:
454:             J = jac(y, p, y_middle, f, f_middle, bc_res)
455:             njev += 1
456:             try:
457:                 LU = splu(J)
458:             except RuntimeError:
459:                 singular = True
460:                 break
461: 
462:             step = LU.solve(res)
463:             cost = np.dot(step, step)
464: 
465:         y_step = step[:m * n].reshape((n, m), order='F')
466:         p_step = step[m * n:]
467: 
468:         alpha = 1
469:         for trial in range(n_trial + 1):
470:             y_new = y - alpha * y_step
471:             if B is not None:
472:                 y_new[:, 0] = np.dot(B, y_new[:, 0])
473:             p_new = p - alpha * p_step
474: 
475:             col_res, y_middle, f, f_middle = col_fun(y_new, p_new)
476:             bc_res = bc(y_new[:, 0], y_new[:, -1], p_new)
477:             res = np.hstack((col_res.ravel(order='F'), bc_res))
478: 
479:             step_new = LU.solve(res)
480:             cost_new = np.dot(step_new, step_new)
481:             if cost_new < (1 - 2 * alpha * sigma) * cost:
482:                 break
483: 
484:             if trial < n_trial:
485:                 alpha *= tau
486: 
487:         y = y_new
488:         p = p_new
489: 
490:         if njev == max_njev:
491:             break
492: 
493:         if (np.all(np.abs(col_res) < tol_r * (1 + np.abs(f_middle))) and
494:                 np.all(bc_res < tol_bc)):
495:             break
496: 
497:         # If the full step was taken, then we are going to continue with
498:         # the same Jacobian. This is the approach of BVP_SOLVER.
499:         if alpha == 1:
500:             step = step_new
501:             cost = cost_new
502:             recompute_jac = False
503:         else:
504:             recompute_jac = True
505: 
506:     return y, p, singular
507: 
508: 
509: def print_iteration_header():
510:     print("{:^15}{:^15}{:^15}{:^15}".format(
511:         "Iteration", "Max residual", "Total nodes", "Nodes added"))
512: 
513: 
514: def print_iteration_progress(iteration, residual, total_nodes, nodes_added):
515:     print("{:^15}{:^15.2e}{:^15}{:^15}".format(
516:         iteration, residual, total_nodes, nodes_added))
517: 
518: 
519: class BVPResult(OptimizeResult):
520:     pass
521: 
522: 
523: TERMINATION_MESSAGES = {
524:     0: "The algorithm converged to the desired accuracy.",
525:     1: "The maximum number of mesh nodes is exceeded.",
526:     2: "A singular Jacobian encountered when solving the collocation system."
527: }
528: 
529: 
530: def estimate_rms_residuals(fun, sol, x, h, p, r_middle, f_middle):
531:     '''Estimate rms values of collocation residuals using Lobatto quadrature.
532: 
533:     The residuals are defined as the difference between the derivatives of
534:     our solution and rhs of the ODE system. We use relative residuals, i.e.
535:     normalized by 1 + np.abs(f). RMS values are computed as sqrt from the
536:     normalized integrals of the squared relative residuals over each interval.
537:     Integrals are estimated using 5-point Lobatto quadrature [1]_, we use the
538:     fact that residuals at the mesh nodes are identically zero.
539: 
540:     In [2] they don't normalize integrals by interval lengths, which gives
541:     a higher rate of convergence of the residuals by the factor of h**0.5.
542:     I chose to do such normalization for an ease of interpretation of return
543:     values as RMS estimates.
544: 
545:     Returns
546:     -------
547:     rms_res : ndarray, shape (m - 1,)
548:         Estimated rms values of the relative residuals over each interval.
549: 
550:     References
551:     ----------
552:     .. [1] http://mathworld.wolfram.com/LobattoQuadrature.html
553:     .. [2] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
554:        Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
555:        Number 3, pp. 299-316, 2001.
556:     '''
557:     x_middle = x[:-1] + 0.5 * h
558:     s = 0.5 * h * (3/7)**0.5
559:     x1 = x_middle + s
560:     x2 = x_middle - s
561:     y1 = sol(x1)
562:     y2 = sol(x2)
563:     y1_prime = sol(x1, 1)
564:     y2_prime = sol(x2, 1)
565:     f1 = fun(x1, y1, p)
566:     f2 = fun(x2, y2, p)
567:     r1 = y1_prime - f1
568:     r2 = y2_prime - f2
569: 
570:     r_middle /= 1 + np.abs(f_middle)
571:     r1 /= 1 + np.abs(f1)
572:     r2 /= 1 + np.abs(f2)
573: 
574:     r1 = np.sum(np.real(r1 * np.conj(r1)), axis=0)
575:     r2 = np.sum(np.real(r2 * np.conj(r2)), axis=0)
576:     r_middle = np.sum(np.real(r_middle * np.conj(r_middle)), axis=0)
577: 
578:     return (0.5 * (32 / 45 * r_middle + 49 / 90 * (r1 + r2))) ** 0.5
579: 
580: 
581: def create_spline(y, yp, x, h):
582:     '''Create a cubic spline given values and derivatives.
583: 
584:     Formulas for the coefficients are taken from interpolate.CubicSpline.
585: 
586:     Returns
587:     -------
588:     sol : PPoly
589:         Constructed spline as a PPoly instance.
590:     '''
591:     from scipy.interpolate import PPoly
592: 
593:     n, m = y.shape
594:     c = np.empty((4, n, m - 1), dtype=y.dtype)
595:     slope = (y[:, 1:] - y[:, :-1]) / h
596:     t = (yp[:, :-1] + yp[:, 1:] - 2 * slope) / h
597:     c[0] = t / h
598:     c[1] = (slope - yp[:, :-1]) / h - t
599:     c[2] = yp[:, :-1]
600:     c[3] = y[:, :-1]
601:     c = np.rollaxis(c, 1)
602: 
603:     return PPoly(c, x, extrapolate=True, axis=1)
604: 
605: 
606: def modify_mesh(x, insert_1, insert_2):
607:     '''Insert nodes into a mesh.
608: 
609:     Nodes removal logic is not established, its impact on the solver is
610:     presumably negligible. So only insertion is done in this function.
611: 
612:     Parameters
613:     ----------
614:     x : ndarray, shape (m,)
615:         Mesh nodes.
616:     insert_1 : ndarray
617:         Intervals to each insert 1 new node in the middle.
618:     insert_2 : ndarray
619:         Intervals to each insert 2 new nodes, such that divide an interval
620:         into 3 equal parts.
621: 
622:     Returns
623:     -------
624:     x_new : ndarray
625:         New mesh nodes.
626: 
627:     Notes
628:     -----
629:     `insert_1` and `insert_2` should not have common values.
630:     '''
631:     # Because np.insert implementation apparently varies with a version of
632:     # numpy, we use a simple and reliable approach with sorting.
633:     return np.sort(np.hstack((
634:         x,
635:         0.5 * (x[insert_1] + x[insert_1 + 1]),
636:         (2 * x[insert_2] + x[insert_2 + 1]) / 3,
637:         (x[insert_2] + 2 * x[insert_2 + 1]) / 3
638:     )))
639: 
640: 
641: def wrap_functions(fun, bc, fun_jac, bc_jac, k, a, S, D, dtype):
642:     '''Wrap functions for unified usage in the solver.'''
643:     if fun_jac is None:
644:         fun_jac_wrapped = None
645: 
646:     if bc_jac is None:
647:         bc_jac_wrapped = None
648: 
649:     if k == 0:
650:         def fun_p(x, y, _):
651:             return np.asarray(fun(x, y), dtype)
652: 
653:         def bc_wrapped(ya, yb, _):
654:             return np.asarray(bc(ya, yb), dtype)
655: 
656:         if fun_jac is not None:
657:             def fun_jac_p(x, y, _):
658:                 return np.asarray(fun_jac(x, y), dtype), None
659: 
660:         if bc_jac is not None:
661:             def bc_jac_wrapped(ya, yb, _):
662:                 dbc_dya, dbc_dyb = bc_jac(ya, yb)
663:                 return (np.asarray(dbc_dya, dtype),
664:                         np.asarray(dbc_dyb, dtype), None)
665:     else:
666:         def fun_p(x, y, p):
667:             return np.asarray(fun(x, y, p), dtype)
668: 
669:         def bc_wrapped(x, y, p):
670:             return np.asarray(bc(x, y, p), dtype)
671: 
672:         if fun_jac is not None:
673:             def fun_jac_p(x, y, p):
674:                 df_dy, df_dp = fun_jac(x, y, p)
675:                 return np.asarray(df_dy, dtype), np.asarray(df_dp, dtype)
676: 
677:         if bc_jac is not None:
678:             def bc_jac_wrapped(ya, yb, p):
679:                 dbc_dya, dbc_dyb, dbc_dp = bc_jac(ya, yb, p)
680:                 return (np.asarray(dbc_dya, dtype), np.asarray(dbc_dyb, dtype),
681:                         np.asarray(dbc_dp, dtype))
682: 
683:     if S is None:
684:         fun_wrapped = fun_p
685:     else:
686:         def fun_wrapped(x, y, p):
687:             f = fun_p(x, y, p)
688:             if x[0] == a:
689:                 f[:, 0] = np.dot(D, f[:, 0])
690:                 f[:, 1:] += np.dot(S, y[:, 1:]) / (x[1:] - a)
691:             else:
692:                 f += np.dot(S, y) / (x - a)
693:             return f
694: 
695:     if fun_jac is not None:
696:         if S is None:
697:             fun_jac_wrapped = fun_jac_p
698:         else:
699:             Sr = S[:, :, np.newaxis]
700: 
701:             def fun_jac_wrapped(x, y, p):
702:                 df_dy, df_dp = fun_jac_p(x, y, p)
703:                 if x[0] == a:
704:                     df_dy[:, :, 0] = np.dot(D, df_dy[:, :, 0])
705:                     df_dy[:, :, 1:] += Sr / (x[1:] - a)
706:                 else:
707:                     df_dy += Sr / (x - a)
708: 
709:                 return df_dy, df_dp
710: 
711:     return fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped
712: 
713: 
714: def solve_bvp(fun, bc, x, y, p=None, S=None, fun_jac=None, bc_jac=None,
715:               tol=1e-3, max_nodes=1000, verbose=0):
716:     '''Solve a boundary-value problem for a system of ODEs.
717: 
718:     This function numerically solves a first order system of ODEs subject to
719:     two-point boundary conditions::
720: 
721:         dy / dx = f(x, y, p) + S * y / (x - a), a <= x <= b
722:         bc(y(a), y(b), p) = 0
723: 
724:     Here x is a 1-dimensional independent variable, y(x) is a n-dimensional
725:     vector-valued function and p is a k-dimensional vector of unknown
726:     parameters which is to be found along with y(x). For the problem to be
727:     determined there must be n + k boundary conditions, i.e. bc must be
728:     (n + k)-dimensional function.
729: 
730:     The last singular term in the right-hand side of the system is optional.
731:     It is defined by an n-by-n matrix S, such that the solution must satisfy
732:     S y(a) = 0. This condition will be forced during iterations, so it must not
733:     contradict boundary conditions. See [2]_ for the explanation how this term
734:     is handled when solving BVPs numerically.
735: 
736:     Problems in a complex domain can be solved as well. In this case y and p
737:     are considered to be complex, and f and bc are assumed to be complex-valued
738:     functions, but x stays real. Note that f and bc must be complex
739:     differentiable (satisfy Cauchy-Riemann equations [4]_), otherwise you
740:     should rewrite your problem for real and imaginary parts separately. To
741:     solve a problem in a complex domain, pass an initial guess for y with a
742:     complex data type (see below).
743: 
744:     Parameters
745:     ----------
746:     fun : callable
747:         Right-hand side of the system. The calling signature is ``fun(x, y)``,
748:         or ``fun(x, y, p)`` if parameters are present. All arguments are
749:         ndarray: ``x`` with shape (m,), ``y`` with shape (n, m), meaning that
750:         ``y[:, i]`` corresponds to ``x[i]``, and ``p`` with shape (k,). The
751:         return value must be an array with shape (n, m) and with the same
752:         layout as ``y``.
753:     bc : callable
754:         Function evaluating residuals of the boundary conditions. The calling
755:         signature is ``bc(ya, yb)``, or ``bc(ya, yb, p)`` if parameters are
756:         present. All arguments are ndarray: ``ya`` and ``yb`` with shape (n,),
757:         and ``p`` with shape (k,). The return value must be an array with
758:         shape (n + k,).
759:     x : array_like, shape (m,)
760:         Initial mesh. Must be a strictly increasing sequence of real numbers
761:         with ``x[0]=a`` and ``x[-1]=b``.
762:     y : array_like, shape (n, m)
763:         Initial guess for the function values at the mesh nodes, i-th column
764:         corresponds to ``x[i]``. For problems in a complex domain pass `y`
765:         with a complex data type (even if the initial guess is purely real).
766:     p : array_like with shape (k,) or None, optional
767:         Initial guess for the unknown parameters. If None (default), it is
768:         assumed that the problem doesn't depend on any parameters.
769:     S : array_like with shape (n, n) or None
770:         Matrix defining the singular term. If None (default), the problem is
771:         solved without the singular term.
772:     fun_jac : callable or None, optional
773:         Function computing derivatives of f with respect to y and p. The
774:         calling signature is ``fun_jac(x, y)``, or ``fun_jac(x, y, p)`` if
775:         parameters are present. The return must contain 1 or 2 elements in the
776:         following order:
777: 
778:             * df_dy : array_like with shape (n, n, m) where an element
779:               (i, j, q) equals to d f_i(x_q, y_q, p) / d (y_q)_j.
780:             * df_dp : array_like with shape (n, k, m) where an element
781:               (i, j, q) equals to d f_i(x_q, y_q, p) / d p_j.
782: 
783:         Here q numbers nodes at which x and y are defined, whereas i and j
784:         number vector components. If the problem is solved without unknown
785:         parameters df_dp should not be returned.
786: 
787:         If `fun_jac` is None (default), the derivatives will be estimated
788:         by the forward finite differences.
789:     bc_jac : callable or None, optional
790:         Function computing derivatives of bc with respect to ya, yb and p.
791:         The calling signature is ``bc_jac(ya, yb)``, or ``bc_jac(ya, yb, p)``
792:         if parameters are present. The return must contain 2 or 3 elements in
793:         the following order:
794: 
795:             * dbc_dya : array_like with shape (n, n) where an element (i, j)
796:               equals to d bc_i(ya, yb, p) / d ya_j.
797:             * dbc_dyb : array_like with shape (n, n) where an element (i, j)
798:               equals to d bc_i(ya, yb, p) / d yb_j.
799:             * dbc_dp : array_like with shape (n, k) where an element (i, j)
800:               equals to d bc_i(ya, yb, p) / d p_j.
801: 
802:         If the problem is solved without unknown parameters dbc_dp should not
803:         be returned.
804: 
805:         If `bc_jac` is None (default), the derivatives will be estimated by
806:         the forward finite differences.
807:     tol : float, optional
808:         Desired tolerance of the solution. If we define ``r = y' - f(x, y)``
809:         where y is the found solution, then the solver tries to achieve on each
810:         mesh interval ``norm(r / (1 + abs(f)) < tol``, where ``norm`` is
811:         estimated in a root mean squared sense (using a numerical quadrature
812:         formula). Default is 1e-3.
813:     max_nodes : int, optional
814:         Maximum allowed number of the mesh nodes. If exceeded, the algorithm
815:         terminates. Default is 1000.
816:     verbose : {0, 1, 2}, optional
817:         Level of algorithm's verbosity:
818: 
819:             * 0 (default) : work silently.
820:             * 1 : display a termination report.
821:             * 2 : display progress during iterations.
822: 
823:     Returns
824:     -------
825:     Bunch object with the following fields defined:
826:     sol : PPoly
827:         Found solution for y as `scipy.interpolate.PPoly` instance, a C1
828:         continuous cubic spline.
829:     p : ndarray or None, shape (k,)
830:         Found parameters. None, if the parameters were not present in the
831:         problem.
832:     x : ndarray, shape (m,)
833:         Nodes of the final mesh.
834:     y : ndarray, shape (n, m)
835:         Solution values at the mesh nodes.
836:     yp : ndarray, shape (n, m)
837:         Solution derivatives at the mesh nodes.
838:     rms_residuals : ndarray, shape (m - 1,)
839:         RMS values of the relative residuals over each mesh interval (see the
840:         description of `tol` parameter).
841:     niter : int
842:         Number of completed iterations.
843:     status : int
844:         Reason for algorithm termination:
845: 
846:             * 0: The algorithm converged to the desired accuracy.
847:             * 1: The maximum number of mesh nodes is exceeded.
848:             * 2: A singular Jacobian encountered when solving the collocation
849:               system.
850: 
851:     message : string
852:         Verbal description of the termination reason.
853:     success : bool
854:         True if the algorithm converged to the desired accuracy (``status=0``).
855: 
856:     Notes
857:     -----
858:     This function implements a 4-th order collocation algorithm with the
859:     control of residuals similar to [1]_. A collocation system is solved
860:     by a damped Newton method with an affine-invariant criterion function as
861:     described in [3]_.
862: 
863:     Note that in [1]_  integral residuals are defined without normalization
864:     by interval lengths. So their definition is different by a multiplier of
865:     h**0.5 (h is an interval length) from the definition used here.
866: 
867:     .. versionadded:: 0.18.0
868: 
869:     References
870:     ----------
871:     .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
872:            Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
873:            Number 3, pp. 299-316, 2001.
874:     .. [2] L.F. Shampine, P. H. Muir and H. Xu, "A User-Friendly Fortran BVP
875:            Solver".
876:     .. [3] U. Ascher, R. Mattheij and R. Russell "Numerical Solution of
877:            Boundary Value Problems for Ordinary Differential Equations".
878:     .. [4] `Cauchy-Riemann equations
879:             <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ on
880:             Wikipedia.
881: 
882:     Examples
883:     --------
884:     In the first example we solve Bratu's problem::
885: 
886:         y'' + k * exp(y) = 0
887:         y(0) = y(1) = 0
888: 
889:     for k = 1.
890: 
891:     We rewrite the equation as a first order system and implement its
892:     right-hand side evaluation::
893: 
894:         y1' = y2
895:         y2' = -exp(y1)
896: 
897:     >>> def fun(x, y):
898:     ...     return np.vstack((y[1], -np.exp(y[0])))
899: 
900:     Implement evaluation of the boundary condition residuals:
901: 
902:     >>> def bc(ya, yb):
903:     ...     return np.array([ya[0], yb[0]])
904: 
905:     Define the initial mesh with 5 nodes:
906: 
907:     >>> x = np.linspace(0, 1, 5)
908: 
909:     This problem is known to have two solutions. To obtain both of them we
910:     use two different initial guesses for y. We denote them by subscripts
911:     a and b.
912: 
913:     >>> y_a = np.zeros((2, x.size))
914:     >>> y_b = np.zeros((2, x.size))
915:     >>> y_b[0] = 3
916: 
917:     Now we are ready to run the solver.
918: 
919:     >>> from scipy.integrate import solve_bvp
920:     >>> res_a = solve_bvp(fun, bc, x, y_a)
921:     >>> res_b = solve_bvp(fun, bc, x, y_b)
922: 
923:     Let's plot the two found solutions. We take an advantage of having the
924:     solution in a spline form to produce a smooth plot.
925: 
926:     >>> x_plot = np.linspace(0, 1, 100)
927:     >>> y_plot_a = res_a.sol(x_plot)[0]
928:     >>> y_plot_b = res_b.sol(x_plot)[0]
929:     >>> import matplotlib.pyplot as plt
930:     >>> plt.plot(x_plot, y_plot_a, label='y_a')
931:     >>> plt.plot(x_plot, y_plot_b, label='y_b')
932:     >>> plt.legend()
933:     >>> plt.xlabel("x")
934:     >>> plt.ylabel("y")
935:     >>> plt.show()
936: 
937:     We see that the two solutions have similar shape, but differ in scale
938:     significantly.
939: 
940:     In the second example we solve a simple Sturm-Liouville problem::
941: 
942:         y'' + k**2 * y = 0
943:         y(0) = y(1) = 0
944: 
945:     It is known that a non-trivial solution y = A * sin(k * x) is possible for
946:     k = pi * n, where n is an integer. To establish the normalization constant
947:     A = 1 we add a boundary condition::
948: 
949:         y'(0) = k
950: 
951:     Again we rewrite our equation as a first order system and implement its
952:     right-hand side evaluation::
953: 
954:         y1' = y2
955:         y2' = -k**2 * y1
956: 
957:     >>> def fun(x, y, p):
958:     ...     k = p[0]
959:     ...     return np.vstack((y[1], -k**2 * y[0]))
960: 
961:     Note that parameters p are passed as a vector (with one element in our
962:     case).
963: 
964:     Implement the boundary conditions:
965: 
966:     >>> def bc(ya, yb, p):
967:     ...     k = p[0]
968:     ...     return np.array([ya[0], yb[0], ya[1] - k])
969: 
970:     Setup the initial mesh and guess for y. We aim to find the solution for
971:     k = 2 * pi, to achieve that we set values of y to approximately follow
972:     sin(2 * pi * x):
973: 
974:     >>> x = np.linspace(0, 1, 5)
975:     >>> y = np.zeros((2, x.size))
976:     >>> y[0, 1] = 1
977:     >>> y[0, 3] = -1
978: 
979:     Run the solver with 6 as an initial guess for k.
980: 
981:     >>> sol = solve_bvp(fun, bc, x, y, p=[6])
982: 
983:     We see that the found k is approximately correct:
984: 
985:     >>> sol.p[0]
986:     6.28329460046
987: 
988:     And finally plot the solution to see the anticipated sinusoid:
989: 
990:     >>> x_plot = np.linspace(0, 1, 100)
991:     >>> y_plot = sol.sol(x_plot)[0]
992:     >>> plt.plot(x_plot, y_plot)
993:     >>> plt.xlabel("x")
994:     >>> plt.ylabel("y")
995:     >>> plt.show()
996:     '''
997:     x = np.asarray(x, dtype=float)
998:     if x.ndim != 1:
999:         raise ValueError("`x` must be 1 dimensional.")
1000:     h = np.diff(x)
1001:     if np.any(h <= 0):
1002:         raise ValueError("`x` must be strictly increasing.")
1003:     a = x[0]
1004: 
1005:     y = np.asarray(y)
1006:     if np.issubdtype(y.dtype, np.complexfloating):
1007:         dtype = complex
1008:     else:
1009:         dtype = float
1010:     y = y.astype(dtype, copy=False)
1011: 
1012:     if y.ndim != 2:
1013:         raise ValueError("`y` must be 2 dimensional.")
1014:     if y.shape[1] != x.shape[0]:
1015:         raise ValueError("`y` is expected to have {} columns, but actually "
1016:                          "has {}.".format(x.shape[0], y.shape[1]))
1017: 
1018:     if p is None:
1019:         p = np.array([])
1020:     else:
1021:         p = np.asarray(p, dtype=dtype)
1022:     if p.ndim != 1:
1023:         raise ValueError("`p` must be 1 dimensional.")
1024: 
1025:     if tol < 100 * EPS:
1026:         warn("`tol` is too low, setting to {:.2e}".format(100 * EPS))
1027:         tol = 100 * EPS
1028: 
1029:     if verbose not in [0, 1, 2]:
1030:         raise ValueError("`verbose` must be in [0, 1, 2].")
1031: 
1032:     n = y.shape[0]
1033:     k = p.shape[0]
1034: 
1035:     if S is not None:
1036:         S = np.asarray(S, dtype=dtype)
1037:         if S.shape != (n, n):
1038:             raise ValueError("`S` is expected to have shape {}, "
1039:                              "but actually has {}".format((n, n), S.shape))
1040: 
1041:         # Compute I - S^+ S to impose necessary boundary conditions.
1042:         B = np.identity(n) - np.dot(pinv(S), S)
1043: 
1044:         y[:, 0] = np.dot(B, y[:, 0])
1045: 
1046:         # Compute (I - S)^+ to correct derivatives at x=a.
1047:         D = pinv(np.identity(n) - S)
1048:     else:
1049:         B = None
1050:         D = None
1051: 
1052:     fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped = wrap_functions(
1053:         fun, bc, fun_jac, bc_jac, k, a, S, D, dtype)
1054: 
1055:     f = fun_wrapped(x, y, p)
1056:     if f.shape != y.shape:
1057:         raise ValueError("`fun` return is expected to have shape {}, "
1058:                          "but actually has {}.".format(y.shape, f.shape))
1059: 
1060:     bc_res = bc_wrapped(y[:, 0], y[:, -1], p)
1061:     if bc_res.shape != (n + k,):
1062:         raise ValueError("`bc` return is expected to have shape {}, "
1063:                          "but actually has {}.".format((n + k,), bc_res.shape))
1064: 
1065:     status = 0
1066:     iteration = 0
1067:     if verbose == 2:
1068:         print_iteration_header()
1069: 
1070:     while True:
1071:         m = x.shape[0]
1072: 
1073:         col_fun, jac_sys = prepare_sys(n, m, k, fun_wrapped, bc_wrapped,
1074:                                        fun_jac_wrapped, bc_jac_wrapped, x, h)
1075:         y, p, singular = solve_newton(n, m, h, col_fun, bc_wrapped, jac_sys,
1076:                                       y, p, B, tol)
1077:         iteration += 1
1078: 
1079:         col_res, y_middle, f, f_middle = collocation_fun(fun_wrapped, y,
1080:                                                          p, x, h)
1081:         # This relation is not trivial, but can be verified.
1082:         r_middle = 1.5 * col_res / h
1083:         sol = create_spline(y, f, x, h)
1084:         rms_res = estimate_rms_residuals(fun_wrapped, sol, x, h, p,
1085:                                          r_middle, f_middle)
1086:         max_rms_res = np.max(rms_res)
1087: 
1088:         if singular:
1089:             status = 2
1090:             break
1091: 
1092:         insert_1, = np.nonzero((rms_res > tol) & (rms_res < 100 * tol))
1093:         insert_2, = np.nonzero(rms_res >= 100 * tol)
1094:         nodes_added = insert_1.shape[0] + 2 * insert_2.shape[0]
1095: 
1096:         if m + nodes_added > max_nodes:
1097:             status = 1
1098:             if verbose == 2:
1099:                 nodes_added = "({})".format(nodes_added)
1100:                 print_iteration_progress(iteration, max_rms_res, m,
1101:                                          nodes_added)
1102:             break
1103: 
1104:         if verbose == 2:
1105:             print_iteration_progress(iteration, max_rms_res, m, nodes_added)
1106: 
1107:         if nodes_added > 0:
1108:             x = modify_mesh(x, insert_1, insert_2)
1109:             h = np.diff(x)
1110:             y = sol(x)
1111:         else:
1112:             status = 0
1113:             break
1114: 
1115:     if verbose > 0:
1116:         if status == 0:
1117:             print("Solved in {} iterations, number of nodes {}, "
1118:                   "maximum relative residual {:.2e}."
1119:                   .format(iteration, x.shape[0], max_rms_res))
1120:         elif status == 1:
1121:             print("Number of nodes is exceeded after iteration {}, "
1122:                   "maximum relative residual {:.2e}."
1123:                   .format(iteration, max_rms_res))
1124:         elif status == 2:
1125:             print("Singular Jacobian encountered when solving the collocation "
1126:                   "system on iteration {}, maximum relative residual {:.2e}."
1127:                   .format(iteration, max_rms_res))
1128: 
1129:     if p.size == 0:
1130:         p = None
1131: 
1132:     return BVPResult(sol=sol, p=p, x=x, y=y, yp=f, rms_residuals=rms_res,
1133:                      niter=iteration, status=status,
1134:                      message=TERMINATION_MESSAGES[status], success=status == 0)
1135: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_32434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Boundary value problem solver.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from warnings import warn' statement (line 4)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_32435 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_32435) is not StypyTypeError):

    if (import_32435 != 'pyd_module'):
        __import__(import_32435)
        sys_modules_32436 = sys.modules[import_32435]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_32436.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_32435)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.linalg import norm, pinv' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_32437 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg')

if (type(import_32437) is not StypyTypeError):

    if (import_32437 != 'pyd_module'):
        __import__(import_32437)
        sys_modules_32438 = sys.modules[import_32437]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', sys_modules_32438.module_type_store, module_type_store, ['norm', 'pinv'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_32438, sys_modules_32438.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm, pinv

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', None, module_type_store, ['norm', 'pinv'], [norm, pinv])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', import_32437)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.sparse import coo_matrix, csc_matrix' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_32439 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse')

if (type(import_32439) is not StypyTypeError):

    if (import_32439 != 'pyd_module'):
        __import__(import_32439)
        sys_modules_32440 = sys.modules[import_32439]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse', sys_modules_32440.module_type_store, module_type_store, ['coo_matrix', 'csc_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_32440, sys_modules_32440.module_type_store, module_type_store)
    else:
        from scipy.sparse import coo_matrix, csc_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse', None, module_type_store, ['coo_matrix', 'csc_matrix'], [coo_matrix, csc_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse', import_32439)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.sparse.linalg import splu' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_32441 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg')

if (type(import_32441) is not StypyTypeError):

    if (import_32441 != 'pyd_module'):
        __import__(import_32441)
        sys_modules_32442 = sys.modules[import_32441]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', sys_modules_32442.module_type_store, module_type_store, ['splu'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_32442, sys_modules_32442.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import splu

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', None, module_type_store, ['splu'], [splu])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', import_32441)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.optimize import OptimizeResult' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_32443 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize')

if (type(import_32443) is not StypyTypeError):

    if (import_32443 != 'pyd_module'):
        __import__(import_32443)
        sys_modules_32444 = sys.modules[import_32443]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', sys_modules_32444.module_type_store, module_type_store, ['OptimizeResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_32444, sys_modules_32444.module_type_store, module_type_store)
    else:
        from scipy.optimize import OptimizeResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', None, module_type_store, ['OptimizeResult'], [OptimizeResult])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', import_32443)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')


# Assigning a Attribute to a Name (line 14):

# Assigning a Attribute to a Name (line 14):

# Call to finfo(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'float' (line 14)
float_32447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'float', False)
# Processing the call keyword arguments (line 14)
kwargs_32448 = {}
# Getting the type of 'np' (line 14)
np_32445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 6), 'np', False)
# Obtaining the member 'finfo' of a type (line 14)
finfo_32446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 6), np_32445, 'finfo')
# Calling finfo(args, kwargs) (line 14)
finfo_call_result_32449 = invoke(stypy.reporting.localization.Localization(__file__, 14, 6), finfo_32446, *[float_32447], **kwargs_32448)

# Obtaining the member 'eps' of a type (line 14)
eps_32450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 6), finfo_call_result_32449, 'eps')
# Assigning a type to the variable 'EPS' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'EPS', eps_32450)

@norecursion
def estimate_fun_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 17)
    None_32451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 38), 'None')
    defaults = [None_32451]
    # Create a new context for function 'estimate_fun_jac'
    module_type_store = module_type_store.open_function_context('estimate_fun_jac', 17, 0, False)
    
    # Passed parameters checking function
    estimate_fun_jac.stypy_localization = localization
    estimate_fun_jac.stypy_type_of_self = None
    estimate_fun_jac.stypy_type_store = module_type_store
    estimate_fun_jac.stypy_function_name = 'estimate_fun_jac'
    estimate_fun_jac.stypy_param_names_list = ['fun', 'x', 'y', 'p', 'f0']
    estimate_fun_jac.stypy_varargs_param_name = None
    estimate_fun_jac.stypy_kwargs_param_name = None
    estimate_fun_jac.stypy_call_defaults = defaults
    estimate_fun_jac.stypy_call_varargs = varargs
    estimate_fun_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'estimate_fun_jac', ['fun', 'x', 'y', 'p', 'f0'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'estimate_fun_jac', localization, ['fun', 'x', 'y', 'p', 'f0'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'estimate_fun_jac(...)' code ##################

    str_32452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, (-1)), 'str', 'Estimate derivatives of an ODE system rhs with forward differences.\n\n    Returns\n    -------\n    df_dy : ndarray, shape (n, n, m)\n        Derivatives with respect to y. An element (i, j, q) corresponds to\n        d f_i(x_q, y_q) / d (y_q)_j.\n    df_dp : ndarray with shape (n, k, m) or None\n        Derivatives with respect to p. An element (i, j, q) corresponds to\n        d f_i(x_q, y_q, p) / d p_j. If `p` is empty, None is returned.\n    ')
    
    # Assigning a Attribute to a Tuple (line 29):
    
    # Assigning a Subscript to a Name (line 29):
    
    # Obtaining the type of the subscript
    int_32453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'int')
    # Getting the type of 'y' (line 29)
    y_32454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'y')
    # Obtaining the member 'shape' of a type (line 29)
    shape_32455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 11), y_32454, 'shape')
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___32456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), shape_32455, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_32457 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), getitem___32456, int_32453)
    
    # Assigning a type to the variable 'tuple_var_assignment_32382' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_32382', subscript_call_result_32457)
    
    # Assigning a Subscript to a Name (line 29):
    
    # Obtaining the type of the subscript
    int_32458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'int')
    # Getting the type of 'y' (line 29)
    y_32459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'y')
    # Obtaining the member 'shape' of a type (line 29)
    shape_32460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 11), y_32459, 'shape')
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___32461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), shape_32460, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_32462 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), getitem___32461, int_32458)
    
    # Assigning a type to the variable 'tuple_var_assignment_32383' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_32383', subscript_call_result_32462)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'tuple_var_assignment_32382' (line 29)
    tuple_var_assignment_32382_32463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_32382')
    # Assigning a type to the variable 'n' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'n', tuple_var_assignment_32382_32463)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'tuple_var_assignment_32383' (line 29)
    tuple_var_assignment_32383_32464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_32383')
    # Assigning a type to the variable 'm' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 7), 'm', tuple_var_assignment_32383_32464)
    
    # Type idiom detected: calculating its left and rigth part (line 30)
    # Getting the type of 'f0' (line 30)
    f0_32465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 7), 'f0')
    # Getting the type of 'None' (line 30)
    None_32466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 13), 'None')
    
    (may_be_32467, more_types_in_union_32468) = may_be_none(f0_32465, None_32466)

    if may_be_32467:

        if more_types_in_union_32468:
            # Runtime conditional SSA (line 30)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 31):
        
        # Assigning a Call to a Name (line 31):
        
        # Call to fun(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'x' (line 31)
        x_32470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'x', False)
        # Getting the type of 'y' (line 31)
        y_32471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'y', False)
        # Getting the type of 'p' (line 31)
        p_32472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'p', False)
        # Processing the call keyword arguments (line 31)
        kwargs_32473 = {}
        # Getting the type of 'fun' (line 31)
        fun_32469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 'fun', False)
        # Calling fun(args, kwargs) (line 31)
        fun_call_result_32474 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), fun_32469, *[x_32470, y_32471, p_32472], **kwargs_32473)
        
        # Assigning a type to the variable 'f0' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'f0', fun_call_result_32474)

        if more_types_in_union_32468:
            # SSA join for if statement (line 30)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 33):
    
    # Assigning a Attribute to a Name (line 33):
    # Getting the type of 'y' (line 33)
    y_32475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'y')
    # Obtaining the member 'dtype' of a type (line 33)
    dtype_32476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), y_32475, 'dtype')
    # Assigning a type to the variable 'dtype' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'dtype', dtype_32476)
    
    # Assigning a Call to a Name (line 35):
    
    # Assigning a Call to a Name (line 35):
    
    # Call to empty(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Obtaining an instance of the builtin type 'tuple' (line 35)
    tuple_32479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 35)
    # Adding element type (line 35)
    # Getting the type of 'n' (line 35)
    n_32480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 22), tuple_32479, n_32480)
    # Adding element type (line 35)
    # Getting the type of 'n' (line 35)
    n_32481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 22), tuple_32479, n_32481)
    # Adding element type (line 35)
    # Getting the type of 'm' (line 35)
    m_32482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 28), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 22), tuple_32479, m_32482)
    
    # Processing the call keyword arguments (line 35)
    # Getting the type of 'dtype' (line 35)
    dtype_32483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 38), 'dtype', False)
    keyword_32484 = dtype_32483
    kwargs_32485 = {'dtype': keyword_32484}
    # Getting the type of 'np' (line 35)
    np_32477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'np', False)
    # Obtaining the member 'empty' of a type (line 35)
    empty_32478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), np_32477, 'empty')
    # Calling empty(args, kwargs) (line 35)
    empty_call_result_32486 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), empty_32478, *[tuple_32479], **kwargs_32485)
    
    # Assigning a type to the variable 'df_dy' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'df_dy', empty_call_result_32486)
    
    # Assigning a BinOp to a Name (line 36):
    
    # Assigning a BinOp to a Name (line 36):
    # Getting the type of 'EPS' (line 36)
    EPS_32487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'EPS')
    float_32488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'float')
    # Applying the binary operator '**' (line 36)
    result_pow_32489 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 8), '**', EPS_32487, float_32488)
    
    int_32490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 20), 'int')
    
    # Call to abs(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'y' (line 36)
    y_32493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'y', False)
    # Processing the call keyword arguments (line 36)
    kwargs_32494 = {}
    # Getting the type of 'np' (line 36)
    np_32491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'np', False)
    # Obtaining the member 'abs' of a type (line 36)
    abs_32492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 24), np_32491, 'abs')
    # Calling abs(args, kwargs) (line 36)
    abs_call_result_32495 = invoke(stypy.reporting.localization.Localization(__file__, 36, 24), abs_32492, *[y_32493], **kwargs_32494)
    
    # Applying the binary operator '+' (line 36)
    result_add_32496 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 20), '+', int_32490, abs_call_result_32495)
    
    # Applying the binary operator '*' (line 36)
    result_mul_32497 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 8), '*', result_pow_32489, result_add_32496)
    
    # Assigning a type to the variable 'h' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'h', result_mul_32497)
    
    
    # Call to range(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'n' (line 37)
    n_32499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), 'n', False)
    # Processing the call keyword arguments (line 37)
    kwargs_32500 = {}
    # Getting the type of 'range' (line 37)
    range_32498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'range', False)
    # Calling range(args, kwargs) (line 37)
    range_call_result_32501 = invoke(stypy.reporting.localization.Localization(__file__, 37, 13), range_32498, *[n_32499], **kwargs_32500)
    
    # Testing the type of a for loop iterable (line 37)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 37, 4), range_call_result_32501)
    # Getting the type of the for loop variable (line 37)
    for_loop_var_32502 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 37, 4), range_call_result_32501)
    # Assigning a type to the variable 'i' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'i', for_loop_var_32502)
    # SSA begins for a for statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 38):
    
    # Assigning a Call to a Name (line 38):
    
    # Call to copy(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_32505 = {}
    # Getting the type of 'y' (line 38)
    y_32503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'y', False)
    # Obtaining the member 'copy' of a type (line 38)
    copy_32504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 16), y_32503, 'copy')
    # Calling copy(args, kwargs) (line 38)
    copy_call_result_32506 = invoke(stypy.reporting.localization.Localization(__file__, 38, 16), copy_32504, *[], **kwargs_32505)
    
    # Assigning a type to the variable 'y_new' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'y_new', copy_call_result_32506)
    
    # Getting the type of 'y_new' (line 39)
    y_new_32507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'y_new')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 39)
    i_32508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'i')
    # Getting the type of 'y_new' (line 39)
    y_new_32509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'y_new')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___32510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), y_new_32509, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_32511 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), getitem___32510, i_32508)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 39)
    i_32512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 22), 'i')
    # Getting the type of 'h' (line 39)
    h_32513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'h')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___32514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), h_32513, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_32515 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), getitem___32514, i_32512)
    
    # Applying the binary operator '+=' (line 39)
    result_iadd_32516 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 8), '+=', subscript_call_result_32511, subscript_call_result_32515)
    # Getting the type of 'y_new' (line 39)
    y_new_32517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'y_new')
    # Getting the type of 'i' (line 39)
    i_32518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'i')
    # Storing an element on a container (line 39)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 8), y_new_32517, (i_32518, result_iadd_32516))
    
    
    # Assigning a BinOp to a Name (line 40):
    
    # Assigning a BinOp to a Name (line 40):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 40)
    i_32519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'i')
    # Getting the type of 'y_new' (line 40)
    y_new_32520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'y_new')
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___32521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 13), y_new_32520, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_32522 = invoke(stypy.reporting.localization.Localization(__file__, 40, 13), getitem___32521, i_32519)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 40)
    i_32523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'i')
    # Getting the type of 'y' (line 40)
    y_32524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'y')
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___32525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 24), y_32524, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_32526 = invoke(stypy.reporting.localization.Localization(__file__, 40, 24), getitem___32525, i_32523)
    
    # Applying the binary operator '-' (line 40)
    result_sub_32527 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 13), '-', subscript_call_result_32522, subscript_call_result_32526)
    
    # Assigning a type to the variable 'hi' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'hi', result_sub_32527)
    
    # Assigning a Call to a Name (line 41):
    
    # Assigning a Call to a Name (line 41):
    
    # Call to fun(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'x' (line 41)
    x_32529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'x', False)
    # Getting the type of 'y_new' (line 41)
    y_new_32530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'y_new', False)
    # Getting the type of 'p' (line 41)
    p_32531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 30), 'p', False)
    # Processing the call keyword arguments (line 41)
    kwargs_32532 = {}
    # Getting the type of 'fun' (line 41)
    fun_32528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'fun', False)
    # Calling fun(args, kwargs) (line 41)
    fun_call_result_32533 = invoke(stypy.reporting.localization.Localization(__file__, 41, 16), fun_32528, *[x_32529, y_new_32530, p_32531], **kwargs_32532)
    
    # Assigning a type to the variable 'f_new' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'f_new', fun_call_result_32533)
    
    # Assigning a BinOp to a Subscript (line 42):
    
    # Assigning a BinOp to a Subscript (line 42):
    # Getting the type of 'f_new' (line 42)
    f_new_32534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'f_new')
    # Getting the type of 'f0' (line 42)
    f0_32535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 34), 'f0')
    # Applying the binary operator '-' (line 42)
    result_sub_32536 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 26), '-', f_new_32534, f0_32535)
    
    # Getting the type of 'hi' (line 42)
    hi_32537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 40), 'hi')
    # Applying the binary operator 'div' (line 42)
    result_div_32538 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 25), 'div', result_sub_32536, hi_32537)
    
    # Getting the type of 'df_dy' (line 42)
    df_dy_32539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'df_dy')
    slice_32540 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 8), None, None, None)
    # Getting the type of 'i' (line 42)
    i_32541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'i')
    slice_32542 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 8), None, None, None)
    # Storing an element on a container (line 42)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 8), df_dy_32539, ((slice_32540, i_32541, slice_32542), result_div_32538))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 44):
    
    # Assigning a Subscript to a Name (line 44):
    
    # Obtaining the type of the subscript
    int_32543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'int')
    # Getting the type of 'p' (line 44)
    p_32544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'p')
    # Obtaining the member 'shape' of a type (line 44)
    shape_32545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), p_32544, 'shape')
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___32546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), shape_32545, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_32547 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), getitem___32546, int_32543)
    
    # Assigning a type to the variable 'k' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'k', subscript_call_result_32547)
    
    
    # Getting the type of 'k' (line 45)
    k_32548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 7), 'k')
    int_32549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 12), 'int')
    # Applying the binary operator '==' (line 45)
    result_eq_32550 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 7), '==', k_32548, int_32549)
    
    # Testing the type of an if condition (line 45)
    if_condition_32551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 4), result_eq_32550)
    # Assigning a type to the variable 'if_condition_32551' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'if_condition_32551', if_condition_32551)
    # SSA begins for if statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 46):
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'None' (line 46)
    None_32552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'None')
    # Assigning a type to the variable 'df_dp' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'df_dp', None_32552)
    # SSA branch for the else part of an if statement (line 45)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 48):
    
    # Assigning a Call to a Name (line 48):
    
    # Call to empty(...): (line 48)
    # Processing the call arguments (line 48)
    
    # Obtaining an instance of the builtin type 'tuple' (line 48)
    tuple_32555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 48)
    # Adding element type (line 48)
    # Getting the type of 'n' (line 48)
    n_32556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 26), tuple_32555, n_32556)
    # Adding element type (line 48)
    # Getting the type of 'k' (line 48)
    k_32557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 29), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 26), tuple_32555, k_32557)
    # Adding element type (line 48)
    # Getting the type of 'm' (line 48)
    m_32558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 32), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 26), tuple_32555, m_32558)
    
    # Processing the call keyword arguments (line 48)
    # Getting the type of 'dtype' (line 48)
    dtype_32559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 42), 'dtype', False)
    keyword_32560 = dtype_32559
    kwargs_32561 = {'dtype': keyword_32560}
    # Getting the type of 'np' (line 48)
    np_32553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'np', False)
    # Obtaining the member 'empty' of a type (line 48)
    empty_32554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), np_32553, 'empty')
    # Calling empty(args, kwargs) (line 48)
    empty_call_result_32562 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), empty_32554, *[tuple_32555], **kwargs_32561)
    
    # Assigning a type to the variable 'df_dp' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'df_dp', empty_call_result_32562)
    
    # Assigning a BinOp to a Name (line 49):
    
    # Assigning a BinOp to a Name (line 49):
    # Getting the type of 'EPS' (line 49)
    EPS_32563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'EPS')
    float_32564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 17), 'float')
    # Applying the binary operator '**' (line 49)
    result_pow_32565 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 12), '**', EPS_32563, float_32564)
    
    int_32566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 24), 'int')
    
    # Call to abs(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'p' (line 49)
    p_32569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 35), 'p', False)
    # Processing the call keyword arguments (line 49)
    kwargs_32570 = {}
    # Getting the type of 'np' (line 49)
    np_32567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 28), 'np', False)
    # Obtaining the member 'abs' of a type (line 49)
    abs_32568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 28), np_32567, 'abs')
    # Calling abs(args, kwargs) (line 49)
    abs_call_result_32571 = invoke(stypy.reporting.localization.Localization(__file__, 49, 28), abs_32568, *[p_32569], **kwargs_32570)
    
    # Applying the binary operator '+' (line 49)
    result_add_32572 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 24), '+', int_32566, abs_call_result_32571)
    
    # Applying the binary operator '*' (line 49)
    result_mul_32573 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 12), '*', result_pow_32565, result_add_32572)
    
    # Assigning a type to the variable 'h' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'h', result_mul_32573)
    
    
    # Call to range(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'k' (line 50)
    k_32575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 23), 'k', False)
    # Processing the call keyword arguments (line 50)
    kwargs_32576 = {}
    # Getting the type of 'range' (line 50)
    range_32574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'range', False)
    # Calling range(args, kwargs) (line 50)
    range_call_result_32577 = invoke(stypy.reporting.localization.Localization(__file__, 50, 17), range_32574, *[k_32575], **kwargs_32576)
    
    # Testing the type of a for loop iterable (line 50)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 8), range_call_result_32577)
    # Getting the type of the for loop variable (line 50)
    for_loop_var_32578 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 8), range_call_result_32577)
    # Assigning a type to the variable 'i' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'i', for_loop_var_32578)
    # SSA begins for a for statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 51):
    
    # Assigning a Call to a Name (line 51):
    
    # Call to copy(...): (line 51)
    # Processing the call keyword arguments (line 51)
    kwargs_32581 = {}
    # Getting the type of 'p' (line 51)
    p_32579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'p', False)
    # Obtaining the member 'copy' of a type (line 51)
    copy_32580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 20), p_32579, 'copy')
    # Calling copy(args, kwargs) (line 51)
    copy_call_result_32582 = invoke(stypy.reporting.localization.Localization(__file__, 51, 20), copy_32580, *[], **kwargs_32581)
    
    # Assigning a type to the variable 'p_new' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'p_new', copy_call_result_32582)
    
    # Getting the type of 'p_new' (line 52)
    p_new_32583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'p_new')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 52)
    i_32584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'i')
    # Getting the type of 'p_new' (line 52)
    p_new_32585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'p_new')
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___32586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), p_new_32585, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 52)
    subscript_call_result_32587 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), getitem___32586, i_32584)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 52)
    i_32588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 26), 'i')
    # Getting the type of 'h' (line 52)
    h_32589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 24), 'h')
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___32590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 24), h_32589, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 52)
    subscript_call_result_32591 = invoke(stypy.reporting.localization.Localization(__file__, 52, 24), getitem___32590, i_32588)
    
    # Applying the binary operator '+=' (line 52)
    result_iadd_32592 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 12), '+=', subscript_call_result_32587, subscript_call_result_32591)
    # Getting the type of 'p_new' (line 52)
    p_new_32593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'p_new')
    # Getting the type of 'i' (line 52)
    i_32594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'i')
    # Storing an element on a container (line 52)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), p_new_32593, (i_32594, result_iadd_32592))
    
    
    # Assigning a BinOp to a Name (line 53):
    
    # Assigning a BinOp to a Name (line 53):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 53)
    i_32595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 23), 'i')
    # Getting the type of 'p_new' (line 53)
    p_new_32596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'p_new')
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___32597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 17), p_new_32596, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_32598 = invoke(stypy.reporting.localization.Localization(__file__, 53, 17), getitem___32597, i_32595)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 53)
    i_32599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 30), 'i')
    # Getting the type of 'p' (line 53)
    p_32600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 28), 'p')
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___32601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 28), p_32600, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_32602 = invoke(stypy.reporting.localization.Localization(__file__, 53, 28), getitem___32601, i_32599)
    
    # Applying the binary operator '-' (line 53)
    result_sub_32603 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 17), '-', subscript_call_result_32598, subscript_call_result_32602)
    
    # Assigning a type to the variable 'hi' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'hi', result_sub_32603)
    
    # Assigning a Call to a Name (line 54):
    
    # Assigning a Call to a Name (line 54):
    
    # Call to fun(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'x' (line 54)
    x_32605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'x', False)
    # Getting the type of 'y' (line 54)
    y_32606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 27), 'y', False)
    # Getting the type of 'p_new' (line 54)
    p_new_32607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 30), 'p_new', False)
    # Processing the call keyword arguments (line 54)
    kwargs_32608 = {}
    # Getting the type of 'fun' (line 54)
    fun_32604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'fun', False)
    # Calling fun(args, kwargs) (line 54)
    fun_call_result_32609 = invoke(stypy.reporting.localization.Localization(__file__, 54, 20), fun_32604, *[x_32605, y_32606, p_new_32607], **kwargs_32608)
    
    # Assigning a type to the variable 'f_new' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'f_new', fun_call_result_32609)
    
    # Assigning a BinOp to a Subscript (line 55):
    
    # Assigning a BinOp to a Subscript (line 55):
    # Getting the type of 'f_new' (line 55)
    f_new_32610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 30), 'f_new')
    # Getting the type of 'f0' (line 55)
    f0_32611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 38), 'f0')
    # Applying the binary operator '-' (line 55)
    result_sub_32612 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 30), '-', f_new_32610, f0_32611)
    
    # Getting the type of 'hi' (line 55)
    hi_32613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 44), 'hi')
    # Applying the binary operator 'div' (line 55)
    result_div_32614 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 29), 'div', result_sub_32612, hi_32613)
    
    # Getting the type of 'df_dp' (line 55)
    df_dp_32615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'df_dp')
    slice_32616 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 55, 12), None, None, None)
    # Getting the type of 'i' (line 55)
    i_32617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'i')
    slice_32618 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 55, 12), None, None, None)
    # Storing an element on a container (line 55)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 12), df_dp_32615, ((slice_32616, i_32617, slice_32618), result_div_32614))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 45)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 57)
    tuple_32619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 57)
    # Adding element type (line 57)
    # Getting the type of 'df_dy' (line 57)
    df_dy_32620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'df_dy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 11), tuple_32619, df_dy_32620)
    # Adding element type (line 57)
    # Getting the type of 'df_dp' (line 57)
    df_dp_32621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'df_dp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 11), tuple_32619, df_dp_32621)
    
    # Assigning a type to the variable 'stypy_return_type' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type', tuple_32619)
    
    # ################# End of 'estimate_fun_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'estimate_fun_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_32622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32622)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'estimate_fun_jac'
    return stypy_return_type_32622

# Assigning a type to the variable 'estimate_fun_jac' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'estimate_fun_jac', estimate_fun_jac)

@norecursion
def estimate_bc_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 60)
    None_32623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 39), 'None')
    defaults = [None_32623]
    # Create a new context for function 'estimate_bc_jac'
    module_type_store = module_type_store.open_function_context('estimate_bc_jac', 60, 0, False)
    
    # Passed parameters checking function
    estimate_bc_jac.stypy_localization = localization
    estimate_bc_jac.stypy_type_of_self = None
    estimate_bc_jac.stypy_type_store = module_type_store
    estimate_bc_jac.stypy_function_name = 'estimate_bc_jac'
    estimate_bc_jac.stypy_param_names_list = ['bc', 'ya', 'yb', 'p', 'bc0']
    estimate_bc_jac.stypy_varargs_param_name = None
    estimate_bc_jac.stypy_kwargs_param_name = None
    estimate_bc_jac.stypy_call_defaults = defaults
    estimate_bc_jac.stypy_call_varargs = varargs
    estimate_bc_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'estimate_bc_jac', ['bc', 'ya', 'yb', 'p', 'bc0'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'estimate_bc_jac', localization, ['bc', 'ya', 'yb', 'p', 'bc0'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'estimate_bc_jac(...)' code ##################

    str_32624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, (-1)), 'str', 'Estimate derivatives of boundary conditions with forward differences.\n\n    Returns\n    -------\n    dbc_dya : ndarray, shape (n + k, n)\n        Derivatives with respect to ya. An element (i, j) corresponds to\n        d bc_i / d ya_j.\n    dbc_dyb : ndarray, shape (n + k, n)\n        Derivatives with respect to yb. An element (i, j) corresponds to\n        d bc_i / d ya_j.\n    dbc_dp : ndarray with shape (n + k, k) or None\n        Derivatives with respect to p. An element (i, j) corresponds to\n        d bc_i / d p_j. If `p` is empty, None is returned.\n    ')
    
    # Assigning a Subscript to a Name (line 75):
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    int_32625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 17), 'int')
    # Getting the type of 'ya' (line 75)
    ya_32626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'ya')
    # Obtaining the member 'shape' of a type (line 75)
    shape_32627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), ya_32626, 'shape')
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___32628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), shape_32627, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_32629 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), getitem___32628, int_32625)
    
    # Assigning a type to the variable 'n' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'n', subscript_call_result_32629)
    
    # Assigning a Subscript to a Name (line 76):
    
    # Assigning a Subscript to a Name (line 76):
    
    # Obtaining the type of the subscript
    int_32630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 16), 'int')
    # Getting the type of 'p' (line 76)
    p_32631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'p')
    # Obtaining the member 'shape' of a type (line 76)
    shape_32632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), p_32631, 'shape')
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___32633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), shape_32632, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_32634 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), getitem___32633, int_32630)
    
    # Assigning a type to the variable 'k' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'k', subscript_call_result_32634)
    
    # Type idiom detected: calculating its left and rigth part (line 78)
    # Getting the type of 'bc0' (line 78)
    bc0_32635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 7), 'bc0')
    # Getting the type of 'None' (line 78)
    None_32636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 14), 'None')
    
    (may_be_32637, more_types_in_union_32638) = may_be_none(bc0_32635, None_32636)

    if may_be_32637:

        if more_types_in_union_32638:
            # Runtime conditional SSA (line 78)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to bc(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'ya' (line 79)
        ya_32640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'ya', False)
        # Getting the type of 'yb' (line 79)
        yb_32641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'yb', False)
        # Getting the type of 'p' (line 79)
        p_32642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'p', False)
        # Processing the call keyword arguments (line 79)
        kwargs_32643 = {}
        # Getting the type of 'bc' (line 79)
        bc_32639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 14), 'bc', False)
        # Calling bc(args, kwargs) (line 79)
        bc_call_result_32644 = invoke(stypy.reporting.localization.Localization(__file__, 79, 14), bc_32639, *[ya_32640, yb_32641, p_32642], **kwargs_32643)
        
        # Assigning a type to the variable 'bc0' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'bc0', bc_call_result_32644)

        if more_types_in_union_32638:
            # SSA join for if statement (line 78)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 81):
    
    # Assigning a Attribute to a Name (line 81):
    # Getting the type of 'ya' (line 81)
    ya_32645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'ya')
    # Obtaining the member 'dtype' of a type (line 81)
    dtype_32646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), ya_32645, 'dtype')
    # Assigning a type to the variable 'dtype' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'dtype', dtype_32646)
    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to empty(...): (line 83)
    # Processing the call arguments (line 83)
    
    # Obtaining an instance of the builtin type 'tuple' (line 83)
    tuple_32649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 83)
    # Adding element type (line 83)
    # Getting the type of 'n' (line 83)
    n_32650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 24), tuple_32649, n_32650)
    # Adding element type (line 83)
    # Getting the type of 'n' (line 83)
    n_32651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'n', False)
    # Getting the type of 'k' (line 83)
    k_32652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 31), 'k', False)
    # Applying the binary operator '+' (line 83)
    result_add_32653 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 27), '+', n_32651, k_32652)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 24), tuple_32649, result_add_32653)
    
    # Processing the call keyword arguments (line 83)
    # Getting the type of 'dtype' (line 83)
    dtype_32654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 41), 'dtype', False)
    keyword_32655 = dtype_32654
    kwargs_32656 = {'dtype': keyword_32655}
    # Getting the type of 'np' (line 83)
    np_32647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 14), 'np', False)
    # Obtaining the member 'empty' of a type (line 83)
    empty_32648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 14), np_32647, 'empty')
    # Calling empty(args, kwargs) (line 83)
    empty_call_result_32657 = invoke(stypy.reporting.localization.Localization(__file__, 83, 14), empty_32648, *[tuple_32649], **kwargs_32656)
    
    # Assigning a type to the variable 'dbc_dya' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'dbc_dya', empty_call_result_32657)
    
    # Assigning a BinOp to a Name (line 84):
    
    # Assigning a BinOp to a Name (line 84):
    # Getting the type of 'EPS' (line 84)
    EPS_32658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'EPS')
    float_32659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 13), 'float')
    # Applying the binary operator '**' (line 84)
    result_pow_32660 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 8), '**', EPS_32658, float_32659)
    
    int_32661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 20), 'int')
    
    # Call to abs(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'ya' (line 84)
    ya_32664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'ya', False)
    # Processing the call keyword arguments (line 84)
    kwargs_32665 = {}
    # Getting the type of 'np' (line 84)
    np_32662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 24), 'np', False)
    # Obtaining the member 'abs' of a type (line 84)
    abs_32663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 24), np_32662, 'abs')
    # Calling abs(args, kwargs) (line 84)
    abs_call_result_32666 = invoke(stypy.reporting.localization.Localization(__file__, 84, 24), abs_32663, *[ya_32664], **kwargs_32665)
    
    # Applying the binary operator '+' (line 84)
    result_add_32667 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 20), '+', int_32661, abs_call_result_32666)
    
    # Applying the binary operator '*' (line 84)
    result_mul_32668 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 8), '*', result_pow_32660, result_add_32667)
    
    # Assigning a type to the variable 'h' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'h', result_mul_32668)
    
    
    # Call to range(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'n' (line 85)
    n_32670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 19), 'n', False)
    # Processing the call keyword arguments (line 85)
    kwargs_32671 = {}
    # Getting the type of 'range' (line 85)
    range_32669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'range', False)
    # Calling range(args, kwargs) (line 85)
    range_call_result_32672 = invoke(stypy.reporting.localization.Localization(__file__, 85, 13), range_32669, *[n_32670], **kwargs_32671)
    
    # Testing the type of a for loop iterable (line 85)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 85, 4), range_call_result_32672)
    # Getting the type of the for loop variable (line 85)
    for_loop_var_32673 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 85, 4), range_call_result_32672)
    # Assigning a type to the variable 'i' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'i', for_loop_var_32673)
    # SSA begins for a for statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to copy(...): (line 86)
    # Processing the call keyword arguments (line 86)
    kwargs_32676 = {}
    # Getting the type of 'ya' (line 86)
    ya_32674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 17), 'ya', False)
    # Obtaining the member 'copy' of a type (line 86)
    copy_32675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 17), ya_32674, 'copy')
    # Calling copy(args, kwargs) (line 86)
    copy_call_result_32677 = invoke(stypy.reporting.localization.Localization(__file__, 86, 17), copy_32675, *[], **kwargs_32676)
    
    # Assigning a type to the variable 'ya_new' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'ya_new', copy_call_result_32677)
    
    # Getting the type of 'ya_new' (line 87)
    ya_new_32678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'ya_new')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 87)
    i_32679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'i')
    # Getting the type of 'ya_new' (line 87)
    ya_new_32680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'ya_new')
    # Obtaining the member '__getitem__' of a type (line 87)
    getitem___32681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), ya_new_32680, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 87)
    subscript_call_result_32682 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), getitem___32681, i_32679)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 87)
    i_32683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'i')
    # Getting the type of 'h' (line 87)
    h_32684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 21), 'h')
    # Obtaining the member '__getitem__' of a type (line 87)
    getitem___32685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 21), h_32684, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 87)
    subscript_call_result_32686 = invoke(stypy.reporting.localization.Localization(__file__, 87, 21), getitem___32685, i_32683)
    
    # Applying the binary operator '+=' (line 87)
    result_iadd_32687 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 8), '+=', subscript_call_result_32682, subscript_call_result_32686)
    # Getting the type of 'ya_new' (line 87)
    ya_new_32688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'ya_new')
    # Getting the type of 'i' (line 87)
    i_32689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'i')
    # Storing an element on a container (line 87)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), ya_new_32688, (i_32689, result_iadd_32687))
    
    
    # Assigning a BinOp to a Name (line 88):
    
    # Assigning a BinOp to a Name (line 88):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 88)
    i_32690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'i')
    # Getting the type of 'ya_new' (line 88)
    ya_new_32691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'ya_new')
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___32692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 13), ya_new_32691, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_32693 = invoke(stypy.reporting.localization.Localization(__file__, 88, 13), getitem___32692, i_32690)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 88)
    i_32694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 28), 'i')
    # Getting the type of 'ya' (line 88)
    ya_32695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 25), 'ya')
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___32696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 25), ya_32695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_32697 = invoke(stypy.reporting.localization.Localization(__file__, 88, 25), getitem___32696, i_32694)
    
    # Applying the binary operator '-' (line 88)
    result_sub_32698 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), '-', subscript_call_result_32693, subscript_call_result_32697)
    
    # Assigning a type to the variable 'hi' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'hi', result_sub_32698)
    
    # Assigning a Call to a Name (line 89):
    
    # Assigning a Call to a Name (line 89):
    
    # Call to bc(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'ya_new' (line 89)
    ya_new_32700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'ya_new', False)
    # Getting the type of 'yb' (line 89)
    yb_32701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'yb', False)
    # Getting the type of 'p' (line 89)
    p_32702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 32), 'p', False)
    # Processing the call keyword arguments (line 89)
    kwargs_32703 = {}
    # Getting the type of 'bc' (line 89)
    bc_32699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'bc', False)
    # Calling bc(args, kwargs) (line 89)
    bc_call_result_32704 = invoke(stypy.reporting.localization.Localization(__file__, 89, 17), bc_32699, *[ya_new_32700, yb_32701, p_32702], **kwargs_32703)
    
    # Assigning a type to the variable 'bc_new' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'bc_new', bc_call_result_32704)
    
    # Assigning a BinOp to a Subscript (line 90):
    
    # Assigning a BinOp to a Subscript (line 90):
    # Getting the type of 'bc_new' (line 90)
    bc_new_32705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 22), 'bc_new')
    # Getting the type of 'bc0' (line 90)
    bc0_32706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 31), 'bc0')
    # Applying the binary operator '-' (line 90)
    result_sub_32707 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 22), '-', bc_new_32705, bc0_32706)
    
    # Getting the type of 'hi' (line 90)
    hi_32708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 38), 'hi')
    # Applying the binary operator 'div' (line 90)
    result_div_32709 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), 'div', result_sub_32707, hi_32708)
    
    # Getting the type of 'dbc_dya' (line 90)
    dbc_dya_32710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'dbc_dya')
    # Getting the type of 'i' (line 90)
    i_32711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'i')
    # Storing an element on a container (line 90)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), dbc_dya_32710, (i_32711, result_div_32709))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 91):
    
    # Assigning a Attribute to a Name (line 91):
    # Getting the type of 'dbc_dya' (line 91)
    dbc_dya_32712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 14), 'dbc_dya')
    # Obtaining the member 'T' of a type (line 91)
    T_32713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 14), dbc_dya_32712, 'T')
    # Assigning a type to the variable 'dbc_dya' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'dbc_dya', T_32713)
    
    # Assigning a BinOp to a Name (line 93):
    
    # Assigning a BinOp to a Name (line 93):
    # Getting the type of 'EPS' (line 93)
    EPS_32714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'EPS')
    float_32715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 13), 'float')
    # Applying the binary operator '**' (line 93)
    result_pow_32716 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 8), '**', EPS_32714, float_32715)
    
    int_32717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'int')
    
    # Call to abs(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'yb' (line 93)
    yb_32720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'yb', False)
    # Processing the call keyword arguments (line 93)
    kwargs_32721 = {}
    # Getting the type of 'np' (line 93)
    np_32718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'np', False)
    # Obtaining the member 'abs' of a type (line 93)
    abs_32719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 24), np_32718, 'abs')
    # Calling abs(args, kwargs) (line 93)
    abs_call_result_32722 = invoke(stypy.reporting.localization.Localization(__file__, 93, 24), abs_32719, *[yb_32720], **kwargs_32721)
    
    # Applying the binary operator '+' (line 93)
    result_add_32723 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 20), '+', int_32717, abs_call_result_32722)
    
    # Applying the binary operator '*' (line 93)
    result_mul_32724 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 8), '*', result_pow_32716, result_add_32723)
    
    # Assigning a type to the variable 'h' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'h', result_mul_32724)
    
    # Assigning a Call to a Name (line 94):
    
    # Assigning a Call to a Name (line 94):
    
    # Call to empty(...): (line 94)
    # Processing the call arguments (line 94)
    
    # Obtaining an instance of the builtin type 'tuple' (line 94)
    tuple_32727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 94)
    # Adding element type (line 94)
    # Getting the type of 'n' (line 94)
    n_32728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 24), tuple_32727, n_32728)
    # Adding element type (line 94)
    # Getting the type of 'n' (line 94)
    n_32729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'n', False)
    # Getting the type of 'k' (line 94)
    k_32730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 31), 'k', False)
    # Applying the binary operator '+' (line 94)
    result_add_32731 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 27), '+', n_32729, k_32730)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 24), tuple_32727, result_add_32731)
    
    # Processing the call keyword arguments (line 94)
    # Getting the type of 'dtype' (line 94)
    dtype_32732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 41), 'dtype', False)
    keyword_32733 = dtype_32732
    kwargs_32734 = {'dtype': keyword_32733}
    # Getting the type of 'np' (line 94)
    np_32725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'np', False)
    # Obtaining the member 'empty' of a type (line 94)
    empty_32726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 14), np_32725, 'empty')
    # Calling empty(args, kwargs) (line 94)
    empty_call_result_32735 = invoke(stypy.reporting.localization.Localization(__file__, 94, 14), empty_32726, *[tuple_32727], **kwargs_32734)
    
    # Assigning a type to the variable 'dbc_dyb' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'dbc_dyb', empty_call_result_32735)
    
    
    # Call to range(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'n' (line 95)
    n_32737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'n', False)
    # Processing the call keyword arguments (line 95)
    kwargs_32738 = {}
    # Getting the type of 'range' (line 95)
    range_32736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'range', False)
    # Calling range(args, kwargs) (line 95)
    range_call_result_32739 = invoke(stypy.reporting.localization.Localization(__file__, 95, 13), range_32736, *[n_32737], **kwargs_32738)
    
    # Testing the type of a for loop iterable (line 95)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 95, 4), range_call_result_32739)
    # Getting the type of the for loop variable (line 95)
    for_loop_var_32740 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 95, 4), range_call_result_32739)
    # Assigning a type to the variable 'i' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'i', for_loop_var_32740)
    # SSA begins for a for statement (line 95)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to copy(...): (line 96)
    # Processing the call keyword arguments (line 96)
    kwargs_32743 = {}
    # Getting the type of 'yb' (line 96)
    yb_32741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'yb', False)
    # Obtaining the member 'copy' of a type (line 96)
    copy_32742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 17), yb_32741, 'copy')
    # Calling copy(args, kwargs) (line 96)
    copy_call_result_32744 = invoke(stypy.reporting.localization.Localization(__file__, 96, 17), copy_32742, *[], **kwargs_32743)
    
    # Assigning a type to the variable 'yb_new' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'yb_new', copy_call_result_32744)
    
    # Getting the type of 'yb_new' (line 97)
    yb_new_32745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'yb_new')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 97)
    i_32746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'i')
    # Getting the type of 'yb_new' (line 97)
    yb_new_32747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'yb_new')
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___32748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), yb_new_32747, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_32749 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), getitem___32748, i_32746)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 97)
    i_32750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'i')
    # Getting the type of 'h' (line 97)
    h_32751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 'h')
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___32752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 21), h_32751, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_32753 = invoke(stypy.reporting.localization.Localization(__file__, 97, 21), getitem___32752, i_32750)
    
    # Applying the binary operator '+=' (line 97)
    result_iadd_32754 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 8), '+=', subscript_call_result_32749, subscript_call_result_32753)
    # Getting the type of 'yb_new' (line 97)
    yb_new_32755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'yb_new')
    # Getting the type of 'i' (line 97)
    i_32756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'i')
    # Storing an element on a container (line 97)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 8), yb_new_32755, (i_32756, result_iadd_32754))
    
    
    # Assigning a BinOp to a Name (line 98):
    
    # Assigning a BinOp to a Name (line 98):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 98)
    i_32757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'i')
    # Getting the type of 'yb_new' (line 98)
    yb_new_32758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'yb_new')
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___32759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 13), yb_new_32758, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_32760 = invoke(stypy.reporting.localization.Localization(__file__, 98, 13), getitem___32759, i_32757)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 98)
    i_32761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'i')
    # Getting the type of 'yb' (line 98)
    yb_32762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'yb')
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___32763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 25), yb_32762, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_32764 = invoke(stypy.reporting.localization.Localization(__file__, 98, 25), getitem___32763, i_32761)
    
    # Applying the binary operator '-' (line 98)
    result_sub_32765 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 13), '-', subscript_call_result_32760, subscript_call_result_32764)
    
    # Assigning a type to the variable 'hi' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'hi', result_sub_32765)
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to bc(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'ya' (line 99)
    ya_32767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'ya', False)
    # Getting the type of 'yb_new' (line 99)
    yb_new_32768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'yb_new', False)
    # Getting the type of 'p' (line 99)
    p_32769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 32), 'p', False)
    # Processing the call keyword arguments (line 99)
    kwargs_32770 = {}
    # Getting the type of 'bc' (line 99)
    bc_32766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'bc', False)
    # Calling bc(args, kwargs) (line 99)
    bc_call_result_32771 = invoke(stypy.reporting.localization.Localization(__file__, 99, 17), bc_32766, *[ya_32767, yb_new_32768, p_32769], **kwargs_32770)
    
    # Assigning a type to the variable 'bc_new' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'bc_new', bc_call_result_32771)
    
    # Assigning a BinOp to a Subscript (line 100):
    
    # Assigning a BinOp to a Subscript (line 100):
    # Getting the type of 'bc_new' (line 100)
    bc_new_32772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'bc_new')
    # Getting the type of 'bc0' (line 100)
    bc0_32773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 31), 'bc0')
    # Applying the binary operator '-' (line 100)
    result_sub_32774 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 22), '-', bc_new_32772, bc0_32773)
    
    # Getting the type of 'hi' (line 100)
    hi_32775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 38), 'hi')
    # Applying the binary operator 'div' (line 100)
    result_div_32776 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 21), 'div', result_sub_32774, hi_32775)
    
    # Getting the type of 'dbc_dyb' (line 100)
    dbc_dyb_32777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'dbc_dyb')
    # Getting the type of 'i' (line 100)
    i_32778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'i')
    # Storing an element on a container (line 100)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 8), dbc_dyb_32777, (i_32778, result_div_32776))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 101):
    
    # Assigning a Attribute to a Name (line 101):
    # Getting the type of 'dbc_dyb' (line 101)
    dbc_dyb_32779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 14), 'dbc_dyb')
    # Obtaining the member 'T' of a type (line 101)
    T_32780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 14), dbc_dyb_32779, 'T')
    # Assigning a type to the variable 'dbc_dyb' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'dbc_dyb', T_32780)
    
    
    # Getting the type of 'k' (line 103)
    k_32781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 7), 'k')
    int_32782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 12), 'int')
    # Applying the binary operator '==' (line 103)
    result_eq_32783 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 7), '==', k_32781, int_32782)
    
    # Testing the type of an if condition (line 103)
    if_condition_32784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 4), result_eq_32783)
    # Assigning a type to the variable 'if_condition_32784' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'if_condition_32784', if_condition_32784)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 104):
    
    # Assigning a Name to a Name (line 104):
    # Getting the type of 'None' (line 104)
    None_32785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'None')
    # Assigning a type to the variable 'dbc_dp' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'dbc_dp', None_32785)
    # SSA branch for the else part of an if statement (line 103)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 106):
    
    # Assigning a BinOp to a Name (line 106):
    # Getting the type of 'EPS' (line 106)
    EPS_32786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'EPS')
    float_32787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 17), 'float')
    # Applying the binary operator '**' (line 106)
    result_pow_32788 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 12), '**', EPS_32786, float_32787)
    
    int_32789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'int')
    
    # Call to abs(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'p' (line 106)
    p_32792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 35), 'p', False)
    # Processing the call keyword arguments (line 106)
    kwargs_32793 = {}
    # Getting the type of 'np' (line 106)
    np_32790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 28), 'np', False)
    # Obtaining the member 'abs' of a type (line 106)
    abs_32791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 28), np_32790, 'abs')
    # Calling abs(args, kwargs) (line 106)
    abs_call_result_32794 = invoke(stypy.reporting.localization.Localization(__file__, 106, 28), abs_32791, *[p_32792], **kwargs_32793)
    
    # Applying the binary operator '+' (line 106)
    result_add_32795 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 24), '+', int_32789, abs_call_result_32794)
    
    # Applying the binary operator '*' (line 106)
    result_mul_32796 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 12), '*', result_pow_32788, result_add_32795)
    
    # Assigning a type to the variable 'h' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'h', result_mul_32796)
    
    # Assigning a Call to a Name (line 107):
    
    # Assigning a Call to a Name (line 107):
    
    # Call to empty(...): (line 107)
    # Processing the call arguments (line 107)
    
    # Obtaining an instance of the builtin type 'tuple' (line 107)
    tuple_32799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 107)
    # Adding element type (line 107)
    # Getting the type of 'k' (line 107)
    k_32800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 27), tuple_32799, k_32800)
    # Adding element type (line 107)
    # Getting the type of 'n' (line 107)
    n_32801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), 'n', False)
    # Getting the type of 'k' (line 107)
    k_32802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 34), 'k', False)
    # Applying the binary operator '+' (line 107)
    result_add_32803 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 30), '+', n_32801, k_32802)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 27), tuple_32799, result_add_32803)
    
    # Processing the call keyword arguments (line 107)
    # Getting the type of 'dtype' (line 107)
    dtype_32804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 44), 'dtype', False)
    keyword_32805 = dtype_32804
    kwargs_32806 = {'dtype': keyword_32805}
    # Getting the type of 'np' (line 107)
    np_32797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'np', False)
    # Obtaining the member 'empty' of a type (line 107)
    empty_32798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 17), np_32797, 'empty')
    # Calling empty(args, kwargs) (line 107)
    empty_call_result_32807 = invoke(stypy.reporting.localization.Localization(__file__, 107, 17), empty_32798, *[tuple_32799], **kwargs_32806)
    
    # Assigning a type to the variable 'dbc_dp' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'dbc_dp', empty_call_result_32807)
    
    
    # Call to range(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'k' (line 108)
    k_32809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'k', False)
    # Processing the call keyword arguments (line 108)
    kwargs_32810 = {}
    # Getting the type of 'range' (line 108)
    range_32808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 17), 'range', False)
    # Calling range(args, kwargs) (line 108)
    range_call_result_32811 = invoke(stypy.reporting.localization.Localization(__file__, 108, 17), range_32808, *[k_32809], **kwargs_32810)
    
    # Testing the type of a for loop iterable (line 108)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 8), range_call_result_32811)
    # Getting the type of the for loop variable (line 108)
    for_loop_var_32812 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 8), range_call_result_32811)
    # Assigning a type to the variable 'i' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'i', for_loop_var_32812)
    # SSA begins for a for statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 109):
    
    # Assigning a Call to a Name (line 109):
    
    # Call to copy(...): (line 109)
    # Processing the call keyword arguments (line 109)
    kwargs_32815 = {}
    # Getting the type of 'p' (line 109)
    p_32813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), 'p', False)
    # Obtaining the member 'copy' of a type (line 109)
    copy_32814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 20), p_32813, 'copy')
    # Calling copy(args, kwargs) (line 109)
    copy_call_result_32816 = invoke(stypy.reporting.localization.Localization(__file__, 109, 20), copy_32814, *[], **kwargs_32815)
    
    # Assigning a type to the variable 'p_new' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'p_new', copy_call_result_32816)
    
    # Getting the type of 'p_new' (line 110)
    p_new_32817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'p_new')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 110)
    i_32818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 18), 'i')
    # Getting the type of 'p_new' (line 110)
    p_new_32819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'p_new')
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___32820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), p_new_32819, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_32821 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), getitem___32820, i_32818)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 110)
    i_32822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 26), 'i')
    # Getting the type of 'h' (line 110)
    h_32823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 24), 'h')
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___32824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 24), h_32823, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_32825 = invoke(stypy.reporting.localization.Localization(__file__, 110, 24), getitem___32824, i_32822)
    
    # Applying the binary operator '+=' (line 110)
    result_iadd_32826 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 12), '+=', subscript_call_result_32821, subscript_call_result_32825)
    # Getting the type of 'p_new' (line 110)
    p_new_32827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'p_new')
    # Getting the type of 'i' (line 110)
    i_32828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 18), 'i')
    # Storing an element on a container (line 110)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 12), p_new_32827, (i_32828, result_iadd_32826))
    
    
    # Assigning a BinOp to a Name (line 111):
    
    # Assigning a BinOp to a Name (line 111):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 111)
    i_32829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'i')
    # Getting the type of 'p_new' (line 111)
    p_new_32830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 17), 'p_new')
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___32831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 17), p_new_32830, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_32832 = invoke(stypy.reporting.localization.Localization(__file__, 111, 17), getitem___32831, i_32829)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 111)
    i_32833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'i')
    # Getting the type of 'p' (line 111)
    p_32834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 28), 'p')
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___32835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 28), p_32834, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_32836 = invoke(stypy.reporting.localization.Localization(__file__, 111, 28), getitem___32835, i_32833)
    
    # Applying the binary operator '-' (line 111)
    result_sub_32837 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 17), '-', subscript_call_result_32832, subscript_call_result_32836)
    
    # Assigning a type to the variable 'hi' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'hi', result_sub_32837)
    
    # Assigning a Call to a Name (line 112):
    
    # Assigning a Call to a Name (line 112):
    
    # Call to bc(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'ya' (line 112)
    ya_32839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'ya', False)
    # Getting the type of 'yb' (line 112)
    yb_32840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'yb', False)
    # Getting the type of 'p_new' (line 112)
    p_new_32841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 32), 'p_new', False)
    # Processing the call keyword arguments (line 112)
    kwargs_32842 = {}
    # Getting the type of 'bc' (line 112)
    bc_32838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 21), 'bc', False)
    # Calling bc(args, kwargs) (line 112)
    bc_call_result_32843 = invoke(stypy.reporting.localization.Localization(__file__, 112, 21), bc_32838, *[ya_32839, yb_32840, p_new_32841], **kwargs_32842)
    
    # Assigning a type to the variable 'bc_new' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'bc_new', bc_call_result_32843)
    
    # Assigning a BinOp to a Subscript (line 113):
    
    # Assigning a BinOp to a Subscript (line 113):
    # Getting the type of 'bc_new' (line 113)
    bc_new_32844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'bc_new')
    # Getting the type of 'bc0' (line 113)
    bc0_32845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 34), 'bc0')
    # Applying the binary operator '-' (line 113)
    result_sub_32846 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 25), '-', bc_new_32844, bc0_32845)
    
    # Getting the type of 'hi' (line 113)
    hi_32847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 41), 'hi')
    # Applying the binary operator 'div' (line 113)
    result_div_32848 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 24), 'div', result_sub_32846, hi_32847)
    
    # Getting the type of 'dbc_dp' (line 113)
    dbc_dp_32849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'dbc_dp')
    # Getting the type of 'i' (line 113)
    i_32850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'i')
    # Storing an element on a container (line 113)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 12), dbc_dp_32849, (i_32850, result_div_32848))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 114):
    
    # Assigning a Attribute to a Name (line 114):
    # Getting the type of 'dbc_dp' (line 114)
    dbc_dp_32851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'dbc_dp')
    # Obtaining the member 'T' of a type (line 114)
    T_32852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 17), dbc_dp_32851, 'T')
    # Assigning a type to the variable 'dbc_dp' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'dbc_dp', T_32852)
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 116)
    tuple_32853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 116)
    # Adding element type (line 116)
    # Getting the type of 'dbc_dya' (line 116)
    dbc_dya_32854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'dbc_dya')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 11), tuple_32853, dbc_dya_32854)
    # Adding element type (line 116)
    # Getting the type of 'dbc_dyb' (line 116)
    dbc_dyb_32855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'dbc_dyb')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 11), tuple_32853, dbc_dyb_32855)
    # Adding element type (line 116)
    # Getting the type of 'dbc_dp' (line 116)
    dbc_dp_32856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 29), 'dbc_dp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 11), tuple_32853, dbc_dp_32856)
    
    # Assigning a type to the variable 'stypy_return_type' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type', tuple_32853)
    
    # ################# End of 'estimate_bc_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'estimate_bc_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_32857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32857)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'estimate_bc_jac'
    return stypy_return_type_32857

# Assigning a type to the variable 'estimate_bc_jac' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'estimate_bc_jac', estimate_bc_jac)

@norecursion
def compute_jac_indices(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'compute_jac_indices'
    module_type_store = module_type_store.open_function_context('compute_jac_indices', 119, 0, False)
    
    # Passed parameters checking function
    compute_jac_indices.stypy_localization = localization
    compute_jac_indices.stypy_type_of_self = None
    compute_jac_indices.stypy_type_store = module_type_store
    compute_jac_indices.stypy_function_name = 'compute_jac_indices'
    compute_jac_indices.stypy_param_names_list = ['n', 'm', 'k']
    compute_jac_indices.stypy_varargs_param_name = None
    compute_jac_indices.stypy_kwargs_param_name = None
    compute_jac_indices.stypy_call_defaults = defaults
    compute_jac_indices.stypy_call_varargs = varargs
    compute_jac_indices.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compute_jac_indices', ['n', 'm', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compute_jac_indices', localization, ['n', 'm', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compute_jac_indices(...)' code ##################

    str_32858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, (-1)), 'str', 'Compute indices for the collocation system Jacobian construction.\n\n    See `construct_global_jac` for the explanation.\n    ')
    
    # Assigning a Call to a Name (line 124):
    
    # Assigning a Call to a Name (line 124):
    
    # Call to repeat(...): (line 124)
    # Processing the call arguments (line 124)
    
    # Call to arange(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'm' (line 124)
    m_32863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 33), 'm', False)
    int_32864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 37), 'int')
    # Applying the binary operator '-' (line 124)
    result_sub_32865 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 33), '-', m_32863, int_32864)
    
    # Getting the type of 'n' (line 124)
    n_32866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 42), 'n', False)
    # Applying the binary operator '*' (line 124)
    result_mul_32867 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 32), '*', result_sub_32865, n_32866)
    
    # Processing the call keyword arguments (line 124)
    kwargs_32868 = {}
    # Getting the type of 'np' (line 124)
    np_32861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 22), 'np', False)
    # Obtaining the member 'arange' of a type (line 124)
    arange_32862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 22), np_32861, 'arange')
    # Calling arange(args, kwargs) (line 124)
    arange_call_result_32869 = invoke(stypy.reporting.localization.Localization(__file__, 124, 22), arange_32862, *[result_mul_32867], **kwargs_32868)
    
    # Getting the type of 'n' (line 124)
    n_32870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 46), 'n', False)
    # Processing the call keyword arguments (line 124)
    kwargs_32871 = {}
    # Getting the type of 'np' (line 124)
    np_32859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'np', False)
    # Obtaining the member 'repeat' of a type (line 124)
    repeat_32860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), np_32859, 'repeat')
    # Calling repeat(args, kwargs) (line 124)
    repeat_call_result_32872 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), repeat_32860, *[arange_call_result_32869, n_32870], **kwargs_32871)
    
    # Assigning a type to the variable 'i_col' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'i_col', repeat_call_result_32872)
    
    # Assigning a BinOp to a Name (line 125):
    
    # Assigning a BinOp to a Name (line 125):
    
    # Call to tile(...): (line 125)
    # Processing the call arguments (line 125)
    
    # Call to arange(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'n' (line 125)
    n_32877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 31), 'n', False)
    # Processing the call keyword arguments (line 125)
    kwargs_32878 = {}
    # Getting the type of 'np' (line 125)
    np_32875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 21), 'np', False)
    # Obtaining the member 'arange' of a type (line 125)
    arange_32876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 21), np_32875, 'arange')
    # Calling arange(args, kwargs) (line 125)
    arange_call_result_32879 = invoke(stypy.reporting.localization.Localization(__file__, 125, 21), arange_32876, *[n_32877], **kwargs_32878)
    
    # Getting the type of 'n' (line 125)
    n_32880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 35), 'n', False)
    # Getting the type of 'm' (line 125)
    m_32881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 40), 'm', False)
    int_32882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 44), 'int')
    # Applying the binary operator '-' (line 125)
    result_sub_32883 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 40), '-', m_32881, int_32882)
    
    # Applying the binary operator '*' (line 125)
    result_mul_32884 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 35), '*', n_32880, result_sub_32883)
    
    # Processing the call keyword arguments (line 125)
    kwargs_32885 = {}
    # Getting the type of 'np' (line 125)
    np_32873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'np', False)
    # Obtaining the member 'tile' of a type (line 125)
    tile_32874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), np_32873, 'tile')
    # Calling tile(args, kwargs) (line 125)
    tile_call_result_32886 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), tile_32874, *[arange_call_result_32879, result_mul_32884], **kwargs_32885)
    
    
    # Call to repeat(...): (line 126)
    # Processing the call arguments (line 126)
    
    # Call to arange(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'm' (line 126)
    m_32891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 'm', False)
    int_32892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 37), 'int')
    # Applying the binary operator '-' (line 126)
    result_sub_32893 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 33), '-', m_32891, int_32892)
    
    # Processing the call keyword arguments (line 126)
    kwargs_32894 = {}
    # Getting the type of 'np' (line 126)
    np_32889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'np', False)
    # Obtaining the member 'arange' of a type (line 126)
    arange_32890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), np_32889, 'arange')
    # Calling arange(args, kwargs) (line 126)
    arange_call_result_32895 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), arange_32890, *[result_sub_32893], **kwargs_32894)
    
    # Getting the type of 'n' (line 126)
    n_32896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 42), 'n', False)
    # Applying the binary operator '*' (line 126)
    result_mul_32897 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 23), '*', arange_call_result_32895, n_32896)
    
    # Getting the type of 'n' (line 126)
    n_32898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 45), 'n', False)
    int_32899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 48), 'int')
    # Applying the binary operator '**' (line 126)
    result_pow_32900 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 45), '**', n_32898, int_32899)
    
    # Processing the call keyword arguments (line 126)
    kwargs_32901 = {}
    # Getting the type of 'np' (line 126)
    np_32887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 13), 'np', False)
    # Obtaining the member 'repeat' of a type (line 126)
    repeat_32888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 13), np_32887, 'repeat')
    # Calling repeat(args, kwargs) (line 126)
    repeat_call_result_32902 = invoke(stypy.reporting.localization.Localization(__file__, 126, 13), repeat_32888, *[result_mul_32897, result_pow_32900], **kwargs_32901)
    
    # Applying the binary operator '+' (line 125)
    result_add_32903 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 13), '+', tile_call_result_32886, repeat_call_result_32902)
    
    # Assigning a type to the variable 'j_col' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'j_col', result_add_32903)
    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to repeat(...): (line 128)
    # Processing the call arguments (line 128)
    
    # Call to arange(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'm' (line 128)
    m_32908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'm', False)
    int_32909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 36), 'int')
    # Applying the binary operator '-' (line 128)
    result_sub_32910 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 32), '-', m_32908, int_32909)
    
    # Getting the type of 'n' (line 128)
    n_32911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'n', False)
    # Applying the binary operator '*' (line 128)
    result_mul_32912 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 31), '*', result_sub_32910, n_32911)
    
    # Getting the type of 'm' (line 128)
    m_32913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 44), 'm', False)
    # Getting the type of 'n' (line 128)
    n_32914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 48), 'n', False)
    # Applying the binary operator '*' (line 128)
    result_mul_32915 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 44), '*', m_32913, n_32914)
    
    # Getting the type of 'k' (line 128)
    k_32916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 52), 'k', False)
    # Applying the binary operator '+' (line 128)
    result_add_32917 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 44), '+', result_mul_32915, k_32916)
    
    # Processing the call keyword arguments (line 128)
    kwargs_32918 = {}
    # Getting the type of 'np' (line 128)
    np_32906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'np', False)
    # Obtaining the member 'arange' of a type (line 128)
    arange_32907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 21), np_32906, 'arange')
    # Calling arange(args, kwargs) (line 128)
    arange_call_result_32919 = invoke(stypy.reporting.localization.Localization(__file__, 128, 21), arange_32907, *[result_mul_32912, result_add_32917], **kwargs_32918)
    
    # Getting the type of 'n' (line 128)
    n_32920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 56), 'n', False)
    # Processing the call keyword arguments (line 128)
    kwargs_32921 = {}
    # Getting the type of 'np' (line 128)
    np_32904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'np', False)
    # Obtaining the member 'repeat' of a type (line 128)
    repeat_32905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 11), np_32904, 'repeat')
    # Calling repeat(args, kwargs) (line 128)
    repeat_call_result_32922 = invoke(stypy.reporting.localization.Localization(__file__, 128, 11), repeat_32905, *[arange_call_result_32919, n_32920], **kwargs_32921)
    
    # Assigning a type to the variable 'i_bc' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'i_bc', repeat_call_result_32922)
    
    # Assigning a Call to a Name (line 129):
    
    # Assigning a Call to a Name (line 129):
    
    # Call to tile(...): (line 129)
    # Processing the call arguments (line 129)
    
    # Call to arange(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'n' (line 129)
    n_32927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 29), 'n', False)
    # Processing the call keyword arguments (line 129)
    kwargs_32928 = {}
    # Getting the type of 'np' (line 129)
    np_32925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), 'np', False)
    # Obtaining the member 'arange' of a type (line 129)
    arange_32926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 19), np_32925, 'arange')
    # Calling arange(args, kwargs) (line 129)
    arange_call_result_32929 = invoke(stypy.reporting.localization.Localization(__file__, 129, 19), arange_32926, *[n_32927], **kwargs_32928)
    
    # Getting the type of 'n' (line 129)
    n_32930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 33), 'n', False)
    # Getting the type of 'k' (line 129)
    k_32931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 37), 'k', False)
    # Applying the binary operator '+' (line 129)
    result_add_32932 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 33), '+', n_32930, k_32931)
    
    # Processing the call keyword arguments (line 129)
    kwargs_32933 = {}
    # Getting the type of 'np' (line 129)
    np_32923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'np', False)
    # Obtaining the member 'tile' of a type (line 129)
    tile_32924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 11), np_32923, 'tile')
    # Calling tile(args, kwargs) (line 129)
    tile_call_result_32934 = invoke(stypy.reporting.localization.Localization(__file__, 129, 11), tile_32924, *[arange_call_result_32929, result_add_32932], **kwargs_32933)
    
    # Assigning a type to the variable 'j_bc' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'j_bc', tile_call_result_32934)
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to repeat(...): (line 131)
    # Processing the call arguments (line 131)
    
    # Call to arange(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'm' (line 131)
    m_32939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'm', False)
    int_32940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 39), 'int')
    # Applying the binary operator '-' (line 131)
    result_sub_32941 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 35), '-', m_32939, int_32940)
    
    # Getting the type of 'n' (line 131)
    n_32942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 44), 'n', False)
    # Applying the binary operator '*' (line 131)
    result_mul_32943 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 34), '*', result_sub_32941, n_32942)
    
    # Processing the call keyword arguments (line 131)
    kwargs_32944 = {}
    # Getting the type of 'np' (line 131)
    np_32937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'np', False)
    # Obtaining the member 'arange' of a type (line 131)
    arange_32938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 24), np_32937, 'arange')
    # Calling arange(args, kwargs) (line 131)
    arange_call_result_32945 = invoke(stypy.reporting.localization.Localization(__file__, 131, 24), arange_32938, *[result_mul_32943], **kwargs_32944)
    
    # Getting the type of 'k' (line 131)
    k_32946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 48), 'k', False)
    # Processing the call keyword arguments (line 131)
    kwargs_32947 = {}
    # Getting the type of 'np' (line 131)
    np_32935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 14), 'np', False)
    # Obtaining the member 'repeat' of a type (line 131)
    repeat_32936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 14), np_32935, 'repeat')
    # Calling repeat(args, kwargs) (line 131)
    repeat_call_result_32948 = invoke(stypy.reporting.localization.Localization(__file__, 131, 14), repeat_32936, *[arange_call_result_32945, k_32946], **kwargs_32947)
    
    # Assigning a type to the variable 'i_p_col' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'i_p_col', repeat_call_result_32948)
    
    # Assigning a Call to a Name (line 132):
    
    # Assigning a Call to a Name (line 132):
    
    # Call to tile(...): (line 132)
    # Processing the call arguments (line 132)
    
    # Call to arange(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'm' (line 132)
    m_32953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 32), 'm', False)
    # Getting the type of 'n' (line 132)
    n_32954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 36), 'n', False)
    # Applying the binary operator '*' (line 132)
    result_mul_32955 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 32), '*', m_32953, n_32954)
    
    # Getting the type of 'm' (line 132)
    m_32956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 39), 'm', False)
    # Getting the type of 'n' (line 132)
    n_32957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 43), 'n', False)
    # Applying the binary operator '*' (line 132)
    result_mul_32958 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 39), '*', m_32956, n_32957)
    
    # Getting the type of 'k' (line 132)
    k_32959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 47), 'k', False)
    # Applying the binary operator '+' (line 132)
    result_add_32960 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 39), '+', result_mul_32958, k_32959)
    
    # Processing the call keyword arguments (line 132)
    kwargs_32961 = {}
    # Getting the type of 'np' (line 132)
    np_32951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 22), 'np', False)
    # Obtaining the member 'arange' of a type (line 132)
    arange_32952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 22), np_32951, 'arange')
    # Calling arange(args, kwargs) (line 132)
    arange_call_result_32962 = invoke(stypy.reporting.localization.Localization(__file__, 132, 22), arange_32952, *[result_mul_32955, result_add_32960], **kwargs_32961)
    
    # Getting the type of 'm' (line 132)
    m_32963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 52), 'm', False)
    int_32964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 56), 'int')
    # Applying the binary operator '-' (line 132)
    result_sub_32965 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 52), '-', m_32963, int_32964)
    
    # Getting the type of 'n' (line 132)
    n_32966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 61), 'n', False)
    # Applying the binary operator '*' (line 132)
    result_mul_32967 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 51), '*', result_sub_32965, n_32966)
    
    # Processing the call keyword arguments (line 132)
    kwargs_32968 = {}
    # Getting the type of 'np' (line 132)
    np_32949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'np', False)
    # Obtaining the member 'tile' of a type (line 132)
    tile_32950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 14), np_32949, 'tile')
    # Calling tile(args, kwargs) (line 132)
    tile_call_result_32969 = invoke(stypy.reporting.localization.Localization(__file__, 132, 14), tile_32950, *[arange_call_result_32962, result_mul_32967], **kwargs_32968)
    
    # Assigning a type to the variable 'j_p_col' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'j_p_col', tile_call_result_32969)
    
    # Assigning a Call to a Name (line 134):
    
    # Assigning a Call to a Name (line 134):
    
    # Call to repeat(...): (line 134)
    # Processing the call arguments (line 134)
    
    # Call to arange(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'm' (line 134)
    m_32974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 34), 'm', False)
    int_32975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 38), 'int')
    # Applying the binary operator '-' (line 134)
    result_sub_32976 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 34), '-', m_32974, int_32975)
    
    # Getting the type of 'n' (line 134)
    n_32977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 43), 'n', False)
    # Applying the binary operator '*' (line 134)
    result_mul_32978 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 33), '*', result_sub_32976, n_32977)
    
    # Getting the type of 'm' (line 134)
    m_32979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 46), 'm', False)
    # Getting the type of 'n' (line 134)
    n_32980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 50), 'n', False)
    # Applying the binary operator '*' (line 134)
    result_mul_32981 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 46), '*', m_32979, n_32980)
    
    # Getting the type of 'k' (line 134)
    k_32982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 54), 'k', False)
    # Applying the binary operator '+' (line 134)
    result_add_32983 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 46), '+', result_mul_32981, k_32982)
    
    # Processing the call keyword arguments (line 134)
    kwargs_32984 = {}
    # Getting the type of 'np' (line 134)
    np_32972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'np', False)
    # Obtaining the member 'arange' of a type (line 134)
    arange_32973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 23), np_32972, 'arange')
    # Calling arange(args, kwargs) (line 134)
    arange_call_result_32985 = invoke(stypy.reporting.localization.Localization(__file__, 134, 23), arange_32973, *[result_mul_32978, result_add_32983], **kwargs_32984)
    
    # Getting the type of 'k' (line 134)
    k_32986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 58), 'k', False)
    # Processing the call keyword arguments (line 134)
    kwargs_32987 = {}
    # Getting the type of 'np' (line 134)
    np_32970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 13), 'np', False)
    # Obtaining the member 'repeat' of a type (line 134)
    repeat_32971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 13), np_32970, 'repeat')
    # Calling repeat(args, kwargs) (line 134)
    repeat_call_result_32988 = invoke(stypy.reporting.localization.Localization(__file__, 134, 13), repeat_32971, *[arange_call_result_32985, k_32986], **kwargs_32987)
    
    # Assigning a type to the variable 'i_p_bc' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'i_p_bc', repeat_call_result_32988)
    
    # Assigning a Call to a Name (line 135):
    
    # Assigning a Call to a Name (line 135):
    
    # Call to tile(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Call to arange(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'm' (line 135)
    m_32993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'm', False)
    # Getting the type of 'n' (line 135)
    n_32994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 35), 'n', False)
    # Applying the binary operator '*' (line 135)
    result_mul_32995 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 31), '*', m_32993, n_32994)
    
    # Getting the type of 'm' (line 135)
    m_32996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 38), 'm', False)
    # Getting the type of 'n' (line 135)
    n_32997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 42), 'n', False)
    # Applying the binary operator '*' (line 135)
    result_mul_32998 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 38), '*', m_32996, n_32997)
    
    # Getting the type of 'k' (line 135)
    k_32999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 46), 'k', False)
    # Applying the binary operator '+' (line 135)
    result_add_33000 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 38), '+', result_mul_32998, k_32999)
    
    # Processing the call keyword arguments (line 135)
    kwargs_33001 = {}
    # Getting the type of 'np' (line 135)
    np_32991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 21), 'np', False)
    # Obtaining the member 'arange' of a type (line 135)
    arange_32992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 21), np_32991, 'arange')
    # Calling arange(args, kwargs) (line 135)
    arange_call_result_33002 = invoke(stypy.reporting.localization.Localization(__file__, 135, 21), arange_32992, *[result_mul_32995, result_add_33000], **kwargs_33001)
    
    # Getting the type of 'n' (line 135)
    n_33003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 50), 'n', False)
    # Getting the type of 'k' (line 135)
    k_33004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 54), 'k', False)
    # Applying the binary operator '+' (line 135)
    result_add_33005 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 50), '+', n_33003, k_33004)
    
    # Processing the call keyword arguments (line 135)
    kwargs_33006 = {}
    # Getting the type of 'np' (line 135)
    np_32989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 13), 'np', False)
    # Obtaining the member 'tile' of a type (line 135)
    tile_32990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 13), np_32989, 'tile')
    # Calling tile(args, kwargs) (line 135)
    tile_call_result_33007 = invoke(stypy.reporting.localization.Localization(__file__, 135, 13), tile_32990, *[arange_call_result_33002, result_add_33005], **kwargs_33006)
    
    # Assigning a type to the variable 'j_p_bc' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'j_p_bc', tile_call_result_33007)
    
    # Assigning a Call to a Name (line 137):
    
    # Assigning a Call to a Name (line 137):
    
    # Call to hstack(...): (line 137)
    # Processing the call arguments (line 137)
    
    # Obtaining an instance of the builtin type 'tuple' (line 137)
    tuple_33010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 137)
    # Adding element type (line 137)
    # Getting the type of 'i_col' (line 137)
    i_col_33011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 19), 'i_col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 19), tuple_33010, i_col_33011)
    # Adding element type (line 137)
    # Getting the type of 'i_col' (line 137)
    i_col_33012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 26), 'i_col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 19), tuple_33010, i_col_33012)
    # Adding element type (line 137)
    # Getting the type of 'i_bc' (line 137)
    i_bc_33013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'i_bc', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 19), tuple_33010, i_bc_33013)
    # Adding element type (line 137)
    # Getting the type of 'i_bc' (line 137)
    i_bc_33014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 39), 'i_bc', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 19), tuple_33010, i_bc_33014)
    # Adding element type (line 137)
    # Getting the type of 'i_p_col' (line 137)
    i_p_col_33015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 45), 'i_p_col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 19), tuple_33010, i_p_col_33015)
    # Adding element type (line 137)
    # Getting the type of 'i_p_bc' (line 137)
    i_p_bc_33016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 54), 'i_p_bc', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 19), tuple_33010, i_p_bc_33016)
    
    # Processing the call keyword arguments (line 137)
    kwargs_33017 = {}
    # Getting the type of 'np' (line 137)
    np_33008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'np', False)
    # Obtaining the member 'hstack' of a type (line 137)
    hstack_33009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), np_33008, 'hstack')
    # Calling hstack(args, kwargs) (line 137)
    hstack_call_result_33018 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), hstack_33009, *[tuple_33010], **kwargs_33017)
    
    # Assigning a type to the variable 'i' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'i', hstack_call_result_33018)
    
    # Assigning a Call to a Name (line 138):
    
    # Assigning a Call to a Name (line 138):
    
    # Call to hstack(...): (line 138)
    # Processing the call arguments (line 138)
    
    # Obtaining an instance of the builtin type 'tuple' (line 138)
    tuple_33021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 138)
    # Adding element type (line 138)
    # Getting the type of 'j_col' (line 138)
    j_col_33022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'j_col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), tuple_33021, j_col_33022)
    # Adding element type (line 138)
    # Getting the type of 'j_col' (line 138)
    j_col_33023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 26), 'j_col', False)
    # Getting the type of 'n' (line 138)
    n_33024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 34), 'n', False)
    # Applying the binary operator '+' (line 138)
    result_add_33025 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 26), '+', j_col_33023, n_33024)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), tuple_33021, result_add_33025)
    # Adding element type (line 138)
    # Getting the type of 'j_bc' (line 139)
    j_bc_33026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'j_bc', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), tuple_33021, j_bc_33026)
    # Adding element type (line 138)
    # Getting the type of 'j_bc' (line 139)
    j_bc_33027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 25), 'j_bc', False)
    # Getting the type of 'm' (line 139)
    m_33028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 33), 'm', False)
    int_33029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 37), 'int')
    # Applying the binary operator '-' (line 139)
    result_sub_33030 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 33), '-', m_33028, int_33029)
    
    # Getting the type of 'n' (line 139)
    n_33031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 42), 'n', False)
    # Applying the binary operator '*' (line 139)
    result_mul_33032 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 32), '*', result_sub_33030, n_33031)
    
    # Applying the binary operator '+' (line 139)
    result_add_33033 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 25), '+', j_bc_33027, result_mul_33032)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), tuple_33021, result_add_33033)
    # Adding element type (line 138)
    # Getting the type of 'j_p_col' (line 140)
    j_p_col_33034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 19), 'j_p_col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), tuple_33021, j_p_col_33034)
    # Adding element type (line 138)
    # Getting the type of 'j_p_bc' (line 140)
    j_p_bc_33035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 28), 'j_p_bc', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), tuple_33021, j_p_bc_33035)
    
    # Processing the call keyword arguments (line 138)
    kwargs_33036 = {}
    # Getting the type of 'np' (line 138)
    np_33019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'np', False)
    # Obtaining the member 'hstack' of a type (line 138)
    hstack_33020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), np_33019, 'hstack')
    # Calling hstack(args, kwargs) (line 138)
    hstack_call_result_33037 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), hstack_33020, *[tuple_33021], **kwargs_33036)
    
    # Assigning a type to the variable 'j' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'j', hstack_call_result_33037)
    
    # Obtaining an instance of the builtin type 'tuple' (line 142)
    tuple_33038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 142)
    # Adding element type (line 142)
    # Getting the type of 'i' (line 142)
    i_33039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 11), tuple_33038, i_33039)
    # Adding element type (line 142)
    # Getting the type of 'j' (line 142)
    j_33040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 14), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 11), tuple_33038, j_33040)
    
    # Assigning a type to the variable 'stypy_return_type' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'stypy_return_type', tuple_33038)
    
    # ################# End of 'compute_jac_indices(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compute_jac_indices' in the type store
    # Getting the type of 'stypy_return_type' (line 119)
    stypy_return_type_33041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33041)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compute_jac_indices'
    return stypy_return_type_33041

# Assigning a type to the variable 'compute_jac_indices' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'compute_jac_indices', compute_jac_indices)

@norecursion
def stacked_matmul(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'stacked_matmul'
    module_type_store = module_type_store.open_function_context('stacked_matmul', 145, 0, False)
    
    # Passed parameters checking function
    stacked_matmul.stypy_localization = localization
    stacked_matmul.stypy_type_of_self = None
    stacked_matmul.stypy_type_store = module_type_store
    stacked_matmul.stypy_function_name = 'stacked_matmul'
    stacked_matmul.stypy_param_names_list = ['a', 'b']
    stacked_matmul.stypy_varargs_param_name = None
    stacked_matmul.stypy_kwargs_param_name = None
    stacked_matmul.stypy_call_defaults = defaults
    stacked_matmul.stypy_call_varargs = varargs
    stacked_matmul.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'stacked_matmul', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'stacked_matmul', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'stacked_matmul(...)' code ##################

    str_33042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, (-1)), 'str', 'Stacked matrix multiply: out[i,:,:] = np.dot(a[i,:,:], b[i,:,:]).\n\n    In our case a[i, :, :] and b[i, :, :] are always square.\n    ')
    
    
    
    # Obtaining the type of the subscript
    int_33043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 15), 'int')
    # Getting the type of 'a' (line 152)
    a_33044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 7), 'a')
    # Obtaining the member 'shape' of a type (line 152)
    shape_33045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 7), a_33044, 'shape')
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___33046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 7), shape_33045, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_33047 = invoke(stypy.reporting.localization.Localization(__file__, 152, 7), getitem___33046, int_33043)
    
    int_33048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 20), 'int')
    # Applying the binary operator '>' (line 152)
    result_gt_33049 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 7), '>', subscript_call_result_33047, int_33048)
    
    # Testing the type of an if condition (line 152)
    if_condition_33050 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 4), result_gt_33049)
    # Assigning a type to the variable 'if_condition_33050' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'if_condition_33050', if_condition_33050)
    # SSA begins for if statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 153):
    
    # Assigning a Call to a Name (line 153):
    
    # Call to empty_like(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'a' (line 153)
    a_33053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 28), 'a', False)
    # Processing the call keyword arguments (line 153)
    kwargs_33054 = {}
    # Getting the type of 'np' (line 153)
    np_33051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 14), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 153)
    empty_like_33052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 14), np_33051, 'empty_like')
    # Calling empty_like(args, kwargs) (line 153)
    empty_like_call_result_33055 = invoke(stypy.reporting.localization.Localization(__file__, 153, 14), empty_like_33052, *[a_33053], **kwargs_33054)
    
    # Assigning a type to the variable 'out' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'out', empty_like_call_result_33055)
    
    
    # Call to range(...): (line 154)
    # Processing the call arguments (line 154)
    
    # Obtaining the type of the subscript
    int_33057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 31), 'int')
    # Getting the type of 'a' (line 154)
    a_33058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 23), 'a', False)
    # Obtaining the member 'shape' of a type (line 154)
    shape_33059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 23), a_33058, 'shape')
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___33060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 23), shape_33059, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_33061 = invoke(stypy.reporting.localization.Localization(__file__, 154, 23), getitem___33060, int_33057)
    
    # Processing the call keyword arguments (line 154)
    kwargs_33062 = {}
    # Getting the type of 'range' (line 154)
    range_33056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 17), 'range', False)
    # Calling range(args, kwargs) (line 154)
    range_call_result_33063 = invoke(stypy.reporting.localization.Localization(__file__, 154, 17), range_33056, *[subscript_call_result_33061], **kwargs_33062)
    
    # Testing the type of a for loop iterable (line 154)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 154, 8), range_call_result_33063)
    # Getting the type of the for loop variable (line 154)
    for_loop_var_33064 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 154, 8), range_call_result_33063)
    # Assigning a type to the variable 'i' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'i', for_loop_var_33064)
    # SSA begins for a for statement (line 154)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 155):
    
    # Assigning a Call to a Subscript (line 155):
    
    # Call to dot(...): (line 155)
    # Processing the call arguments (line 155)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 155)
    i_33067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 30), 'i', False)
    # Getting the type of 'a' (line 155)
    a_33068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___33069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 28), a_33068, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_33070 = invoke(stypy.reporting.localization.Localization(__file__, 155, 28), getitem___33069, i_33067)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 155)
    i_33071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 36), 'i', False)
    # Getting the type of 'b' (line 155)
    b_33072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___33073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 34), b_33072, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_33074 = invoke(stypy.reporting.localization.Localization(__file__, 155, 34), getitem___33073, i_33071)
    
    # Processing the call keyword arguments (line 155)
    kwargs_33075 = {}
    # Getting the type of 'np' (line 155)
    np_33065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'np', False)
    # Obtaining the member 'dot' of a type (line 155)
    dot_33066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 21), np_33065, 'dot')
    # Calling dot(args, kwargs) (line 155)
    dot_call_result_33076 = invoke(stypy.reporting.localization.Localization(__file__, 155, 21), dot_33066, *[subscript_call_result_33070, subscript_call_result_33074], **kwargs_33075)
    
    # Getting the type of 'out' (line 155)
    out_33077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'out')
    # Getting the type of 'i' (line 155)
    i_33078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'i')
    # Storing an element on a container (line 155)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 12), out_33077, (i_33078, dot_call_result_33076))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'out' (line 156)
    out_33079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'stypy_return_type', out_33079)
    # SSA branch for the else part of an if statement (line 152)
    module_type_store.open_ssa_branch('else')
    
    # Call to einsum(...): (line 158)
    # Processing the call arguments (line 158)
    str_33082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 25), 'str', '...ij,...jk->...ik')
    # Getting the type of 'a' (line 158)
    a_33083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 47), 'a', False)
    # Getting the type of 'b' (line 158)
    b_33084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 50), 'b', False)
    # Processing the call keyword arguments (line 158)
    kwargs_33085 = {}
    # Getting the type of 'np' (line 158)
    np_33080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'np', False)
    # Obtaining the member 'einsum' of a type (line 158)
    einsum_33081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 15), np_33080, 'einsum')
    # Calling einsum(args, kwargs) (line 158)
    einsum_call_result_33086 = invoke(stypy.reporting.localization.Localization(__file__, 158, 15), einsum_33081, *[str_33082, a_33083, b_33084], **kwargs_33085)
    
    # Assigning a type to the variable 'stypy_return_type' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'stypy_return_type', einsum_call_result_33086)
    # SSA join for if statement (line 152)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'stacked_matmul(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'stacked_matmul' in the type store
    # Getting the type of 'stypy_return_type' (line 145)
    stypy_return_type_33087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33087)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'stacked_matmul'
    return stypy_return_type_33087

# Assigning a type to the variable 'stacked_matmul' (line 145)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'stacked_matmul', stacked_matmul)

@norecursion
def construct_global_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'construct_global_jac'
    module_type_store = module_type_store.open_function_context('construct_global_jac', 161, 0, False)
    
    # Passed parameters checking function
    construct_global_jac.stypy_localization = localization
    construct_global_jac.stypy_type_of_self = None
    construct_global_jac.stypy_type_store = module_type_store
    construct_global_jac.stypy_function_name = 'construct_global_jac'
    construct_global_jac.stypy_param_names_list = ['n', 'm', 'k', 'i_jac', 'j_jac', 'h', 'df_dy', 'df_dy_middle', 'df_dp', 'df_dp_middle', 'dbc_dya', 'dbc_dyb', 'dbc_dp']
    construct_global_jac.stypy_varargs_param_name = None
    construct_global_jac.stypy_kwargs_param_name = None
    construct_global_jac.stypy_call_defaults = defaults
    construct_global_jac.stypy_call_varargs = varargs
    construct_global_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'construct_global_jac', ['n', 'm', 'k', 'i_jac', 'j_jac', 'h', 'df_dy', 'df_dy_middle', 'df_dp', 'df_dp_middle', 'dbc_dya', 'dbc_dyb', 'dbc_dp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'construct_global_jac', localization, ['n', 'm', 'k', 'i_jac', 'j_jac', 'h', 'df_dy', 'df_dy_middle', 'df_dp', 'df_dp_middle', 'dbc_dya', 'dbc_dyb', 'dbc_dp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'construct_global_jac(...)' code ##################

    str_33088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, (-1)), 'str', 'Construct the Jacobian of the collocation system.\n\n    There are n * m + k functions: m - 1 collocations residuals, each\n    containing n components, followed by n + k boundary condition residuals.\n\n    There are n * m + k variables: m vectors of y, each containing n\n    components, followed by k values of vector p.\n\n    For example, let m = 4, n = 2 and k = 1, then the Jacobian will have\n    the following sparsity structure:\n\n        1 1 2 2 0 0 0 0  5\n        1 1 2 2 0 0 0 0  5\n        0 0 1 1 2 2 0 0  5\n        0 0 1 1 2 2 0 0  5\n        0 0 0 0 1 1 2 2  5\n        0 0 0 0 1 1 2 2  5\n\n        3 3 0 0 0 0 4 4  6\n        3 3 0 0 0 0 4 4  6\n        3 3 0 0 0 0 4 4  6\n\n    Zeros denote identically zero values, other values denote different kinds\n    of blocks in the matrix (see below). The blank row indicates the separation\n    of collocation residuals from boundary conditions. And the blank column\n    indicates the separation of y values from p values.\n\n    Refer to [1]_  (p. 306) for the formula of n x n blocks for derivatives\n    of collocation residuals with respect to y.\n\n    Parameters\n    ----------\n    n : int\n        Number of equations in the ODE system.\n    m : int\n        Number of nodes in the mesh.\n    k : int\n        Number of the unknown parameters.\n    i_jac, j_jac : ndarray\n        Row and column indices returned by `compute_jac_indices`. They\n        represent different blocks in the Jacobian matrix in the following\n        order (see the scheme above):\n\n            * 1: m - 1 diagonal n x n blocks for the collocation residuals.\n            * 2: m - 1 off-diagonal n x n blocks for the collocation residuals.\n            * 3 : (n + k) x n block for the dependency of the boundary\n              conditions on ya.\n            * 4: (n + k) x n block for the dependency of the boundary\n              conditions on yb.\n            * 5: (m - 1) * n x k block for the dependency of the collocation\n              residuals on p.\n            * 6: (n + k) x k block for the dependency of the boundary\n              conditions on p.\n\n    df_dy : ndarray, shape (n, n, m)\n        Jacobian of f with respect to y computed at the mesh nodes.\n    df_dy_middle : ndarray, shape (n, n, m - 1)\n        Jacobian of f with respect to y computed at the middle between the\n        mesh nodes.\n    df_dp : ndarray with shape (n, k, m) or None\n        Jacobian of f with respect to p computed at the mesh nodes.\n    df_dp_middle: ndarray with shape (n, k, m - 1) or None\n        Jacobian of f with respect to p computed at the middle between the\n        mesh nodes.\n    dbc_dya, dbc_dyb : ndarray, shape (n, n)\n        Jacobian of bc with respect to ya and yb.\n    dbc_dp: ndarray with shape (n, k) or None\n        Jacobian of bc with respect to p.\n\n    Returns\n    -------\n    J : csc_matrix, shape (n * m + k, n * m + k)\n        Jacobian of the collocation system in a sparse form.\n\n    References\n    ----------\n    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual\n       Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,\n       Number 3, pp. 299-316, 2001.\n    ')
    
    # Assigning a Call to a Name (line 243):
    
    # Assigning a Call to a Name (line 243):
    
    # Call to transpose(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'df_dy' (line 243)
    df_dy_33091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 25), 'df_dy', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 243)
    tuple_33092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 243)
    # Adding element type (line 243)
    int_33093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 33), tuple_33092, int_33093)
    # Adding element type (line 243)
    int_33094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 33), tuple_33092, int_33094)
    # Adding element type (line 243)
    int_33095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 33), tuple_33092, int_33095)
    
    # Processing the call keyword arguments (line 243)
    kwargs_33096 = {}
    # Getting the type of 'np' (line 243)
    np_33089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'np', False)
    # Obtaining the member 'transpose' of a type (line 243)
    transpose_33090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), np_33089, 'transpose')
    # Calling transpose(args, kwargs) (line 243)
    transpose_call_result_33097 = invoke(stypy.reporting.localization.Localization(__file__, 243, 12), transpose_33090, *[df_dy_33091, tuple_33092], **kwargs_33096)
    
    # Assigning a type to the variable 'df_dy' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'df_dy', transpose_call_result_33097)
    
    # Assigning a Call to a Name (line 244):
    
    # Assigning a Call to a Name (line 244):
    
    # Call to transpose(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'df_dy_middle' (line 244)
    df_dy_middle_33100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 32), 'df_dy_middle', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 244)
    tuple_33101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 244)
    # Adding element type (line 244)
    int_33102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 47), tuple_33101, int_33102)
    # Adding element type (line 244)
    int_33103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 47), tuple_33101, int_33103)
    # Adding element type (line 244)
    int_33104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 47), tuple_33101, int_33104)
    
    # Processing the call keyword arguments (line 244)
    kwargs_33105 = {}
    # Getting the type of 'np' (line 244)
    np_33098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'np', False)
    # Obtaining the member 'transpose' of a type (line 244)
    transpose_33099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 19), np_33098, 'transpose')
    # Calling transpose(args, kwargs) (line 244)
    transpose_call_result_33106 = invoke(stypy.reporting.localization.Localization(__file__, 244, 19), transpose_33099, *[df_dy_middle_33100, tuple_33101], **kwargs_33105)
    
    # Assigning a type to the variable 'df_dy_middle' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'df_dy_middle', transpose_call_result_33106)
    
    # Assigning a Subscript to a Name (line 246):
    
    # Assigning a Subscript to a Name (line 246):
    
    # Obtaining the type of the subscript
    slice_33107 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 246, 8), None, None, None)
    # Getting the type of 'np' (line 246)
    np_33108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 13), 'np')
    # Obtaining the member 'newaxis' of a type (line 246)
    newaxis_33109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 13), np_33108, 'newaxis')
    # Getting the type of 'np' (line 246)
    np_33110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 25), 'np')
    # Obtaining the member 'newaxis' of a type (line 246)
    newaxis_33111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 25), np_33110, 'newaxis')
    # Getting the type of 'h' (line 246)
    h_33112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'h')
    # Obtaining the member '__getitem__' of a type (line 246)
    getitem___33113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), h_33112, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 246)
    subscript_call_result_33114 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), getitem___33113, (slice_33107, newaxis_33109, newaxis_33111))
    
    # Assigning a type to the variable 'h' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'h', subscript_call_result_33114)
    
    # Assigning a Attribute to a Name (line 248):
    
    # Assigning a Attribute to a Name (line 248):
    # Getting the type of 'df_dy' (line 248)
    df_dy_33115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'df_dy')
    # Obtaining the member 'dtype' of a type (line 248)
    dtype_33116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), df_dy_33115, 'dtype')
    # Assigning a type to the variable 'dtype' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'dtype', dtype_33116)
    
    # Assigning a Call to a Name (line 251):
    
    # Assigning a Call to a Name (line 251):
    
    # Call to empty(...): (line 251)
    # Processing the call arguments (line 251)
    
    # Obtaining an instance of the builtin type 'tuple' (line 251)
    tuple_33119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 251)
    # Adding element type (line 251)
    # Getting the type of 'm' (line 251)
    m_33120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 26), 'm', False)
    int_33121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 30), 'int')
    # Applying the binary operator '-' (line 251)
    result_sub_33122 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 26), '-', m_33120, int_33121)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 26), tuple_33119, result_sub_33122)
    # Adding element type (line 251)
    # Getting the type of 'n' (line 251)
    n_33123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 33), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 26), tuple_33119, n_33123)
    # Adding element type (line 251)
    # Getting the type of 'n' (line 251)
    n_33124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 36), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 26), tuple_33119, n_33124)
    
    # Processing the call keyword arguments (line 251)
    # Getting the type of 'dtype' (line 251)
    dtype_33125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 46), 'dtype', False)
    keyword_33126 = dtype_33125
    kwargs_33127 = {'dtype': keyword_33126}
    # Getting the type of 'np' (line 251)
    np_33117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'np', False)
    # Obtaining the member 'empty' of a type (line 251)
    empty_33118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 16), np_33117, 'empty')
    # Calling empty(args, kwargs) (line 251)
    empty_call_result_33128 = invoke(stypy.reporting.localization.Localization(__file__, 251, 16), empty_33118, *[tuple_33119], **kwargs_33127)
    
    # Assigning a type to the variable 'dPhi_dy_0' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'dPhi_dy_0', empty_call_result_33128)
    
    # Assigning a UnaryOp to a Subscript (line 252):
    
    # Assigning a UnaryOp to a Subscript (line 252):
    
    
    # Call to identity(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'n' (line 252)
    n_33131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 32), 'n', False)
    # Processing the call keyword arguments (line 252)
    kwargs_33132 = {}
    # Getting the type of 'np' (line 252)
    np_33129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 20), 'np', False)
    # Obtaining the member 'identity' of a type (line 252)
    identity_33130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 20), np_33129, 'identity')
    # Calling identity(args, kwargs) (line 252)
    identity_call_result_33133 = invoke(stypy.reporting.localization.Localization(__file__, 252, 20), identity_33130, *[n_33131], **kwargs_33132)
    
    # Applying the 'usub' unary operator (line 252)
    result___neg___33134 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 19), 'usub', identity_call_result_33133)
    
    # Getting the type of 'dPhi_dy_0' (line 252)
    dPhi_dy_0_33135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'dPhi_dy_0')
    slice_33136 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 252, 4), None, None, None)
    # Storing an element on a container (line 252)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 4), dPhi_dy_0_33135, (slice_33136, result___neg___33134))
    
    # Getting the type of 'dPhi_dy_0' (line 253)
    dPhi_dy_0_33137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'dPhi_dy_0')
    # Getting the type of 'h' (line 253)
    h_33138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 17), 'h')
    int_33139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 21), 'int')
    # Applying the binary operator 'div' (line 253)
    result_div_33140 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 17), 'div', h_33138, int_33139)
    
    
    # Obtaining the type of the subscript
    int_33141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 33), 'int')
    slice_33142 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 253, 26), None, int_33141, None)
    # Getting the type of 'df_dy' (line 253)
    df_dy_33143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 26), 'df_dy')
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___33144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 26), df_dy_33143, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_33145 = invoke(stypy.reporting.localization.Localization(__file__, 253, 26), getitem___33144, slice_33142)
    
    int_33146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 39), 'int')
    # Getting the type of 'df_dy_middle' (line 253)
    df_dy_middle_33147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 43), 'df_dy_middle')
    # Applying the binary operator '*' (line 253)
    result_mul_33148 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 39), '*', int_33146, df_dy_middle_33147)
    
    # Applying the binary operator '+' (line 253)
    result_add_33149 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 26), '+', subscript_call_result_33145, result_mul_33148)
    
    # Applying the binary operator '*' (line 253)
    result_mul_33150 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 23), '*', result_div_33140, result_add_33149)
    
    # Applying the binary operator '-=' (line 253)
    result_isub_33151 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 4), '-=', dPhi_dy_0_33137, result_mul_33150)
    # Assigning a type to the variable 'dPhi_dy_0' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'dPhi_dy_0', result_isub_33151)
    
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to stacked_matmul(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'df_dy_middle' (line 254)
    df_dy_middle_33153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 23), 'df_dy_middle', False)
    
    # Obtaining the type of the subscript
    int_33154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 44), 'int')
    slice_33155 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 254, 37), None, int_33154, None)
    # Getting the type of 'df_dy' (line 254)
    df_dy_33156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 37), 'df_dy', False)
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___33157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 37), df_dy_33156, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_33158 = invoke(stypy.reporting.localization.Localization(__file__, 254, 37), getitem___33157, slice_33155)
    
    # Processing the call keyword arguments (line 254)
    kwargs_33159 = {}
    # Getting the type of 'stacked_matmul' (line 254)
    stacked_matmul_33152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'stacked_matmul', False)
    # Calling stacked_matmul(args, kwargs) (line 254)
    stacked_matmul_call_result_33160 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), stacked_matmul_33152, *[df_dy_middle_33153, subscript_call_result_33158], **kwargs_33159)
    
    # Assigning a type to the variable 'T' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'T', stacked_matmul_call_result_33160)
    
    # Getting the type of 'dPhi_dy_0' (line 255)
    dPhi_dy_0_33161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'dPhi_dy_0')
    # Getting the type of 'h' (line 255)
    h_33162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 17), 'h')
    int_33163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 20), 'int')
    # Applying the binary operator '**' (line 255)
    result_pow_33164 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 17), '**', h_33162, int_33163)
    
    int_33165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 24), 'int')
    # Applying the binary operator 'div' (line 255)
    result_div_33166 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 17), 'div', result_pow_33164, int_33165)
    
    # Getting the type of 'T' (line 255)
    T_33167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 29), 'T')
    # Applying the binary operator '*' (line 255)
    result_mul_33168 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 27), '*', result_div_33166, T_33167)
    
    # Applying the binary operator '-=' (line 255)
    result_isub_33169 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 4), '-=', dPhi_dy_0_33161, result_mul_33168)
    # Assigning a type to the variable 'dPhi_dy_0' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'dPhi_dy_0', result_isub_33169)
    
    
    # Assigning a Call to a Name (line 258):
    
    # Assigning a Call to a Name (line 258):
    
    # Call to empty(...): (line 258)
    # Processing the call arguments (line 258)
    
    # Obtaining an instance of the builtin type 'tuple' (line 258)
    tuple_33172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 258)
    # Adding element type (line 258)
    # Getting the type of 'm' (line 258)
    m_33173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 26), 'm', False)
    int_33174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 30), 'int')
    # Applying the binary operator '-' (line 258)
    result_sub_33175 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 26), '-', m_33173, int_33174)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 26), tuple_33172, result_sub_33175)
    # Adding element type (line 258)
    # Getting the type of 'n' (line 258)
    n_33176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 33), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 26), tuple_33172, n_33176)
    # Adding element type (line 258)
    # Getting the type of 'n' (line 258)
    n_33177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 36), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 26), tuple_33172, n_33177)
    
    # Processing the call keyword arguments (line 258)
    # Getting the type of 'dtype' (line 258)
    dtype_33178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 46), 'dtype', False)
    keyword_33179 = dtype_33178
    kwargs_33180 = {'dtype': keyword_33179}
    # Getting the type of 'np' (line 258)
    np_33170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), 'np', False)
    # Obtaining the member 'empty' of a type (line 258)
    empty_33171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 16), np_33170, 'empty')
    # Calling empty(args, kwargs) (line 258)
    empty_call_result_33181 = invoke(stypy.reporting.localization.Localization(__file__, 258, 16), empty_33171, *[tuple_33172], **kwargs_33180)
    
    # Assigning a type to the variable 'dPhi_dy_1' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'dPhi_dy_1', empty_call_result_33181)
    
    # Assigning a Call to a Subscript (line 259):
    
    # Assigning a Call to a Subscript (line 259):
    
    # Call to identity(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'n' (line 259)
    n_33184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 31), 'n', False)
    # Processing the call keyword arguments (line 259)
    kwargs_33185 = {}
    # Getting the type of 'np' (line 259)
    np_33182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 19), 'np', False)
    # Obtaining the member 'identity' of a type (line 259)
    identity_33183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 19), np_33182, 'identity')
    # Calling identity(args, kwargs) (line 259)
    identity_call_result_33186 = invoke(stypy.reporting.localization.Localization(__file__, 259, 19), identity_33183, *[n_33184], **kwargs_33185)
    
    # Getting the type of 'dPhi_dy_1' (line 259)
    dPhi_dy_1_33187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'dPhi_dy_1')
    slice_33188 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 259, 4), None, None, None)
    # Storing an element on a container (line 259)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 4), dPhi_dy_1_33187, (slice_33188, identity_call_result_33186))
    
    # Getting the type of 'dPhi_dy_1' (line 260)
    dPhi_dy_1_33189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'dPhi_dy_1')
    # Getting the type of 'h' (line 260)
    h_33190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 17), 'h')
    int_33191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 21), 'int')
    # Applying the binary operator 'div' (line 260)
    result_div_33192 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 17), 'div', h_33190, int_33191)
    
    
    # Obtaining the type of the subscript
    int_33193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 32), 'int')
    slice_33194 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 260, 26), int_33193, None, None)
    # Getting the type of 'df_dy' (line 260)
    df_dy_33195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 26), 'df_dy')
    # Obtaining the member '__getitem__' of a type (line 260)
    getitem___33196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 26), df_dy_33195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 260)
    subscript_call_result_33197 = invoke(stypy.reporting.localization.Localization(__file__, 260, 26), getitem___33196, slice_33194)
    
    int_33198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 38), 'int')
    # Getting the type of 'df_dy_middle' (line 260)
    df_dy_middle_33199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 42), 'df_dy_middle')
    # Applying the binary operator '*' (line 260)
    result_mul_33200 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 38), '*', int_33198, df_dy_middle_33199)
    
    # Applying the binary operator '+' (line 260)
    result_add_33201 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 26), '+', subscript_call_result_33197, result_mul_33200)
    
    # Applying the binary operator '*' (line 260)
    result_mul_33202 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 23), '*', result_div_33192, result_add_33201)
    
    # Applying the binary operator '-=' (line 260)
    result_isub_33203 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 4), '-=', dPhi_dy_1_33189, result_mul_33202)
    # Assigning a type to the variable 'dPhi_dy_1' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'dPhi_dy_1', result_isub_33203)
    
    
    # Assigning a Call to a Name (line 261):
    
    # Assigning a Call to a Name (line 261):
    
    # Call to stacked_matmul(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'df_dy_middle' (line 261)
    df_dy_middle_33205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'df_dy_middle', False)
    
    # Obtaining the type of the subscript
    int_33206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 43), 'int')
    slice_33207 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 261, 37), int_33206, None, None)
    # Getting the type of 'df_dy' (line 261)
    df_dy_33208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 37), 'df_dy', False)
    # Obtaining the member '__getitem__' of a type (line 261)
    getitem___33209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 37), df_dy_33208, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 261)
    subscript_call_result_33210 = invoke(stypy.reporting.localization.Localization(__file__, 261, 37), getitem___33209, slice_33207)
    
    # Processing the call keyword arguments (line 261)
    kwargs_33211 = {}
    # Getting the type of 'stacked_matmul' (line 261)
    stacked_matmul_33204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'stacked_matmul', False)
    # Calling stacked_matmul(args, kwargs) (line 261)
    stacked_matmul_call_result_33212 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), stacked_matmul_33204, *[df_dy_middle_33205, subscript_call_result_33210], **kwargs_33211)
    
    # Assigning a type to the variable 'T' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'T', stacked_matmul_call_result_33212)
    
    # Getting the type of 'dPhi_dy_1' (line 262)
    dPhi_dy_1_33213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'dPhi_dy_1')
    # Getting the type of 'h' (line 262)
    h_33214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 17), 'h')
    int_33215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 20), 'int')
    # Applying the binary operator '**' (line 262)
    result_pow_33216 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 17), '**', h_33214, int_33215)
    
    int_33217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 24), 'int')
    # Applying the binary operator 'div' (line 262)
    result_div_33218 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 17), 'div', result_pow_33216, int_33217)
    
    # Getting the type of 'T' (line 262)
    T_33219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 29), 'T')
    # Applying the binary operator '*' (line 262)
    result_mul_33220 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 27), '*', result_div_33218, T_33219)
    
    # Applying the binary operator '+=' (line 262)
    result_iadd_33221 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 4), '+=', dPhi_dy_1_33213, result_mul_33220)
    # Assigning a type to the variable 'dPhi_dy_1' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'dPhi_dy_1', result_iadd_33221)
    
    
    # Assigning a Call to a Name (line 264):
    
    # Assigning a Call to a Name (line 264):
    
    # Call to hstack(...): (line 264)
    # Processing the call arguments (line 264)
    
    # Obtaining an instance of the builtin type 'tuple' (line 264)
    tuple_33224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 264)
    # Adding element type (line 264)
    
    # Call to ravel(...): (line 264)
    # Processing the call keyword arguments (line 264)
    kwargs_33227 = {}
    # Getting the type of 'dPhi_dy_0' (line 264)
    dPhi_dy_0_33225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'dPhi_dy_0', False)
    # Obtaining the member 'ravel' of a type (line 264)
    ravel_33226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 24), dPhi_dy_0_33225, 'ravel')
    # Calling ravel(args, kwargs) (line 264)
    ravel_call_result_33228 = invoke(stypy.reporting.localization.Localization(__file__, 264, 24), ravel_33226, *[], **kwargs_33227)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 24), tuple_33224, ravel_call_result_33228)
    # Adding element type (line 264)
    
    # Call to ravel(...): (line 264)
    # Processing the call keyword arguments (line 264)
    kwargs_33231 = {}
    # Getting the type of 'dPhi_dy_1' (line 264)
    dPhi_dy_1_33229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 43), 'dPhi_dy_1', False)
    # Obtaining the member 'ravel' of a type (line 264)
    ravel_33230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 43), dPhi_dy_1_33229, 'ravel')
    # Calling ravel(args, kwargs) (line 264)
    ravel_call_result_33232 = invoke(stypy.reporting.localization.Localization(__file__, 264, 43), ravel_33230, *[], **kwargs_33231)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 24), tuple_33224, ravel_call_result_33232)
    # Adding element type (line 264)
    
    # Call to ravel(...): (line 264)
    # Processing the call keyword arguments (line 264)
    kwargs_33235 = {}
    # Getting the type of 'dbc_dya' (line 264)
    dbc_dya_33233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 62), 'dbc_dya', False)
    # Obtaining the member 'ravel' of a type (line 264)
    ravel_33234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 62), dbc_dya_33233, 'ravel')
    # Calling ravel(args, kwargs) (line 264)
    ravel_call_result_33236 = invoke(stypy.reporting.localization.Localization(__file__, 264, 62), ravel_33234, *[], **kwargs_33235)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 24), tuple_33224, ravel_call_result_33236)
    # Adding element type (line 264)
    
    # Call to ravel(...): (line 265)
    # Processing the call keyword arguments (line 265)
    kwargs_33239 = {}
    # Getting the type of 'dbc_dyb' (line 265)
    dbc_dyb_33237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 24), 'dbc_dyb', False)
    # Obtaining the member 'ravel' of a type (line 265)
    ravel_33238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 24), dbc_dyb_33237, 'ravel')
    # Calling ravel(args, kwargs) (line 265)
    ravel_call_result_33240 = invoke(stypy.reporting.localization.Localization(__file__, 265, 24), ravel_33238, *[], **kwargs_33239)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 24), tuple_33224, ravel_call_result_33240)
    
    # Processing the call keyword arguments (line 264)
    kwargs_33241 = {}
    # Getting the type of 'np' (line 264)
    np_33222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 13), 'np', False)
    # Obtaining the member 'hstack' of a type (line 264)
    hstack_33223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 13), np_33222, 'hstack')
    # Calling hstack(args, kwargs) (line 264)
    hstack_call_result_33242 = invoke(stypy.reporting.localization.Localization(__file__, 264, 13), hstack_33223, *[tuple_33224], **kwargs_33241)
    
    # Assigning a type to the variable 'values' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'values', hstack_call_result_33242)
    
    
    # Getting the type of 'k' (line 267)
    k_33243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 7), 'k')
    int_33244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 11), 'int')
    # Applying the binary operator '>' (line 267)
    result_gt_33245 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 7), '>', k_33243, int_33244)
    
    # Testing the type of an if condition (line 267)
    if_condition_33246 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 4), result_gt_33245)
    # Assigning a type to the variable 'if_condition_33246' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'if_condition_33246', if_condition_33246)
    # SSA begins for if statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 268):
    
    # Assigning a Call to a Name (line 268):
    
    # Call to transpose(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'df_dp' (line 268)
    df_dp_33249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 29), 'df_dp', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 268)
    tuple_33250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 268)
    # Adding element type (line 268)
    int_33251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 37), tuple_33250, int_33251)
    # Adding element type (line 268)
    int_33252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 37), tuple_33250, int_33252)
    # Adding element type (line 268)
    int_33253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 37), tuple_33250, int_33253)
    
    # Processing the call keyword arguments (line 268)
    kwargs_33254 = {}
    # Getting the type of 'np' (line 268)
    np_33247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'np', False)
    # Obtaining the member 'transpose' of a type (line 268)
    transpose_33248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 16), np_33247, 'transpose')
    # Calling transpose(args, kwargs) (line 268)
    transpose_call_result_33255 = invoke(stypy.reporting.localization.Localization(__file__, 268, 16), transpose_33248, *[df_dp_33249, tuple_33250], **kwargs_33254)
    
    # Assigning a type to the variable 'df_dp' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'df_dp', transpose_call_result_33255)
    
    # Assigning a Call to a Name (line 269):
    
    # Assigning a Call to a Name (line 269):
    
    # Call to transpose(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'df_dp_middle' (line 269)
    df_dp_middle_33258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 36), 'df_dp_middle', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 269)
    tuple_33259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 269)
    # Adding element type (line 269)
    int_33260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 51), tuple_33259, int_33260)
    # Adding element type (line 269)
    int_33261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 51), tuple_33259, int_33261)
    # Adding element type (line 269)
    int_33262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 51), tuple_33259, int_33262)
    
    # Processing the call keyword arguments (line 269)
    kwargs_33263 = {}
    # Getting the type of 'np' (line 269)
    np_33256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 23), 'np', False)
    # Obtaining the member 'transpose' of a type (line 269)
    transpose_33257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 23), np_33256, 'transpose')
    # Calling transpose(args, kwargs) (line 269)
    transpose_call_result_33264 = invoke(stypy.reporting.localization.Localization(__file__, 269, 23), transpose_33257, *[df_dp_middle_33258, tuple_33259], **kwargs_33263)
    
    # Assigning a type to the variable 'df_dp_middle' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'df_dp_middle', transpose_call_result_33264)
    
    # Assigning a Call to a Name (line 270):
    
    # Assigning a Call to a Name (line 270):
    
    # Call to stacked_matmul(...): (line 270)
    # Processing the call arguments (line 270)
    # Getting the type of 'df_dy_middle' (line 270)
    df_dy_middle_33266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 27), 'df_dy_middle', False)
    
    # Obtaining the type of the subscript
    int_33267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 48), 'int')
    slice_33268 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 270, 41), None, int_33267, None)
    # Getting the type of 'df_dp' (line 270)
    df_dp_33269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 41), 'df_dp', False)
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___33270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 41), df_dp_33269, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_33271 = invoke(stypy.reporting.localization.Localization(__file__, 270, 41), getitem___33270, slice_33268)
    
    
    # Obtaining the type of the subscript
    int_33272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 60), 'int')
    slice_33273 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 270, 54), int_33272, None, None)
    # Getting the type of 'df_dp' (line 270)
    df_dp_33274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 54), 'df_dp', False)
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___33275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 54), df_dp_33274, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_33276 = invoke(stypy.reporting.localization.Localization(__file__, 270, 54), getitem___33275, slice_33273)
    
    # Applying the binary operator '-' (line 270)
    result_sub_33277 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 41), '-', subscript_call_result_33271, subscript_call_result_33276)
    
    # Processing the call keyword arguments (line 270)
    kwargs_33278 = {}
    # Getting the type of 'stacked_matmul' (line 270)
    stacked_matmul_33265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'stacked_matmul', False)
    # Calling stacked_matmul(args, kwargs) (line 270)
    stacked_matmul_call_result_33279 = invoke(stypy.reporting.localization.Localization(__file__, 270, 12), stacked_matmul_33265, *[df_dy_middle_33266, result_sub_33277], **kwargs_33278)
    
    # Assigning a type to the variable 'T' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'T', stacked_matmul_call_result_33279)
    
    # Getting the type of 'df_dp_middle' (line 271)
    df_dp_middle_33280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'df_dp_middle')
    float_33281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 24), 'float')
    # Getting the type of 'h' (line 271)
    h_33282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 32), 'h')
    # Applying the binary operator '*' (line 271)
    result_mul_33283 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 24), '*', float_33281, h_33282)
    
    # Getting the type of 'T' (line 271)
    T_33284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 36), 'T')
    # Applying the binary operator '*' (line 271)
    result_mul_33285 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 34), '*', result_mul_33283, T_33284)
    
    # Applying the binary operator '+=' (line 271)
    result_iadd_33286 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 8), '+=', df_dp_middle_33280, result_mul_33285)
    # Assigning a type to the variable 'df_dp_middle' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'df_dp_middle', result_iadd_33286)
    
    
    # Assigning a BinOp to a Name (line 272):
    
    # Assigning a BinOp to a Name (line 272):
    
    # Getting the type of 'h' (line 272)
    h_33287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 19), 'h')
    # Applying the 'usub' unary operator (line 272)
    result___neg___33288 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 18), 'usub', h_33287)
    
    int_33289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 21), 'int')
    # Applying the binary operator 'div' (line 272)
    result_div_33290 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 18), 'div', result___neg___33288, int_33289)
    
    
    # Obtaining the type of the subscript
    int_33291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 33), 'int')
    slice_33292 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 272, 26), None, int_33291, None)
    # Getting the type of 'df_dp' (line 272)
    df_dp_33293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 26), 'df_dp')
    # Obtaining the member '__getitem__' of a type (line 272)
    getitem___33294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 26), df_dp_33293, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 272)
    subscript_call_result_33295 = invoke(stypy.reporting.localization.Localization(__file__, 272, 26), getitem___33294, slice_33292)
    
    
    # Obtaining the type of the subscript
    int_33296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 45), 'int')
    slice_33297 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 272, 39), int_33296, None, None)
    # Getting the type of 'df_dp' (line 272)
    df_dp_33298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 39), 'df_dp')
    # Obtaining the member '__getitem__' of a type (line 272)
    getitem___33299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 39), df_dp_33298, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 272)
    subscript_call_result_33300 = invoke(stypy.reporting.localization.Localization(__file__, 272, 39), getitem___33299, slice_33297)
    
    # Applying the binary operator '+' (line 272)
    result_add_33301 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 26), '+', subscript_call_result_33295, subscript_call_result_33300)
    
    int_33302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 51), 'int')
    # Getting the type of 'df_dp_middle' (line 272)
    df_dp_middle_33303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 55), 'df_dp_middle')
    # Applying the binary operator '*' (line 272)
    result_mul_33304 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 51), '*', int_33302, df_dp_middle_33303)
    
    # Applying the binary operator '+' (line 272)
    result_add_33305 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 49), '+', result_add_33301, result_mul_33304)
    
    # Applying the binary operator '*' (line 272)
    result_mul_33306 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 23), '*', result_div_33290, result_add_33305)
    
    # Assigning a type to the variable 'dPhi_dp' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'dPhi_dp', result_mul_33306)
    
    # Assigning a Call to a Name (line 273):
    
    # Assigning a Call to a Name (line 273):
    
    # Call to hstack(...): (line 273)
    # Processing the call arguments (line 273)
    
    # Obtaining an instance of the builtin type 'tuple' (line 273)
    tuple_33309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 273)
    # Adding element type (line 273)
    # Getting the type of 'values' (line 273)
    values_33310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 28), 'values', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 28), tuple_33309, values_33310)
    # Adding element type (line 273)
    
    # Call to ravel(...): (line 273)
    # Processing the call keyword arguments (line 273)
    kwargs_33313 = {}
    # Getting the type of 'dPhi_dp' (line 273)
    dPhi_dp_33311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 36), 'dPhi_dp', False)
    # Obtaining the member 'ravel' of a type (line 273)
    ravel_33312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 36), dPhi_dp_33311, 'ravel')
    # Calling ravel(args, kwargs) (line 273)
    ravel_call_result_33314 = invoke(stypy.reporting.localization.Localization(__file__, 273, 36), ravel_33312, *[], **kwargs_33313)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 28), tuple_33309, ravel_call_result_33314)
    # Adding element type (line 273)
    
    # Call to ravel(...): (line 273)
    # Processing the call keyword arguments (line 273)
    kwargs_33317 = {}
    # Getting the type of 'dbc_dp' (line 273)
    dbc_dp_33315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 53), 'dbc_dp', False)
    # Obtaining the member 'ravel' of a type (line 273)
    ravel_33316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 53), dbc_dp_33315, 'ravel')
    # Calling ravel(args, kwargs) (line 273)
    ravel_call_result_33318 = invoke(stypy.reporting.localization.Localization(__file__, 273, 53), ravel_33316, *[], **kwargs_33317)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 28), tuple_33309, ravel_call_result_33318)
    
    # Processing the call keyword arguments (line 273)
    kwargs_33319 = {}
    # Getting the type of 'np' (line 273)
    np_33307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 17), 'np', False)
    # Obtaining the member 'hstack' of a type (line 273)
    hstack_33308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 17), np_33307, 'hstack')
    # Calling hstack(args, kwargs) (line 273)
    hstack_call_result_33320 = invoke(stypy.reporting.localization.Localization(__file__, 273, 17), hstack_33308, *[tuple_33309], **kwargs_33319)
    
    # Assigning a type to the variable 'values' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'values', hstack_call_result_33320)
    # SSA join for if statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 275):
    
    # Assigning a Call to a Name (line 275):
    
    # Call to coo_matrix(...): (line 275)
    # Processing the call arguments (line 275)
    
    # Obtaining an instance of the builtin type 'tuple' (line 275)
    tuple_33322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 275)
    # Adding element type (line 275)
    # Getting the type of 'values' (line 275)
    values_33323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 20), 'values', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 20), tuple_33322, values_33323)
    # Adding element type (line 275)
    
    # Obtaining an instance of the builtin type 'tuple' (line 275)
    tuple_33324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 275)
    # Adding element type (line 275)
    # Getting the type of 'i_jac' (line 275)
    i_jac_33325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 29), 'i_jac', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 29), tuple_33324, i_jac_33325)
    # Adding element type (line 275)
    # Getting the type of 'j_jac' (line 275)
    j_jac_33326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 36), 'j_jac', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 29), tuple_33324, j_jac_33326)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 20), tuple_33322, tuple_33324)
    
    # Processing the call keyword arguments (line 275)
    kwargs_33327 = {}
    # Getting the type of 'coo_matrix' (line 275)
    coo_matrix_33321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 275)
    coo_matrix_call_result_33328 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), coo_matrix_33321, *[tuple_33322], **kwargs_33327)
    
    # Assigning a type to the variable 'J' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'J', coo_matrix_call_result_33328)
    
    # Call to csc_matrix(...): (line 276)
    # Processing the call arguments (line 276)
    # Getting the type of 'J' (line 276)
    J_33330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'J', False)
    # Processing the call keyword arguments (line 276)
    kwargs_33331 = {}
    # Getting the type of 'csc_matrix' (line 276)
    csc_matrix_33329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 11), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 276)
    csc_matrix_call_result_33332 = invoke(stypy.reporting.localization.Localization(__file__, 276, 11), csc_matrix_33329, *[J_33330], **kwargs_33331)
    
    # Assigning a type to the variable 'stypy_return_type' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'stypy_return_type', csc_matrix_call_result_33332)
    
    # ################# End of 'construct_global_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'construct_global_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 161)
    stypy_return_type_33333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33333)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'construct_global_jac'
    return stypy_return_type_33333

# Assigning a type to the variable 'construct_global_jac' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'construct_global_jac', construct_global_jac)

@norecursion
def collocation_fun(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'collocation_fun'
    module_type_store = module_type_store.open_function_context('collocation_fun', 279, 0, False)
    
    # Passed parameters checking function
    collocation_fun.stypy_localization = localization
    collocation_fun.stypy_type_of_self = None
    collocation_fun.stypy_type_store = module_type_store
    collocation_fun.stypy_function_name = 'collocation_fun'
    collocation_fun.stypy_param_names_list = ['fun', 'y', 'p', 'x', 'h']
    collocation_fun.stypy_varargs_param_name = None
    collocation_fun.stypy_kwargs_param_name = None
    collocation_fun.stypy_call_defaults = defaults
    collocation_fun.stypy_call_varargs = varargs
    collocation_fun.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'collocation_fun', ['fun', 'y', 'p', 'x', 'h'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'collocation_fun', localization, ['fun', 'y', 'p', 'x', 'h'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'collocation_fun(...)' code ##################

    str_33334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, (-1)), 'str', 'Evaluate collocation residuals.\n\n    This function lies in the core of the method. The solution is sought\n    as a cubic C1 continuous spline with derivatives matching the ODE rhs\n    at given nodes `x`. Collocation conditions are formed from the equality\n    of the spline derivatives and rhs of the ODE system in the middle points\n    between nodes.\n\n    Such method is classified to Lobbato IIIA family in ODE literature.\n    Refer to [1]_ for the formula and some discussion.\n\n    Returns\n    -------\n    col_res : ndarray, shape (n, m - 1)\n        Collocation residuals at the middle points of the mesh intervals.\n    y_middle : ndarray, shape (n, m - 1)\n        Values of the cubic spline evaluated at the middle points of the mesh\n        intervals.\n    f : ndarray, shape (n, m)\n        RHS of the ODE system evaluated at the mesh nodes.\n    f_middle : ndarray, shape (n, m - 1)\n        RHS of the ODE system evaluated at the middle points of the mesh\n        intervals (and using `y_middle`).\n\n    References\n    ----------\n    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual\n           Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,\n           Number 3, pp. 299-316, 2001.\n    ')
    
    # Assigning a Call to a Name (line 310):
    
    # Assigning a Call to a Name (line 310):
    
    # Call to fun(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'x' (line 310)
    x_33336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'x', False)
    # Getting the type of 'y' (line 310)
    y_33337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'y', False)
    # Getting the type of 'p' (line 310)
    p_33338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 18), 'p', False)
    # Processing the call keyword arguments (line 310)
    kwargs_33339 = {}
    # Getting the type of 'fun' (line 310)
    fun_33335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'fun', False)
    # Calling fun(args, kwargs) (line 310)
    fun_call_result_33340 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), fun_33335, *[x_33336, y_33337, p_33338], **kwargs_33339)
    
    # Assigning a type to the variable 'f' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'f', fun_call_result_33340)
    
    # Assigning a BinOp to a Name (line 311):
    
    # Assigning a BinOp to a Name (line 311):
    float_33341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 16), 'float')
    
    # Obtaining the type of the subscript
    slice_33342 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 311, 23), None, None, None)
    int_33343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 28), 'int')
    slice_33344 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 311, 23), int_33343, None, None)
    # Getting the type of 'y' (line 311)
    y_33345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 23), 'y')
    # Obtaining the member '__getitem__' of a type (line 311)
    getitem___33346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 23), y_33345, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 311)
    subscript_call_result_33347 = invoke(stypy.reporting.localization.Localization(__file__, 311, 23), getitem___33346, (slice_33342, slice_33344))
    
    
    # Obtaining the type of the subscript
    slice_33348 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 311, 34), None, None, None)
    int_33349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 40), 'int')
    slice_33350 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 311, 34), None, int_33349, None)
    # Getting the type of 'y' (line 311)
    y_33351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 34), 'y')
    # Obtaining the member '__getitem__' of a type (line 311)
    getitem___33352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 34), y_33351, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 311)
    subscript_call_result_33353 = invoke(stypy.reporting.localization.Localization(__file__, 311, 34), getitem___33352, (slice_33348, slice_33350))
    
    # Applying the binary operator '+' (line 311)
    result_add_33354 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 23), '+', subscript_call_result_33347, subscript_call_result_33353)
    
    # Applying the binary operator '*' (line 311)
    result_mul_33355 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 16), '*', float_33341, result_add_33354)
    
    float_33356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 16), 'float')
    # Getting the type of 'h' (line 312)
    h_33357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 24), 'h')
    # Applying the binary operator '*' (line 312)
    result_mul_33358 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 16), '*', float_33356, h_33357)
    
    
    # Obtaining the type of the subscript
    slice_33359 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 312, 29), None, None, None)
    int_33360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 34), 'int')
    slice_33361 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 312, 29), int_33360, None, None)
    # Getting the type of 'f' (line 312)
    f_33362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 29), 'f')
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___33363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 29), f_33362, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_33364 = invoke(stypy.reporting.localization.Localization(__file__, 312, 29), getitem___33363, (slice_33359, slice_33361))
    
    
    # Obtaining the type of the subscript
    slice_33365 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 312, 40), None, None, None)
    int_33366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 46), 'int')
    slice_33367 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 312, 40), None, int_33366, None)
    # Getting the type of 'f' (line 312)
    f_33368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 40), 'f')
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___33369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 40), f_33368, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_33370 = invoke(stypy.reporting.localization.Localization(__file__, 312, 40), getitem___33369, (slice_33365, slice_33367))
    
    # Applying the binary operator '-' (line 312)
    result_sub_33371 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 29), '-', subscript_call_result_33364, subscript_call_result_33370)
    
    # Applying the binary operator '*' (line 312)
    result_mul_33372 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 26), '*', result_mul_33358, result_sub_33371)
    
    # Applying the binary operator '-' (line 311)
    result_sub_33373 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 16), '-', result_mul_33355, result_mul_33372)
    
    # Assigning a type to the variable 'y_middle' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'y_middle', result_sub_33373)
    
    # Assigning a Call to a Name (line 313):
    
    # Assigning a Call to a Name (line 313):
    
    # Call to fun(...): (line 313)
    # Processing the call arguments (line 313)
    
    # Obtaining the type of the subscript
    int_33375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 22), 'int')
    slice_33376 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 313, 19), None, int_33375, None)
    # Getting the type of 'x' (line 313)
    x_33377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___33378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 19), x_33377, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_33379 = invoke(stypy.reporting.localization.Localization(__file__, 313, 19), getitem___33378, slice_33376)
    
    float_33380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 28), 'float')
    # Getting the type of 'h' (line 313)
    h_33381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 34), 'h', False)
    # Applying the binary operator '*' (line 313)
    result_mul_33382 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 28), '*', float_33380, h_33381)
    
    # Applying the binary operator '+' (line 313)
    result_add_33383 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 19), '+', subscript_call_result_33379, result_mul_33382)
    
    # Getting the type of 'y_middle' (line 313)
    y_middle_33384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 37), 'y_middle', False)
    # Getting the type of 'p' (line 313)
    p_33385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 47), 'p', False)
    # Processing the call keyword arguments (line 313)
    kwargs_33386 = {}
    # Getting the type of 'fun' (line 313)
    fun_33374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'fun', False)
    # Calling fun(args, kwargs) (line 313)
    fun_call_result_33387 = invoke(stypy.reporting.localization.Localization(__file__, 313, 15), fun_33374, *[result_add_33383, y_middle_33384, p_33385], **kwargs_33386)
    
    # Assigning a type to the variable 'f_middle' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'f_middle', fun_call_result_33387)
    
    # Assigning a BinOp to a Name (line 314):
    
    # Assigning a BinOp to a Name (line 314):
    
    # Obtaining the type of the subscript
    slice_33388 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 314, 14), None, None, None)
    int_33389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 19), 'int')
    slice_33390 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 314, 14), int_33389, None, None)
    # Getting the type of 'y' (line 314)
    y_33391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 14), 'y')
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___33392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 14), y_33391, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_33393 = invoke(stypy.reporting.localization.Localization(__file__, 314, 14), getitem___33392, (slice_33388, slice_33390))
    
    
    # Obtaining the type of the subscript
    slice_33394 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 314, 25), None, None, None)
    int_33395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 31), 'int')
    slice_33396 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 314, 25), None, int_33395, None)
    # Getting the type of 'y' (line 314)
    y_33397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 25), 'y')
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___33398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 25), y_33397, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_33399 = invoke(stypy.reporting.localization.Localization(__file__, 314, 25), getitem___33398, (slice_33394, slice_33396))
    
    # Applying the binary operator '-' (line 314)
    result_sub_33400 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 14), '-', subscript_call_result_33393, subscript_call_result_33399)
    
    # Getting the type of 'h' (line 314)
    h_33401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 37), 'h')
    int_33402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 41), 'int')
    # Applying the binary operator 'div' (line 314)
    result_div_33403 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 37), 'div', h_33401, int_33402)
    
    
    # Obtaining the type of the subscript
    slice_33404 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 314, 46), None, None, None)
    int_33405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 52), 'int')
    slice_33406 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 314, 46), None, int_33405, None)
    # Getting the type of 'f' (line 314)
    f_33407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 46), 'f')
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___33408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 46), f_33407, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_33409 = invoke(stypy.reporting.localization.Localization(__file__, 314, 46), getitem___33408, (slice_33404, slice_33406))
    
    
    # Obtaining the type of the subscript
    slice_33410 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 314, 58), None, None, None)
    int_33411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 63), 'int')
    slice_33412 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 314, 58), int_33411, None, None)
    # Getting the type of 'f' (line 314)
    f_33413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 58), 'f')
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___33414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 58), f_33413, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_33415 = invoke(stypy.reporting.localization.Localization(__file__, 314, 58), getitem___33414, (slice_33410, slice_33412))
    
    # Applying the binary operator '+' (line 314)
    result_add_33416 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 46), '+', subscript_call_result_33409, subscript_call_result_33415)
    
    int_33417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 46), 'int')
    # Getting the type of 'f_middle' (line 315)
    f_middle_33418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 50), 'f_middle')
    # Applying the binary operator '*' (line 315)
    result_mul_33419 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 46), '*', int_33417, f_middle_33418)
    
    # Applying the binary operator '+' (line 314)
    result_add_33420 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 67), '+', result_add_33416, result_mul_33419)
    
    # Applying the binary operator '*' (line 314)
    result_mul_33421 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 43), '*', result_div_33403, result_add_33420)
    
    # Applying the binary operator '-' (line 314)
    result_sub_33422 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 35), '-', result_sub_33400, result_mul_33421)
    
    # Assigning a type to the variable 'col_res' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'col_res', result_sub_33422)
    
    # Obtaining an instance of the builtin type 'tuple' (line 317)
    tuple_33423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 317)
    # Adding element type (line 317)
    # Getting the type of 'col_res' (line 317)
    col_res_33424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), 'col_res')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 11), tuple_33423, col_res_33424)
    # Adding element type (line 317)
    # Getting the type of 'y_middle' (line 317)
    y_middle_33425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'y_middle')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 11), tuple_33423, y_middle_33425)
    # Adding element type (line 317)
    # Getting the type of 'f' (line 317)
    f_33426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 30), 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 11), tuple_33423, f_33426)
    # Adding element type (line 317)
    # Getting the type of 'f_middle' (line 317)
    f_middle_33427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 33), 'f_middle')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 11), tuple_33423, f_middle_33427)
    
    # Assigning a type to the variable 'stypy_return_type' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type', tuple_33423)
    
    # ################# End of 'collocation_fun(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'collocation_fun' in the type store
    # Getting the type of 'stypy_return_type' (line 279)
    stypy_return_type_33428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33428)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'collocation_fun'
    return stypy_return_type_33428

# Assigning a type to the variable 'collocation_fun' (line 279)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 0), 'collocation_fun', collocation_fun)

@norecursion
def prepare_sys(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'prepare_sys'
    module_type_store = module_type_store.open_function_context('prepare_sys', 320, 0, False)
    
    # Passed parameters checking function
    prepare_sys.stypy_localization = localization
    prepare_sys.stypy_type_of_self = None
    prepare_sys.stypy_type_store = module_type_store
    prepare_sys.stypy_function_name = 'prepare_sys'
    prepare_sys.stypy_param_names_list = ['n', 'm', 'k', 'fun', 'bc', 'fun_jac', 'bc_jac', 'x', 'h']
    prepare_sys.stypy_varargs_param_name = None
    prepare_sys.stypy_kwargs_param_name = None
    prepare_sys.stypy_call_defaults = defaults
    prepare_sys.stypy_call_varargs = varargs
    prepare_sys.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'prepare_sys', ['n', 'm', 'k', 'fun', 'bc', 'fun_jac', 'bc_jac', 'x', 'h'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'prepare_sys', localization, ['n', 'm', 'k', 'fun', 'bc', 'fun_jac', 'bc_jac', 'x', 'h'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'prepare_sys(...)' code ##################

    str_33429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 4), 'str', 'Create the function and the Jacobian for the collocation system.')
    
    # Assigning a BinOp to a Name (line 322):
    
    # Assigning a BinOp to a Name (line 322):
    
    # Obtaining the type of the subscript
    int_33430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 18), 'int')
    slice_33431 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 322, 15), None, int_33430, None)
    # Getting the type of 'x' (line 322)
    x_33432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 15), 'x')
    # Obtaining the member '__getitem__' of a type (line 322)
    getitem___33433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 15), x_33432, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 322)
    subscript_call_result_33434 = invoke(stypy.reporting.localization.Localization(__file__, 322, 15), getitem___33433, slice_33431)
    
    float_33435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 24), 'float')
    # Getting the type of 'h' (line 322)
    h_33436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 30), 'h')
    # Applying the binary operator '*' (line 322)
    result_mul_33437 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 24), '*', float_33435, h_33436)
    
    # Applying the binary operator '+' (line 322)
    result_add_33438 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 15), '+', subscript_call_result_33434, result_mul_33437)
    
    # Assigning a type to the variable 'x_middle' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'x_middle', result_add_33438)
    
    # Assigning a Call to a Tuple (line 323):
    
    # Assigning a Subscript to a Name (line 323):
    
    # Obtaining the type of the subscript
    int_33439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 4), 'int')
    
    # Call to compute_jac_indices(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'n' (line 323)
    n_33441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 39), 'n', False)
    # Getting the type of 'm' (line 323)
    m_33442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 42), 'm', False)
    # Getting the type of 'k' (line 323)
    k_33443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 45), 'k', False)
    # Processing the call keyword arguments (line 323)
    kwargs_33444 = {}
    # Getting the type of 'compute_jac_indices' (line 323)
    compute_jac_indices_33440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'compute_jac_indices', False)
    # Calling compute_jac_indices(args, kwargs) (line 323)
    compute_jac_indices_call_result_33445 = invoke(stypy.reporting.localization.Localization(__file__, 323, 19), compute_jac_indices_33440, *[n_33441, m_33442, k_33443], **kwargs_33444)
    
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___33446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 4), compute_jac_indices_call_result_33445, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
    subscript_call_result_33447 = invoke(stypy.reporting.localization.Localization(__file__, 323, 4), getitem___33446, int_33439)
    
    # Assigning a type to the variable 'tuple_var_assignment_32384' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'tuple_var_assignment_32384', subscript_call_result_33447)
    
    # Assigning a Subscript to a Name (line 323):
    
    # Obtaining the type of the subscript
    int_33448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 4), 'int')
    
    # Call to compute_jac_indices(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'n' (line 323)
    n_33450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 39), 'n', False)
    # Getting the type of 'm' (line 323)
    m_33451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 42), 'm', False)
    # Getting the type of 'k' (line 323)
    k_33452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 45), 'k', False)
    # Processing the call keyword arguments (line 323)
    kwargs_33453 = {}
    # Getting the type of 'compute_jac_indices' (line 323)
    compute_jac_indices_33449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'compute_jac_indices', False)
    # Calling compute_jac_indices(args, kwargs) (line 323)
    compute_jac_indices_call_result_33454 = invoke(stypy.reporting.localization.Localization(__file__, 323, 19), compute_jac_indices_33449, *[n_33450, m_33451, k_33452], **kwargs_33453)
    
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___33455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 4), compute_jac_indices_call_result_33454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
    subscript_call_result_33456 = invoke(stypy.reporting.localization.Localization(__file__, 323, 4), getitem___33455, int_33448)
    
    # Assigning a type to the variable 'tuple_var_assignment_32385' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'tuple_var_assignment_32385', subscript_call_result_33456)
    
    # Assigning a Name to a Name (line 323):
    # Getting the type of 'tuple_var_assignment_32384' (line 323)
    tuple_var_assignment_32384_33457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'tuple_var_assignment_32384')
    # Assigning a type to the variable 'i_jac' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'i_jac', tuple_var_assignment_32384_33457)
    
    # Assigning a Name to a Name (line 323):
    # Getting the type of 'tuple_var_assignment_32385' (line 323)
    tuple_var_assignment_32385_33458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'tuple_var_assignment_32385')
    # Assigning a type to the variable 'j_jac' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 11), 'j_jac', tuple_var_assignment_32385_33458)

    @norecursion
    def col_fun(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'col_fun'
        module_type_store = module_type_store.open_function_context('col_fun', 325, 4, False)
        
        # Passed parameters checking function
        col_fun.stypy_localization = localization
        col_fun.stypy_type_of_self = None
        col_fun.stypy_type_store = module_type_store
        col_fun.stypy_function_name = 'col_fun'
        col_fun.stypy_param_names_list = ['y', 'p']
        col_fun.stypy_varargs_param_name = None
        col_fun.stypy_kwargs_param_name = None
        col_fun.stypy_call_defaults = defaults
        col_fun.stypy_call_varargs = varargs
        col_fun.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'col_fun', ['y', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'col_fun', localization, ['y', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'col_fun(...)' code ##################

        
        # Call to collocation_fun(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'fun' (line 326)
        fun_33460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 31), 'fun', False)
        # Getting the type of 'y' (line 326)
        y_33461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'y', False)
        # Getting the type of 'p' (line 326)
        p_33462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 39), 'p', False)
        # Getting the type of 'x' (line 326)
        x_33463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 42), 'x', False)
        # Getting the type of 'h' (line 326)
        h_33464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 45), 'h', False)
        # Processing the call keyword arguments (line 326)
        kwargs_33465 = {}
        # Getting the type of 'collocation_fun' (line 326)
        collocation_fun_33459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 15), 'collocation_fun', False)
        # Calling collocation_fun(args, kwargs) (line 326)
        collocation_fun_call_result_33466 = invoke(stypy.reporting.localization.Localization(__file__, 326, 15), collocation_fun_33459, *[fun_33460, y_33461, p_33462, x_33463, h_33464], **kwargs_33465)
        
        # Assigning a type to the variable 'stypy_return_type' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'stypy_return_type', collocation_fun_call_result_33466)
        
        # ################# End of 'col_fun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'col_fun' in the type store
        # Getting the type of 'stypy_return_type' (line 325)
        stypy_return_type_33467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33467)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'col_fun'
        return stypy_return_type_33467

    # Assigning a type to the variable 'col_fun' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'col_fun', col_fun)

    @norecursion
    def sys_jac(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sys_jac'
        module_type_store = module_type_store.open_function_context('sys_jac', 328, 4, False)
        
        # Passed parameters checking function
        sys_jac.stypy_localization = localization
        sys_jac.stypy_type_of_self = None
        sys_jac.stypy_type_store = module_type_store
        sys_jac.stypy_function_name = 'sys_jac'
        sys_jac.stypy_param_names_list = ['y', 'p', 'y_middle', 'f', 'f_middle', 'bc0']
        sys_jac.stypy_varargs_param_name = None
        sys_jac.stypy_kwargs_param_name = None
        sys_jac.stypy_call_defaults = defaults
        sys_jac.stypy_call_varargs = varargs
        sys_jac.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'sys_jac', ['y', 'p', 'y_middle', 'f', 'f_middle', 'bc0'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sys_jac', localization, ['y', 'p', 'y_middle', 'f', 'f_middle', 'bc0'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sys_jac(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 329)
        # Getting the type of 'fun_jac' (line 329)
        fun_jac_33468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 11), 'fun_jac')
        # Getting the type of 'None' (line 329)
        None_33469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 22), 'None')
        
        (may_be_33470, more_types_in_union_33471) = may_be_none(fun_jac_33468, None_33469)

        if may_be_33470:

            if more_types_in_union_33471:
                # Runtime conditional SSA (line 329)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Tuple (line 330):
            
            # Assigning a Subscript to a Name (line 330):
            
            # Obtaining the type of the subscript
            int_33472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 12), 'int')
            
            # Call to estimate_fun_jac(...): (line 330)
            # Processing the call arguments (line 330)
            # Getting the type of 'fun' (line 330)
            fun_33474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 44), 'fun', False)
            # Getting the type of 'x' (line 330)
            x_33475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 49), 'x', False)
            # Getting the type of 'y' (line 330)
            y_33476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 52), 'y', False)
            # Getting the type of 'p' (line 330)
            p_33477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 55), 'p', False)
            # Getting the type of 'f' (line 330)
            f_33478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 58), 'f', False)
            # Processing the call keyword arguments (line 330)
            kwargs_33479 = {}
            # Getting the type of 'estimate_fun_jac' (line 330)
            estimate_fun_jac_33473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 27), 'estimate_fun_jac', False)
            # Calling estimate_fun_jac(args, kwargs) (line 330)
            estimate_fun_jac_call_result_33480 = invoke(stypy.reporting.localization.Localization(__file__, 330, 27), estimate_fun_jac_33473, *[fun_33474, x_33475, y_33476, p_33477, f_33478], **kwargs_33479)
            
            # Obtaining the member '__getitem__' of a type (line 330)
            getitem___33481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), estimate_fun_jac_call_result_33480, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 330)
            subscript_call_result_33482 = invoke(stypy.reporting.localization.Localization(__file__, 330, 12), getitem___33481, int_33472)
            
            # Assigning a type to the variable 'tuple_var_assignment_32386' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'tuple_var_assignment_32386', subscript_call_result_33482)
            
            # Assigning a Subscript to a Name (line 330):
            
            # Obtaining the type of the subscript
            int_33483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 12), 'int')
            
            # Call to estimate_fun_jac(...): (line 330)
            # Processing the call arguments (line 330)
            # Getting the type of 'fun' (line 330)
            fun_33485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 44), 'fun', False)
            # Getting the type of 'x' (line 330)
            x_33486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 49), 'x', False)
            # Getting the type of 'y' (line 330)
            y_33487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 52), 'y', False)
            # Getting the type of 'p' (line 330)
            p_33488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 55), 'p', False)
            # Getting the type of 'f' (line 330)
            f_33489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 58), 'f', False)
            # Processing the call keyword arguments (line 330)
            kwargs_33490 = {}
            # Getting the type of 'estimate_fun_jac' (line 330)
            estimate_fun_jac_33484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 27), 'estimate_fun_jac', False)
            # Calling estimate_fun_jac(args, kwargs) (line 330)
            estimate_fun_jac_call_result_33491 = invoke(stypy.reporting.localization.Localization(__file__, 330, 27), estimate_fun_jac_33484, *[fun_33485, x_33486, y_33487, p_33488, f_33489], **kwargs_33490)
            
            # Obtaining the member '__getitem__' of a type (line 330)
            getitem___33492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), estimate_fun_jac_call_result_33491, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 330)
            subscript_call_result_33493 = invoke(stypy.reporting.localization.Localization(__file__, 330, 12), getitem___33492, int_33483)
            
            # Assigning a type to the variable 'tuple_var_assignment_32387' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'tuple_var_assignment_32387', subscript_call_result_33493)
            
            # Assigning a Name to a Name (line 330):
            # Getting the type of 'tuple_var_assignment_32386' (line 330)
            tuple_var_assignment_32386_33494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'tuple_var_assignment_32386')
            # Assigning a type to the variable 'df_dy' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'df_dy', tuple_var_assignment_32386_33494)
            
            # Assigning a Name to a Name (line 330):
            # Getting the type of 'tuple_var_assignment_32387' (line 330)
            tuple_var_assignment_32387_33495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'tuple_var_assignment_32387')
            # Assigning a type to the variable 'df_dp' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'df_dp', tuple_var_assignment_32387_33495)
            
            # Assigning a Call to a Tuple (line 331):
            
            # Assigning a Subscript to a Name (line 331):
            
            # Obtaining the type of the subscript
            int_33496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 12), 'int')
            
            # Call to estimate_fun_jac(...): (line 331)
            # Processing the call arguments (line 331)
            # Getting the type of 'fun' (line 332)
            fun_33498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 'fun', False)
            # Getting the type of 'x_middle' (line 332)
            x_middle_33499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 21), 'x_middle', False)
            # Getting the type of 'y_middle' (line 332)
            y_middle_33500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 31), 'y_middle', False)
            # Getting the type of 'p' (line 332)
            p_33501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 41), 'p', False)
            # Getting the type of 'f_middle' (line 332)
            f_middle_33502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 44), 'f_middle', False)
            # Processing the call keyword arguments (line 331)
            kwargs_33503 = {}
            # Getting the type of 'estimate_fun_jac' (line 331)
            estimate_fun_jac_33497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 41), 'estimate_fun_jac', False)
            # Calling estimate_fun_jac(args, kwargs) (line 331)
            estimate_fun_jac_call_result_33504 = invoke(stypy.reporting.localization.Localization(__file__, 331, 41), estimate_fun_jac_33497, *[fun_33498, x_middle_33499, y_middle_33500, p_33501, f_middle_33502], **kwargs_33503)
            
            # Obtaining the member '__getitem__' of a type (line 331)
            getitem___33505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), estimate_fun_jac_call_result_33504, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 331)
            subscript_call_result_33506 = invoke(stypy.reporting.localization.Localization(__file__, 331, 12), getitem___33505, int_33496)
            
            # Assigning a type to the variable 'tuple_var_assignment_32388' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'tuple_var_assignment_32388', subscript_call_result_33506)
            
            # Assigning a Subscript to a Name (line 331):
            
            # Obtaining the type of the subscript
            int_33507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 12), 'int')
            
            # Call to estimate_fun_jac(...): (line 331)
            # Processing the call arguments (line 331)
            # Getting the type of 'fun' (line 332)
            fun_33509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 'fun', False)
            # Getting the type of 'x_middle' (line 332)
            x_middle_33510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 21), 'x_middle', False)
            # Getting the type of 'y_middle' (line 332)
            y_middle_33511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 31), 'y_middle', False)
            # Getting the type of 'p' (line 332)
            p_33512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 41), 'p', False)
            # Getting the type of 'f_middle' (line 332)
            f_middle_33513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 44), 'f_middle', False)
            # Processing the call keyword arguments (line 331)
            kwargs_33514 = {}
            # Getting the type of 'estimate_fun_jac' (line 331)
            estimate_fun_jac_33508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 41), 'estimate_fun_jac', False)
            # Calling estimate_fun_jac(args, kwargs) (line 331)
            estimate_fun_jac_call_result_33515 = invoke(stypy.reporting.localization.Localization(__file__, 331, 41), estimate_fun_jac_33508, *[fun_33509, x_middle_33510, y_middle_33511, p_33512, f_middle_33513], **kwargs_33514)
            
            # Obtaining the member '__getitem__' of a type (line 331)
            getitem___33516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), estimate_fun_jac_call_result_33515, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 331)
            subscript_call_result_33517 = invoke(stypy.reporting.localization.Localization(__file__, 331, 12), getitem___33516, int_33507)
            
            # Assigning a type to the variable 'tuple_var_assignment_32389' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'tuple_var_assignment_32389', subscript_call_result_33517)
            
            # Assigning a Name to a Name (line 331):
            # Getting the type of 'tuple_var_assignment_32388' (line 331)
            tuple_var_assignment_32388_33518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'tuple_var_assignment_32388')
            # Assigning a type to the variable 'df_dy_middle' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'df_dy_middle', tuple_var_assignment_32388_33518)
            
            # Assigning a Name to a Name (line 331):
            # Getting the type of 'tuple_var_assignment_32389' (line 331)
            tuple_var_assignment_32389_33519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'tuple_var_assignment_32389')
            # Assigning a type to the variable 'df_dp_middle' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'df_dp_middle', tuple_var_assignment_32389_33519)

            if more_types_in_union_33471:
                # Runtime conditional SSA for else branch (line 329)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_33470) or more_types_in_union_33471):
            
            # Assigning a Call to a Tuple (line 334):
            
            # Assigning a Subscript to a Name (line 334):
            
            # Obtaining the type of the subscript
            int_33520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 12), 'int')
            
            # Call to fun_jac(...): (line 334)
            # Processing the call arguments (line 334)
            # Getting the type of 'x' (line 334)
            x_33522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 35), 'x', False)
            # Getting the type of 'y' (line 334)
            y_33523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 38), 'y', False)
            # Getting the type of 'p' (line 334)
            p_33524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 41), 'p', False)
            # Processing the call keyword arguments (line 334)
            kwargs_33525 = {}
            # Getting the type of 'fun_jac' (line 334)
            fun_jac_33521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 27), 'fun_jac', False)
            # Calling fun_jac(args, kwargs) (line 334)
            fun_jac_call_result_33526 = invoke(stypy.reporting.localization.Localization(__file__, 334, 27), fun_jac_33521, *[x_33522, y_33523, p_33524], **kwargs_33525)
            
            # Obtaining the member '__getitem__' of a type (line 334)
            getitem___33527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 12), fun_jac_call_result_33526, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 334)
            subscript_call_result_33528 = invoke(stypy.reporting.localization.Localization(__file__, 334, 12), getitem___33527, int_33520)
            
            # Assigning a type to the variable 'tuple_var_assignment_32390' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'tuple_var_assignment_32390', subscript_call_result_33528)
            
            # Assigning a Subscript to a Name (line 334):
            
            # Obtaining the type of the subscript
            int_33529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 12), 'int')
            
            # Call to fun_jac(...): (line 334)
            # Processing the call arguments (line 334)
            # Getting the type of 'x' (line 334)
            x_33531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 35), 'x', False)
            # Getting the type of 'y' (line 334)
            y_33532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 38), 'y', False)
            # Getting the type of 'p' (line 334)
            p_33533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 41), 'p', False)
            # Processing the call keyword arguments (line 334)
            kwargs_33534 = {}
            # Getting the type of 'fun_jac' (line 334)
            fun_jac_33530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 27), 'fun_jac', False)
            # Calling fun_jac(args, kwargs) (line 334)
            fun_jac_call_result_33535 = invoke(stypy.reporting.localization.Localization(__file__, 334, 27), fun_jac_33530, *[x_33531, y_33532, p_33533], **kwargs_33534)
            
            # Obtaining the member '__getitem__' of a type (line 334)
            getitem___33536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 12), fun_jac_call_result_33535, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 334)
            subscript_call_result_33537 = invoke(stypy.reporting.localization.Localization(__file__, 334, 12), getitem___33536, int_33529)
            
            # Assigning a type to the variable 'tuple_var_assignment_32391' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'tuple_var_assignment_32391', subscript_call_result_33537)
            
            # Assigning a Name to a Name (line 334):
            # Getting the type of 'tuple_var_assignment_32390' (line 334)
            tuple_var_assignment_32390_33538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'tuple_var_assignment_32390')
            # Assigning a type to the variable 'df_dy' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'df_dy', tuple_var_assignment_32390_33538)
            
            # Assigning a Name to a Name (line 334):
            # Getting the type of 'tuple_var_assignment_32391' (line 334)
            tuple_var_assignment_32391_33539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'tuple_var_assignment_32391')
            # Assigning a type to the variable 'df_dp' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'df_dp', tuple_var_assignment_32391_33539)
            
            # Assigning a Call to a Tuple (line 335):
            
            # Assigning a Subscript to a Name (line 335):
            
            # Obtaining the type of the subscript
            int_33540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 12), 'int')
            
            # Call to fun_jac(...): (line 335)
            # Processing the call arguments (line 335)
            # Getting the type of 'x_middle' (line 335)
            x_middle_33542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 49), 'x_middle', False)
            # Getting the type of 'y_middle' (line 335)
            y_middle_33543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 59), 'y_middle', False)
            # Getting the type of 'p' (line 335)
            p_33544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 69), 'p', False)
            # Processing the call keyword arguments (line 335)
            kwargs_33545 = {}
            # Getting the type of 'fun_jac' (line 335)
            fun_jac_33541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 41), 'fun_jac', False)
            # Calling fun_jac(args, kwargs) (line 335)
            fun_jac_call_result_33546 = invoke(stypy.reporting.localization.Localization(__file__, 335, 41), fun_jac_33541, *[x_middle_33542, y_middle_33543, p_33544], **kwargs_33545)
            
            # Obtaining the member '__getitem__' of a type (line 335)
            getitem___33547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), fun_jac_call_result_33546, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 335)
            subscript_call_result_33548 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), getitem___33547, int_33540)
            
            # Assigning a type to the variable 'tuple_var_assignment_32392' (line 335)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'tuple_var_assignment_32392', subscript_call_result_33548)
            
            # Assigning a Subscript to a Name (line 335):
            
            # Obtaining the type of the subscript
            int_33549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 12), 'int')
            
            # Call to fun_jac(...): (line 335)
            # Processing the call arguments (line 335)
            # Getting the type of 'x_middle' (line 335)
            x_middle_33551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 49), 'x_middle', False)
            # Getting the type of 'y_middle' (line 335)
            y_middle_33552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 59), 'y_middle', False)
            # Getting the type of 'p' (line 335)
            p_33553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 69), 'p', False)
            # Processing the call keyword arguments (line 335)
            kwargs_33554 = {}
            # Getting the type of 'fun_jac' (line 335)
            fun_jac_33550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 41), 'fun_jac', False)
            # Calling fun_jac(args, kwargs) (line 335)
            fun_jac_call_result_33555 = invoke(stypy.reporting.localization.Localization(__file__, 335, 41), fun_jac_33550, *[x_middle_33551, y_middle_33552, p_33553], **kwargs_33554)
            
            # Obtaining the member '__getitem__' of a type (line 335)
            getitem___33556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), fun_jac_call_result_33555, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 335)
            subscript_call_result_33557 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), getitem___33556, int_33549)
            
            # Assigning a type to the variable 'tuple_var_assignment_32393' (line 335)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'tuple_var_assignment_32393', subscript_call_result_33557)
            
            # Assigning a Name to a Name (line 335):
            # Getting the type of 'tuple_var_assignment_32392' (line 335)
            tuple_var_assignment_32392_33558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'tuple_var_assignment_32392')
            # Assigning a type to the variable 'df_dy_middle' (line 335)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'df_dy_middle', tuple_var_assignment_32392_33558)
            
            # Assigning a Name to a Name (line 335):
            # Getting the type of 'tuple_var_assignment_32393' (line 335)
            tuple_var_assignment_32393_33559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'tuple_var_assignment_32393')
            # Assigning a type to the variable 'df_dp_middle' (line 335)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 26), 'df_dp_middle', tuple_var_assignment_32393_33559)

            if (may_be_33470 and more_types_in_union_33471):
                # SSA join for if statement (line 329)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 337)
        # Getting the type of 'bc_jac' (line 337)
        bc_jac_33560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 11), 'bc_jac')
        # Getting the type of 'None' (line 337)
        None_33561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 21), 'None')
        
        (may_be_33562, more_types_in_union_33563) = may_be_none(bc_jac_33560, None_33561)

        if may_be_33562:

            if more_types_in_union_33563:
                # Runtime conditional SSA (line 337)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Tuple (line 338):
            
            # Assigning a Subscript to a Name (line 338):
            
            # Obtaining the type of the subscript
            int_33564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 12), 'int')
            
            # Call to estimate_bc_jac(...): (line 338)
            # Processing the call arguments (line 338)
            # Getting the type of 'bc' (line 338)
            bc_33566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 55), 'bc', False)
            
            # Obtaining the type of the subscript
            slice_33567 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 338, 59), None, None, None)
            int_33568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 64), 'int')
            # Getting the type of 'y' (line 338)
            y_33569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 59), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 338)
            getitem___33570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 59), y_33569, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 338)
            subscript_call_result_33571 = invoke(stypy.reporting.localization.Localization(__file__, 338, 59), getitem___33570, (slice_33567, int_33568))
            
            
            # Obtaining the type of the subscript
            slice_33572 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 338, 68), None, None, None)
            int_33573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 73), 'int')
            # Getting the type of 'y' (line 338)
            y_33574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 68), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 338)
            getitem___33575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 68), y_33574, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 338)
            subscript_call_result_33576 = invoke(stypy.reporting.localization.Localization(__file__, 338, 68), getitem___33575, (slice_33572, int_33573))
            
            # Getting the type of 'p' (line 339)
            p_33577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 55), 'p', False)
            # Getting the type of 'bc0' (line 339)
            bc0_33578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 58), 'bc0', False)
            # Processing the call keyword arguments (line 338)
            kwargs_33579 = {}
            # Getting the type of 'estimate_bc_jac' (line 338)
            estimate_bc_jac_33565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 39), 'estimate_bc_jac', False)
            # Calling estimate_bc_jac(args, kwargs) (line 338)
            estimate_bc_jac_call_result_33580 = invoke(stypy.reporting.localization.Localization(__file__, 338, 39), estimate_bc_jac_33565, *[bc_33566, subscript_call_result_33571, subscript_call_result_33576, p_33577, bc0_33578], **kwargs_33579)
            
            # Obtaining the member '__getitem__' of a type (line 338)
            getitem___33581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), estimate_bc_jac_call_result_33580, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 338)
            subscript_call_result_33582 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), getitem___33581, int_33564)
            
            # Assigning a type to the variable 'tuple_var_assignment_32394' (line 338)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'tuple_var_assignment_32394', subscript_call_result_33582)
            
            # Assigning a Subscript to a Name (line 338):
            
            # Obtaining the type of the subscript
            int_33583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 12), 'int')
            
            # Call to estimate_bc_jac(...): (line 338)
            # Processing the call arguments (line 338)
            # Getting the type of 'bc' (line 338)
            bc_33585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 55), 'bc', False)
            
            # Obtaining the type of the subscript
            slice_33586 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 338, 59), None, None, None)
            int_33587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 64), 'int')
            # Getting the type of 'y' (line 338)
            y_33588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 59), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 338)
            getitem___33589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 59), y_33588, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 338)
            subscript_call_result_33590 = invoke(stypy.reporting.localization.Localization(__file__, 338, 59), getitem___33589, (slice_33586, int_33587))
            
            
            # Obtaining the type of the subscript
            slice_33591 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 338, 68), None, None, None)
            int_33592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 73), 'int')
            # Getting the type of 'y' (line 338)
            y_33593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 68), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 338)
            getitem___33594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 68), y_33593, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 338)
            subscript_call_result_33595 = invoke(stypy.reporting.localization.Localization(__file__, 338, 68), getitem___33594, (slice_33591, int_33592))
            
            # Getting the type of 'p' (line 339)
            p_33596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 55), 'p', False)
            # Getting the type of 'bc0' (line 339)
            bc0_33597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 58), 'bc0', False)
            # Processing the call keyword arguments (line 338)
            kwargs_33598 = {}
            # Getting the type of 'estimate_bc_jac' (line 338)
            estimate_bc_jac_33584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 39), 'estimate_bc_jac', False)
            # Calling estimate_bc_jac(args, kwargs) (line 338)
            estimate_bc_jac_call_result_33599 = invoke(stypy.reporting.localization.Localization(__file__, 338, 39), estimate_bc_jac_33584, *[bc_33585, subscript_call_result_33590, subscript_call_result_33595, p_33596, bc0_33597], **kwargs_33598)
            
            # Obtaining the member '__getitem__' of a type (line 338)
            getitem___33600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), estimate_bc_jac_call_result_33599, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 338)
            subscript_call_result_33601 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), getitem___33600, int_33583)
            
            # Assigning a type to the variable 'tuple_var_assignment_32395' (line 338)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'tuple_var_assignment_32395', subscript_call_result_33601)
            
            # Assigning a Subscript to a Name (line 338):
            
            # Obtaining the type of the subscript
            int_33602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 12), 'int')
            
            # Call to estimate_bc_jac(...): (line 338)
            # Processing the call arguments (line 338)
            # Getting the type of 'bc' (line 338)
            bc_33604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 55), 'bc', False)
            
            # Obtaining the type of the subscript
            slice_33605 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 338, 59), None, None, None)
            int_33606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 64), 'int')
            # Getting the type of 'y' (line 338)
            y_33607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 59), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 338)
            getitem___33608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 59), y_33607, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 338)
            subscript_call_result_33609 = invoke(stypy.reporting.localization.Localization(__file__, 338, 59), getitem___33608, (slice_33605, int_33606))
            
            
            # Obtaining the type of the subscript
            slice_33610 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 338, 68), None, None, None)
            int_33611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 73), 'int')
            # Getting the type of 'y' (line 338)
            y_33612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 68), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 338)
            getitem___33613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 68), y_33612, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 338)
            subscript_call_result_33614 = invoke(stypy.reporting.localization.Localization(__file__, 338, 68), getitem___33613, (slice_33610, int_33611))
            
            # Getting the type of 'p' (line 339)
            p_33615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 55), 'p', False)
            # Getting the type of 'bc0' (line 339)
            bc0_33616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 58), 'bc0', False)
            # Processing the call keyword arguments (line 338)
            kwargs_33617 = {}
            # Getting the type of 'estimate_bc_jac' (line 338)
            estimate_bc_jac_33603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 39), 'estimate_bc_jac', False)
            # Calling estimate_bc_jac(args, kwargs) (line 338)
            estimate_bc_jac_call_result_33618 = invoke(stypy.reporting.localization.Localization(__file__, 338, 39), estimate_bc_jac_33603, *[bc_33604, subscript_call_result_33609, subscript_call_result_33614, p_33615, bc0_33616], **kwargs_33617)
            
            # Obtaining the member '__getitem__' of a type (line 338)
            getitem___33619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), estimate_bc_jac_call_result_33618, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 338)
            subscript_call_result_33620 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), getitem___33619, int_33602)
            
            # Assigning a type to the variable 'tuple_var_assignment_32396' (line 338)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'tuple_var_assignment_32396', subscript_call_result_33620)
            
            # Assigning a Name to a Name (line 338):
            # Getting the type of 'tuple_var_assignment_32394' (line 338)
            tuple_var_assignment_32394_33621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'tuple_var_assignment_32394')
            # Assigning a type to the variable 'dbc_dya' (line 338)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'dbc_dya', tuple_var_assignment_32394_33621)
            
            # Assigning a Name to a Name (line 338):
            # Getting the type of 'tuple_var_assignment_32395' (line 338)
            tuple_var_assignment_32395_33622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'tuple_var_assignment_32395')
            # Assigning a type to the variable 'dbc_dyb' (line 338)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 21), 'dbc_dyb', tuple_var_assignment_32395_33622)
            
            # Assigning a Name to a Name (line 338):
            # Getting the type of 'tuple_var_assignment_32396' (line 338)
            tuple_var_assignment_32396_33623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'tuple_var_assignment_32396')
            # Assigning a type to the variable 'dbc_dp' (line 338)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 30), 'dbc_dp', tuple_var_assignment_32396_33623)

            if more_types_in_union_33563:
                # Runtime conditional SSA for else branch (line 337)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_33562) or more_types_in_union_33563):
            
            # Assigning a Call to a Tuple (line 341):
            
            # Assigning a Subscript to a Name (line 341):
            
            # Obtaining the type of the subscript
            int_33624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 12), 'int')
            
            # Call to bc_jac(...): (line 341)
            # Processing the call arguments (line 341)
            
            # Obtaining the type of the subscript
            slice_33626 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 46), None, None, None)
            int_33627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 51), 'int')
            # Getting the type of 'y' (line 341)
            y_33628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 46), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___33629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 46), y_33628, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_33630 = invoke(stypy.reporting.localization.Localization(__file__, 341, 46), getitem___33629, (slice_33626, int_33627))
            
            
            # Obtaining the type of the subscript
            slice_33631 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 55), None, None, None)
            int_33632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 60), 'int')
            # Getting the type of 'y' (line 341)
            y_33633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 55), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___33634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 55), y_33633, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_33635 = invoke(stypy.reporting.localization.Localization(__file__, 341, 55), getitem___33634, (slice_33631, int_33632))
            
            # Getting the type of 'p' (line 341)
            p_33636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 65), 'p', False)
            # Processing the call keyword arguments (line 341)
            kwargs_33637 = {}
            # Getting the type of 'bc_jac' (line 341)
            bc_jac_33625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 39), 'bc_jac', False)
            # Calling bc_jac(args, kwargs) (line 341)
            bc_jac_call_result_33638 = invoke(stypy.reporting.localization.Localization(__file__, 341, 39), bc_jac_33625, *[subscript_call_result_33630, subscript_call_result_33635, p_33636], **kwargs_33637)
            
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___33639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 12), bc_jac_call_result_33638, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_33640 = invoke(stypy.reporting.localization.Localization(__file__, 341, 12), getitem___33639, int_33624)
            
            # Assigning a type to the variable 'tuple_var_assignment_32397' (line 341)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'tuple_var_assignment_32397', subscript_call_result_33640)
            
            # Assigning a Subscript to a Name (line 341):
            
            # Obtaining the type of the subscript
            int_33641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 12), 'int')
            
            # Call to bc_jac(...): (line 341)
            # Processing the call arguments (line 341)
            
            # Obtaining the type of the subscript
            slice_33643 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 46), None, None, None)
            int_33644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 51), 'int')
            # Getting the type of 'y' (line 341)
            y_33645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 46), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___33646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 46), y_33645, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_33647 = invoke(stypy.reporting.localization.Localization(__file__, 341, 46), getitem___33646, (slice_33643, int_33644))
            
            
            # Obtaining the type of the subscript
            slice_33648 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 55), None, None, None)
            int_33649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 60), 'int')
            # Getting the type of 'y' (line 341)
            y_33650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 55), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___33651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 55), y_33650, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_33652 = invoke(stypy.reporting.localization.Localization(__file__, 341, 55), getitem___33651, (slice_33648, int_33649))
            
            # Getting the type of 'p' (line 341)
            p_33653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 65), 'p', False)
            # Processing the call keyword arguments (line 341)
            kwargs_33654 = {}
            # Getting the type of 'bc_jac' (line 341)
            bc_jac_33642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 39), 'bc_jac', False)
            # Calling bc_jac(args, kwargs) (line 341)
            bc_jac_call_result_33655 = invoke(stypy.reporting.localization.Localization(__file__, 341, 39), bc_jac_33642, *[subscript_call_result_33647, subscript_call_result_33652, p_33653], **kwargs_33654)
            
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___33656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 12), bc_jac_call_result_33655, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_33657 = invoke(stypy.reporting.localization.Localization(__file__, 341, 12), getitem___33656, int_33641)
            
            # Assigning a type to the variable 'tuple_var_assignment_32398' (line 341)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'tuple_var_assignment_32398', subscript_call_result_33657)
            
            # Assigning a Subscript to a Name (line 341):
            
            # Obtaining the type of the subscript
            int_33658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 12), 'int')
            
            # Call to bc_jac(...): (line 341)
            # Processing the call arguments (line 341)
            
            # Obtaining the type of the subscript
            slice_33660 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 46), None, None, None)
            int_33661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 51), 'int')
            # Getting the type of 'y' (line 341)
            y_33662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 46), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___33663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 46), y_33662, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_33664 = invoke(stypy.reporting.localization.Localization(__file__, 341, 46), getitem___33663, (slice_33660, int_33661))
            
            
            # Obtaining the type of the subscript
            slice_33665 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 55), None, None, None)
            int_33666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 60), 'int')
            # Getting the type of 'y' (line 341)
            y_33667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 55), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___33668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 55), y_33667, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_33669 = invoke(stypy.reporting.localization.Localization(__file__, 341, 55), getitem___33668, (slice_33665, int_33666))
            
            # Getting the type of 'p' (line 341)
            p_33670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 65), 'p', False)
            # Processing the call keyword arguments (line 341)
            kwargs_33671 = {}
            # Getting the type of 'bc_jac' (line 341)
            bc_jac_33659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 39), 'bc_jac', False)
            # Calling bc_jac(args, kwargs) (line 341)
            bc_jac_call_result_33672 = invoke(stypy.reporting.localization.Localization(__file__, 341, 39), bc_jac_33659, *[subscript_call_result_33664, subscript_call_result_33669, p_33670], **kwargs_33671)
            
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___33673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 12), bc_jac_call_result_33672, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_33674 = invoke(stypy.reporting.localization.Localization(__file__, 341, 12), getitem___33673, int_33658)
            
            # Assigning a type to the variable 'tuple_var_assignment_32399' (line 341)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'tuple_var_assignment_32399', subscript_call_result_33674)
            
            # Assigning a Name to a Name (line 341):
            # Getting the type of 'tuple_var_assignment_32397' (line 341)
            tuple_var_assignment_32397_33675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'tuple_var_assignment_32397')
            # Assigning a type to the variable 'dbc_dya' (line 341)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'dbc_dya', tuple_var_assignment_32397_33675)
            
            # Assigning a Name to a Name (line 341):
            # Getting the type of 'tuple_var_assignment_32398' (line 341)
            tuple_var_assignment_32398_33676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'tuple_var_assignment_32398')
            # Assigning a type to the variable 'dbc_dyb' (line 341)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 21), 'dbc_dyb', tuple_var_assignment_32398_33676)
            
            # Assigning a Name to a Name (line 341):
            # Getting the type of 'tuple_var_assignment_32399' (line 341)
            tuple_var_assignment_32399_33677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'tuple_var_assignment_32399')
            # Assigning a type to the variable 'dbc_dp' (line 341)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 30), 'dbc_dp', tuple_var_assignment_32399_33677)

            if (may_be_33562 and more_types_in_union_33563):
                # SSA join for if statement (line 337)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to construct_global_jac(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'n' (line 343)
        n_33679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 36), 'n', False)
        # Getting the type of 'm' (line 343)
        m_33680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 39), 'm', False)
        # Getting the type of 'k' (line 343)
        k_33681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 42), 'k', False)
        # Getting the type of 'i_jac' (line 343)
        i_jac_33682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 45), 'i_jac', False)
        # Getting the type of 'j_jac' (line 343)
        j_jac_33683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 52), 'j_jac', False)
        # Getting the type of 'h' (line 343)
        h_33684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 59), 'h', False)
        # Getting the type of 'df_dy' (line 343)
        df_dy_33685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 62), 'df_dy', False)
        # Getting the type of 'df_dy_middle' (line 344)
        df_dy_middle_33686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 36), 'df_dy_middle', False)
        # Getting the type of 'df_dp' (line 344)
        df_dp_33687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 50), 'df_dp', False)
        # Getting the type of 'df_dp_middle' (line 344)
        df_dp_middle_33688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 57), 'df_dp_middle', False)
        # Getting the type of 'dbc_dya' (line 344)
        dbc_dya_33689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 71), 'dbc_dya', False)
        # Getting the type of 'dbc_dyb' (line 345)
        dbc_dyb_33690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 36), 'dbc_dyb', False)
        # Getting the type of 'dbc_dp' (line 345)
        dbc_dp_33691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 45), 'dbc_dp', False)
        # Processing the call keyword arguments (line 343)
        kwargs_33692 = {}
        # Getting the type of 'construct_global_jac' (line 343)
        construct_global_jac_33678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 15), 'construct_global_jac', False)
        # Calling construct_global_jac(args, kwargs) (line 343)
        construct_global_jac_call_result_33693 = invoke(stypy.reporting.localization.Localization(__file__, 343, 15), construct_global_jac_33678, *[n_33679, m_33680, k_33681, i_jac_33682, j_jac_33683, h_33684, df_dy_33685, df_dy_middle_33686, df_dp_33687, df_dp_middle_33688, dbc_dya_33689, dbc_dyb_33690, dbc_dp_33691], **kwargs_33692)
        
        # Assigning a type to the variable 'stypy_return_type' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'stypy_return_type', construct_global_jac_call_result_33693)
        
        # ################# End of 'sys_jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sys_jac' in the type store
        # Getting the type of 'stypy_return_type' (line 328)
        stypy_return_type_33694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33694)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sys_jac'
        return stypy_return_type_33694

    # Assigning a type to the variable 'sys_jac' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'sys_jac', sys_jac)
    
    # Obtaining an instance of the builtin type 'tuple' (line 347)
    tuple_33695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 347)
    # Adding element type (line 347)
    # Getting the type of 'col_fun' (line 347)
    col_fun_33696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 11), 'col_fun')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 11), tuple_33695, col_fun_33696)
    # Adding element type (line 347)
    # Getting the type of 'sys_jac' (line 347)
    sys_jac_33697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 20), 'sys_jac')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 11), tuple_33695, sys_jac_33697)
    
    # Assigning a type to the variable 'stypy_return_type' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'stypy_return_type', tuple_33695)
    
    # ################# End of 'prepare_sys(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'prepare_sys' in the type store
    # Getting the type of 'stypy_return_type' (line 320)
    stypy_return_type_33698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33698)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'prepare_sys'
    return stypy_return_type_33698

# Assigning a type to the variable 'prepare_sys' (line 320)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 0), 'prepare_sys', prepare_sys)

@norecursion
def solve_newton(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'solve_newton'
    module_type_store = module_type_store.open_function_context('solve_newton', 350, 0, False)
    
    # Passed parameters checking function
    solve_newton.stypy_localization = localization
    solve_newton.stypy_type_of_self = None
    solve_newton.stypy_type_store = module_type_store
    solve_newton.stypy_function_name = 'solve_newton'
    solve_newton.stypy_param_names_list = ['n', 'm', 'h', 'col_fun', 'bc', 'jac', 'y', 'p', 'B', 'bvp_tol']
    solve_newton.stypy_varargs_param_name = None
    solve_newton.stypy_kwargs_param_name = None
    solve_newton.stypy_call_defaults = defaults
    solve_newton.stypy_call_varargs = varargs
    solve_newton.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_newton', ['n', 'm', 'h', 'col_fun', 'bc', 'jac', 'y', 'p', 'B', 'bvp_tol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_newton', localization, ['n', 'm', 'h', 'col_fun', 'bc', 'jac', 'y', 'p', 'B', 'bvp_tol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_newton(...)' code ##################

    str_33699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, (-1)), 'str', 'Solve the nonlinear collocation system by a Newton method.\n\n    This is a simple Newton method with a backtracking line search. As\n    advised in [1]_, an affine-invariant criterion function F = ||J^-1 r||^2\n    is used, where J is the Jacobian matrix at the current iteration and r is\n    the vector or collocation residuals (values of the system lhs).\n\n    The method alters between full Newton iterations and the fixed-Jacobian\n    iterations based\n\n    There are other tricks proposed in [1]_, but they are not used as they\n    don\'t seem to improve anything significantly, and even break the\n    convergence on some test problems I tried.\n\n    All important parameters of the algorithm are defined inside the function.\n\n    Parameters\n    ----------\n    n : int\n        Number of equations in the ODE system.\n    m : int\n        Number of nodes in the mesh.\n    h : ndarray, shape (m-1,)\n        Mesh intervals.\n    col_fun : callable\n        Function computing collocation residuals.\n    bc : callable\n        Function computing boundary condition residuals.\n    jac : callable\n        Function computing the Jacobian of the whole system (including\n        collocation and boundary condition residuals). It is supposed to\n        return csc_matrix.\n    y : ndarray, shape (n, m)\n        Initial guess for the function values at the mesh nodes.\n    p : ndarray, shape (k,)\n        Initial guess for the unknown parameters.\n    B : ndarray with shape (n, n) or None\n        Matrix to force the S y(a) = 0 condition for a problems with the\n        singular term. If None, the singular term is assumed to be absent.\n    bvp_tol : float\n        Tolerance to which we want to solve a BVP.\n\n    Returns\n    -------\n    y : ndarray, shape (n, m)\n        Final iterate for the function values at the mesh nodes.\n    p : ndarray, shape (k,)\n        Final iterate for the unknown parameters.\n    singular : bool\n        True, if the LU decomposition failed because Jacobian turned out\n        to be singular.\n\n    References\n    ----------\n    .. [1]  U. Ascher, R. Mattheij and R. Russell "Numerical Solution of\n       Boundary Value Problems for Ordinary Differential Equations"\n    ')
    
    # Assigning a BinOp to a Name (line 416):
    
    # Assigning a BinOp to a Name (line 416):
    int_33700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 12), 'int')
    int_33701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 14), 'int')
    # Applying the binary operator 'div' (line 416)
    result_div_33702 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 12), 'div', int_33700, int_33701)
    
    # Getting the type of 'h' (line 416)
    h_33703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 18), 'h')
    # Applying the binary operator '*' (line 416)
    result_mul_33704 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 16), '*', result_div_33702, h_33703)
    
    float_33705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 22), 'float')
    # Applying the binary operator '*' (line 416)
    result_mul_33706 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 20), '*', result_mul_33704, float_33705)
    
    # Getting the type of 'bvp_tol' (line 416)
    bvp_tol_33707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 29), 'bvp_tol')
    # Applying the binary operator '*' (line 416)
    result_mul_33708 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 27), '*', result_mul_33706, bvp_tol_33707)
    
    # Assigning a type to the variable 'tol_r' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'tol_r', result_mul_33708)
    
    # Assigning a BinOp to a Name (line 422):
    
    # Assigning a BinOp to a Name (line 422):
    float_33709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 13), 'float')
    # Getting the type of 'bvp_tol' (line 422)
    bvp_tol_33710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 20), 'bvp_tol')
    # Applying the binary operator '*' (line 422)
    result_mul_33711 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 13), '*', float_33709, bvp_tol_33710)
    
    # Assigning a type to the variable 'tol_bc' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'tol_bc', result_mul_33711)
    
    # Assigning a Num to a Name (line 427):
    
    # Assigning a Num to a Name (line 427):
    int_33712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 15), 'int')
    # Assigning a type to the variable 'max_njev' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'max_njev', int_33712)
    
    # Assigning a Num to a Name (line 432):
    
    # Assigning a Num to a Name (line 432):
    int_33713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 15), 'int')
    # Assigning a type to the variable 'max_iter' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'max_iter', int_33713)
    
    # Assigning a Num to a Name (line 436):
    
    # Assigning a Num to a Name (line 436):
    float_33714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 12), 'float')
    # Assigning a type to the variable 'sigma' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'sigma', float_33714)
    
    # Assigning a Num to a Name (line 439):
    
    # Assigning a Num to a Name (line 439):
    float_33715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 10), 'float')
    # Assigning a type to the variable 'tau' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'tau', float_33715)
    
    # Assigning a Num to a Name (line 443):
    
    # Assigning a Num to a Name (line 443):
    int_33716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 14), 'int')
    # Assigning a type to the variable 'n_trial' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'n_trial', int_33716)
    
    # Assigning a Call to a Tuple (line 445):
    
    # Assigning a Subscript to a Name (line 445):
    
    # Obtaining the type of the subscript
    int_33717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 4), 'int')
    
    # Call to col_fun(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'y' (line 445)
    y_33719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 45), 'y', False)
    # Getting the type of 'p' (line 445)
    p_33720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 48), 'p', False)
    # Processing the call keyword arguments (line 445)
    kwargs_33721 = {}
    # Getting the type of 'col_fun' (line 445)
    col_fun_33718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 37), 'col_fun', False)
    # Calling col_fun(args, kwargs) (line 445)
    col_fun_call_result_33722 = invoke(stypy.reporting.localization.Localization(__file__, 445, 37), col_fun_33718, *[y_33719, p_33720], **kwargs_33721)
    
    # Obtaining the member '__getitem__' of a type (line 445)
    getitem___33723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 4), col_fun_call_result_33722, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 445)
    subscript_call_result_33724 = invoke(stypy.reporting.localization.Localization(__file__, 445, 4), getitem___33723, int_33717)
    
    # Assigning a type to the variable 'tuple_var_assignment_32400' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'tuple_var_assignment_32400', subscript_call_result_33724)
    
    # Assigning a Subscript to a Name (line 445):
    
    # Obtaining the type of the subscript
    int_33725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 4), 'int')
    
    # Call to col_fun(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'y' (line 445)
    y_33727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 45), 'y', False)
    # Getting the type of 'p' (line 445)
    p_33728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 48), 'p', False)
    # Processing the call keyword arguments (line 445)
    kwargs_33729 = {}
    # Getting the type of 'col_fun' (line 445)
    col_fun_33726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 37), 'col_fun', False)
    # Calling col_fun(args, kwargs) (line 445)
    col_fun_call_result_33730 = invoke(stypy.reporting.localization.Localization(__file__, 445, 37), col_fun_33726, *[y_33727, p_33728], **kwargs_33729)
    
    # Obtaining the member '__getitem__' of a type (line 445)
    getitem___33731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 4), col_fun_call_result_33730, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 445)
    subscript_call_result_33732 = invoke(stypy.reporting.localization.Localization(__file__, 445, 4), getitem___33731, int_33725)
    
    # Assigning a type to the variable 'tuple_var_assignment_32401' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'tuple_var_assignment_32401', subscript_call_result_33732)
    
    # Assigning a Subscript to a Name (line 445):
    
    # Obtaining the type of the subscript
    int_33733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 4), 'int')
    
    # Call to col_fun(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'y' (line 445)
    y_33735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 45), 'y', False)
    # Getting the type of 'p' (line 445)
    p_33736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 48), 'p', False)
    # Processing the call keyword arguments (line 445)
    kwargs_33737 = {}
    # Getting the type of 'col_fun' (line 445)
    col_fun_33734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 37), 'col_fun', False)
    # Calling col_fun(args, kwargs) (line 445)
    col_fun_call_result_33738 = invoke(stypy.reporting.localization.Localization(__file__, 445, 37), col_fun_33734, *[y_33735, p_33736], **kwargs_33737)
    
    # Obtaining the member '__getitem__' of a type (line 445)
    getitem___33739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 4), col_fun_call_result_33738, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 445)
    subscript_call_result_33740 = invoke(stypy.reporting.localization.Localization(__file__, 445, 4), getitem___33739, int_33733)
    
    # Assigning a type to the variable 'tuple_var_assignment_32402' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'tuple_var_assignment_32402', subscript_call_result_33740)
    
    # Assigning a Subscript to a Name (line 445):
    
    # Obtaining the type of the subscript
    int_33741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 4), 'int')
    
    # Call to col_fun(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'y' (line 445)
    y_33743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 45), 'y', False)
    # Getting the type of 'p' (line 445)
    p_33744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 48), 'p', False)
    # Processing the call keyword arguments (line 445)
    kwargs_33745 = {}
    # Getting the type of 'col_fun' (line 445)
    col_fun_33742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 37), 'col_fun', False)
    # Calling col_fun(args, kwargs) (line 445)
    col_fun_call_result_33746 = invoke(stypy.reporting.localization.Localization(__file__, 445, 37), col_fun_33742, *[y_33743, p_33744], **kwargs_33745)
    
    # Obtaining the member '__getitem__' of a type (line 445)
    getitem___33747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 4), col_fun_call_result_33746, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 445)
    subscript_call_result_33748 = invoke(stypy.reporting.localization.Localization(__file__, 445, 4), getitem___33747, int_33741)
    
    # Assigning a type to the variable 'tuple_var_assignment_32403' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'tuple_var_assignment_32403', subscript_call_result_33748)
    
    # Assigning a Name to a Name (line 445):
    # Getting the type of 'tuple_var_assignment_32400' (line 445)
    tuple_var_assignment_32400_33749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'tuple_var_assignment_32400')
    # Assigning a type to the variable 'col_res' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'col_res', tuple_var_assignment_32400_33749)
    
    # Assigning a Name to a Name (line 445):
    # Getting the type of 'tuple_var_assignment_32401' (line 445)
    tuple_var_assignment_32401_33750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'tuple_var_assignment_32401')
    # Assigning a type to the variable 'y_middle' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 13), 'y_middle', tuple_var_assignment_32401_33750)
    
    # Assigning a Name to a Name (line 445):
    # Getting the type of 'tuple_var_assignment_32402' (line 445)
    tuple_var_assignment_32402_33751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'tuple_var_assignment_32402')
    # Assigning a type to the variable 'f' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 23), 'f', tuple_var_assignment_32402_33751)
    
    # Assigning a Name to a Name (line 445):
    # Getting the type of 'tuple_var_assignment_32403' (line 445)
    tuple_var_assignment_32403_33752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'tuple_var_assignment_32403')
    # Assigning a type to the variable 'f_middle' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 26), 'f_middle', tuple_var_assignment_32403_33752)
    
    # Assigning a Call to a Name (line 446):
    
    # Assigning a Call to a Name (line 446):
    
    # Call to bc(...): (line 446)
    # Processing the call arguments (line 446)
    
    # Obtaining the type of the subscript
    slice_33754 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 446, 16), None, None, None)
    int_33755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 21), 'int')
    # Getting the type of 'y' (line 446)
    y_33756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 16), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 446)
    getitem___33757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 16), y_33756, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 446)
    subscript_call_result_33758 = invoke(stypy.reporting.localization.Localization(__file__, 446, 16), getitem___33757, (slice_33754, int_33755))
    
    
    # Obtaining the type of the subscript
    slice_33759 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 446, 25), None, None, None)
    int_33760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 30), 'int')
    # Getting the type of 'y' (line 446)
    y_33761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 25), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 446)
    getitem___33762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 25), y_33761, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 446)
    subscript_call_result_33763 = invoke(stypy.reporting.localization.Localization(__file__, 446, 25), getitem___33762, (slice_33759, int_33760))
    
    # Getting the type of 'p' (line 446)
    p_33764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 35), 'p', False)
    # Processing the call keyword arguments (line 446)
    kwargs_33765 = {}
    # Getting the type of 'bc' (line 446)
    bc_33753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 13), 'bc', False)
    # Calling bc(args, kwargs) (line 446)
    bc_call_result_33766 = invoke(stypy.reporting.localization.Localization(__file__, 446, 13), bc_33753, *[subscript_call_result_33758, subscript_call_result_33763, p_33764], **kwargs_33765)
    
    # Assigning a type to the variable 'bc_res' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'bc_res', bc_call_result_33766)
    
    # Assigning a Call to a Name (line 447):
    
    # Assigning a Call to a Name (line 447):
    
    # Call to hstack(...): (line 447)
    # Processing the call arguments (line 447)
    
    # Obtaining an instance of the builtin type 'tuple' (line 447)
    tuple_33769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 447)
    # Adding element type (line 447)
    
    # Call to ravel(...): (line 447)
    # Processing the call keyword arguments (line 447)
    str_33772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 41), 'str', 'F')
    keyword_33773 = str_33772
    kwargs_33774 = {'order': keyword_33773}
    # Getting the type of 'col_res' (line 447)
    col_res_33770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 21), 'col_res', False)
    # Obtaining the member 'ravel' of a type (line 447)
    ravel_33771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 21), col_res_33770, 'ravel')
    # Calling ravel(args, kwargs) (line 447)
    ravel_call_result_33775 = invoke(stypy.reporting.localization.Localization(__file__, 447, 21), ravel_33771, *[], **kwargs_33774)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 21), tuple_33769, ravel_call_result_33775)
    # Adding element type (line 447)
    # Getting the type of 'bc_res' (line 447)
    bc_res_33776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 47), 'bc_res', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 21), tuple_33769, bc_res_33776)
    
    # Processing the call keyword arguments (line 447)
    kwargs_33777 = {}
    # Getting the type of 'np' (line 447)
    np_33767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 10), 'np', False)
    # Obtaining the member 'hstack' of a type (line 447)
    hstack_33768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 10), np_33767, 'hstack')
    # Calling hstack(args, kwargs) (line 447)
    hstack_call_result_33778 = invoke(stypy.reporting.localization.Localization(__file__, 447, 10), hstack_33768, *[tuple_33769], **kwargs_33777)
    
    # Assigning a type to the variable 'res' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'res', hstack_call_result_33778)
    
    # Assigning a Num to a Name (line 449):
    
    # Assigning a Num to a Name (line 449):
    int_33779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 11), 'int')
    # Assigning a type to the variable 'njev' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'njev', int_33779)
    
    # Assigning a Name to a Name (line 450):
    
    # Assigning a Name to a Name (line 450):
    # Getting the type of 'False' (line 450)
    False_33780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 15), 'False')
    # Assigning a type to the variable 'singular' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'singular', False_33780)
    
    # Assigning a Name to a Name (line 451):
    
    # Assigning a Name to a Name (line 451):
    # Getting the type of 'True' (line 451)
    True_33781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 20), 'True')
    # Assigning a type to the variable 'recompute_jac' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'recompute_jac', True_33781)
    
    
    # Call to range(...): (line 452)
    # Processing the call arguments (line 452)
    # Getting the type of 'max_iter' (line 452)
    max_iter_33783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 27), 'max_iter', False)
    # Processing the call keyword arguments (line 452)
    kwargs_33784 = {}
    # Getting the type of 'range' (line 452)
    range_33782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 21), 'range', False)
    # Calling range(args, kwargs) (line 452)
    range_call_result_33785 = invoke(stypy.reporting.localization.Localization(__file__, 452, 21), range_33782, *[max_iter_33783], **kwargs_33784)
    
    # Testing the type of a for loop iterable (line 452)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 452, 4), range_call_result_33785)
    # Getting the type of the for loop variable (line 452)
    for_loop_var_33786 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 452, 4), range_call_result_33785)
    # Assigning a type to the variable 'iteration' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'iteration', for_loop_var_33786)
    # SSA begins for a for statement (line 452)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'recompute_jac' (line 453)
    recompute_jac_33787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 11), 'recompute_jac')
    # Testing the type of an if condition (line 453)
    if_condition_33788 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 8), recompute_jac_33787)
    # Assigning a type to the variable 'if_condition_33788' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'if_condition_33788', if_condition_33788)
    # SSA begins for if statement (line 453)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 454):
    
    # Assigning a Call to a Name (line 454):
    
    # Call to jac(...): (line 454)
    # Processing the call arguments (line 454)
    # Getting the type of 'y' (line 454)
    y_33790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 20), 'y', False)
    # Getting the type of 'p' (line 454)
    p_33791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 23), 'p', False)
    # Getting the type of 'y_middle' (line 454)
    y_middle_33792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 26), 'y_middle', False)
    # Getting the type of 'f' (line 454)
    f_33793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 36), 'f', False)
    # Getting the type of 'f_middle' (line 454)
    f_middle_33794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 39), 'f_middle', False)
    # Getting the type of 'bc_res' (line 454)
    bc_res_33795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 49), 'bc_res', False)
    # Processing the call keyword arguments (line 454)
    kwargs_33796 = {}
    # Getting the type of 'jac' (line 454)
    jac_33789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 16), 'jac', False)
    # Calling jac(args, kwargs) (line 454)
    jac_call_result_33797 = invoke(stypy.reporting.localization.Localization(__file__, 454, 16), jac_33789, *[y_33790, p_33791, y_middle_33792, f_33793, f_middle_33794, bc_res_33795], **kwargs_33796)
    
    # Assigning a type to the variable 'J' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'J', jac_call_result_33797)
    
    # Getting the type of 'njev' (line 455)
    njev_33798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'njev')
    int_33799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 20), 'int')
    # Applying the binary operator '+=' (line 455)
    result_iadd_33800 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 12), '+=', njev_33798, int_33799)
    # Assigning a type to the variable 'njev' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'njev', result_iadd_33800)
    
    
    
    # SSA begins for try-except statement (line 456)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 457):
    
    # Assigning a Call to a Name (line 457):
    
    # Call to splu(...): (line 457)
    # Processing the call arguments (line 457)
    # Getting the type of 'J' (line 457)
    J_33802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 26), 'J', False)
    # Processing the call keyword arguments (line 457)
    kwargs_33803 = {}
    # Getting the type of 'splu' (line 457)
    splu_33801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 21), 'splu', False)
    # Calling splu(args, kwargs) (line 457)
    splu_call_result_33804 = invoke(stypy.reporting.localization.Localization(__file__, 457, 21), splu_33801, *[J_33802], **kwargs_33803)
    
    # Assigning a type to the variable 'LU' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 16), 'LU', splu_call_result_33804)
    # SSA branch for the except part of a try statement (line 456)
    # SSA branch for the except 'RuntimeError' branch of a try statement (line 456)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 459):
    
    # Assigning a Name to a Name (line 459):
    # Getting the type of 'True' (line 459)
    True_33805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 27), 'True')
    # Assigning a type to the variable 'singular' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 16), 'singular', True_33805)
    # SSA join for try-except statement (line 456)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 462):
    
    # Assigning a Call to a Name (line 462):
    
    # Call to solve(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'res' (line 462)
    res_33808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 28), 'res', False)
    # Processing the call keyword arguments (line 462)
    kwargs_33809 = {}
    # Getting the type of 'LU' (line 462)
    LU_33806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 19), 'LU', False)
    # Obtaining the member 'solve' of a type (line 462)
    solve_33807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 19), LU_33806, 'solve')
    # Calling solve(args, kwargs) (line 462)
    solve_call_result_33810 = invoke(stypy.reporting.localization.Localization(__file__, 462, 19), solve_33807, *[res_33808], **kwargs_33809)
    
    # Assigning a type to the variable 'step' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'step', solve_call_result_33810)
    
    # Assigning a Call to a Name (line 463):
    
    # Assigning a Call to a Name (line 463):
    
    # Call to dot(...): (line 463)
    # Processing the call arguments (line 463)
    # Getting the type of 'step' (line 463)
    step_33813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 26), 'step', False)
    # Getting the type of 'step' (line 463)
    step_33814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 32), 'step', False)
    # Processing the call keyword arguments (line 463)
    kwargs_33815 = {}
    # Getting the type of 'np' (line 463)
    np_33811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 19), 'np', False)
    # Obtaining the member 'dot' of a type (line 463)
    dot_33812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 19), np_33811, 'dot')
    # Calling dot(args, kwargs) (line 463)
    dot_call_result_33816 = invoke(stypy.reporting.localization.Localization(__file__, 463, 19), dot_33812, *[step_33813, step_33814], **kwargs_33815)
    
    # Assigning a type to the variable 'cost' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'cost', dot_call_result_33816)
    # SSA join for if statement (line 453)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 465):
    
    # Assigning a Call to a Name (line 465):
    
    # Call to reshape(...): (line 465)
    # Processing the call arguments (line 465)
    
    # Obtaining an instance of the builtin type 'tuple' (line 465)
    tuple_33825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 465)
    # Adding element type (line 465)
    # Getting the type of 'n' (line 465)
    n_33826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 39), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 39), tuple_33825, n_33826)
    # Adding element type (line 465)
    # Getting the type of 'm' (line 465)
    m_33827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 42), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 39), tuple_33825, m_33827)
    
    # Processing the call keyword arguments (line 465)
    str_33828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 52), 'str', 'F')
    keyword_33829 = str_33828
    kwargs_33830 = {'order': keyword_33829}
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 465)
    m_33817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 23), 'm', False)
    # Getting the type of 'n' (line 465)
    n_33818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 27), 'n', False)
    # Applying the binary operator '*' (line 465)
    result_mul_33819 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 23), '*', m_33817, n_33818)
    
    slice_33820 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 465, 17), None, result_mul_33819, None)
    # Getting the type of 'step' (line 465)
    step_33821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 17), 'step', False)
    # Obtaining the member '__getitem__' of a type (line 465)
    getitem___33822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 17), step_33821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 465)
    subscript_call_result_33823 = invoke(stypy.reporting.localization.Localization(__file__, 465, 17), getitem___33822, slice_33820)
    
    # Obtaining the member 'reshape' of a type (line 465)
    reshape_33824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 17), subscript_call_result_33823, 'reshape')
    # Calling reshape(args, kwargs) (line 465)
    reshape_call_result_33831 = invoke(stypy.reporting.localization.Localization(__file__, 465, 17), reshape_33824, *[tuple_33825], **kwargs_33830)
    
    # Assigning a type to the variable 'y_step' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'y_step', reshape_call_result_33831)
    
    # Assigning a Subscript to a Name (line 466):
    
    # Assigning a Subscript to a Name (line 466):
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 466)
    m_33832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 22), 'm')
    # Getting the type of 'n' (line 466)
    n_33833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 26), 'n')
    # Applying the binary operator '*' (line 466)
    result_mul_33834 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 22), '*', m_33832, n_33833)
    
    slice_33835 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 466, 17), result_mul_33834, None, None)
    # Getting the type of 'step' (line 466)
    step_33836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 17), 'step')
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___33837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 17), step_33836, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_33838 = invoke(stypy.reporting.localization.Localization(__file__, 466, 17), getitem___33837, slice_33835)
    
    # Assigning a type to the variable 'p_step' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'p_step', subscript_call_result_33838)
    
    # Assigning a Num to a Name (line 468):
    
    # Assigning a Num to a Name (line 468):
    int_33839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 16), 'int')
    # Assigning a type to the variable 'alpha' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'alpha', int_33839)
    
    
    # Call to range(...): (line 469)
    # Processing the call arguments (line 469)
    # Getting the type of 'n_trial' (line 469)
    n_trial_33841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 27), 'n_trial', False)
    int_33842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 37), 'int')
    # Applying the binary operator '+' (line 469)
    result_add_33843 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 27), '+', n_trial_33841, int_33842)
    
    # Processing the call keyword arguments (line 469)
    kwargs_33844 = {}
    # Getting the type of 'range' (line 469)
    range_33840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 21), 'range', False)
    # Calling range(args, kwargs) (line 469)
    range_call_result_33845 = invoke(stypy.reporting.localization.Localization(__file__, 469, 21), range_33840, *[result_add_33843], **kwargs_33844)
    
    # Testing the type of a for loop iterable (line 469)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 469, 8), range_call_result_33845)
    # Getting the type of the for loop variable (line 469)
    for_loop_var_33846 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 469, 8), range_call_result_33845)
    # Assigning a type to the variable 'trial' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'trial', for_loop_var_33846)
    # SSA begins for a for statement (line 469)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 470):
    
    # Assigning a BinOp to a Name (line 470):
    # Getting the type of 'y' (line 470)
    y_33847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 20), 'y')
    # Getting the type of 'alpha' (line 470)
    alpha_33848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 24), 'alpha')
    # Getting the type of 'y_step' (line 470)
    y_step_33849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 32), 'y_step')
    # Applying the binary operator '*' (line 470)
    result_mul_33850 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 24), '*', alpha_33848, y_step_33849)
    
    # Applying the binary operator '-' (line 470)
    result_sub_33851 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 20), '-', y_33847, result_mul_33850)
    
    # Assigning a type to the variable 'y_new' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'y_new', result_sub_33851)
    
    # Type idiom detected: calculating its left and rigth part (line 471)
    # Getting the type of 'B' (line 471)
    B_33852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'B')
    # Getting the type of 'None' (line 471)
    None_33853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 24), 'None')
    
    (may_be_33854, more_types_in_union_33855) = may_not_be_none(B_33852, None_33853)

    if may_be_33854:

        if more_types_in_union_33855:
            # Runtime conditional SSA (line 471)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Subscript (line 472):
        
        # Assigning a Call to a Subscript (line 472):
        
        # Call to dot(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'B' (line 472)
        B_33858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 37), 'B', False)
        
        # Obtaining the type of the subscript
        slice_33859 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 472, 40), None, None, None)
        int_33860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 49), 'int')
        # Getting the type of 'y_new' (line 472)
        y_new_33861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 40), 'y_new', False)
        # Obtaining the member '__getitem__' of a type (line 472)
        getitem___33862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 40), y_new_33861, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 472)
        subscript_call_result_33863 = invoke(stypy.reporting.localization.Localization(__file__, 472, 40), getitem___33862, (slice_33859, int_33860))
        
        # Processing the call keyword arguments (line 472)
        kwargs_33864 = {}
        # Getting the type of 'np' (line 472)
        np_33856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 30), 'np', False)
        # Obtaining the member 'dot' of a type (line 472)
        dot_33857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 30), np_33856, 'dot')
        # Calling dot(args, kwargs) (line 472)
        dot_call_result_33865 = invoke(stypy.reporting.localization.Localization(__file__, 472, 30), dot_33857, *[B_33858, subscript_call_result_33863], **kwargs_33864)
        
        # Getting the type of 'y_new' (line 472)
        y_new_33866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 16), 'y_new')
        slice_33867 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 472, 16), None, None, None)
        int_33868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 25), 'int')
        # Storing an element on a container (line 472)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 16), y_new_33866, ((slice_33867, int_33868), dot_call_result_33865))

        if more_types_in_union_33855:
            # SSA join for if statement (line 471)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 473):
    
    # Assigning a BinOp to a Name (line 473):
    # Getting the type of 'p' (line 473)
    p_33869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 20), 'p')
    # Getting the type of 'alpha' (line 473)
    alpha_33870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 24), 'alpha')
    # Getting the type of 'p_step' (line 473)
    p_step_33871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 32), 'p_step')
    # Applying the binary operator '*' (line 473)
    result_mul_33872 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 24), '*', alpha_33870, p_step_33871)
    
    # Applying the binary operator '-' (line 473)
    result_sub_33873 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 20), '-', p_33869, result_mul_33872)
    
    # Assigning a type to the variable 'p_new' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'p_new', result_sub_33873)
    
    # Assigning a Call to a Tuple (line 475):
    
    # Assigning a Subscript to a Name (line 475):
    
    # Obtaining the type of the subscript
    int_33874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 12), 'int')
    
    # Call to col_fun(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'y_new' (line 475)
    y_new_33876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 53), 'y_new', False)
    # Getting the type of 'p_new' (line 475)
    p_new_33877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 60), 'p_new', False)
    # Processing the call keyword arguments (line 475)
    kwargs_33878 = {}
    # Getting the type of 'col_fun' (line 475)
    col_fun_33875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 45), 'col_fun', False)
    # Calling col_fun(args, kwargs) (line 475)
    col_fun_call_result_33879 = invoke(stypy.reporting.localization.Localization(__file__, 475, 45), col_fun_33875, *[y_new_33876, p_new_33877], **kwargs_33878)
    
    # Obtaining the member '__getitem__' of a type (line 475)
    getitem___33880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), col_fun_call_result_33879, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 475)
    subscript_call_result_33881 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), getitem___33880, int_33874)
    
    # Assigning a type to the variable 'tuple_var_assignment_32404' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_32404', subscript_call_result_33881)
    
    # Assigning a Subscript to a Name (line 475):
    
    # Obtaining the type of the subscript
    int_33882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 12), 'int')
    
    # Call to col_fun(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'y_new' (line 475)
    y_new_33884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 53), 'y_new', False)
    # Getting the type of 'p_new' (line 475)
    p_new_33885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 60), 'p_new', False)
    # Processing the call keyword arguments (line 475)
    kwargs_33886 = {}
    # Getting the type of 'col_fun' (line 475)
    col_fun_33883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 45), 'col_fun', False)
    # Calling col_fun(args, kwargs) (line 475)
    col_fun_call_result_33887 = invoke(stypy.reporting.localization.Localization(__file__, 475, 45), col_fun_33883, *[y_new_33884, p_new_33885], **kwargs_33886)
    
    # Obtaining the member '__getitem__' of a type (line 475)
    getitem___33888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), col_fun_call_result_33887, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 475)
    subscript_call_result_33889 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), getitem___33888, int_33882)
    
    # Assigning a type to the variable 'tuple_var_assignment_32405' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_32405', subscript_call_result_33889)
    
    # Assigning a Subscript to a Name (line 475):
    
    # Obtaining the type of the subscript
    int_33890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 12), 'int')
    
    # Call to col_fun(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'y_new' (line 475)
    y_new_33892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 53), 'y_new', False)
    # Getting the type of 'p_new' (line 475)
    p_new_33893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 60), 'p_new', False)
    # Processing the call keyword arguments (line 475)
    kwargs_33894 = {}
    # Getting the type of 'col_fun' (line 475)
    col_fun_33891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 45), 'col_fun', False)
    # Calling col_fun(args, kwargs) (line 475)
    col_fun_call_result_33895 = invoke(stypy.reporting.localization.Localization(__file__, 475, 45), col_fun_33891, *[y_new_33892, p_new_33893], **kwargs_33894)
    
    # Obtaining the member '__getitem__' of a type (line 475)
    getitem___33896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), col_fun_call_result_33895, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 475)
    subscript_call_result_33897 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), getitem___33896, int_33890)
    
    # Assigning a type to the variable 'tuple_var_assignment_32406' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_32406', subscript_call_result_33897)
    
    # Assigning a Subscript to a Name (line 475):
    
    # Obtaining the type of the subscript
    int_33898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 12), 'int')
    
    # Call to col_fun(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'y_new' (line 475)
    y_new_33900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 53), 'y_new', False)
    # Getting the type of 'p_new' (line 475)
    p_new_33901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 60), 'p_new', False)
    # Processing the call keyword arguments (line 475)
    kwargs_33902 = {}
    # Getting the type of 'col_fun' (line 475)
    col_fun_33899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 45), 'col_fun', False)
    # Calling col_fun(args, kwargs) (line 475)
    col_fun_call_result_33903 = invoke(stypy.reporting.localization.Localization(__file__, 475, 45), col_fun_33899, *[y_new_33900, p_new_33901], **kwargs_33902)
    
    # Obtaining the member '__getitem__' of a type (line 475)
    getitem___33904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), col_fun_call_result_33903, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 475)
    subscript_call_result_33905 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), getitem___33904, int_33898)
    
    # Assigning a type to the variable 'tuple_var_assignment_32407' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_32407', subscript_call_result_33905)
    
    # Assigning a Name to a Name (line 475):
    # Getting the type of 'tuple_var_assignment_32404' (line 475)
    tuple_var_assignment_32404_33906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_32404')
    # Assigning a type to the variable 'col_res' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'col_res', tuple_var_assignment_32404_33906)
    
    # Assigning a Name to a Name (line 475):
    # Getting the type of 'tuple_var_assignment_32405' (line 475)
    tuple_var_assignment_32405_33907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_32405')
    # Assigning a type to the variable 'y_middle' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 21), 'y_middle', tuple_var_assignment_32405_33907)
    
    # Assigning a Name to a Name (line 475):
    # Getting the type of 'tuple_var_assignment_32406' (line 475)
    tuple_var_assignment_32406_33908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_32406')
    # Assigning a type to the variable 'f' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 31), 'f', tuple_var_assignment_32406_33908)
    
    # Assigning a Name to a Name (line 475):
    # Getting the type of 'tuple_var_assignment_32407' (line 475)
    tuple_var_assignment_32407_33909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_32407')
    # Assigning a type to the variable 'f_middle' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 34), 'f_middle', tuple_var_assignment_32407_33909)
    
    # Assigning a Call to a Name (line 476):
    
    # Assigning a Call to a Name (line 476):
    
    # Call to bc(...): (line 476)
    # Processing the call arguments (line 476)
    
    # Obtaining the type of the subscript
    slice_33911 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 476, 24), None, None, None)
    int_33912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 33), 'int')
    # Getting the type of 'y_new' (line 476)
    y_new_33913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 24), 'y_new', False)
    # Obtaining the member '__getitem__' of a type (line 476)
    getitem___33914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 24), y_new_33913, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 476)
    subscript_call_result_33915 = invoke(stypy.reporting.localization.Localization(__file__, 476, 24), getitem___33914, (slice_33911, int_33912))
    
    
    # Obtaining the type of the subscript
    slice_33916 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 476, 37), None, None, None)
    int_33917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 46), 'int')
    # Getting the type of 'y_new' (line 476)
    y_new_33918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 37), 'y_new', False)
    # Obtaining the member '__getitem__' of a type (line 476)
    getitem___33919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 37), y_new_33918, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 476)
    subscript_call_result_33920 = invoke(stypy.reporting.localization.Localization(__file__, 476, 37), getitem___33919, (slice_33916, int_33917))
    
    # Getting the type of 'p_new' (line 476)
    p_new_33921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 51), 'p_new', False)
    # Processing the call keyword arguments (line 476)
    kwargs_33922 = {}
    # Getting the type of 'bc' (line 476)
    bc_33910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 21), 'bc', False)
    # Calling bc(args, kwargs) (line 476)
    bc_call_result_33923 = invoke(stypy.reporting.localization.Localization(__file__, 476, 21), bc_33910, *[subscript_call_result_33915, subscript_call_result_33920, p_new_33921], **kwargs_33922)
    
    # Assigning a type to the variable 'bc_res' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'bc_res', bc_call_result_33923)
    
    # Assigning a Call to a Name (line 477):
    
    # Assigning a Call to a Name (line 477):
    
    # Call to hstack(...): (line 477)
    # Processing the call arguments (line 477)
    
    # Obtaining an instance of the builtin type 'tuple' (line 477)
    tuple_33926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 477)
    # Adding element type (line 477)
    
    # Call to ravel(...): (line 477)
    # Processing the call keyword arguments (line 477)
    str_33929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 49), 'str', 'F')
    keyword_33930 = str_33929
    kwargs_33931 = {'order': keyword_33930}
    # Getting the type of 'col_res' (line 477)
    col_res_33927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 29), 'col_res', False)
    # Obtaining the member 'ravel' of a type (line 477)
    ravel_33928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 29), col_res_33927, 'ravel')
    # Calling ravel(args, kwargs) (line 477)
    ravel_call_result_33932 = invoke(stypy.reporting.localization.Localization(__file__, 477, 29), ravel_33928, *[], **kwargs_33931)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 29), tuple_33926, ravel_call_result_33932)
    # Adding element type (line 477)
    # Getting the type of 'bc_res' (line 477)
    bc_res_33933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 55), 'bc_res', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 29), tuple_33926, bc_res_33933)
    
    # Processing the call keyword arguments (line 477)
    kwargs_33934 = {}
    # Getting the type of 'np' (line 477)
    np_33924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 18), 'np', False)
    # Obtaining the member 'hstack' of a type (line 477)
    hstack_33925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 18), np_33924, 'hstack')
    # Calling hstack(args, kwargs) (line 477)
    hstack_call_result_33935 = invoke(stypy.reporting.localization.Localization(__file__, 477, 18), hstack_33925, *[tuple_33926], **kwargs_33934)
    
    # Assigning a type to the variable 'res' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'res', hstack_call_result_33935)
    
    # Assigning a Call to a Name (line 479):
    
    # Assigning a Call to a Name (line 479):
    
    # Call to solve(...): (line 479)
    # Processing the call arguments (line 479)
    # Getting the type of 'res' (line 479)
    res_33938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 32), 'res', False)
    # Processing the call keyword arguments (line 479)
    kwargs_33939 = {}
    # Getting the type of 'LU' (line 479)
    LU_33936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 23), 'LU', False)
    # Obtaining the member 'solve' of a type (line 479)
    solve_33937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 23), LU_33936, 'solve')
    # Calling solve(args, kwargs) (line 479)
    solve_call_result_33940 = invoke(stypy.reporting.localization.Localization(__file__, 479, 23), solve_33937, *[res_33938], **kwargs_33939)
    
    # Assigning a type to the variable 'step_new' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'step_new', solve_call_result_33940)
    
    # Assigning a Call to a Name (line 480):
    
    # Assigning a Call to a Name (line 480):
    
    # Call to dot(...): (line 480)
    # Processing the call arguments (line 480)
    # Getting the type of 'step_new' (line 480)
    step_new_33943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 30), 'step_new', False)
    # Getting the type of 'step_new' (line 480)
    step_new_33944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 40), 'step_new', False)
    # Processing the call keyword arguments (line 480)
    kwargs_33945 = {}
    # Getting the type of 'np' (line 480)
    np_33941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 23), 'np', False)
    # Obtaining the member 'dot' of a type (line 480)
    dot_33942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 23), np_33941, 'dot')
    # Calling dot(args, kwargs) (line 480)
    dot_call_result_33946 = invoke(stypy.reporting.localization.Localization(__file__, 480, 23), dot_33942, *[step_new_33943, step_new_33944], **kwargs_33945)
    
    # Assigning a type to the variable 'cost_new' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'cost_new', dot_call_result_33946)
    
    
    # Getting the type of 'cost_new' (line 481)
    cost_new_33947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 15), 'cost_new')
    int_33948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 27), 'int')
    int_33949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 31), 'int')
    # Getting the type of 'alpha' (line 481)
    alpha_33950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 35), 'alpha')
    # Applying the binary operator '*' (line 481)
    result_mul_33951 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 31), '*', int_33949, alpha_33950)
    
    # Getting the type of 'sigma' (line 481)
    sigma_33952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 43), 'sigma')
    # Applying the binary operator '*' (line 481)
    result_mul_33953 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 41), '*', result_mul_33951, sigma_33952)
    
    # Applying the binary operator '-' (line 481)
    result_sub_33954 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 27), '-', int_33948, result_mul_33953)
    
    # Getting the type of 'cost' (line 481)
    cost_33955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 52), 'cost')
    # Applying the binary operator '*' (line 481)
    result_mul_33956 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 26), '*', result_sub_33954, cost_33955)
    
    # Applying the binary operator '<' (line 481)
    result_lt_33957 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 15), '<', cost_new_33947, result_mul_33956)
    
    # Testing the type of an if condition (line 481)
    if_condition_33958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 481, 12), result_lt_33957)
    # Assigning a type to the variable 'if_condition_33958' (line 481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'if_condition_33958', if_condition_33958)
    # SSA begins for if statement (line 481)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 481)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'trial' (line 484)
    trial_33959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 15), 'trial')
    # Getting the type of 'n_trial' (line 484)
    n_trial_33960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 23), 'n_trial')
    # Applying the binary operator '<' (line 484)
    result_lt_33961 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 15), '<', trial_33959, n_trial_33960)
    
    # Testing the type of an if condition (line 484)
    if_condition_33962 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 484, 12), result_lt_33961)
    # Assigning a type to the variable 'if_condition_33962' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'if_condition_33962', if_condition_33962)
    # SSA begins for if statement (line 484)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'alpha' (line 485)
    alpha_33963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 16), 'alpha')
    # Getting the type of 'tau' (line 485)
    tau_33964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 25), 'tau')
    # Applying the binary operator '*=' (line 485)
    result_imul_33965 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 16), '*=', alpha_33963, tau_33964)
    # Assigning a type to the variable 'alpha' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 16), 'alpha', result_imul_33965)
    
    # SSA join for if statement (line 484)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 487):
    
    # Assigning a Name to a Name (line 487):
    # Getting the type of 'y_new' (line 487)
    y_new_33966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'y_new')
    # Assigning a type to the variable 'y' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'y', y_new_33966)
    
    # Assigning a Name to a Name (line 488):
    
    # Assigning a Name to a Name (line 488):
    # Getting the type of 'p_new' (line 488)
    p_new_33967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'p_new')
    # Assigning a type to the variable 'p' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'p', p_new_33967)
    
    
    # Getting the type of 'njev' (line 490)
    njev_33968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 11), 'njev')
    # Getting the type of 'max_njev' (line 490)
    max_njev_33969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 19), 'max_njev')
    # Applying the binary operator '==' (line 490)
    result_eq_33970 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 11), '==', njev_33968, max_njev_33969)
    
    # Testing the type of an if condition (line 490)
    if_condition_33971 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 490, 8), result_eq_33970)
    # Assigning a type to the variable 'if_condition_33971' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'if_condition_33971', if_condition_33971)
    # SSA begins for if statement (line 490)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 490)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to all(...): (line 493)
    # Processing the call arguments (line 493)
    
    
    # Call to abs(...): (line 493)
    # Processing the call arguments (line 493)
    # Getting the type of 'col_res' (line 493)
    col_res_33976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 26), 'col_res', False)
    # Processing the call keyword arguments (line 493)
    kwargs_33977 = {}
    # Getting the type of 'np' (line 493)
    np_33974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 19), 'np', False)
    # Obtaining the member 'abs' of a type (line 493)
    abs_33975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 19), np_33974, 'abs')
    # Calling abs(args, kwargs) (line 493)
    abs_call_result_33978 = invoke(stypy.reporting.localization.Localization(__file__, 493, 19), abs_33975, *[col_res_33976], **kwargs_33977)
    
    # Getting the type of 'tol_r' (line 493)
    tol_r_33979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 37), 'tol_r', False)
    int_33980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 46), 'int')
    
    # Call to abs(...): (line 493)
    # Processing the call arguments (line 493)
    # Getting the type of 'f_middle' (line 493)
    f_middle_33983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 57), 'f_middle', False)
    # Processing the call keyword arguments (line 493)
    kwargs_33984 = {}
    # Getting the type of 'np' (line 493)
    np_33981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 50), 'np', False)
    # Obtaining the member 'abs' of a type (line 493)
    abs_33982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 50), np_33981, 'abs')
    # Calling abs(args, kwargs) (line 493)
    abs_call_result_33985 = invoke(stypy.reporting.localization.Localization(__file__, 493, 50), abs_33982, *[f_middle_33983], **kwargs_33984)
    
    # Applying the binary operator '+' (line 493)
    result_add_33986 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 46), '+', int_33980, abs_call_result_33985)
    
    # Applying the binary operator '*' (line 493)
    result_mul_33987 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 37), '*', tol_r_33979, result_add_33986)
    
    # Applying the binary operator '<' (line 493)
    result_lt_33988 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 19), '<', abs_call_result_33978, result_mul_33987)
    
    # Processing the call keyword arguments (line 493)
    kwargs_33989 = {}
    # Getting the type of 'np' (line 493)
    np_33972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'np', False)
    # Obtaining the member 'all' of a type (line 493)
    all_33973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 12), np_33972, 'all')
    # Calling all(args, kwargs) (line 493)
    all_call_result_33990 = invoke(stypy.reporting.localization.Localization(__file__, 493, 12), all_33973, *[result_lt_33988], **kwargs_33989)
    
    
    # Call to all(...): (line 494)
    # Processing the call arguments (line 494)
    
    # Getting the type of 'bc_res' (line 494)
    bc_res_33993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 23), 'bc_res', False)
    # Getting the type of 'tol_bc' (line 494)
    tol_bc_33994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 32), 'tol_bc', False)
    # Applying the binary operator '<' (line 494)
    result_lt_33995 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 23), '<', bc_res_33993, tol_bc_33994)
    
    # Processing the call keyword arguments (line 494)
    kwargs_33996 = {}
    # Getting the type of 'np' (line 494)
    np_33991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 16), 'np', False)
    # Obtaining the member 'all' of a type (line 494)
    all_33992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 16), np_33991, 'all')
    # Calling all(args, kwargs) (line 494)
    all_call_result_33997 = invoke(stypy.reporting.localization.Localization(__file__, 494, 16), all_33992, *[result_lt_33995], **kwargs_33996)
    
    # Applying the binary operator 'and' (line 493)
    result_and_keyword_33998 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 12), 'and', all_call_result_33990, all_call_result_33997)
    
    # Testing the type of an if condition (line 493)
    if_condition_33999 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 493, 8), result_and_keyword_33998)
    # Assigning a type to the variable 'if_condition_33999' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'if_condition_33999', if_condition_33999)
    # SSA begins for if statement (line 493)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 493)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'alpha' (line 499)
    alpha_34000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 11), 'alpha')
    int_34001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 20), 'int')
    # Applying the binary operator '==' (line 499)
    result_eq_34002 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 11), '==', alpha_34000, int_34001)
    
    # Testing the type of an if condition (line 499)
    if_condition_34003 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 499, 8), result_eq_34002)
    # Assigning a type to the variable 'if_condition_34003' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'if_condition_34003', if_condition_34003)
    # SSA begins for if statement (line 499)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 500):
    
    # Assigning a Name to a Name (line 500):
    # Getting the type of 'step_new' (line 500)
    step_new_34004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 19), 'step_new')
    # Assigning a type to the variable 'step' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'step', step_new_34004)
    
    # Assigning a Name to a Name (line 501):
    
    # Assigning a Name to a Name (line 501):
    # Getting the type of 'cost_new' (line 501)
    cost_new_34005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 19), 'cost_new')
    # Assigning a type to the variable 'cost' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'cost', cost_new_34005)
    
    # Assigning a Name to a Name (line 502):
    
    # Assigning a Name to a Name (line 502):
    # Getting the type of 'False' (line 502)
    False_34006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 28), 'False')
    # Assigning a type to the variable 'recompute_jac' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'recompute_jac', False_34006)
    # SSA branch for the else part of an if statement (line 499)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 504):
    
    # Assigning a Name to a Name (line 504):
    # Getting the type of 'True' (line 504)
    True_34007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 28), 'True')
    # Assigning a type to the variable 'recompute_jac' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'recompute_jac', True_34007)
    # SSA join for if statement (line 499)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 506)
    tuple_34008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 506)
    # Adding element type (line 506)
    # Getting the type of 'y' (line 506)
    y_34009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 11), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 11), tuple_34008, y_34009)
    # Adding element type (line 506)
    # Getting the type of 'p' (line 506)
    p_34010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 14), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 11), tuple_34008, p_34010)
    # Adding element type (line 506)
    # Getting the type of 'singular' (line 506)
    singular_34011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 17), 'singular')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 11), tuple_34008, singular_34011)
    
    # Assigning a type to the variable 'stypy_return_type' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'stypy_return_type', tuple_34008)
    
    # ################# End of 'solve_newton(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_newton' in the type store
    # Getting the type of 'stypy_return_type' (line 350)
    stypy_return_type_34012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34012)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_newton'
    return stypy_return_type_34012

# Assigning a type to the variable 'solve_newton' (line 350)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 0), 'solve_newton', solve_newton)

@norecursion
def print_iteration_header(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_iteration_header'
    module_type_store = module_type_store.open_function_context('print_iteration_header', 509, 0, False)
    
    # Passed parameters checking function
    print_iteration_header.stypy_localization = localization
    print_iteration_header.stypy_type_of_self = None
    print_iteration_header.stypy_type_store = module_type_store
    print_iteration_header.stypy_function_name = 'print_iteration_header'
    print_iteration_header.stypy_param_names_list = []
    print_iteration_header.stypy_varargs_param_name = None
    print_iteration_header.stypy_kwargs_param_name = None
    print_iteration_header.stypy_call_defaults = defaults
    print_iteration_header.stypy_call_varargs = varargs
    print_iteration_header.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_iteration_header', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_iteration_header', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_iteration_header(...)' code ##################

    
    # Call to print(...): (line 510)
    # Processing the call arguments (line 510)
    
    # Call to format(...): (line 510)
    # Processing the call arguments (line 510)
    str_34016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 8), 'str', 'Iteration')
    str_34017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 21), 'str', 'Max residual')
    str_34018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 37), 'str', 'Total nodes')
    str_34019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 52), 'str', 'Nodes added')
    # Processing the call keyword arguments (line 510)
    kwargs_34020 = {}
    str_34014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 10), 'str', '{:^15}{:^15}{:^15}{:^15}')
    # Obtaining the member 'format' of a type (line 510)
    format_34015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 10), str_34014, 'format')
    # Calling format(args, kwargs) (line 510)
    format_call_result_34021 = invoke(stypy.reporting.localization.Localization(__file__, 510, 10), format_34015, *[str_34016, str_34017, str_34018, str_34019], **kwargs_34020)
    
    # Processing the call keyword arguments (line 510)
    kwargs_34022 = {}
    # Getting the type of 'print' (line 510)
    print_34013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'print', False)
    # Calling print(args, kwargs) (line 510)
    print_call_result_34023 = invoke(stypy.reporting.localization.Localization(__file__, 510, 4), print_34013, *[format_call_result_34021], **kwargs_34022)
    
    
    # ################# End of 'print_iteration_header(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_iteration_header' in the type store
    # Getting the type of 'stypy_return_type' (line 509)
    stypy_return_type_34024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34024)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_iteration_header'
    return stypy_return_type_34024

# Assigning a type to the variable 'print_iteration_header' (line 509)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 0), 'print_iteration_header', print_iteration_header)

@norecursion
def print_iteration_progress(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_iteration_progress'
    module_type_store = module_type_store.open_function_context('print_iteration_progress', 514, 0, False)
    
    # Passed parameters checking function
    print_iteration_progress.stypy_localization = localization
    print_iteration_progress.stypy_type_of_self = None
    print_iteration_progress.stypy_type_store = module_type_store
    print_iteration_progress.stypy_function_name = 'print_iteration_progress'
    print_iteration_progress.stypy_param_names_list = ['iteration', 'residual', 'total_nodes', 'nodes_added']
    print_iteration_progress.stypy_varargs_param_name = None
    print_iteration_progress.stypy_kwargs_param_name = None
    print_iteration_progress.stypy_call_defaults = defaults
    print_iteration_progress.stypy_call_varargs = varargs
    print_iteration_progress.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_iteration_progress', ['iteration', 'residual', 'total_nodes', 'nodes_added'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_iteration_progress', localization, ['iteration', 'residual', 'total_nodes', 'nodes_added'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_iteration_progress(...)' code ##################

    
    # Call to print(...): (line 515)
    # Processing the call arguments (line 515)
    
    # Call to format(...): (line 515)
    # Processing the call arguments (line 515)
    # Getting the type of 'iteration' (line 516)
    iteration_34028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'iteration', False)
    # Getting the type of 'residual' (line 516)
    residual_34029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 19), 'residual', False)
    # Getting the type of 'total_nodes' (line 516)
    total_nodes_34030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 29), 'total_nodes', False)
    # Getting the type of 'nodes_added' (line 516)
    nodes_added_34031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 42), 'nodes_added', False)
    # Processing the call keyword arguments (line 515)
    kwargs_34032 = {}
    str_34026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 10), 'str', '{:^15}{:^15.2e}{:^15}{:^15}')
    # Obtaining the member 'format' of a type (line 515)
    format_34027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 10), str_34026, 'format')
    # Calling format(args, kwargs) (line 515)
    format_call_result_34033 = invoke(stypy.reporting.localization.Localization(__file__, 515, 10), format_34027, *[iteration_34028, residual_34029, total_nodes_34030, nodes_added_34031], **kwargs_34032)
    
    # Processing the call keyword arguments (line 515)
    kwargs_34034 = {}
    # Getting the type of 'print' (line 515)
    print_34025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'print', False)
    # Calling print(args, kwargs) (line 515)
    print_call_result_34035 = invoke(stypy.reporting.localization.Localization(__file__, 515, 4), print_34025, *[format_call_result_34033], **kwargs_34034)
    
    
    # ################# End of 'print_iteration_progress(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_iteration_progress' in the type store
    # Getting the type of 'stypy_return_type' (line 514)
    stypy_return_type_34036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34036)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_iteration_progress'
    return stypy_return_type_34036

# Assigning a type to the variable 'print_iteration_progress' (line 514)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 0), 'print_iteration_progress', print_iteration_progress)
# Declaration of the 'BVPResult' class
# Getting the type of 'OptimizeResult' (line 519)
OptimizeResult_34037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 16), 'OptimizeResult')

class BVPResult(OptimizeResult_34037, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 519, 0, False)
        # Assigning a type to the variable 'self' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BVPResult.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BVPResult' (line 519)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 0), 'BVPResult', BVPResult)

# Assigning a Dict to a Name (line 523):

# Assigning a Dict to a Name (line 523):

# Obtaining an instance of the builtin type 'dict' (line 523)
dict_34038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 23), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 523)
# Adding element type (key, value) (line 523)
int_34039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 4), 'int')
str_34040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 7), 'str', 'The algorithm converged to the desired accuracy.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 23), dict_34038, (int_34039, str_34040))
# Adding element type (key, value) (line 523)
int_34041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 4), 'int')
str_34042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 7), 'str', 'The maximum number of mesh nodes is exceeded.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 23), dict_34038, (int_34041, str_34042))
# Adding element type (key, value) (line 523)
int_34043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 4), 'int')
str_34044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 7), 'str', 'A singular Jacobian encountered when solving the collocation system.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 23), dict_34038, (int_34043, str_34044))

# Assigning a type to the variable 'TERMINATION_MESSAGES' (line 523)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 0), 'TERMINATION_MESSAGES', dict_34038)

@norecursion
def estimate_rms_residuals(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'estimate_rms_residuals'
    module_type_store = module_type_store.open_function_context('estimate_rms_residuals', 530, 0, False)
    
    # Passed parameters checking function
    estimate_rms_residuals.stypy_localization = localization
    estimate_rms_residuals.stypy_type_of_self = None
    estimate_rms_residuals.stypy_type_store = module_type_store
    estimate_rms_residuals.stypy_function_name = 'estimate_rms_residuals'
    estimate_rms_residuals.stypy_param_names_list = ['fun', 'sol', 'x', 'h', 'p', 'r_middle', 'f_middle']
    estimate_rms_residuals.stypy_varargs_param_name = None
    estimate_rms_residuals.stypy_kwargs_param_name = None
    estimate_rms_residuals.stypy_call_defaults = defaults
    estimate_rms_residuals.stypy_call_varargs = varargs
    estimate_rms_residuals.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'estimate_rms_residuals', ['fun', 'sol', 'x', 'h', 'p', 'r_middle', 'f_middle'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'estimate_rms_residuals', localization, ['fun', 'sol', 'x', 'h', 'p', 'r_middle', 'f_middle'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'estimate_rms_residuals(...)' code ##################

    str_34045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, (-1)), 'str', 'Estimate rms values of collocation residuals using Lobatto quadrature.\n\n    The residuals are defined as the difference between the derivatives of\n    our solution and rhs of the ODE system. We use relative residuals, i.e.\n    normalized by 1 + np.abs(f). RMS values are computed as sqrt from the\n    normalized integrals of the squared relative residuals over each interval.\n    Integrals are estimated using 5-point Lobatto quadrature [1]_, we use the\n    fact that residuals at the mesh nodes are identically zero.\n\n    In [2] they don\'t normalize integrals by interval lengths, which gives\n    a higher rate of convergence of the residuals by the factor of h**0.5.\n    I chose to do such normalization for an ease of interpretation of return\n    values as RMS estimates.\n\n    Returns\n    -------\n    rms_res : ndarray, shape (m - 1,)\n        Estimated rms values of the relative residuals over each interval.\n\n    References\n    ----------\n    .. [1] http://mathworld.wolfram.com/LobattoQuadrature.html\n    .. [2] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual\n       Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,\n       Number 3, pp. 299-316, 2001.\n    ')
    
    # Assigning a BinOp to a Name (line 557):
    
    # Assigning a BinOp to a Name (line 557):
    
    # Obtaining the type of the subscript
    int_34046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 18), 'int')
    slice_34047 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 557, 15), None, int_34046, None)
    # Getting the type of 'x' (line 557)
    x_34048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 15), 'x')
    # Obtaining the member '__getitem__' of a type (line 557)
    getitem___34049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 15), x_34048, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 557)
    subscript_call_result_34050 = invoke(stypy.reporting.localization.Localization(__file__, 557, 15), getitem___34049, slice_34047)
    
    float_34051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 24), 'float')
    # Getting the type of 'h' (line 557)
    h_34052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 30), 'h')
    # Applying the binary operator '*' (line 557)
    result_mul_34053 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 24), '*', float_34051, h_34052)
    
    # Applying the binary operator '+' (line 557)
    result_add_34054 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 15), '+', subscript_call_result_34050, result_mul_34053)
    
    # Assigning a type to the variable 'x_middle' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'x_middle', result_add_34054)
    
    # Assigning a BinOp to a Name (line 558):
    
    # Assigning a BinOp to a Name (line 558):
    float_34055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 8), 'float')
    # Getting the type of 'h' (line 558)
    h_34056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 14), 'h')
    # Applying the binary operator '*' (line 558)
    result_mul_34057 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 8), '*', float_34055, h_34056)
    
    int_34058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 19), 'int')
    int_34059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 21), 'int')
    # Applying the binary operator 'div' (line 558)
    result_div_34060 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 19), 'div', int_34058, int_34059)
    
    float_34061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 25), 'float')
    # Applying the binary operator '**' (line 558)
    result_pow_34062 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 18), '**', result_div_34060, float_34061)
    
    # Applying the binary operator '*' (line 558)
    result_mul_34063 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 16), '*', result_mul_34057, result_pow_34062)
    
    # Assigning a type to the variable 's' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 4), 's', result_mul_34063)
    
    # Assigning a BinOp to a Name (line 559):
    
    # Assigning a BinOp to a Name (line 559):
    # Getting the type of 'x_middle' (line 559)
    x_middle_34064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 9), 'x_middle')
    # Getting the type of 's' (line 559)
    s_34065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 20), 's')
    # Applying the binary operator '+' (line 559)
    result_add_34066 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 9), '+', x_middle_34064, s_34065)
    
    # Assigning a type to the variable 'x1' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'x1', result_add_34066)
    
    # Assigning a BinOp to a Name (line 560):
    
    # Assigning a BinOp to a Name (line 560):
    # Getting the type of 'x_middle' (line 560)
    x_middle_34067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 9), 'x_middle')
    # Getting the type of 's' (line 560)
    s_34068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 20), 's')
    # Applying the binary operator '-' (line 560)
    result_sub_34069 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 9), '-', x_middle_34067, s_34068)
    
    # Assigning a type to the variable 'x2' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'x2', result_sub_34069)
    
    # Assigning a Call to a Name (line 561):
    
    # Assigning a Call to a Name (line 561):
    
    # Call to sol(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'x1' (line 561)
    x1_34071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 13), 'x1', False)
    # Processing the call keyword arguments (line 561)
    kwargs_34072 = {}
    # Getting the type of 'sol' (line 561)
    sol_34070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 9), 'sol', False)
    # Calling sol(args, kwargs) (line 561)
    sol_call_result_34073 = invoke(stypy.reporting.localization.Localization(__file__, 561, 9), sol_34070, *[x1_34071], **kwargs_34072)
    
    # Assigning a type to the variable 'y1' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'y1', sol_call_result_34073)
    
    # Assigning a Call to a Name (line 562):
    
    # Assigning a Call to a Name (line 562):
    
    # Call to sol(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'x2' (line 562)
    x2_34075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 13), 'x2', False)
    # Processing the call keyword arguments (line 562)
    kwargs_34076 = {}
    # Getting the type of 'sol' (line 562)
    sol_34074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 9), 'sol', False)
    # Calling sol(args, kwargs) (line 562)
    sol_call_result_34077 = invoke(stypy.reporting.localization.Localization(__file__, 562, 9), sol_34074, *[x2_34075], **kwargs_34076)
    
    # Assigning a type to the variable 'y2' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'y2', sol_call_result_34077)
    
    # Assigning a Call to a Name (line 563):
    
    # Assigning a Call to a Name (line 563):
    
    # Call to sol(...): (line 563)
    # Processing the call arguments (line 563)
    # Getting the type of 'x1' (line 563)
    x1_34079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 19), 'x1', False)
    int_34080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 23), 'int')
    # Processing the call keyword arguments (line 563)
    kwargs_34081 = {}
    # Getting the type of 'sol' (line 563)
    sol_34078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 15), 'sol', False)
    # Calling sol(args, kwargs) (line 563)
    sol_call_result_34082 = invoke(stypy.reporting.localization.Localization(__file__, 563, 15), sol_34078, *[x1_34079, int_34080], **kwargs_34081)
    
    # Assigning a type to the variable 'y1_prime' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'y1_prime', sol_call_result_34082)
    
    # Assigning a Call to a Name (line 564):
    
    # Assigning a Call to a Name (line 564):
    
    # Call to sol(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'x2' (line 564)
    x2_34084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 19), 'x2', False)
    int_34085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 23), 'int')
    # Processing the call keyword arguments (line 564)
    kwargs_34086 = {}
    # Getting the type of 'sol' (line 564)
    sol_34083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 15), 'sol', False)
    # Calling sol(args, kwargs) (line 564)
    sol_call_result_34087 = invoke(stypy.reporting.localization.Localization(__file__, 564, 15), sol_34083, *[x2_34084, int_34085], **kwargs_34086)
    
    # Assigning a type to the variable 'y2_prime' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'y2_prime', sol_call_result_34087)
    
    # Assigning a Call to a Name (line 565):
    
    # Assigning a Call to a Name (line 565):
    
    # Call to fun(...): (line 565)
    # Processing the call arguments (line 565)
    # Getting the type of 'x1' (line 565)
    x1_34089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 13), 'x1', False)
    # Getting the type of 'y1' (line 565)
    y1_34090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 17), 'y1', False)
    # Getting the type of 'p' (line 565)
    p_34091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 21), 'p', False)
    # Processing the call keyword arguments (line 565)
    kwargs_34092 = {}
    # Getting the type of 'fun' (line 565)
    fun_34088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 9), 'fun', False)
    # Calling fun(args, kwargs) (line 565)
    fun_call_result_34093 = invoke(stypy.reporting.localization.Localization(__file__, 565, 9), fun_34088, *[x1_34089, y1_34090, p_34091], **kwargs_34092)
    
    # Assigning a type to the variable 'f1' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'f1', fun_call_result_34093)
    
    # Assigning a Call to a Name (line 566):
    
    # Assigning a Call to a Name (line 566):
    
    # Call to fun(...): (line 566)
    # Processing the call arguments (line 566)
    # Getting the type of 'x2' (line 566)
    x2_34095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 13), 'x2', False)
    # Getting the type of 'y2' (line 566)
    y2_34096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 17), 'y2', False)
    # Getting the type of 'p' (line 566)
    p_34097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 21), 'p', False)
    # Processing the call keyword arguments (line 566)
    kwargs_34098 = {}
    # Getting the type of 'fun' (line 566)
    fun_34094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 9), 'fun', False)
    # Calling fun(args, kwargs) (line 566)
    fun_call_result_34099 = invoke(stypy.reporting.localization.Localization(__file__, 566, 9), fun_34094, *[x2_34095, y2_34096, p_34097], **kwargs_34098)
    
    # Assigning a type to the variable 'f2' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'f2', fun_call_result_34099)
    
    # Assigning a BinOp to a Name (line 567):
    
    # Assigning a BinOp to a Name (line 567):
    # Getting the type of 'y1_prime' (line 567)
    y1_prime_34100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 9), 'y1_prime')
    # Getting the type of 'f1' (line 567)
    f1_34101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 20), 'f1')
    # Applying the binary operator '-' (line 567)
    result_sub_34102 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 9), '-', y1_prime_34100, f1_34101)
    
    # Assigning a type to the variable 'r1' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'r1', result_sub_34102)
    
    # Assigning a BinOp to a Name (line 568):
    
    # Assigning a BinOp to a Name (line 568):
    # Getting the type of 'y2_prime' (line 568)
    y2_prime_34103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 9), 'y2_prime')
    # Getting the type of 'f2' (line 568)
    f2_34104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 20), 'f2')
    # Applying the binary operator '-' (line 568)
    result_sub_34105 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 9), '-', y2_prime_34103, f2_34104)
    
    # Assigning a type to the variable 'r2' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'r2', result_sub_34105)
    
    # Getting the type of 'r_middle' (line 570)
    r_middle_34106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'r_middle')
    int_34107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 16), 'int')
    
    # Call to abs(...): (line 570)
    # Processing the call arguments (line 570)
    # Getting the type of 'f_middle' (line 570)
    f_middle_34110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 27), 'f_middle', False)
    # Processing the call keyword arguments (line 570)
    kwargs_34111 = {}
    # Getting the type of 'np' (line 570)
    np_34108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 20), 'np', False)
    # Obtaining the member 'abs' of a type (line 570)
    abs_34109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 20), np_34108, 'abs')
    # Calling abs(args, kwargs) (line 570)
    abs_call_result_34112 = invoke(stypy.reporting.localization.Localization(__file__, 570, 20), abs_34109, *[f_middle_34110], **kwargs_34111)
    
    # Applying the binary operator '+' (line 570)
    result_add_34113 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 16), '+', int_34107, abs_call_result_34112)
    
    # Applying the binary operator 'div=' (line 570)
    result_div_34114 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 4), 'div=', r_middle_34106, result_add_34113)
    # Assigning a type to the variable 'r_middle' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'r_middle', result_div_34114)
    
    
    # Getting the type of 'r1' (line 571)
    r1_34115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 4), 'r1')
    int_34116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 10), 'int')
    
    # Call to abs(...): (line 571)
    # Processing the call arguments (line 571)
    # Getting the type of 'f1' (line 571)
    f1_34119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 21), 'f1', False)
    # Processing the call keyword arguments (line 571)
    kwargs_34120 = {}
    # Getting the type of 'np' (line 571)
    np_34117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 14), 'np', False)
    # Obtaining the member 'abs' of a type (line 571)
    abs_34118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 14), np_34117, 'abs')
    # Calling abs(args, kwargs) (line 571)
    abs_call_result_34121 = invoke(stypy.reporting.localization.Localization(__file__, 571, 14), abs_34118, *[f1_34119], **kwargs_34120)
    
    # Applying the binary operator '+' (line 571)
    result_add_34122 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 10), '+', int_34116, abs_call_result_34121)
    
    # Applying the binary operator 'div=' (line 571)
    result_div_34123 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 4), 'div=', r1_34115, result_add_34122)
    # Assigning a type to the variable 'r1' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 4), 'r1', result_div_34123)
    
    
    # Getting the type of 'r2' (line 572)
    r2_34124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'r2')
    int_34125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 10), 'int')
    
    # Call to abs(...): (line 572)
    # Processing the call arguments (line 572)
    # Getting the type of 'f2' (line 572)
    f2_34128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 21), 'f2', False)
    # Processing the call keyword arguments (line 572)
    kwargs_34129 = {}
    # Getting the type of 'np' (line 572)
    np_34126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 14), 'np', False)
    # Obtaining the member 'abs' of a type (line 572)
    abs_34127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 14), np_34126, 'abs')
    # Calling abs(args, kwargs) (line 572)
    abs_call_result_34130 = invoke(stypy.reporting.localization.Localization(__file__, 572, 14), abs_34127, *[f2_34128], **kwargs_34129)
    
    # Applying the binary operator '+' (line 572)
    result_add_34131 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 10), '+', int_34125, abs_call_result_34130)
    
    # Applying the binary operator 'div=' (line 572)
    result_div_34132 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 4), 'div=', r2_34124, result_add_34131)
    # Assigning a type to the variable 'r2' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'r2', result_div_34132)
    
    
    # Assigning a Call to a Name (line 574):
    
    # Assigning a Call to a Name (line 574):
    
    # Call to sum(...): (line 574)
    # Processing the call arguments (line 574)
    
    # Call to real(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'r1' (line 574)
    r1_34137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 24), 'r1', False)
    
    # Call to conj(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'r1' (line 574)
    r1_34140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 37), 'r1', False)
    # Processing the call keyword arguments (line 574)
    kwargs_34141 = {}
    # Getting the type of 'np' (line 574)
    np_34138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 29), 'np', False)
    # Obtaining the member 'conj' of a type (line 574)
    conj_34139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 29), np_34138, 'conj')
    # Calling conj(args, kwargs) (line 574)
    conj_call_result_34142 = invoke(stypy.reporting.localization.Localization(__file__, 574, 29), conj_34139, *[r1_34140], **kwargs_34141)
    
    # Applying the binary operator '*' (line 574)
    result_mul_34143 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 24), '*', r1_34137, conj_call_result_34142)
    
    # Processing the call keyword arguments (line 574)
    kwargs_34144 = {}
    # Getting the type of 'np' (line 574)
    np_34135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 16), 'np', False)
    # Obtaining the member 'real' of a type (line 574)
    real_34136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 16), np_34135, 'real')
    # Calling real(args, kwargs) (line 574)
    real_call_result_34145 = invoke(stypy.reporting.localization.Localization(__file__, 574, 16), real_34136, *[result_mul_34143], **kwargs_34144)
    
    # Processing the call keyword arguments (line 574)
    int_34146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 48), 'int')
    keyword_34147 = int_34146
    kwargs_34148 = {'axis': keyword_34147}
    # Getting the type of 'np' (line 574)
    np_34133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 9), 'np', False)
    # Obtaining the member 'sum' of a type (line 574)
    sum_34134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 9), np_34133, 'sum')
    # Calling sum(args, kwargs) (line 574)
    sum_call_result_34149 = invoke(stypy.reporting.localization.Localization(__file__, 574, 9), sum_34134, *[real_call_result_34145], **kwargs_34148)
    
    # Assigning a type to the variable 'r1' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'r1', sum_call_result_34149)
    
    # Assigning a Call to a Name (line 575):
    
    # Assigning a Call to a Name (line 575):
    
    # Call to sum(...): (line 575)
    # Processing the call arguments (line 575)
    
    # Call to real(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'r2' (line 575)
    r2_34154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 24), 'r2', False)
    
    # Call to conj(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'r2' (line 575)
    r2_34157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 37), 'r2', False)
    # Processing the call keyword arguments (line 575)
    kwargs_34158 = {}
    # Getting the type of 'np' (line 575)
    np_34155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 29), 'np', False)
    # Obtaining the member 'conj' of a type (line 575)
    conj_34156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 29), np_34155, 'conj')
    # Calling conj(args, kwargs) (line 575)
    conj_call_result_34159 = invoke(stypy.reporting.localization.Localization(__file__, 575, 29), conj_34156, *[r2_34157], **kwargs_34158)
    
    # Applying the binary operator '*' (line 575)
    result_mul_34160 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 24), '*', r2_34154, conj_call_result_34159)
    
    # Processing the call keyword arguments (line 575)
    kwargs_34161 = {}
    # Getting the type of 'np' (line 575)
    np_34152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'np', False)
    # Obtaining the member 'real' of a type (line 575)
    real_34153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 16), np_34152, 'real')
    # Calling real(args, kwargs) (line 575)
    real_call_result_34162 = invoke(stypy.reporting.localization.Localization(__file__, 575, 16), real_34153, *[result_mul_34160], **kwargs_34161)
    
    # Processing the call keyword arguments (line 575)
    int_34163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 48), 'int')
    keyword_34164 = int_34163
    kwargs_34165 = {'axis': keyword_34164}
    # Getting the type of 'np' (line 575)
    np_34150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 9), 'np', False)
    # Obtaining the member 'sum' of a type (line 575)
    sum_34151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 9), np_34150, 'sum')
    # Calling sum(args, kwargs) (line 575)
    sum_call_result_34166 = invoke(stypy.reporting.localization.Localization(__file__, 575, 9), sum_34151, *[real_call_result_34162], **kwargs_34165)
    
    # Assigning a type to the variable 'r2' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'r2', sum_call_result_34166)
    
    # Assigning a Call to a Name (line 576):
    
    # Assigning a Call to a Name (line 576):
    
    # Call to sum(...): (line 576)
    # Processing the call arguments (line 576)
    
    # Call to real(...): (line 576)
    # Processing the call arguments (line 576)
    # Getting the type of 'r_middle' (line 576)
    r_middle_34171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 30), 'r_middle', False)
    
    # Call to conj(...): (line 576)
    # Processing the call arguments (line 576)
    # Getting the type of 'r_middle' (line 576)
    r_middle_34174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 49), 'r_middle', False)
    # Processing the call keyword arguments (line 576)
    kwargs_34175 = {}
    # Getting the type of 'np' (line 576)
    np_34172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 41), 'np', False)
    # Obtaining the member 'conj' of a type (line 576)
    conj_34173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 41), np_34172, 'conj')
    # Calling conj(args, kwargs) (line 576)
    conj_call_result_34176 = invoke(stypy.reporting.localization.Localization(__file__, 576, 41), conj_34173, *[r_middle_34174], **kwargs_34175)
    
    # Applying the binary operator '*' (line 576)
    result_mul_34177 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 30), '*', r_middle_34171, conj_call_result_34176)
    
    # Processing the call keyword arguments (line 576)
    kwargs_34178 = {}
    # Getting the type of 'np' (line 576)
    np_34169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 22), 'np', False)
    # Obtaining the member 'real' of a type (line 576)
    real_34170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 22), np_34169, 'real')
    # Calling real(args, kwargs) (line 576)
    real_call_result_34179 = invoke(stypy.reporting.localization.Localization(__file__, 576, 22), real_34170, *[result_mul_34177], **kwargs_34178)
    
    # Processing the call keyword arguments (line 576)
    int_34180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 66), 'int')
    keyword_34181 = int_34180
    kwargs_34182 = {'axis': keyword_34181}
    # Getting the type of 'np' (line 576)
    np_34167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 15), 'np', False)
    # Obtaining the member 'sum' of a type (line 576)
    sum_34168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 15), np_34167, 'sum')
    # Calling sum(args, kwargs) (line 576)
    sum_call_result_34183 = invoke(stypy.reporting.localization.Localization(__file__, 576, 15), sum_34168, *[real_call_result_34179], **kwargs_34182)
    
    # Assigning a type to the variable 'r_middle' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'r_middle', sum_call_result_34183)
    float_34184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 12), 'float')
    int_34185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 19), 'int')
    int_34186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 24), 'int')
    # Applying the binary operator 'div' (line 578)
    result_div_34187 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 19), 'div', int_34185, int_34186)
    
    # Getting the type of 'r_middle' (line 578)
    r_middle_34188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 29), 'r_middle')
    # Applying the binary operator '*' (line 578)
    result_mul_34189 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 27), '*', result_div_34187, r_middle_34188)
    
    int_34190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 40), 'int')
    int_34191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 45), 'int')
    # Applying the binary operator 'div' (line 578)
    result_div_34192 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 40), 'div', int_34190, int_34191)
    
    # Getting the type of 'r1' (line 578)
    r1_34193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 51), 'r1')
    # Getting the type of 'r2' (line 578)
    r2_34194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 56), 'r2')
    # Applying the binary operator '+' (line 578)
    result_add_34195 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 51), '+', r1_34193, r2_34194)
    
    # Applying the binary operator '*' (line 578)
    result_mul_34196 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 48), '*', result_div_34192, result_add_34195)
    
    # Applying the binary operator '+' (line 578)
    result_add_34197 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 19), '+', result_mul_34189, result_mul_34196)
    
    # Applying the binary operator '*' (line 578)
    result_mul_34198 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 12), '*', float_34184, result_add_34197)
    
    float_34199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 65), 'float')
    # Applying the binary operator '**' (line 578)
    result_pow_34200 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 11), '**', result_mul_34198, float_34199)
    
    # Assigning a type to the variable 'stypy_return_type' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'stypy_return_type', result_pow_34200)
    
    # ################# End of 'estimate_rms_residuals(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'estimate_rms_residuals' in the type store
    # Getting the type of 'stypy_return_type' (line 530)
    stypy_return_type_34201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34201)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'estimate_rms_residuals'
    return stypy_return_type_34201

# Assigning a type to the variable 'estimate_rms_residuals' (line 530)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 0), 'estimate_rms_residuals', estimate_rms_residuals)

@norecursion
def create_spline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_spline'
    module_type_store = module_type_store.open_function_context('create_spline', 581, 0, False)
    
    # Passed parameters checking function
    create_spline.stypy_localization = localization
    create_spline.stypy_type_of_self = None
    create_spline.stypy_type_store = module_type_store
    create_spline.stypy_function_name = 'create_spline'
    create_spline.stypy_param_names_list = ['y', 'yp', 'x', 'h']
    create_spline.stypy_varargs_param_name = None
    create_spline.stypy_kwargs_param_name = None
    create_spline.stypy_call_defaults = defaults
    create_spline.stypy_call_varargs = varargs
    create_spline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_spline', ['y', 'yp', 'x', 'h'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_spline', localization, ['y', 'yp', 'x', 'h'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_spline(...)' code ##################

    str_34202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, (-1)), 'str', 'Create a cubic spline given values and derivatives.\n\n    Formulas for the coefficients are taken from interpolate.CubicSpline.\n\n    Returns\n    -------\n    sol : PPoly\n        Constructed spline as a PPoly instance.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 591, 4))
    
    # 'from scipy.interpolate import PPoly' statement (line 591)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
    import_34203 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 591, 4), 'scipy.interpolate')

    if (type(import_34203) is not StypyTypeError):

        if (import_34203 != 'pyd_module'):
            __import__(import_34203)
            sys_modules_34204 = sys.modules[import_34203]
            import_from_module(stypy.reporting.localization.Localization(__file__, 591, 4), 'scipy.interpolate', sys_modules_34204.module_type_store, module_type_store, ['PPoly'])
            nest_module(stypy.reporting.localization.Localization(__file__, 591, 4), __file__, sys_modules_34204, sys_modules_34204.module_type_store, module_type_store)
        else:
            from scipy.interpolate import PPoly

            import_from_module(stypy.reporting.localization.Localization(__file__, 591, 4), 'scipy.interpolate', None, module_type_store, ['PPoly'], [PPoly])

    else:
        # Assigning a type to the variable 'scipy.interpolate' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 4), 'scipy.interpolate', import_34203)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')
    
    
    # Assigning a Attribute to a Tuple (line 593):
    
    # Assigning a Subscript to a Name (line 593):
    
    # Obtaining the type of the subscript
    int_34205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 4), 'int')
    # Getting the type of 'y' (line 593)
    y_34206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 11), 'y')
    # Obtaining the member 'shape' of a type (line 593)
    shape_34207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 11), y_34206, 'shape')
    # Obtaining the member '__getitem__' of a type (line 593)
    getitem___34208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 4), shape_34207, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 593)
    subscript_call_result_34209 = invoke(stypy.reporting.localization.Localization(__file__, 593, 4), getitem___34208, int_34205)
    
    # Assigning a type to the variable 'tuple_var_assignment_32408' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'tuple_var_assignment_32408', subscript_call_result_34209)
    
    # Assigning a Subscript to a Name (line 593):
    
    # Obtaining the type of the subscript
    int_34210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 4), 'int')
    # Getting the type of 'y' (line 593)
    y_34211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 11), 'y')
    # Obtaining the member 'shape' of a type (line 593)
    shape_34212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 11), y_34211, 'shape')
    # Obtaining the member '__getitem__' of a type (line 593)
    getitem___34213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 4), shape_34212, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 593)
    subscript_call_result_34214 = invoke(stypy.reporting.localization.Localization(__file__, 593, 4), getitem___34213, int_34210)
    
    # Assigning a type to the variable 'tuple_var_assignment_32409' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'tuple_var_assignment_32409', subscript_call_result_34214)
    
    # Assigning a Name to a Name (line 593):
    # Getting the type of 'tuple_var_assignment_32408' (line 593)
    tuple_var_assignment_32408_34215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'tuple_var_assignment_32408')
    # Assigning a type to the variable 'n' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'n', tuple_var_assignment_32408_34215)
    
    # Assigning a Name to a Name (line 593):
    # Getting the type of 'tuple_var_assignment_32409' (line 593)
    tuple_var_assignment_32409_34216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'tuple_var_assignment_32409')
    # Assigning a type to the variable 'm' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 7), 'm', tuple_var_assignment_32409_34216)
    
    # Assigning a Call to a Name (line 594):
    
    # Assigning a Call to a Name (line 594):
    
    # Call to empty(...): (line 594)
    # Processing the call arguments (line 594)
    
    # Obtaining an instance of the builtin type 'tuple' (line 594)
    tuple_34219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 594)
    # Adding element type (line 594)
    int_34220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 18), tuple_34219, int_34220)
    # Adding element type (line 594)
    # Getting the type of 'n' (line 594)
    n_34221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 21), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 18), tuple_34219, n_34221)
    # Adding element type (line 594)
    # Getting the type of 'm' (line 594)
    m_34222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 24), 'm', False)
    int_34223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 28), 'int')
    # Applying the binary operator '-' (line 594)
    result_sub_34224 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 24), '-', m_34222, int_34223)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 18), tuple_34219, result_sub_34224)
    
    # Processing the call keyword arguments (line 594)
    # Getting the type of 'y' (line 594)
    y_34225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 38), 'y', False)
    # Obtaining the member 'dtype' of a type (line 594)
    dtype_34226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 38), y_34225, 'dtype')
    keyword_34227 = dtype_34226
    kwargs_34228 = {'dtype': keyword_34227}
    # Getting the type of 'np' (line 594)
    np_34217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 594)
    empty_34218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 8), np_34217, 'empty')
    # Calling empty(args, kwargs) (line 594)
    empty_call_result_34229 = invoke(stypy.reporting.localization.Localization(__file__, 594, 8), empty_34218, *[tuple_34219], **kwargs_34228)
    
    # Assigning a type to the variable 'c' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'c', empty_call_result_34229)
    
    # Assigning a BinOp to a Name (line 595):
    
    # Assigning a BinOp to a Name (line 595):
    
    # Obtaining the type of the subscript
    slice_34230 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 595, 13), None, None, None)
    int_34231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 18), 'int')
    slice_34232 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 595, 13), int_34231, None, None)
    # Getting the type of 'y' (line 595)
    y_34233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 13), 'y')
    # Obtaining the member '__getitem__' of a type (line 595)
    getitem___34234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 13), y_34233, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 595)
    subscript_call_result_34235 = invoke(stypy.reporting.localization.Localization(__file__, 595, 13), getitem___34234, (slice_34230, slice_34232))
    
    
    # Obtaining the type of the subscript
    slice_34236 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 595, 24), None, None, None)
    int_34237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 30), 'int')
    slice_34238 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 595, 24), None, int_34237, None)
    # Getting the type of 'y' (line 595)
    y_34239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 24), 'y')
    # Obtaining the member '__getitem__' of a type (line 595)
    getitem___34240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 24), y_34239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 595)
    subscript_call_result_34241 = invoke(stypy.reporting.localization.Localization(__file__, 595, 24), getitem___34240, (slice_34236, slice_34238))
    
    # Applying the binary operator '-' (line 595)
    result_sub_34242 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 13), '-', subscript_call_result_34235, subscript_call_result_34241)
    
    # Getting the type of 'h' (line 595)
    h_34243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 37), 'h')
    # Applying the binary operator 'div' (line 595)
    result_div_34244 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 12), 'div', result_sub_34242, h_34243)
    
    # Assigning a type to the variable 'slope' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'slope', result_div_34244)
    
    # Assigning a BinOp to a Name (line 596):
    
    # Assigning a BinOp to a Name (line 596):
    
    # Obtaining the type of the subscript
    slice_34245 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 596, 9), None, None, None)
    int_34246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 16), 'int')
    slice_34247 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 596, 9), None, int_34246, None)
    # Getting the type of 'yp' (line 596)
    yp_34248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 9), 'yp')
    # Obtaining the member '__getitem__' of a type (line 596)
    getitem___34249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 9), yp_34248, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 596)
    subscript_call_result_34250 = invoke(stypy.reporting.localization.Localization(__file__, 596, 9), getitem___34249, (slice_34245, slice_34247))
    
    
    # Obtaining the type of the subscript
    slice_34251 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 596, 22), None, None, None)
    int_34252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 28), 'int')
    slice_34253 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 596, 22), int_34252, None, None)
    # Getting the type of 'yp' (line 596)
    yp_34254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 22), 'yp')
    # Obtaining the member '__getitem__' of a type (line 596)
    getitem___34255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 22), yp_34254, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 596)
    subscript_call_result_34256 = invoke(stypy.reporting.localization.Localization(__file__, 596, 22), getitem___34255, (slice_34251, slice_34253))
    
    # Applying the binary operator '+' (line 596)
    result_add_34257 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 9), '+', subscript_call_result_34250, subscript_call_result_34256)
    
    int_34258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 34), 'int')
    # Getting the type of 'slope' (line 596)
    slope_34259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 38), 'slope')
    # Applying the binary operator '*' (line 596)
    result_mul_34260 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 34), '*', int_34258, slope_34259)
    
    # Applying the binary operator '-' (line 596)
    result_sub_34261 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 32), '-', result_add_34257, result_mul_34260)
    
    # Getting the type of 'h' (line 596)
    h_34262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 47), 'h')
    # Applying the binary operator 'div' (line 596)
    result_div_34263 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 8), 'div', result_sub_34261, h_34262)
    
    # Assigning a type to the variable 't' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 't', result_div_34263)
    
    # Assigning a BinOp to a Subscript (line 597):
    
    # Assigning a BinOp to a Subscript (line 597):
    # Getting the type of 't' (line 597)
    t_34264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 11), 't')
    # Getting the type of 'h' (line 597)
    h_34265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 15), 'h')
    # Applying the binary operator 'div' (line 597)
    result_div_34266 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 11), 'div', t_34264, h_34265)
    
    # Getting the type of 'c' (line 597)
    c_34267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'c')
    int_34268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 6), 'int')
    # Storing an element on a container (line 597)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 4), c_34267, (int_34268, result_div_34266))
    
    # Assigning a BinOp to a Subscript (line 598):
    
    # Assigning a BinOp to a Subscript (line 598):
    # Getting the type of 'slope' (line 598)
    slope_34269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 12), 'slope')
    
    # Obtaining the type of the subscript
    slice_34270 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 598, 20), None, None, None)
    int_34271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 27), 'int')
    slice_34272 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 598, 20), None, int_34271, None)
    # Getting the type of 'yp' (line 598)
    yp_34273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 20), 'yp')
    # Obtaining the member '__getitem__' of a type (line 598)
    getitem___34274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 20), yp_34273, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 598)
    subscript_call_result_34275 = invoke(stypy.reporting.localization.Localization(__file__, 598, 20), getitem___34274, (slice_34270, slice_34272))
    
    # Applying the binary operator '-' (line 598)
    result_sub_34276 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 12), '-', slope_34269, subscript_call_result_34275)
    
    # Getting the type of 'h' (line 598)
    h_34277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 34), 'h')
    # Applying the binary operator 'div' (line 598)
    result_div_34278 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 11), 'div', result_sub_34276, h_34277)
    
    # Getting the type of 't' (line 598)
    t_34279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 38), 't')
    # Applying the binary operator '-' (line 598)
    result_sub_34280 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 11), '-', result_div_34278, t_34279)
    
    # Getting the type of 'c' (line 598)
    c_34281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'c')
    int_34282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 6), 'int')
    # Storing an element on a container (line 598)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 4), c_34281, (int_34282, result_sub_34280))
    
    # Assigning a Subscript to a Subscript (line 599):
    
    # Assigning a Subscript to a Subscript (line 599):
    
    # Obtaining the type of the subscript
    slice_34283 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 599, 11), None, None, None)
    int_34284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 18), 'int')
    slice_34285 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 599, 11), None, int_34284, None)
    # Getting the type of 'yp' (line 599)
    yp_34286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 11), 'yp')
    # Obtaining the member '__getitem__' of a type (line 599)
    getitem___34287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 11), yp_34286, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 599)
    subscript_call_result_34288 = invoke(stypy.reporting.localization.Localization(__file__, 599, 11), getitem___34287, (slice_34283, slice_34285))
    
    # Getting the type of 'c' (line 599)
    c_34289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'c')
    int_34290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 6), 'int')
    # Storing an element on a container (line 599)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 4), c_34289, (int_34290, subscript_call_result_34288))
    
    # Assigning a Subscript to a Subscript (line 600):
    
    # Assigning a Subscript to a Subscript (line 600):
    
    # Obtaining the type of the subscript
    slice_34291 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 600, 11), None, None, None)
    int_34292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 17), 'int')
    slice_34293 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 600, 11), None, int_34292, None)
    # Getting the type of 'y' (line 600)
    y_34294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 11), 'y')
    # Obtaining the member '__getitem__' of a type (line 600)
    getitem___34295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 11), y_34294, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 600)
    subscript_call_result_34296 = invoke(stypy.reporting.localization.Localization(__file__, 600, 11), getitem___34295, (slice_34291, slice_34293))
    
    # Getting the type of 'c' (line 600)
    c_34297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'c')
    int_34298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 6), 'int')
    # Storing an element on a container (line 600)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 4), c_34297, (int_34298, subscript_call_result_34296))
    
    # Assigning a Call to a Name (line 601):
    
    # Assigning a Call to a Name (line 601):
    
    # Call to rollaxis(...): (line 601)
    # Processing the call arguments (line 601)
    # Getting the type of 'c' (line 601)
    c_34301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 20), 'c', False)
    int_34302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 23), 'int')
    # Processing the call keyword arguments (line 601)
    kwargs_34303 = {}
    # Getting the type of 'np' (line 601)
    np_34299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 601)
    rollaxis_34300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 8), np_34299, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 601)
    rollaxis_call_result_34304 = invoke(stypy.reporting.localization.Localization(__file__, 601, 8), rollaxis_34300, *[c_34301, int_34302], **kwargs_34303)
    
    # Assigning a type to the variable 'c' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'c', rollaxis_call_result_34304)
    
    # Call to PPoly(...): (line 603)
    # Processing the call arguments (line 603)
    # Getting the type of 'c' (line 603)
    c_34306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 17), 'c', False)
    # Getting the type of 'x' (line 603)
    x_34307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 20), 'x', False)
    # Processing the call keyword arguments (line 603)
    # Getting the type of 'True' (line 603)
    True_34308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 35), 'True', False)
    keyword_34309 = True_34308
    int_34310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 46), 'int')
    keyword_34311 = int_34310
    kwargs_34312 = {'extrapolate': keyword_34309, 'axis': keyword_34311}
    # Getting the type of 'PPoly' (line 603)
    PPoly_34305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 11), 'PPoly', False)
    # Calling PPoly(args, kwargs) (line 603)
    PPoly_call_result_34313 = invoke(stypy.reporting.localization.Localization(__file__, 603, 11), PPoly_34305, *[c_34306, x_34307], **kwargs_34312)
    
    # Assigning a type to the variable 'stypy_return_type' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'stypy_return_type', PPoly_call_result_34313)
    
    # ################# End of 'create_spline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_spline' in the type store
    # Getting the type of 'stypy_return_type' (line 581)
    stypy_return_type_34314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34314)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_spline'
    return stypy_return_type_34314

# Assigning a type to the variable 'create_spline' (line 581)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 0), 'create_spline', create_spline)

@norecursion
def modify_mesh(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'modify_mesh'
    module_type_store = module_type_store.open_function_context('modify_mesh', 606, 0, False)
    
    # Passed parameters checking function
    modify_mesh.stypy_localization = localization
    modify_mesh.stypy_type_of_self = None
    modify_mesh.stypy_type_store = module_type_store
    modify_mesh.stypy_function_name = 'modify_mesh'
    modify_mesh.stypy_param_names_list = ['x', 'insert_1', 'insert_2']
    modify_mesh.stypy_varargs_param_name = None
    modify_mesh.stypy_kwargs_param_name = None
    modify_mesh.stypy_call_defaults = defaults
    modify_mesh.stypy_call_varargs = varargs
    modify_mesh.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'modify_mesh', ['x', 'insert_1', 'insert_2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'modify_mesh', localization, ['x', 'insert_1', 'insert_2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'modify_mesh(...)' code ##################

    str_34315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, (-1)), 'str', 'Insert nodes into a mesh.\n\n    Nodes removal logic is not established, its impact on the solver is\n    presumably negligible. So only insertion is done in this function.\n\n    Parameters\n    ----------\n    x : ndarray, shape (m,)\n        Mesh nodes.\n    insert_1 : ndarray\n        Intervals to each insert 1 new node in the middle.\n    insert_2 : ndarray\n        Intervals to each insert 2 new nodes, such that divide an interval\n        into 3 equal parts.\n\n    Returns\n    -------\n    x_new : ndarray\n        New mesh nodes.\n\n    Notes\n    -----\n    `insert_1` and `insert_2` should not have common values.\n    ')
    
    # Call to sort(...): (line 633)
    # Processing the call arguments (line 633)
    
    # Call to hstack(...): (line 633)
    # Processing the call arguments (line 633)
    
    # Obtaining an instance of the builtin type 'tuple' (line 634)
    tuple_34320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 634)
    # Adding element type (line 634)
    # Getting the type of 'x' (line 634)
    x_34321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 8), tuple_34320, x_34321)
    # Adding element type (line 634)
    float_34322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 8), 'float')
    
    # Obtaining the type of the subscript
    # Getting the type of 'insert_1' (line 635)
    insert_1_34323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 17), 'insert_1', False)
    # Getting the type of 'x' (line 635)
    x_34324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 15), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 635)
    getitem___34325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 15), x_34324, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 635)
    subscript_call_result_34326 = invoke(stypy.reporting.localization.Localization(__file__, 635, 15), getitem___34325, insert_1_34323)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'insert_1' (line 635)
    insert_1_34327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 31), 'insert_1', False)
    int_34328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 42), 'int')
    # Applying the binary operator '+' (line 635)
    result_add_34329 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 31), '+', insert_1_34327, int_34328)
    
    # Getting the type of 'x' (line 635)
    x_34330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 29), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 635)
    getitem___34331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 29), x_34330, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 635)
    subscript_call_result_34332 = invoke(stypy.reporting.localization.Localization(__file__, 635, 29), getitem___34331, result_add_34329)
    
    # Applying the binary operator '+' (line 635)
    result_add_34333 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 15), '+', subscript_call_result_34326, subscript_call_result_34332)
    
    # Applying the binary operator '*' (line 635)
    result_mul_34334 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 8), '*', float_34322, result_add_34333)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 8), tuple_34320, result_mul_34334)
    # Adding element type (line 634)
    int_34335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 9), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'insert_2' (line 636)
    insert_2_34336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 15), 'insert_2', False)
    # Getting the type of 'x' (line 636)
    x_34337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 13), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 636)
    getitem___34338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 13), x_34337, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 636)
    subscript_call_result_34339 = invoke(stypy.reporting.localization.Localization(__file__, 636, 13), getitem___34338, insert_2_34336)
    
    # Applying the binary operator '*' (line 636)
    result_mul_34340 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 9), '*', int_34335, subscript_call_result_34339)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'insert_2' (line 636)
    insert_2_34341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 29), 'insert_2', False)
    int_34342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 40), 'int')
    # Applying the binary operator '+' (line 636)
    result_add_34343 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 29), '+', insert_2_34341, int_34342)
    
    # Getting the type of 'x' (line 636)
    x_34344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 27), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 636)
    getitem___34345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 27), x_34344, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 636)
    subscript_call_result_34346 = invoke(stypy.reporting.localization.Localization(__file__, 636, 27), getitem___34345, result_add_34343)
    
    # Applying the binary operator '+' (line 636)
    result_add_34347 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 9), '+', result_mul_34340, subscript_call_result_34346)
    
    int_34348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 46), 'int')
    # Applying the binary operator 'div' (line 636)
    result_div_34349 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 8), 'div', result_add_34347, int_34348)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 8), tuple_34320, result_div_34349)
    # Adding element type (line 634)
    
    # Obtaining the type of the subscript
    # Getting the type of 'insert_2' (line 637)
    insert_2_34350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 11), 'insert_2', False)
    # Getting the type of 'x' (line 637)
    x_34351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 9), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 637)
    getitem___34352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 9), x_34351, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 637)
    subscript_call_result_34353 = invoke(stypy.reporting.localization.Localization(__file__, 637, 9), getitem___34352, insert_2_34350)
    
    int_34354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 23), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'insert_2' (line 637)
    insert_2_34355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 29), 'insert_2', False)
    int_34356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 40), 'int')
    # Applying the binary operator '+' (line 637)
    result_add_34357 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 29), '+', insert_2_34355, int_34356)
    
    # Getting the type of 'x' (line 637)
    x_34358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 27), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 637)
    getitem___34359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 27), x_34358, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 637)
    subscript_call_result_34360 = invoke(stypy.reporting.localization.Localization(__file__, 637, 27), getitem___34359, result_add_34357)
    
    # Applying the binary operator '*' (line 637)
    result_mul_34361 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 23), '*', int_34354, subscript_call_result_34360)
    
    # Applying the binary operator '+' (line 637)
    result_add_34362 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 9), '+', subscript_call_result_34353, result_mul_34361)
    
    int_34363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 46), 'int')
    # Applying the binary operator 'div' (line 637)
    result_div_34364 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 8), 'div', result_add_34362, int_34363)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 8), tuple_34320, result_div_34364)
    
    # Processing the call keyword arguments (line 633)
    kwargs_34365 = {}
    # Getting the type of 'np' (line 633)
    np_34318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 19), 'np', False)
    # Obtaining the member 'hstack' of a type (line 633)
    hstack_34319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 19), np_34318, 'hstack')
    # Calling hstack(args, kwargs) (line 633)
    hstack_call_result_34366 = invoke(stypy.reporting.localization.Localization(__file__, 633, 19), hstack_34319, *[tuple_34320], **kwargs_34365)
    
    # Processing the call keyword arguments (line 633)
    kwargs_34367 = {}
    # Getting the type of 'np' (line 633)
    np_34316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 11), 'np', False)
    # Obtaining the member 'sort' of a type (line 633)
    sort_34317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 11), np_34316, 'sort')
    # Calling sort(args, kwargs) (line 633)
    sort_call_result_34368 = invoke(stypy.reporting.localization.Localization(__file__, 633, 11), sort_34317, *[hstack_call_result_34366], **kwargs_34367)
    
    # Assigning a type to the variable 'stypy_return_type' (line 633)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 4), 'stypy_return_type', sort_call_result_34368)
    
    # ################# End of 'modify_mesh(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'modify_mesh' in the type store
    # Getting the type of 'stypy_return_type' (line 606)
    stypy_return_type_34369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34369)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'modify_mesh'
    return stypy_return_type_34369

# Assigning a type to the variable 'modify_mesh' (line 606)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 0), 'modify_mesh', modify_mesh)

@norecursion
def wrap_functions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'wrap_functions'
    module_type_store = module_type_store.open_function_context('wrap_functions', 641, 0, False)
    
    # Passed parameters checking function
    wrap_functions.stypy_localization = localization
    wrap_functions.stypy_type_of_self = None
    wrap_functions.stypy_type_store = module_type_store
    wrap_functions.stypy_function_name = 'wrap_functions'
    wrap_functions.stypy_param_names_list = ['fun', 'bc', 'fun_jac', 'bc_jac', 'k', 'a', 'S', 'D', 'dtype']
    wrap_functions.stypy_varargs_param_name = None
    wrap_functions.stypy_kwargs_param_name = None
    wrap_functions.stypy_call_defaults = defaults
    wrap_functions.stypy_call_varargs = varargs
    wrap_functions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'wrap_functions', ['fun', 'bc', 'fun_jac', 'bc_jac', 'k', 'a', 'S', 'D', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'wrap_functions', localization, ['fun', 'bc', 'fun_jac', 'bc_jac', 'k', 'a', 'S', 'D', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'wrap_functions(...)' code ##################

    str_34370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 4), 'str', 'Wrap functions for unified usage in the solver.')
    
    # Type idiom detected: calculating its left and rigth part (line 643)
    # Getting the type of 'fun_jac' (line 643)
    fun_jac_34371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 7), 'fun_jac')
    # Getting the type of 'None' (line 643)
    None_34372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 18), 'None')
    
    (may_be_34373, more_types_in_union_34374) = may_be_none(fun_jac_34371, None_34372)

    if may_be_34373:

        if more_types_in_union_34374:
            # Runtime conditional SSA (line 643)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 644):
        
        # Assigning a Name to a Name (line 644):
        # Getting the type of 'None' (line 644)
        None_34375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 26), 'None')
        # Assigning a type to the variable 'fun_jac_wrapped' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'fun_jac_wrapped', None_34375)

        if more_types_in_union_34374:
            # SSA join for if statement (line 643)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 646)
    # Getting the type of 'bc_jac' (line 646)
    bc_jac_34376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 7), 'bc_jac')
    # Getting the type of 'None' (line 646)
    None_34377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 17), 'None')
    
    (may_be_34378, more_types_in_union_34379) = may_be_none(bc_jac_34376, None_34377)

    if may_be_34378:

        if more_types_in_union_34379:
            # Runtime conditional SSA (line 646)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 647):
        
        # Assigning a Name to a Name (line 647):
        # Getting the type of 'None' (line 647)
        None_34380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 25), 'None')
        # Assigning a type to the variable 'bc_jac_wrapped' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'bc_jac_wrapped', None_34380)

        if more_types_in_union_34379:
            # SSA join for if statement (line 646)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'k' (line 649)
    k_34381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 7), 'k')
    int_34382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 12), 'int')
    # Applying the binary operator '==' (line 649)
    result_eq_34383 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 7), '==', k_34381, int_34382)
    
    # Testing the type of an if condition (line 649)
    if_condition_34384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 649, 4), result_eq_34383)
    # Assigning a type to the variable 'if_condition_34384' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'if_condition_34384', if_condition_34384)
    # SSA begins for if statement (line 649)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

    @norecursion
    def fun_p(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun_p'
        module_type_store = module_type_store.open_function_context('fun_p', 650, 8, False)
        
        # Passed parameters checking function
        fun_p.stypy_localization = localization
        fun_p.stypy_type_of_self = None
        fun_p.stypy_type_store = module_type_store
        fun_p.stypy_function_name = 'fun_p'
        fun_p.stypy_param_names_list = ['x', 'y', '_']
        fun_p.stypy_varargs_param_name = None
        fun_p.stypy_kwargs_param_name = None
        fun_p.stypy_call_defaults = defaults
        fun_p.stypy_call_varargs = varargs
        fun_p.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fun_p', ['x', 'y', '_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun_p', localization, ['x', 'y', '_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun_p(...)' code ##################

        
        # Call to asarray(...): (line 651)
        # Processing the call arguments (line 651)
        
        # Call to fun(...): (line 651)
        # Processing the call arguments (line 651)
        # Getting the type of 'x' (line 651)
        x_34388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 34), 'x', False)
        # Getting the type of 'y' (line 651)
        y_34389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 37), 'y', False)
        # Processing the call keyword arguments (line 651)
        kwargs_34390 = {}
        # Getting the type of 'fun' (line 651)
        fun_34387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 30), 'fun', False)
        # Calling fun(args, kwargs) (line 651)
        fun_call_result_34391 = invoke(stypy.reporting.localization.Localization(__file__, 651, 30), fun_34387, *[x_34388, y_34389], **kwargs_34390)
        
        # Getting the type of 'dtype' (line 651)
        dtype_34392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 41), 'dtype', False)
        # Processing the call keyword arguments (line 651)
        kwargs_34393 = {}
        # Getting the type of 'np' (line 651)
        np_34385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 19), 'np', False)
        # Obtaining the member 'asarray' of a type (line 651)
        asarray_34386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 19), np_34385, 'asarray')
        # Calling asarray(args, kwargs) (line 651)
        asarray_call_result_34394 = invoke(stypy.reporting.localization.Localization(__file__, 651, 19), asarray_34386, *[fun_call_result_34391, dtype_34392], **kwargs_34393)
        
        # Assigning a type to the variable 'stypy_return_type' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 12), 'stypy_return_type', asarray_call_result_34394)
        
        # ################# End of 'fun_p(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_p' in the type store
        # Getting the type of 'stypy_return_type' (line 650)
        stypy_return_type_34395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34395)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_p'
        return stypy_return_type_34395

    # Assigning a type to the variable 'fun_p' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'fun_p', fun_p)

    @norecursion
    def bc_wrapped(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'bc_wrapped'
        module_type_store = module_type_store.open_function_context('bc_wrapped', 653, 8, False)
        
        # Passed parameters checking function
        bc_wrapped.stypy_localization = localization
        bc_wrapped.stypy_type_of_self = None
        bc_wrapped.stypy_type_store = module_type_store
        bc_wrapped.stypy_function_name = 'bc_wrapped'
        bc_wrapped.stypy_param_names_list = ['ya', 'yb', '_']
        bc_wrapped.stypy_varargs_param_name = None
        bc_wrapped.stypy_kwargs_param_name = None
        bc_wrapped.stypy_call_defaults = defaults
        bc_wrapped.stypy_call_varargs = varargs
        bc_wrapped.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'bc_wrapped', ['ya', 'yb', '_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bc_wrapped', localization, ['ya', 'yb', '_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bc_wrapped(...)' code ##################

        
        # Call to asarray(...): (line 654)
        # Processing the call arguments (line 654)
        
        # Call to bc(...): (line 654)
        # Processing the call arguments (line 654)
        # Getting the type of 'ya' (line 654)
        ya_34399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 33), 'ya', False)
        # Getting the type of 'yb' (line 654)
        yb_34400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 37), 'yb', False)
        # Processing the call keyword arguments (line 654)
        kwargs_34401 = {}
        # Getting the type of 'bc' (line 654)
        bc_34398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 30), 'bc', False)
        # Calling bc(args, kwargs) (line 654)
        bc_call_result_34402 = invoke(stypy.reporting.localization.Localization(__file__, 654, 30), bc_34398, *[ya_34399, yb_34400], **kwargs_34401)
        
        # Getting the type of 'dtype' (line 654)
        dtype_34403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 42), 'dtype', False)
        # Processing the call keyword arguments (line 654)
        kwargs_34404 = {}
        # Getting the type of 'np' (line 654)
        np_34396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 19), 'np', False)
        # Obtaining the member 'asarray' of a type (line 654)
        asarray_34397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 19), np_34396, 'asarray')
        # Calling asarray(args, kwargs) (line 654)
        asarray_call_result_34405 = invoke(stypy.reporting.localization.Localization(__file__, 654, 19), asarray_34397, *[bc_call_result_34402, dtype_34403], **kwargs_34404)
        
        # Assigning a type to the variable 'stypy_return_type' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'stypy_return_type', asarray_call_result_34405)
        
        # ################# End of 'bc_wrapped(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bc_wrapped' in the type store
        # Getting the type of 'stypy_return_type' (line 653)
        stypy_return_type_34406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34406)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bc_wrapped'
        return stypy_return_type_34406

    # Assigning a type to the variable 'bc_wrapped' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'bc_wrapped', bc_wrapped)
    
    # Type idiom detected: calculating its left and rigth part (line 656)
    # Getting the type of 'fun_jac' (line 656)
    fun_jac_34407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'fun_jac')
    # Getting the type of 'None' (line 656)
    None_34408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 26), 'None')
    
    (may_be_34409, more_types_in_union_34410) = may_not_be_none(fun_jac_34407, None_34408)

    if may_be_34409:

        if more_types_in_union_34410:
            # Runtime conditional SSA (line 656)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def fun_jac_p(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fun_jac_p'
            module_type_store = module_type_store.open_function_context('fun_jac_p', 657, 12, False)
            
            # Passed parameters checking function
            fun_jac_p.stypy_localization = localization
            fun_jac_p.stypy_type_of_self = None
            fun_jac_p.stypy_type_store = module_type_store
            fun_jac_p.stypy_function_name = 'fun_jac_p'
            fun_jac_p.stypy_param_names_list = ['x', 'y', '_']
            fun_jac_p.stypy_varargs_param_name = None
            fun_jac_p.stypy_kwargs_param_name = None
            fun_jac_p.stypy_call_defaults = defaults
            fun_jac_p.stypy_call_varargs = varargs
            fun_jac_p.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fun_jac_p', ['x', 'y', '_'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fun_jac_p', localization, ['x', 'y', '_'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fun_jac_p(...)' code ##################

            
            # Obtaining an instance of the builtin type 'tuple' (line 658)
            tuple_34411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 23), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 658)
            # Adding element type (line 658)
            
            # Call to asarray(...): (line 658)
            # Processing the call arguments (line 658)
            
            # Call to fun_jac(...): (line 658)
            # Processing the call arguments (line 658)
            # Getting the type of 'x' (line 658)
            x_34415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 42), 'x', False)
            # Getting the type of 'y' (line 658)
            y_34416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 45), 'y', False)
            # Processing the call keyword arguments (line 658)
            kwargs_34417 = {}
            # Getting the type of 'fun_jac' (line 658)
            fun_jac_34414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 34), 'fun_jac', False)
            # Calling fun_jac(args, kwargs) (line 658)
            fun_jac_call_result_34418 = invoke(stypy.reporting.localization.Localization(__file__, 658, 34), fun_jac_34414, *[x_34415, y_34416], **kwargs_34417)
            
            # Getting the type of 'dtype' (line 658)
            dtype_34419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 49), 'dtype', False)
            # Processing the call keyword arguments (line 658)
            kwargs_34420 = {}
            # Getting the type of 'np' (line 658)
            np_34412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 23), 'np', False)
            # Obtaining the member 'asarray' of a type (line 658)
            asarray_34413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 23), np_34412, 'asarray')
            # Calling asarray(args, kwargs) (line 658)
            asarray_call_result_34421 = invoke(stypy.reporting.localization.Localization(__file__, 658, 23), asarray_34413, *[fun_jac_call_result_34418, dtype_34419], **kwargs_34420)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 23), tuple_34411, asarray_call_result_34421)
            # Adding element type (line 658)
            # Getting the type of 'None' (line 658)
            None_34422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 57), 'None')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 23), tuple_34411, None_34422)
            
            # Assigning a type to the variable 'stypy_return_type' (line 658)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 16), 'stypy_return_type', tuple_34411)
            
            # ################# End of 'fun_jac_p(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fun_jac_p' in the type store
            # Getting the type of 'stypy_return_type' (line 657)
            stypy_return_type_34423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_34423)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fun_jac_p'
            return stypy_return_type_34423

        # Assigning a type to the variable 'fun_jac_p' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'fun_jac_p', fun_jac_p)

        if more_types_in_union_34410:
            # SSA join for if statement (line 656)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 660)
    # Getting the type of 'bc_jac' (line 660)
    bc_jac_34424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'bc_jac')
    # Getting the type of 'None' (line 660)
    None_34425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 25), 'None')
    
    (may_be_34426, more_types_in_union_34427) = may_not_be_none(bc_jac_34424, None_34425)

    if may_be_34426:

        if more_types_in_union_34427:
            # Runtime conditional SSA (line 660)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def bc_jac_wrapped(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'bc_jac_wrapped'
            module_type_store = module_type_store.open_function_context('bc_jac_wrapped', 661, 12, False)
            
            # Passed parameters checking function
            bc_jac_wrapped.stypy_localization = localization
            bc_jac_wrapped.stypy_type_of_self = None
            bc_jac_wrapped.stypy_type_store = module_type_store
            bc_jac_wrapped.stypy_function_name = 'bc_jac_wrapped'
            bc_jac_wrapped.stypy_param_names_list = ['ya', 'yb', '_']
            bc_jac_wrapped.stypy_varargs_param_name = None
            bc_jac_wrapped.stypy_kwargs_param_name = None
            bc_jac_wrapped.stypy_call_defaults = defaults
            bc_jac_wrapped.stypy_call_varargs = varargs
            bc_jac_wrapped.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'bc_jac_wrapped', ['ya', 'yb', '_'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'bc_jac_wrapped', localization, ['ya', 'yb', '_'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'bc_jac_wrapped(...)' code ##################

            
            # Assigning a Call to a Tuple (line 662):
            
            # Assigning a Subscript to a Name (line 662):
            
            # Obtaining the type of the subscript
            int_34428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 16), 'int')
            
            # Call to bc_jac(...): (line 662)
            # Processing the call arguments (line 662)
            # Getting the type of 'ya' (line 662)
            ya_34430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 42), 'ya', False)
            # Getting the type of 'yb' (line 662)
            yb_34431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 46), 'yb', False)
            # Processing the call keyword arguments (line 662)
            kwargs_34432 = {}
            # Getting the type of 'bc_jac' (line 662)
            bc_jac_34429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 35), 'bc_jac', False)
            # Calling bc_jac(args, kwargs) (line 662)
            bc_jac_call_result_34433 = invoke(stypy.reporting.localization.Localization(__file__, 662, 35), bc_jac_34429, *[ya_34430, yb_34431], **kwargs_34432)
            
            # Obtaining the member '__getitem__' of a type (line 662)
            getitem___34434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 16), bc_jac_call_result_34433, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 662)
            subscript_call_result_34435 = invoke(stypy.reporting.localization.Localization(__file__, 662, 16), getitem___34434, int_34428)
            
            # Assigning a type to the variable 'tuple_var_assignment_32410' (line 662)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'tuple_var_assignment_32410', subscript_call_result_34435)
            
            # Assigning a Subscript to a Name (line 662):
            
            # Obtaining the type of the subscript
            int_34436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 16), 'int')
            
            # Call to bc_jac(...): (line 662)
            # Processing the call arguments (line 662)
            # Getting the type of 'ya' (line 662)
            ya_34438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 42), 'ya', False)
            # Getting the type of 'yb' (line 662)
            yb_34439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 46), 'yb', False)
            # Processing the call keyword arguments (line 662)
            kwargs_34440 = {}
            # Getting the type of 'bc_jac' (line 662)
            bc_jac_34437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 35), 'bc_jac', False)
            # Calling bc_jac(args, kwargs) (line 662)
            bc_jac_call_result_34441 = invoke(stypy.reporting.localization.Localization(__file__, 662, 35), bc_jac_34437, *[ya_34438, yb_34439], **kwargs_34440)
            
            # Obtaining the member '__getitem__' of a type (line 662)
            getitem___34442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 16), bc_jac_call_result_34441, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 662)
            subscript_call_result_34443 = invoke(stypy.reporting.localization.Localization(__file__, 662, 16), getitem___34442, int_34436)
            
            # Assigning a type to the variable 'tuple_var_assignment_32411' (line 662)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'tuple_var_assignment_32411', subscript_call_result_34443)
            
            # Assigning a Name to a Name (line 662):
            # Getting the type of 'tuple_var_assignment_32410' (line 662)
            tuple_var_assignment_32410_34444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'tuple_var_assignment_32410')
            # Assigning a type to the variable 'dbc_dya' (line 662)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'dbc_dya', tuple_var_assignment_32410_34444)
            
            # Assigning a Name to a Name (line 662):
            # Getting the type of 'tuple_var_assignment_32411' (line 662)
            tuple_var_assignment_32411_34445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'tuple_var_assignment_32411')
            # Assigning a type to the variable 'dbc_dyb' (line 662)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 25), 'dbc_dyb', tuple_var_assignment_32411_34445)
            
            # Obtaining an instance of the builtin type 'tuple' (line 663)
            tuple_34446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 24), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 663)
            # Adding element type (line 663)
            
            # Call to asarray(...): (line 663)
            # Processing the call arguments (line 663)
            # Getting the type of 'dbc_dya' (line 663)
            dbc_dya_34449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 35), 'dbc_dya', False)
            # Getting the type of 'dtype' (line 663)
            dtype_34450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 44), 'dtype', False)
            # Processing the call keyword arguments (line 663)
            kwargs_34451 = {}
            # Getting the type of 'np' (line 663)
            np_34447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 24), 'np', False)
            # Obtaining the member 'asarray' of a type (line 663)
            asarray_34448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 24), np_34447, 'asarray')
            # Calling asarray(args, kwargs) (line 663)
            asarray_call_result_34452 = invoke(stypy.reporting.localization.Localization(__file__, 663, 24), asarray_34448, *[dbc_dya_34449, dtype_34450], **kwargs_34451)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 24), tuple_34446, asarray_call_result_34452)
            # Adding element type (line 663)
            
            # Call to asarray(...): (line 664)
            # Processing the call arguments (line 664)
            # Getting the type of 'dbc_dyb' (line 664)
            dbc_dyb_34455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 35), 'dbc_dyb', False)
            # Getting the type of 'dtype' (line 664)
            dtype_34456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 44), 'dtype', False)
            # Processing the call keyword arguments (line 664)
            kwargs_34457 = {}
            # Getting the type of 'np' (line 664)
            np_34453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 24), 'np', False)
            # Obtaining the member 'asarray' of a type (line 664)
            asarray_34454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 24), np_34453, 'asarray')
            # Calling asarray(args, kwargs) (line 664)
            asarray_call_result_34458 = invoke(stypy.reporting.localization.Localization(__file__, 664, 24), asarray_34454, *[dbc_dyb_34455, dtype_34456], **kwargs_34457)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 24), tuple_34446, asarray_call_result_34458)
            # Adding element type (line 663)
            # Getting the type of 'None' (line 664)
            None_34459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 52), 'None')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 24), tuple_34446, None_34459)
            
            # Assigning a type to the variable 'stypy_return_type' (line 663)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 16), 'stypy_return_type', tuple_34446)
            
            # ################# End of 'bc_jac_wrapped(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'bc_jac_wrapped' in the type store
            # Getting the type of 'stypy_return_type' (line 661)
            stypy_return_type_34460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_34460)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'bc_jac_wrapped'
            return stypy_return_type_34460

        # Assigning a type to the variable 'bc_jac_wrapped' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'bc_jac_wrapped', bc_jac_wrapped)

        if more_types_in_union_34427:
            # SSA join for if statement (line 660)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 649)
    module_type_store.open_ssa_branch('else')

    @norecursion
    def fun_p(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun_p'
        module_type_store = module_type_store.open_function_context('fun_p', 666, 8, False)
        
        # Passed parameters checking function
        fun_p.stypy_localization = localization
        fun_p.stypy_type_of_self = None
        fun_p.stypy_type_store = module_type_store
        fun_p.stypy_function_name = 'fun_p'
        fun_p.stypy_param_names_list = ['x', 'y', 'p']
        fun_p.stypy_varargs_param_name = None
        fun_p.stypy_kwargs_param_name = None
        fun_p.stypy_call_defaults = defaults
        fun_p.stypy_call_varargs = varargs
        fun_p.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fun_p', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun_p', localization, ['x', 'y', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun_p(...)' code ##################

        
        # Call to asarray(...): (line 667)
        # Processing the call arguments (line 667)
        
        # Call to fun(...): (line 667)
        # Processing the call arguments (line 667)
        # Getting the type of 'x' (line 667)
        x_34464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 34), 'x', False)
        # Getting the type of 'y' (line 667)
        y_34465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 37), 'y', False)
        # Getting the type of 'p' (line 667)
        p_34466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 40), 'p', False)
        # Processing the call keyword arguments (line 667)
        kwargs_34467 = {}
        # Getting the type of 'fun' (line 667)
        fun_34463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 30), 'fun', False)
        # Calling fun(args, kwargs) (line 667)
        fun_call_result_34468 = invoke(stypy.reporting.localization.Localization(__file__, 667, 30), fun_34463, *[x_34464, y_34465, p_34466], **kwargs_34467)
        
        # Getting the type of 'dtype' (line 667)
        dtype_34469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 44), 'dtype', False)
        # Processing the call keyword arguments (line 667)
        kwargs_34470 = {}
        # Getting the type of 'np' (line 667)
        np_34461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 19), 'np', False)
        # Obtaining the member 'asarray' of a type (line 667)
        asarray_34462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 19), np_34461, 'asarray')
        # Calling asarray(args, kwargs) (line 667)
        asarray_call_result_34471 = invoke(stypy.reporting.localization.Localization(__file__, 667, 19), asarray_34462, *[fun_call_result_34468, dtype_34469], **kwargs_34470)
        
        # Assigning a type to the variable 'stypy_return_type' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'stypy_return_type', asarray_call_result_34471)
        
        # ################# End of 'fun_p(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_p' in the type store
        # Getting the type of 'stypy_return_type' (line 666)
        stypy_return_type_34472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34472)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_p'
        return stypy_return_type_34472

    # Assigning a type to the variable 'fun_p' (line 666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'fun_p', fun_p)

    @norecursion
    def bc_wrapped(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'bc_wrapped'
        module_type_store = module_type_store.open_function_context('bc_wrapped', 669, 8, False)
        
        # Passed parameters checking function
        bc_wrapped.stypy_localization = localization
        bc_wrapped.stypy_type_of_self = None
        bc_wrapped.stypy_type_store = module_type_store
        bc_wrapped.stypy_function_name = 'bc_wrapped'
        bc_wrapped.stypy_param_names_list = ['x', 'y', 'p']
        bc_wrapped.stypy_varargs_param_name = None
        bc_wrapped.stypy_kwargs_param_name = None
        bc_wrapped.stypy_call_defaults = defaults
        bc_wrapped.stypy_call_varargs = varargs
        bc_wrapped.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'bc_wrapped', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bc_wrapped', localization, ['x', 'y', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bc_wrapped(...)' code ##################

        
        # Call to asarray(...): (line 670)
        # Processing the call arguments (line 670)
        
        # Call to bc(...): (line 670)
        # Processing the call arguments (line 670)
        # Getting the type of 'x' (line 670)
        x_34476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 33), 'x', False)
        # Getting the type of 'y' (line 670)
        y_34477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 36), 'y', False)
        # Getting the type of 'p' (line 670)
        p_34478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 39), 'p', False)
        # Processing the call keyword arguments (line 670)
        kwargs_34479 = {}
        # Getting the type of 'bc' (line 670)
        bc_34475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 30), 'bc', False)
        # Calling bc(args, kwargs) (line 670)
        bc_call_result_34480 = invoke(stypy.reporting.localization.Localization(__file__, 670, 30), bc_34475, *[x_34476, y_34477, p_34478], **kwargs_34479)
        
        # Getting the type of 'dtype' (line 670)
        dtype_34481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 43), 'dtype', False)
        # Processing the call keyword arguments (line 670)
        kwargs_34482 = {}
        # Getting the type of 'np' (line 670)
        np_34473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 19), 'np', False)
        # Obtaining the member 'asarray' of a type (line 670)
        asarray_34474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 19), np_34473, 'asarray')
        # Calling asarray(args, kwargs) (line 670)
        asarray_call_result_34483 = invoke(stypy.reporting.localization.Localization(__file__, 670, 19), asarray_34474, *[bc_call_result_34480, dtype_34481], **kwargs_34482)
        
        # Assigning a type to the variable 'stypy_return_type' (line 670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 12), 'stypy_return_type', asarray_call_result_34483)
        
        # ################# End of 'bc_wrapped(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bc_wrapped' in the type store
        # Getting the type of 'stypy_return_type' (line 669)
        stypy_return_type_34484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34484)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bc_wrapped'
        return stypy_return_type_34484

    # Assigning a type to the variable 'bc_wrapped' (line 669)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 8), 'bc_wrapped', bc_wrapped)
    
    # Type idiom detected: calculating its left and rigth part (line 672)
    # Getting the type of 'fun_jac' (line 672)
    fun_jac_34485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'fun_jac')
    # Getting the type of 'None' (line 672)
    None_34486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 26), 'None')
    
    (may_be_34487, more_types_in_union_34488) = may_not_be_none(fun_jac_34485, None_34486)

    if may_be_34487:

        if more_types_in_union_34488:
            # Runtime conditional SSA (line 672)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def fun_jac_p(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fun_jac_p'
            module_type_store = module_type_store.open_function_context('fun_jac_p', 673, 12, False)
            
            # Passed parameters checking function
            fun_jac_p.stypy_localization = localization
            fun_jac_p.stypy_type_of_self = None
            fun_jac_p.stypy_type_store = module_type_store
            fun_jac_p.stypy_function_name = 'fun_jac_p'
            fun_jac_p.stypy_param_names_list = ['x', 'y', 'p']
            fun_jac_p.stypy_varargs_param_name = None
            fun_jac_p.stypy_kwargs_param_name = None
            fun_jac_p.stypy_call_defaults = defaults
            fun_jac_p.stypy_call_varargs = varargs
            fun_jac_p.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fun_jac_p', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fun_jac_p', localization, ['x', 'y', 'p'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fun_jac_p(...)' code ##################

            
            # Assigning a Call to a Tuple (line 674):
            
            # Assigning a Subscript to a Name (line 674):
            
            # Obtaining the type of the subscript
            int_34489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 16), 'int')
            
            # Call to fun_jac(...): (line 674)
            # Processing the call arguments (line 674)
            # Getting the type of 'x' (line 674)
            x_34491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 39), 'x', False)
            # Getting the type of 'y' (line 674)
            y_34492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 42), 'y', False)
            # Getting the type of 'p' (line 674)
            p_34493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 45), 'p', False)
            # Processing the call keyword arguments (line 674)
            kwargs_34494 = {}
            # Getting the type of 'fun_jac' (line 674)
            fun_jac_34490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 31), 'fun_jac', False)
            # Calling fun_jac(args, kwargs) (line 674)
            fun_jac_call_result_34495 = invoke(stypy.reporting.localization.Localization(__file__, 674, 31), fun_jac_34490, *[x_34491, y_34492, p_34493], **kwargs_34494)
            
            # Obtaining the member '__getitem__' of a type (line 674)
            getitem___34496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 16), fun_jac_call_result_34495, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 674)
            subscript_call_result_34497 = invoke(stypy.reporting.localization.Localization(__file__, 674, 16), getitem___34496, int_34489)
            
            # Assigning a type to the variable 'tuple_var_assignment_32412' (line 674)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 16), 'tuple_var_assignment_32412', subscript_call_result_34497)
            
            # Assigning a Subscript to a Name (line 674):
            
            # Obtaining the type of the subscript
            int_34498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 16), 'int')
            
            # Call to fun_jac(...): (line 674)
            # Processing the call arguments (line 674)
            # Getting the type of 'x' (line 674)
            x_34500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 39), 'x', False)
            # Getting the type of 'y' (line 674)
            y_34501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 42), 'y', False)
            # Getting the type of 'p' (line 674)
            p_34502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 45), 'p', False)
            # Processing the call keyword arguments (line 674)
            kwargs_34503 = {}
            # Getting the type of 'fun_jac' (line 674)
            fun_jac_34499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 31), 'fun_jac', False)
            # Calling fun_jac(args, kwargs) (line 674)
            fun_jac_call_result_34504 = invoke(stypy.reporting.localization.Localization(__file__, 674, 31), fun_jac_34499, *[x_34500, y_34501, p_34502], **kwargs_34503)
            
            # Obtaining the member '__getitem__' of a type (line 674)
            getitem___34505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 16), fun_jac_call_result_34504, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 674)
            subscript_call_result_34506 = invoke(stypy.reporting.localization.Localization(__file__, 674, 16), getitem___34505, int_34498)
            
            # Assigning a type to the variable 'tuple_var_assignment_32413' (line 674)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 16), 'tuple_var_assignment_32413', subscript_call_result_34506)
            
            # Assigning a Name to a Name (line 674):
            # Getting the type of 'tuple_var_assignment_32412' (line 674)
            tuple_var_assignment_32412_34507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 16), 'tuple_var_assignment_32412')
            # Assigning a type to the variable 'df_dy' (line 674)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 16), 'df_dy', tuple_var_assignment_32412_34507)
            
            # Assigning a Name to a Name (line 674):
            # Getting the type of 'tuple_var_assignment_32413' (line 674)
            tuple_var_assignment_32413_34508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 16), 'tuple_var_assignment_32413')
            # Assigning a type to the variable 'df_dp' (line 674)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 23), 'df_dp', tuple_var_assignment_32413_34508)
            
            # Obtaining an instance of the builtin type 'tuple' (line 675)
            tuple_34509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 23), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 675)
            # Adding element type (line 675)
            
            # Call to asarray(...): (line 675)
            # Processing the call arguments (line 675)
            # Getting the type of 'df_dy' (line 675)
            df_dy_34512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 34), 'df_dy', False)
            # Getting the type of 'dtype' (line 675)
            dtype_34513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 41), 'dtype', False)
            # Processing the call keyword arguments (line 675)
            kwargs_34514 = {}
            # Getting the type of 'np' (line 675)
            np_34510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 23), 'np', False)
            # Obtaining the member 'asarray' of a type (line 675)
            asarray_34511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 23), np_34510, 'asarray')
            # Calling asarray(args, kwargs) (line 675)
            asarray_call_result_34515 = invoke(stypy.reporting.localization.Localization(__file__, 675, 23), asarray_34511, *[df_dy_34512, dtype_34513], **kwargs_34514)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 675, 23), tuple_34509, asarray_call_result_34515)
            # Adding element type (line 675)
            
            # Call to asarray(...): (line 675)
            # Processing the call arguments (line 675)
            # Getting the type of 'df_dp' (line 675)
            df_dp_34518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 60), 'df_dp', False)
            # Getting the type of 'dtype' (line 675)
            dtype_34519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 67), 'dtype', False)
            # Processing the call keyword arguments (line 675)
            kwargs_34520 = {}
            # Getting the type of 'np' (line 675)
            np_34516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 49), 'np', False)
            # Obtaining the member 'asarray' of a type (line 675)
            asarray_34517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 49), np_34516, 'asarray')
            # Calling asarray(args, kwargs) (line 675)
            asarray_call_result_34521 = invoke(stypy.reporting.localization.Localization(__file__, 675, 49), asarray_34517, *[df_dp_34518, dtype_34519], **kwargs_34520)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 675, 23), tuple_34509, asarray_call_result_34521)
            
            # Assigning a type to the variable 'stypy_return_type' (line 675)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 16), 'stypy_return_type', tuple_34509)
            
            # ################# End of 'fun_jac_p(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fun_jac_p' in the type store
            # Getting the type of 'stypy_return_type' (line 673)
            stypy_return_type_34522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_34522)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fun_jac_p'
            return stypy_return_type_34522

        # Assigning a type to the variable 'fun_jac_p' (line 673)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 12), 'fun_jac_p', fun_jac_p)

        if more_types_in_union_34488:
            # SSA join for if statement (line 672)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 677)
    # Getting the type of 'bc_jac' (line 677)
    bc_jac_34523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 8), 'bc_jac')
    # Getting the type of 'None' (line 677)
    None_34524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 25), 'None')
    
    (may_be_34525, more_types_in_union_34526) = may_not_be_none(bc_jac_34523, None_34524)

    if may_be_34525:

        if more_types_in_union_34526:
            # Runtime conditional SSA (line 677)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def bc_jac_wrapped(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'bc_jac_wrapped'
            module_type_store = module_type_store.open_function_context('bc_jac_wrapped', 678, 12, False)
            
            # Passed parameters checking function
            bc_jac_wrapped.stypy_localization = localization
            bc_jac_wrapped.stypy_type_of_self = None
            bc_jac_wrapped.stypy_type_store = module_type_store
            bc_jac_wrapped.stypy_function_name = 'bc_jac_wrapped'
            bc_jac_wrapped.stypy_param_names_list = ['ya', 'yb', 'p']
            bc_jac_wrapped.stypy_varargs_param_name = None
            bc_jac_wrapped.stypy_kwargs_param_name = None
            bc_jac_wrapped.stypy_call_defaults = defaults
            bc_jac_wrapped.stypy_call_varargs = varargs
            bc_jac_wrapped.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'bc_jac_wrapped', ['ya', 'yb', 'p'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'bc_jac_wrapped', localization, ['ya', 'yb', 'p'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'bc_jac_wrapped(...)' code ##################

            
            # Assigning a Call to a Tuple (line 679):
            
            # Assigning a Subscript to a Name (line 679):
            
            # Obtaining the type of the subscript
            int_34527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 16), 'int')
            
            # Call to bc_jac(...): (line 679)
            # Processing the call arguments (line 679)
            # Getting the type of 'ya' (line 679)
            ya_34529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 50), 'ya', False)
            # Getting the type of 'yb' (line 679)
            yb_34530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 54), 'yb', False)
            # Getting the type of 'p' (line 679)
            p_34531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 58), 'p', False)
            # Processing the call keyword arguments (line 679)
            kwargs_34532 = {}
            # Getting the type of 'bc_jac' (line 679)
            bc_jac_34528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 43), 'bc_jac', False)
            # Calling bc_jac(args, kwargs) (line 679)
            bc_jac_call_result_34533 = invoke(stypy.reporting.localization.Localization(__file__, 679, 43), bc_jac_34528, *[ya_34529, yb_34530, p_34531], **kwargs_34532)
            
            # Obtaining the member '__getitem__' of a type (line 679)
            getitem___34534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 16), bc_jac_call_result_34533, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 679)
            subscript_call_result_34535 = invoke(stypy.reporting.localization.Localization(__file__, 679, 16), getitem___34534, int_34527)
            
            # Assigning a type to the variable 'tuple_var_assignment_32414' (line 679)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'tuple_var_assignment_32414', subscript_call_result_34535)
            
            # Assigning a Subscript to a Name (line 679):
            
            # Obtaining the type of the subscript
            int_34536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 16), 'int')
            
            # Call to bc_jac(...): (line 679)
            # Processing the call arguments (line 679)
            # Getting the type of 'ya' (line 679)
            ya_34538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 50), 'ya', False)
            # Getting the type of 'yb' (line 679)
            yb_34539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 54), 'yb', False)
            # Getting the type of 'p' (line 679)
            p_34540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 58), 'p', False)
            # Processing the call keyword arguments (line 679)
            kwargs_34541 = {}
            # Getting the type of 'bc_jac' (line 679)
            bc_jac_34537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 43), 'bc_jac', False)
            # Calling bc_jac(args, kwargs) (line 679)
            bc_jac_call_result_34542 = invoke(stypy.reporting.localization.Localization(__file__, 679, 43), bc_jac_34537, *[ya_34538, yb_34539, p_34540], **kwargs_34541)
            
            # Obtaining the member '__getitem__' of a type (line 679)
            getitem___34543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 16), bc_jac_call_result_34542, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 679)
            subscript_call_result_34544 = invoke(stypy.reporting.localization.Localization(__file__, 679, 16), getitem___34543, int_34536)
            
            # Assigning a type to the variable 'tuple_var_assignment_32415' (line 679)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'tuple_var_assignment_32415', subscript_call_result_34544)
            
            # Assigning a Subscript to a Name (line 679):
            
            # Obtaining the type of the subscript
            int_34545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 16), 'int')
            
            # Call to bc_jac(...): (line 679)
            # Processing the call arguments (line 679)
            # Getting the type of 'ya' (line 679)
            ya_34547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 50), 'ya', False)
            # Getting the type of 'yb' (line 679)
            yb_34548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 54), 'yb', False)
            # Getting the type of 'p' (line 679)
            p_34549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 58), 'p', False)
            # Processing the call keyword arguments (line 679)
            kwargs_34550 = {}
            # Getting the type of 'bc_jac' (line 679)
            bc_jac_34546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 43), 'bc_jac', False)
            # Calling bc_jac(args, kwargs) (line 679)
            bc_jac_call_result_34551 = invoke(stypy.reporting.localization.Localization(__file__, 679, 43), bc_jac_34546, *[ya_34547, yb_34548, p_34549], **kwargs_34550)
            
            # Obtaining the member '__getitem__' of a type (line 679)
            getitem___34552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 16), bc_jac_call_result_34551, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 679)
            subscript_call_result_34553 = invoke(stypy.reporting.localization.Localization(__file__, 679, 16), getitem___34552, int_34545)
            
            # Assigning a type to the variable 'tuple_var_assignment_32416' (line 679)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'tuple_var_assignment_32416', subscript_call_result_34553)
            
            # Assigning a Name to a Name (line 679):
            # Getting the type of 'tuple_var_assignment_32414' (line 679)
            tuple_var_assignment_32414_34554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'tuple_var_assignment_32414')
            # Assigning a type to the variable 'dbc_dya' (line 679)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'dbc_dya', tuple_var_assignment_32414_34554)
            
            # Assigning a Name to a Name (line 679):
            # Getting the type of 'tuple_var_assignment_32415' (line 679)
            tuple_var_assignment_32415_34555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'tuple_var_assignment_32415')
            # Assigning a type to the variable 'dbc_dyb' (line 679)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 25), 'dbc_dyb', tuple_var_assignment_32415_34555)
            
            # Assigning a Name to a Name (line 679):
            # Getting the type of 'tuple_var_assignment_32416' (line 679)
            tuple_var_assignment_32416_34556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'tuple_var_assignment_32416')
            # Assigning a type to the variable 'dbc_dp' (line 679)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 34), 'dbc_dp', tuple_var_assignment_32416_34556)
            
            # Obtaining an instance of the builtin type 'tuple' (line 680)
            tuple_34557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 24), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 680)
            # Adding element type (line 680)
            
            # Call to asarray(...): (line 680)
            # Processing the call arguments (line 680)
            # Getting the type of 'dbc_dya' (line 680)
            dbc_dya_34560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 35), 'dbc_dya', False)
            # Getting the type of 'dtype' (line 680)
            dtype_34561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 44), 'dtype', False)
            # Processing the call keyword arguments (line 680)
            kwargs_34562 = {}
            # Getting the type of 'np' (line 680)
            np_34558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 24), 'np', False)
            # Obtaining the member 'asarray' of a type (line 680)
            asarray_34559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 24), np_34558, 'asarray')
            # Calling asarray(args, kwargs) (line 680)
            asarray_call_result_34563 = invoke(stypy.reporting.localization.Localization(__file__, 680, 24), asarray_34559, *[dbc_dya_34560, dtype_34561], **kwargs_34562)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 24), tuple_34557, asarray_call_result_34563)
            # Adding element type (line 680)
            
            # Call to asarray(...): (line 680)
            # Processing the call arguments (line 680)
            # Getting the type of 'dbc_dyb' (line 680)
            dbc_dyb_34566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 63), 'dbc_dyb', False)
            # Getting the type of 'dtype' (line 680)
            dtype_34567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 72), 'dtype', False)
            # Processing the call keyword arguments (line 680)
            kwargs_34568 = {}
            # Getting the type of 'np' (line 680)
            np_34564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 52), 'np', False)
            # Obtaining the member 'asarray' of a type (line 680)
            asarray_34565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 52), np_34564, 'asarray')
            # Calling asarray(args, kwargs) (line 680)
            asarray_call_result_34569 = invoke(stypy.reporting.localization.Localization(__file__, 680, 52), asarray_34565, *[dbc_dyb_34566, dtype_34567], **kwargs_34568)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 24), tuple_34557, asarray_call_result_34569)
            # Adding element type (line 680)
            
            # Call to asarray(...): (line 681)
            # Processing the call arguments (line 681)
            # Getting the type of 'dbc_dp' (line 681)
            dbc_dp_34572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 35), 'dbc_dp', False)
            # Getting the type of 'dtype' (line 681)
            dtype_34573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 43), 'dtype', False)
            # Processing the call keyword arguments (line 681)
            kwargs_34574 = {}
            # Getting the type of 'np' (line 681)
            np_34570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 24), 'np', False)
            # Obtaining the member 'asarray' of a type (line 681)
            asarray_34571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 24), np_34570, 'asarray')
            # Calling asarray(args, kwargs) (line 681)
            asarray_call_result_34575 = invoke(stypy.reporting.localization.Localization(__file__, 681, 24), asarray_34571, *[dbc_dp_34572, dtype_34573], **kwargs_34574)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 24), tuple_34557, asarray_call_result_34575)
            
            # Assigning a type to the variable 'stypy_return_type' (line 680)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 16), 'stypy_return_type', tuple_34557)
            
            # ################# End of 'bc_jac_wrapped(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'bc_jac_wrapped' in the type store
            # Getting the type of 'stypy_return_type' (line 678)
            stypy_return_type_34576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_34576)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'bc_jac_wrapped'
            return stypy_return_type_34576

        # Assigning a type to the variable 'bc_jac_wrapped' (line 678)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'bc_jac_wrapped', bc_jac_wrapped)

        if more_types_in_union_34526:
            # SSA join for if statement (line 677)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 649)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 683)
    # Getting the type of 'S' (line 683)
    S_34577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 7), 'S')
    # Getting the type of 'None' (line 683)
    None_34578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 12), 'None')
    
    (may_be_34579, more_types_in_union_34580) = may_be_none(S_34577, None_34578)

    if may_be_34579:

        if more_types_in_union_34580:
            # Runtime conditional SSA (line 683)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 684):
        
        # Assigning a Name to a Name (line 684):
        # Getting the type of 'fun_p' (line 684)
        fun_p_34581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 22), 'fun_p')
        # Assigning a type to the variable 'fun_wrapped' (line 684)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'fun_wrapped', fun_p_34581)

        if more_types_in_union_34580:
            # Runtime conditional SSA for else branch (line 683)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_34579) or more_types_in_union_34580):

        @norecursion
        def fun_wrapped(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fun_wrapped'
            module_type_store = module_type_store.open_function_context('fun_wrapped', 686, 8, False)
            
            # Passed parameters checking function
            fun_wrapped.stypy_localization = localization
            fun_wrapped.stypy_type_of_self = None
            fun_wrapped.stypy_type_store = module_type_store
            fun_wrapped.stypy_function_name = 'fun_wrapped'
            fun_wrapped.stypy_param_names_list = ['x', 'y', 'p']
            fun_wrapped.stypy_varargs_param_name = None
            fun_wrapped.stypy_kwargs_param_name = None
            fun_wrapped.stypy_call_defaults = defaults
            fun_wrapped.stypy_call_varargs = varargs
            fun_wrapped.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fun_wrapped', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fun_wrapped', localization, ['x', 'y', 'p'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fun_wrapped(...)' code ##################

            
            # Assigning a Call to a Name (line 687):
            
            # Assigning a Call to a Name (line 687):
            
            # Call to fun_p(...): (line 687)
            # Processing the call arguments (line 687)
            # Getting the type of 'x' (line 687)
            x_34583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 22), 'x', False)
            # Getting the type of 'y' (line 687)
            y_34584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 25), 'y', False)
            # Getting the type of 'p' (line 687)
            p_34585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 28), 'p', False)
            # Processing the call keyword arguments (line 687)
            kwargs_34586 = {}
            # Getting the type of 'fun_p' (line 687)
            fun_p_34582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 16), 'fun_p', False)
            # Calling fun_p(args, kwargs) (line 687)
            fun_p_call_result_34587 = invoke(stypy.reporting.localization.Localization(__file__, 687, 16), fun_p_34582, *[x_34583, y_34584, p_34585], **kwargs_34586)
            
            # Assigning a type to the variable 'f' (line 687)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 12), 'f', fun_p_call_result_34587)
            
            
            
            # Obtaining the type of the subscript
            int_34588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 17), 'int')
            # Getting the type of 'x' (line 688)
            x_34589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 15), 'x')
            # Obtaining the member '__getitem__' of a type (line 688)
            getitem___34590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 15), x_34589, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 688)
            subscript_call_result_34591 = invoke(stypy.reporting.localization.Localization(__file__, 688, 15), getitem___34590, int_34588)
            
            # Getting the type of 'a' (line 688)
            a_34592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 23), 'a')
            # Applying the binary operator '==' (line 688)
            result_eq_34593 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 15), '==', subscript_call_result_34591, a_34592)
            
            # Testing the type of an if condition (line 688)
            if_condition_34594 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 688, 12), result_eq_34593)
            # Assigning a type to the variable 'if_condition_34594' (line 688)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 12), 'if_condition_34594', if_condition_34594)
            # SSA begins for if statement (line 688)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 689):
            
            # Assigning a Call to a Subscript (line 689):
            
            # Call to dot(...): (line 689)
            # Processing the call arguments (line 689)
            # Getting the type of 'D' (line 689)
            D_34597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 33), 'D', False)
            
            # Obtaining the type of the subscript
            slice_34598 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 689, 36), None, None, None)
            int_34599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 41), 'int')
            # Getting the type of 'f' (line 689)
            f_34600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 36), 'f', False)
            # Obtaining the member '__getitem__' of a type (line 689)
            getitem___34601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 36), f_34600, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 689)
            subscript_call_result_34602 = invoke(stypy.reporting.localization.Localization(__file__, 689, 36), getitem___34601, (slice_34598, int_34599))
            
            # Processing the call keyword arguments (line 689)
            kwargs_34603 = {}
            # Getting the type of 'np' (line 689)
            np_34595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 26), 'np', False)
            # Obtaining the member 'dot' of a type (line 689)
            dot_34596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 26), np_34595, 'dot')
            # Calling dot(args, kwargs) (line 689)
            dot_call_result_34604 = invoke(stypy.reporting.localization.Localization(__file__, 689, 26), dot_34596, *[D_34597, subscript_call_result_34602], **kwargs_34603)
            
            # Getting the type of 'f' (line 689)
            f_34605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 16), 'f')
            slice_34606 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 689, 16), None, None, None)
            int_34607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 21), 'int')
            # Storing an element on a container (line 689)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 689, 16), f_34605, ((slice_34606, int_34607), dot_call_result_34604))
            
            # Getting the type of 'f' (line 690)
            f_34608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 16), 'f')
            
            # Obtaining the type of the subscript
            slice_34609 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 690, 16), None, None, None)
            int_34610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 21), 'int')
            slice_34611 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 690, 16), int_34610, None, None)
            # Getting the type of 'f' (line 690)
            f_34612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 16), 'f')
            # Obtaining the member '__getitem__' of a type (line 690)
            getitem___34613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 16), f_34612, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 690)
            subscript_call_result_34614 = invoke(stypy.reporting.localization.Localization(__file__, 690, 16), getitem___34613, (slice_34609, slice_34611))
            
            
            # Call to dot(...): (line 690)
            # Processing the call arguments (line 690)
            # Getting the type of 'S' (line 690)
            S_34617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 35), 'S', False)
            
            # Obtaining the type of the subscript
            slice_34618 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 690, 38), None, None, None)
            int_34619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 43), 'int')
            slice_34620 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 690, 38), int_34619, None, None)
            # Getting the type of 'y' (line 690)
            y_34621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 38), 'y', False)
            # Obtaining the member '__getitem__' of a type (line 690)
            getitem___34622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 38), y_34621, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 690)
            subscript_call_result_34623 = invoke(stypy.reporting.localization.Localization(__file__, 690, 38), getitem___34622, (slice_34618, slice_34620))
            
            # Processing the call keyword arguments (line 690)
            kwargs_34624 = {}
            # Getting the type of 'np' (line 690)
            np_34615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 28), 'np', False)
            # Obtaining the member 'dot' of a type (line 690)
            dot_34616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 28), np_34615, 'dot')
            # Calling dot(args, kwargs) (line 690)
            dot_call_result_34625 = invoke(stypy.reporting.localization.Localization(__file__, 690, 28), dot_34616, *[S_34617, subscript_call_result_34623], **kwargs_34624)
            
            
            # Obtaining the type of the subscript
            int_34626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 53), 'int')
            slice_34627 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 690, 51), int_34626, None, None)
            # Getting the type of 'x' (line 690)
            x_34628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 51), 'x')
            # Obtaining the member '__getitem__' of a type (line 690)
            getitem___34629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 51), x_34628, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 690)
            subscript_call_result_34630 = invoke(stypy.reporting.localization.Localization(__file__, 690, 51), getitem___34629, slice_34627)
            
            # Getting the type of 'a' (line 690)
            a_34631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 59), 'a')
            # Applying the binary operator '-' (line 690)
            result_sub_34632 = python_operator(stypy.reporting.localization.Localization(__file__, 690, 51), '-', subscript_call_result_34630, a_34631)
            
            # Applying the binary operator 'div' (line 690)
            result_div_34633 = python_operator(stypy.reporting.localization.Localization(__file__, 690, 28), 'div', dot_call_result_34625, result_sub_34632)
            
            # Applying the binary operator '+=' (line 690)
            result_iadd_34634 = python_operator(stypy.reporting.localization.Localization(__file__, 690, 16), '+=', subscript_call_result_34614, result_div_34633)
            # Getting the type of 'f' (line 690)
            f_34635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 16), 'f')
            slice_34636 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 690, 16), None, None, None)
            int_34637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 21), 'int')
            slice_34638 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 690, 16), int_34637, None, None)
            # Storing an element on a container (line 690)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 690, 16), f_34635, ((slice_34636, slice_34638), result_iadd_34634))
            
            # SSA branch for the else part of an if statement (line 688)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'f' (line 692)
            f_34639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'f')
            
            # Call to dot(...): (line 692)
            # Processing the call arguments (line 692)
            # Getting the type of 'S' (line 692)
            S_34642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 28), 'S', False)
            # Getting the type of 'y' (line 692)
            y_34643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 31), 'y', False)
            # Processing the call keyword arguments (line 692)
            kwargs_34644 = {}
            # Getting the type of 'np' (line 692)
            np_34640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 21), 'np', False)
            # Obtaining the member 'dot' of a type (line 692)
            dot_34641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 21), np_34640, 'dot')
            # Calling dot(args, kwargs) (line 692)
            dot_call_result_34645 = invoke(stypy.reporting.localization.Localization(__file__, 692, 21), dot_34641, *[S_34642, y_34643], **kwargs_34644)
            
            # Getting the type of 'x' (line 692)
            x_34646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 37), 'x')
            # Getting the type of 'a' (line 692)
            a_34647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 41), 'a')
            # Applying the binary operator '-' (line 692)
            result_sub_34648 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 37), '-', x_34646, a_34647)
            
            # Applying the binary operator 'div' (line 692)
            result_div_34649 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 21), 'div', dot_call_result_34645, result_sub_34648)
            
            # Applying the binary operator '+=' (line 692)
            result_iadd_34650 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 16), '+=', f_34639, result_div_34649)
            # Assigning a type to the variable 'f' (line 692)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'f', result_iadd_34650)
            
            # SSA join for if statement (line 688)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'f' (line 693)
            f_34651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 19), 'f')
            # Assigning a type to the variable 'stypy_return_type' (line 693)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 12), 'stypy_return_type', f_34651)
            
            # ################# End of 'fun_wrapped(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fun_wrapped' in the type store
            # Getting the type of 'stypy_return_type' (line 686)
            stypy_return_type_34652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_34652)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fun_wrapped'
            return stypy_return_type_34652

        # Assigning a type to the variable 'fun_wrapped' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'fun_wrapped', fun_wrapped)

        if (may_be_34579 and more_types_in_union_34580):
            # SSA join for if statement (line 683)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 695)
    # Getting the type of 'fun_jac' (line 695)
    fun_jac_34653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'fun_jac')
    # Getting the type of 'None' (line 695)
    None_34654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 22), 'None')
    
    (may_be_34655, more_types_in_union_34656) = may_not_be_none(fun_jac_34653, None_34654)

    if may_be_34655:

        if more_types_in_union_34656:
            # Runtime conditional SSA (line 695)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 696)
        # Getting the type of 'S' (line 696)
        S_34657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 11), 'S')
        # Getting the type of 'None' (line 696)
        None_34658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 16), 'None')
        
        (may_be_34659, more_types_in_union_34660) = may_be_none(S_34657, None_34658)

        if may_be_34659:

            if more_types_in_union_34660:
                # Runtime conditional SSA (line 696)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 697):
            
            # Assigning a Name to a Name (line 697):
            # Getting the type of 'fun_jac_p' (line 697)
            fun_jac_p_34661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 30), 'fun_jac_p')
            # Assigning a type to the variable 'fun_jac_wrapped' (line 697)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 12), 'fun_jac_wrapped', fun_jac_p_34661)

            if more_types_in_union_34660:
                # Runtime conditional SSA for else branch (line 696)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_34659) or more_types_in_union_34660):
            
            # Assigning a Subscript to a Name (line 699):
            
            # Assigning a Subscript to a Name (line 699):
            
            # Obtaining the type of the subscript
            slice_34662 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 699, 17), None, None, None)
            slice_34663 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 699, 17), None, None, None)
            # Getting the type of 'np' (line 699)
            np_34664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 25), 'np')
            # Obtaining the member 'newaxis' of a type (line 699)
            newaxis_34665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 25), np_34664, 'newaxis')
            # Getting the type of 'S' (line 699)
            S_34666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 17), 'S')
            # Obtaining the member '__getitem__' of a type (line 699)
            getitem___34667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 17), S_34666, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 699)
            subscript_call_result_34668 = invoke(stypy.reporting.localization.Localization(__file__, 699, 17), getitem___34667, (slice_34662, slice_34663, newaxis_34665))
            
            # Assigning a type to the variable 'Sr' (line 699)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'Sr', subscript_call_result_34668)

            @norecursion
            def fun_jac_wrapped(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'fun_jac_wrapped'
                module_type_store = module_type_store.open_function_context('fun_jac_wrapped', 701, 12, False)
                
                # Passed parameters checking function
                fun_jac_wrapped.stypy_localization = localization
                fun_jac_wrapped.stypy_type_of_self = None
                fun_jac_wrapped.stypy_type_store = module_type_store
                fun_jac_wrapped.stypy_function_name = 'fun_jac_wrapped'
                fun_jac_wrapped.stypy_param_names_list = ['x', 'y', 'p']
                fun_jac_wrapped.stypy_varargs_param_name = None
                fun_jac_wrapped.stypy_kwargs_param_name = None
                fun_jac_wrapped.stypy_call_defaults = defaults
                fun_jac_wrapped.stypy_call_varargs = varargs
                fun_jac_wrapped.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'fun_jac_wrapped', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'fun_jac_wrapped', localization, ['x', 'y', 'p'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'fun_jac_wrapped(...)' code ##################

                
                # Assigning a Call to a Tuple (line 702):
                
                # Assigning a Subscript to a Name (line 702):
                
                # Obtaining the type of the subscript
                int_34669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 16), 'int')
                
                # Call to fun_jac_p(...): (line 702)
                # Processing the call arguments (line 702)
                # Getting the type of 'x' (line 702)
                x_34671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 41), 'x', False)
                # Getting the type of 'y' (line 702)
                y_34672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 44), 'y', False)
                # Getting the type of 'p' (line 702)
                p_34673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 47), 'p', False)
                # Processing the call keyword arguments (line 702)
                kwargs_34674 = {}
                # Getting the type of 'fun_jac_p' (line 702)
                fun_jac_p_34670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 31), 'fun_jac_p', False)
                # Calling fun_jac_p(args, kwargs) (line 702)
                fun_jac_p_call_result_34675 = invoke(stypy.reporting.localization.Localization(__file__, 702, 31), fun_jac_p_34670, *[x_34671, y_34672, p_34673], **kwargs_34674)
                
                # Obtaining the member '__getitem__' of a type (line 702)
                getitem___34676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 16), fun_jac_p_call_result_34675, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 702)
                subscript_call_result_34677 = invoke(stypy.reporting.localization.Localization(__file__, 702, 16), getitem___34676, int_34669)
                
                # Assigning a type to the variable 'tuple_var_assignment_32417' (line 702)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 16), 'tuple_var_assignment_32417', subscript_call_result_34677)
                
                # Assigning a Subscript to a Name (line 702):
                
                # Obtaining the type of the subscript
                int_34678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 16), 'int')
                
                # Call to fun_jac_p(...): (line 702)
                # Processing the call arguments (line 702)
                # Getting the type of 'x' (line 702)
                x_34680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 41), 'x', False)
                # Getting the type of 'y' (line 702)
                y_34681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 44), 'y', False)
                # Getting the type of 'p' (line 702)
                p_34682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 47), 'p', False)
                # Processing the call keyword arguments (line 702)
                kwargs_34683 = {}
                # Getting the type of 'fun_jac_p' (line 702)
                fun_jac_p_34679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 31), 'fun_jac_p', False)
                # Calling fun_jac_p(args, kwargs) (line 702)
                fun_jac_p_call_result_34684 = invoke(stypy.reporting.localization.Localization(__file__, 702, 31), fun_jac_p_34679, *[x_34680, y_34681, p_34682], **kwargs_34683)
                
                # Obtaining the member '__getitem__' of a type (line 702)
                getitem___34685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 16), fun_jac_p_call_result_34684, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 702)
                subscript_call_result_34686 = invoke(stypy.reporting.localization.Localization(__file__, 702, 16), getitem___34685, int_34678)
                
                # Assigning a type to the variable 'tuple_var_assignment_32418' (line 702)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 16), 'tuple_var_assignment_32418', subscript_call_result_34686)
                
                # Assigning a Name to a Name (line 702):
                # Getting the type of 'tuple_var_assignment_32417' (line 702)
                tuple_var_assignment_32417_34687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 16), 'tuple_var_assignment_32417')
                # Assigning a type to the variable 'df_dy' (line 702)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 16), 'df_dy', tuple_var_assignment_32417_34687)
                
                # Assigning a Name to a Name (line 702):
                # Getting the type of 'tuple_var_assignment_32418' (line 702)
                tuple_var_assignment_32418_34688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 16), 'tuple_var_assignment_32418')
                # Assigning a type to the variable 'df_dp' (line 702)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 23), 'df_dp', tuple_var_assignment_32418_34688)
                
                
                
                # Obtaining the type of the subscript
                int_34689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 21), 'int')
                # Getting the type of 'x' (line 703)
                x_34690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 19), 'x')
                # Obtaining the member '__getitem__' of a type (line 703)
                getitem___34691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 19), x_34690, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 703)
                subscript_call_result_34692 = invoke(stypy.reporting.localization.Localization(__file__, 703, 19), getitem___34691, int_34689)
                
                # Getting the type of 'a' (line 703)
                a_34693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 27), 'a')
                # Applying the binary operator '==' (line 703)
                result_eq_34694 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 19), '==', subscript_call_result_34692, a_34693)
                
                # Testing the type of an if condition (line 703)
                if_condition_34695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 703, 16), result_eq_34694)
                # Assigning a type to the variable 'if_condition_34695' (line 703)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 16), 'if_condition_34695', if_condition_34695)
                # SSA begins for if statement (line 703)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 704):
                
                # Assigning a Call to a Subscript (line 704):
                
                # Call to dot(...): (line 704)
                # Processing the call arguments (line 704)
                # Getting the type of 'D' (line 704)
                D_34698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 44), 'D', False)
                
                # Obtaining the type of the subscript
                slice_34699 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 704, 47), None, None, None)
                slice_34700 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 704, 47), None, None, None)
                int_34701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 59), 'int')
                # Getting the type of 'df_dy' (line 704)
                df_dy_34702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 47), 'df_dy', False)
                # Obtaining the member '__getitem__' of a type (line 704)
                getitem___34703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 47), df_dy_34702, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 704)
                subscript_call_result_34704 = invoke(stypy.reporting.localization.Localization(__file__, 704, 47), getitem___34703, (slice_34699, slice_34700, int_34701))
                
                # Processing the call keyword arguments (line 704)
                kwargs_34705 = {}
                # Getting the type of 'np' (line 704)
                np_34696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 37), 'np', False)
                # Obtaining the member 'dot' of a type (line 704)
                dot_34697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 37), np_34696, 'dot')
                # Calling dot(args, kwargs) (line 704)
                dot_call_result_34706 = invoke(stypy.reporting.localization.Localization(__file__, 704, 37), dot_34697, *[D_34698, subscript_call_result_34704], **kwargs_34705)
                
                # Getting the type of 'df_dy' (line 704)
                df_dy_34707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 20), 'df_dy')
                slice_34708 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 704, 20), None, None, None)
                slice_34709 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 704, 20), None, None, None)
                int_34710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 32), 'int')
                # Storing an element on a container (line 704)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 704, 20), df_dy_34707, ((slice_34708, slice_34709, int_34710), dot_call_result_34706))
                
                # Getting the type of 'df_dy' (line 705)
                df_dy_34711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 20), 'df_dy')
                
                # Obtaining the type of the subscript
                slice_34712 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 705, 20), None, None, None)
                slice_34713 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 705, 20), None, None, None)
                int_34714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 32), 'int')
                slice_34715 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 705, 20), int_34714, None, None)
                # Getting the type of 'df_dy' (line 705)
                df_dy_34716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 20), 'df_dy')
                # Obtaining the member '__getitem__' of a type (line 705)
                getitem___34717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 20), df_dy_34716, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 705)
                subscript_call_result_34718 = invoke(stypy.reporting.localization.Localization(__file__, 705, 20), getitem___34717, (slice_34712, slice_34713, slice_34715))
                
                # Getting the type of 'Sr' (line 705)
                Sr_34719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 39), 'Sr')
                
                # Obtaining the type of the subscript
                int_34720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 47), 'int')
                slice_34721 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 705, 45), int_34720, None, None)
                # Getting the type of 'x' (line 705)
                x_34722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 45), 'x')
                # Obtaining the member '__getitem__' of a type (line 705)
                getitem___34723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 45), x_34722, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 705)
                subscript_call_result_34724 = invoke(stypy.reporting.localization.Localization(__file__, 705, 45), getitem___34723, slice_34721)
                
                # Getting the type of 'a' (line 705)
                a_34725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 53), 'a')
                # Applying the binary operator '-' (line 705)
                result_sub_34726 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 45), '-', subscript_call_result_34724, a_34725)
                
                # Applying the binary operator 'div' (line 705)
                result_div_34727 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 39), 'div', Sr_34719, result_sub_34726)
                
                # Applying the binary operator '+=' (line 705)
                result_iadd_34728 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 20), '+=', subscript_call_result_34718, result_div_34727)
                # Getting the type of 'df_dy' (line 705)
                df_dy_34729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 20), 'df_dy')
                slice_34730 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 705, 20), None, None, None)
                slice_34731 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 705, 20), None, None, None)
                int_34732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 32), 'int')
                slice_34733 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 705, 20), int_34732, None, None)
                # Storing an element on a container (line 705)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 20), df_dy_34729, ((slice_34730, slice_34731, slice_34733), result_iadd_34728))
                
                # SSA branch for the else part of an if statement (line 703)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'df_dy' (line 707)
                df_dy_34734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 20), 'df_dy')
                # Getting the type of 'Sr' (line 707)
                Sr_34735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 29), 'Sr')
                # Getting the type of 'x' (line 707)
                x_34736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 35), 'x')
                # Getting the type of 'a' (line 707)
                a_34737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 39), 'a')
                # Applying the binary operator '-' (line 707)
                result_sub_34738 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 35), '-', x_34736, a_34737)
                
                # Applying the binary operator 'div' (line 707)
                result_div_34739 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 29), 'div', Sr_34735, result_sub_34738)
                
                # Applying the binary operator '+=' (line 707)
                result_iadd_34740 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 20), '+=', df_dy_34734, result_div_34739)
                # Assigning a type to the variable 'df_dy' (line 707)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 20), 'df_dy', result_iadd_34740)
                
                # SSA join for if statement (line 703)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Obtaining an instance of the builtin type 'tuple' (line 709)
                tuple_34741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 709)
                # Adding element type (line 709)
                # Getting the type of 'df_dy' (line 709)
                df_dy_34742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 23), 'df_dy')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 23), tuple_34741, df_dy_34742)
                # Adding element type (line 709)
                # Getting the type of 'df_dp' (line 709)
                df_dp_34743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 30), 'df_dp')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 23), tuple_34741, df_dp_34743)
                
                # Assigning a type to the variable 'stypy_return_type' (line 709)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 16), 'stypy_return_type', tuple_34741)
                
                # ################# End of 'fun_jac_wrapped(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'fun_jac_wrapped' in the type store
                # Getting the type of 'stypy_return_type' (line 701)
                stypy_return_type_34744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_34744)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'fun_jac_wrapped'
                return stypy_return_type_34744

            # Assigning a type to the variable 'fun_jac_wrapped' (line 701)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 12), 'fun_jac_wrapped', fun_jac_wrapped)

            if (may_be_34659 and more_types_in_union_34660):
                # SSA join for if statement (line 696)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_34656:
            # SSA join for if statement (line 695)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 711)
    tuple_34745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 711)
    # Adding element type (line 711)
    # Getting the type of 'fun_wrapped' (line 711)
    fun_wrapped_34746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 11), 'fun_wrapped')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 11), tuple_34745, fun_wrapped_34746)
    # Adding element type (line 711)
    # Getting the type of 'bc_wrapped' (line 711)
    bc_wrapped_34747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 24), 'bc_wrapped')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 11), tuple_34745, bc_wrapped_34747)
    # Adding element type (line 711)
    # Getting the type of 'fun_jac_wrapped' (line 711)
    fun_jac_wrapped_34748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 36), 'fun_jac_wrapped')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 11), tuple_34745, fun_jac_wrapped_34748)
    # Adding element type (line 711)
    # Getting the type of 'bc_jac_wrapped' (line 711)
    bc_jac_wrapped_34749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 53), 'bc_jac_wrapped')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 11), tuple_34745, bc_jac_wrapped_34749)
    
    # Assigning a type to the variable 'stypy_return_type' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'stypy_return_type', tuple_34745)
    
    # ################# End of 'wrap_functions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'wrap_functions' in the type store
    # Getting the type of 'stypy_return_type' (line 641)
    stypy_return_type_34750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34750)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'wrap_functions'
    return stypy_return_type_34750

# Assigning a type to the variable 'wrap_functions' (line 641)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 0), 'wrap_functions', wrap_functions)

@norecursion
def solve_bvp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 714)
    None_34751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 31), 'None')
    # Getting the type of 'None' (line 714)
    None_34752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 39), 'None')
    # Getting the type of 'None' (line 714)
    None_34753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 53), 'None')
    # Getting the type of 'None' (line 714)
    None_34754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 66), 'None')
    float_34755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 18), 'float')
    int_34756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 34), 'int')
    int_34757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 48), 'int')
    defaults = [None_34751, None_34752, None_34753, None_34754, float_34755, int_34756, int_34757]
    # Create a new context for function 'solve_bvp'
    module_type_store = module_type_store.open_function_context('solve_bvp', 714, 0, False)
    
    # Passed parameters checking function
    solve_bvp.stypy_localization = localization
    solve_bvp.stypy_type_of_self = None
    solve_bvp.stypy_type_store = module_type_store
    solve_bvp.stypy_function_name = 'solve_bvp'
    solve_bvp.stypy_param_names_list = ['fun', 'bc', 'x', 'y', 'p', 'S', 'fun_jac', 'bc_jac', 'tol', 'max_nodes', 'verbose']
    solve_bvp.stypy_varargs_param_name = None
    solve_bvp.stypy_kwargs_param_name = None
    solve_bvp.stypy_call_defaults = defaults
    solve_bvp.stypy_call_varargs = varargs
    solve_bvp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_bvp', ['fun', 'bc', 'x', 'y', 'p', 'S', 'fun_jac', 'bc_jac', 'tol', 'max_nodes', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_bvp', localization, ['fun', 'bc', 'x', 'y', 'p', 'S', 'fun_jac', 'bc_jac', 'tol', 'max_nodes', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_bvp(...)' code ##################

    str_34758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 996, (-1)), 'str', 'Solve a boundary-value problem for a system of ODEs.\n\n    This function numerically solves a first order system of ODEs subject to\n    two-point boundary conditions::\n\n        dy / dx = f(x, y, p) + S * y / (x - a), a <= x <= b\n        bc(y(a), y(b), p) = 0\n\n    Here x is a 1-dimensional independent variable, y(x) is a n-dimensional\n    vector-valued function and p is a k-dimensional vector of unknown\n    parameters which is to be found along with y(x). For the problem to be\n    determined there must be n + k boundary conditions, i.e. bc must be\n    (n + k)-dimensional function.\n\n    The last singular term in the right-hand side of the system is optional.\n    It is defined by an n-by-n matrix S, such that the solution must satisfy\n    S y(a) = 0. This condition will be forced during iterations, so it must not\n    contradict boundary conditions. See [2]_ for the explanation how this term\n    is handled when solving BVPs numerically.\n\n    Problems in a complex domain can be solved as well. In this case y and p\n    are considered to be complex, and f and bc are assumed to be complex-valued\n    functions, but x stays real. Note that f and bc must be complex\n    differentiable (satisfy Cauchy-Riemann equations [4]_), otherwise you\n    should rewrite your problem for real and imaginary parts separately. To\n    solve a problem in a complex domain, pass an initial guess for y with a\n    complex data type (see below).\n\n    Parameters\n    ----------\n    fun : callable\n        Right-hand side of the system. The calling signature is ``fun(x, y)``,\n        or ``fun(x, y, p)`` if parameters are present. All arguments are\n        ndarray: ``x`` with shape (m,), ``y`` with shape (n, m), meaning that\n        ``y[:, i]`` corresponds to ``x[i]``, and ``p`` with shape (k,). The\n        return value must be an array with shape (n, m) and with the same\n        layout as ``y``.\n    bc : callable\n        Function evaluating residuals of the boundary conditions. The calling\n        signature is ``bc(ya, yb)``, or ``bc(ya, yb, p)`` if parameters are\n        present. All arguments are ndarray: ``ya`` and ``yb`` with shape (n,),\n        and ``p`` with shape (k,). The return value must be an array with\n        shape (n + k,).\n    x : array_like, shape (m,)\n        Initial mesh. Must be a strictly increasing sequence of real numbers\n        with ``x[0]=a`` and ``x[-1]=b``.\n    y : array_like, shape (n, m)\n        Initial guess for the function values at the mesh nodes, i-th column\n        corresponds to ``x[i]``. For problems in a complex domain pass `y`\n        with a complex data type (even if the initial guess is purely real).\n    p : array_like with shape (k,) or None, optional\n        Initial guess for the unknown parameters. If None (default), it is\n        assumed that the problem doesn\'t depend on any parameters.\n    S : array_like with shape (n, n) or None\n        Matrix defining the singular term. If None (default), the problem is\n        solved without the singular term.\n    fun_jac : callable or None, optional\n        Function computing derivatives of f with respect to y and p. The\n        calling signature is ``fun_jac(x, y)``, or ``fun_jac(x, y, p)`` if\n        parameters are present. The return must contain 1 or 2 elements in the\n        following order:\n\n            * df_dy : array_like with shape (n, n, m) where an element\n              (i, j, q) equals to d f_i(x_q, y_q, p) / d (y_q)_j.\n            * df_dp : array_like with shape (n, k, m) where an element\n              (i, j, q) equals to d f_i(x_q, y_q, p) / d p_j.\n\n        Here q numbers nodes at which x and y are defined, whereas i and j\n        number vector components. If the problem is solved without unknown\n        parameters df_dp should not be returned.\n\n        If `fun_jac` is None (default), the derivatives will be estimated\n        by the forward finite differences.\n    bc_jac : callable or None, optional\n        Function computing derivatives of bc with respect to ya, yb and p.\n        The calling signature is ``bc_jac(ya, yb)``, or ``bc_jac(ya, yb, p)``\n        if parameters are present. The return must contain 2 or 3 elements in\n        the following order:\n\n            * dbc_dya : array_like with shape (n, n) where an element (i, j)\n              equals to d bc_i(ya, yb, p) / d ya_j.\n            * dbc_dyb : array_like with shape (n, n) where an element (i, j)\n              equals to d bc_i(ya, yb, p) / d yb_j.\n            * dbc_dp : array_like with shape (n, k) where an element (i, j)\n              equals to d bc_i(ya, yb, p) / d p_j.\n\n        If the problem is solved without unknown parameters dbc_dp should not\n        be returned.\n\n        If `bc_jac` is None (default), the derivatives will be estimated by\n        the forward finite differences.\n    tol : float, optional\n        Desired tolerance of the solution. If we define ``r = y\' - f(x, y)``\n        where y is the found solution, then the solver tries to achieve on each\n        mesh interval ``norm(r / (1 + abs(f)) < tol``, where ``norm`` is\n        estimated in a root mean squared sense (using a numerical quadrature\n        formula). Default is 1e-3.\n    max_nodes : int, optional\n        Maximum allowed number of the mesh nodes. If exceeded, the algorithm\n        terminates. Default is 1000.\n    verbose : {0, 1, 2}, optional\n        Level of algorithm\'s verbosity:\n\n            * 0 (default) : work silently.\n            * 1 : display a termination report.\n            * 2 : display progress during iterations.\n\n    Returns\n    -------\n    Bunch object with the following fields defined:\n    sol : PPoly\n        Found solution for y as `scipy.interpolate.PPoly` instance, a C1\n        continuous cubic spline.\n    p : ndarray or None, shape (k,)\n        Found parameters. None, if the parameters were not present in the\n        problem.\n    x : ndarray, shape (m,)\n        Nodes of the final mesh.\n    y : ndarray, shape (n, m)\n        Solution values at the mesh nodes.\n    yp : ndarray, shape (n, m)\n        Solution derivatives at the mesh nodes.\n    rms_residuals : ndarray, shape (m - 1,)\n        RMS values of the relative residuals over each mesh interval (see the\n        description of `tol` parameter).\n    niter : int\n        Number of completed iterations.\n    status : int\n        Reason for algorithm termination:\n\n            * 0: The algorithm converged to the desired accuracy.\n            * 1: The maximum number of mesh nodes is exceeded.\n            * 2: A singular Jacobian encountered when solving the collocation\n              system.\n\n    message : string\n        Verbal description of the termination reason.\n    success : bool\n        True if the algorithm converged to the desired accuracy (``status=0``).\n\n    Notes\n    -----\n    This function implements a 4-th order collocation algorithm with the\n    control of residuals similar to [1]_. A collocation system is solved\n    by a damped Newton method with an affine-invariant criterion function as\n    described in [3]_.\n\n    Note that in [1]_  integral residuals are defined without normalization\n    by interval lengths. So their definition is different by a multiplier of\n    h**0.5 (h is an interval length) from the definition used here.\n\n    .. versionadded:: 0.18.0\n\n    References\n    ----------\n    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual\n           Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,\n           Number 3, pp. 299-316, 2001.\n    .. [2] L.F. Shampine, P. H. Muir and H. Xu, "A User-Friendly Fortran BVP\n           Solver".\n    .. [3] U. Ascher, R. Mattheij and R. Russell "Numerical Solution of\n           Boundary Value Problems for Ordinary Differential Equations".\n    .. [4] `Cauchy-Riemann equations\n            <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ on\n            Wikipedia.\n\n    Examples\n    --------\n    In the first example we solve Bratu\'s problem::\n\n        y\'\' + k * exp(y) = 0\n        y(0) = y(1) = 0\n\n    for k = 1.\n\n    We rewrite the equation as a first order system and implement its\n    right-hand side evaluation::\n\n        y1\' = y2\n        y2\' = -exp(y1)\n\n    >>> def fun(x, y):\n    ...     return np.vstack((y[1], -np.exp(y[0])))\n\n    Implement evaluation of the boundary condition residuals:\n\n    >>> def bc(ya, yb):\n    ...     return np.array([ya[0], yb[0]])\n\n    Define the initial mesh with 5 nodes:\n\n    >>> x = np.linspace(0, 1, 5)\n\n    This problem is known to have two solutions. To obtain both of them we\n    use two different initial guesses for y. We denote them by subscripts\n    a and b.\n\n    >>> y_a = np.zeros((2, x.size))\n    >>> y_b = np.zeros((2, x.size))\n    >>> y_b[0] = 3\n\n    Now we are ready to run the solver.\n\n    >>> from scipy.integrate import solve_bvp\n    >>> res_a = solve_bvp(fun, bc, x, y_a)\n    >>> res_b = solve_bvp(fun, bc, x, y_b)\n\n    Let\'s plot the two found solutions. We take an advantage of having the\n    solution in a spline form to produce a smooth plot.\n\n    >>> x_plot = np.linspace(0, 1, 100)\n    >>> y_plot_a = res_a.sol(x_plot)[0]\n    >>> y_plot_b = res_b.sol(x_plot)[0]\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(x_plot, y_plot_a, label=\'y_a\')\n    >>> plt.plot(x_plot, y_plot_b, label=\'y_b\')\n    >>> plt.legend()\n    >>> plt.xlabel("x")\n    >>> plt.ylabel("y")\n    >>> plt.show()\n\n    We see that the two solutions have similar shape, but differ in scale\n    significantly.\n\n    In the second example we solve a simple Sturm-Liouville problem::\n\n        y\'\' + k**2 * y = 0\n        y(0) = y(1) = 0\n\n    It is known that a non-trivial solution y = A * sin(k * x) is possible for\n    k = pi * n, where n is an integer. To establish the normalization constant\n    A = 1 we add a boundary condition::\n\n        y\'(0) = k\n\n    Again we rewrite our equation as a first order system and implement its\n    right-hand side evaluation::\n\n        y1\' = y2\n        y2\' = -k**2 * y1\n\n    >>> def fun(x, y, p):\n    ...     k = p[0]\n    ...     return np.vstack((y[1], -k**2 * y[0]))\n\n    Note that parameters p are passed as a vector (with one element in our\n    case).\n\n    Implement the boundary conditions:\n\n    >>> def bc(ya, yb, p):\n    ...     k = p[0]\n    ...     return np.array([ya[0], yb[0], ya[1] - k])\n\n    Setup the initial mesh and guess for y. We aim to find the solution for\n    k = 2 * pi, to achieve that we set values of y to approximately follow\n    sin(2 * pi * x):\n\n    >>> x = np.linspace(0, 1, 5)\n    >>> y = np.zeros((2, x.size))\n    >>> y[0, 1] = 1\n    >>> y[0, 3] = -1\n\n    Run the solver with 6 as an initial guess for k.\n\n    >>> sol = solve_bvp(fun, bc, x, y, p=[6])\n\n    We see that the found k is approximately correct:\n\n    >>> sol.p[0]\n    6.28329460046\n\n    And finally plot the solution to see the anticipated sinusoid:\n\n    >>> x_plot = np.linspace(0, 1, 100)\n    >>> y_plot = sol.sol(x_plot)[0]\n    >>> plt.plot(x_plot, y_plot)\n    >>> plt.xlabel("x")\n    >>> plt.ylabel("y")\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Name (line 997):
    
    # Assigning a Call to a Name (line 997):
    
    # Call to asarray(...): (line 997)
    # Processing the call arguments (line 997)
    # Getting the type of 'x' (line 997)
    x_34761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 19), 'x', False)
    # Processing the call keyword arguments (line 997)
    # Getting the type of 'float' (line 997)
    float_34762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 28), 'float', False)
    keyword_34763 = float_34762
    kwargs_34764 = {'dtype': keyword_34763}
    # Getting the type of 'np' (line 997)
    np_34759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 997)
    asarray_34760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 997, 8), np_34759, 'asarray')
    # Calling asarray(args, kwargs) (line 997)
    asarray_call_result_34765 = invoke(stypy.reporting.localization.Localization(__file__, 997, 8), asarray_34760, *[x_34761], **kwargs_34764)
    
    # Assigning a type to the variable 'x' (line 997)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 997, 4), 'x', asarray_call_result_34765)
    
    
    # Getting the type of 'x' (line 998)
    x_34766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 7), 'x')
    # Obtaining the member 'ndim' of a type (line 998)
    ndim_34767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 998, 7), x_34766, 'ndim')
    int_34768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 998, 17), 'int')
    # Applying the binary operator '!=' (line 998)
    result_ne_34769 = python_operator(stypy.reporting.localization.Localization(__file__, 998, 7), '!=', ndim_34767, int_34768)
    
    # Testing the type of an if condition (line 998)
    if_condition_34770 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 998, 4), result_ne_34769)
    # Assigning a type to the variable 'if_condition_34770' (line 998)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 998, 4), 'if_condition_34770', if_condition_34770)
    # SSA begins for if statement (line 998)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 999)
    # Processing the call arguments (line 999)
    str_34772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 999, 25), 'str', '`x` must be 1 dimensional.')
    # Processing the call keyword arguments (line 999)
    kwargs_34773 = {}
    # Getting the type of 'ValueError' (line 999)
    ValueError_34771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 999)
    ValueError_call_result_34774 = invoke(stypy.reporting.localization.Localization(__file__, 999, 14), ValueError_34771, *[str_34772], **kwargs_34773)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 999, 8), ValueError_call_result_34774, 'raise parameter', BaseException)
    # SSA join for if statement (line 998)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1000):
    
    # Assigning a Call to a Name (line 1000):
    
    # Call to diff(...): (line 1000)
    # Processing the call arguments (line 1000)
    # Getting the type of 'x' (line 1000)
    x_34777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 16), 'x', False)
    # Processing the call keyword arguments (line 1000)
    kwargs_34778 = {}
    # Getting the type of 'np' (line 1000)
    np_34775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 8), 'np', False)
    # Obtaining the member 'diff' of a type (line 1000)
    diff_34776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1000, 8), np_34775, 'diff')
    # Calling diff(args, kwargs) (line 1000)
    diff_call_result_34779 = invoke(stypy.reporting.localization.Localization(__file__, 1000, 8), diff_34776, *[x_34777], **kwargs_34778)
    
    # Assigning a type to the variable 'h' (line 1000)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1000, 4), 'h', diff_call_result_34779)
    
    
    # Call to any(...): (line 1001)
    # Processing the call arguments (line 1001)
    
    # Getting the type of 'h' (line 1001)
    h_34782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 14), 'h', False)
    int_34783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1001, 19), 'int')
    # Applying the binary operator '<=' (line 1001)
    result_le_34784 = python_operator(stypy.reporting.localization.Localization(__file__, 1001, 14), '<=', h_34782, int_34783)
    
    # Processing the call keyword arguments (line 1001)
    kwargs_34785 = {}
    # Getting the type of 'np' (line 1001)
    np_34780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 1001)
    any_34781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 7), np_34780, 'any')
    # Calling any(args, kwargs) (line 1001)
    any_call_result_34786 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 7), any_34781, *[result_le_34784], **kwargs_34785)
    
    # Testing the type of an if condition (line 1001)
    if_condition_34787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1001, 4), any_call_result_34786)
    # Assigning a type to the variable 'if_condition_34787' (line 1001)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1001, 4), 'if_condition_34787', if_condition_34787)
    # SSA begins for if statement (line 1001)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1002)
    # Processing the call arguments (line 1002)
    str_34789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1002, 25), 'str', '`x` must be strictly increasing.')
    # Processing the call keyword arguments (line 1002)
    kwargs_34790 = {}
    # Getting the type of 'ValueError' (line 1002)
    ValueError_34788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1002)
    ValueError_call_result_34791 = invoke(stypy.reporting.localization.Localization(__file__, 1002, 14), ValueError_34788, *[str_34789], **kwargs_34790)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1002, 8), ValueError_call_result_34791, 'raise parameter', BaseException)
    # SSA join for if statement (line 1001)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 1003):
    
    # Assigning a Subscript to a Name (line 1003):
    
    # Obtaining the type of the subscript
    int_34792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1003, 10), 'int')
    # Getting the type of 'x' (line 1003)
    x_34793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 8), 'x')
    # Obtaining the member '__getitem__' of a type (line 1003)
    getitem___34794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1003, 8), x_34793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1003)
    subscript_call_result_34795 = invoke(stypy.reporting.localization.Localization(__file__, 1003, 8), getitem___34794, int_34792)
    
    # Assigning a type to the variable 'a' (line 1003)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1003, 4), 'a', subscript_call_result_34795)
    
    # Assigning a Call to a Name (line 1005):
    
    # Assigning a Call to a Name (line 1005):
    
    # Call to asarray(...): (line 1005)
    # Processing the call arguments (line 1005)
    # Getting the type of 'y' (line 1005)
    y_34798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 19), 'y', False)
    # Processing the call keyword arguments (line 1005)
    kwargs_34799 = {}
    # Getting the type of 'np' (line 1005)
    np_34796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1005)
    asarray_34797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1005, 8), np_34796, 'asarray')
    # Calling asarray(args, kwargs) (line 1005)
    asarray_call_result_34800 = invoke(stypy.reporting.localization.Localization(__file__, 1005, 8), asarray_34797, *[y_34798], **kwargs_34799)
    
    # Assigning a type to the variable 'y' (line 1005)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1005, 4), 'y', asarray_call_result_34800)
    
    
    # Call to issubdtype(...): (line 1006)
    # Processing the call arguments (line 1006)
    # Getting the type of 'y' (line 1006)
    y_34803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 21), 'y', False)
    # Obtaining the member 'dtype' of a type (line 1006)
    dtype_34804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1006, 21), y_34803, 'dtype')
    # Getting the type of 'np' (line 1006)
    np_34805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 30), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 1006)
    complexfloating_34806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1006, 30), np_34805, 'complexfloating')
    # Processing the call keyword arguments (line 1006)
    kwargs_34807 = {}
    # Getting the type of 'np' (line 1006)
    np_34801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 7), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 1006)
    issubdtype_34802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1006, 7), np_34801, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 1006)
    issubdtype_call_result_34808 = invoke(stypy.reporting.localization.Localization(__file__, 1006, 7), issubdtype_34802, *[dtype_34804, complexfloating_34806], **kwargs_34807)
    
    # Testing the type of an if condition (line 1006)
    if_condition_34809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1006, 4), issubdtype_call_result_34808)
    # Assigning a type to the variable 'if_condition_34809' (line 1006)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1006, 4), 'if_condition_34809', if_condition_34809)
    # SSA begins for if statement (line 1006)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 1007):
    
    # Assigning a Name to a Name (line 1007):
    # Getting the type of 'complex' (line 1007)
    complex_34810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 16), 'complex')
    # Assigning a type to the variable 'dtype' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 8), 'dtype', complex_34810)
    # SSA branch for the else part of an if statement (line 1006)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 1009):
    
    # Assigning a Name to a Name (line 1009):
    # Getting the type of 'float' (line 1009)
    float_34811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 16), 'float')
    # Assigning a type to the variable 'dtype' (line 1009)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1009, 8), 'dtype', float_34811)
    # SSA join for if statement (line 1006)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1010):
    
    # Assigning a Call to a Name (line 1010):
    
    # Call to astype(...): (line 1010)
    # Processing the call arguments (line 1010)
    # Getting the type of 'dtype' (line 1010)
    dtype_34814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 17), 'dtype', False)
    # Processing the call keyword arguments (line 1010)
    # Getting the type of 'False' (line 1010)
    False_34815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 29), 'False', False)
    keyword_34816 = False_34815
    kwargs_34817 = {'copy': keyword_34816}
    # Getting the type of 'y' (line 1010)
    y_34812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 8), 'y', False)
    # Obtaining the member 'astype' of a type (line 1010)
    astype_34813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1010, 8), y_34812, 'astype')
    # Calling astype(args, kwargs) (line 1010)
    astype_call_result_34818 = invoke(stypy.reporting.localization.Localization(__file__, 1010, 8), astype_34813, *[dtype_34814], **kwargs_34817)
    
    # Assigning a type to the variable 'y' (line 1010)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1010, 4), 'y', astype_call_result_34818)
    
    
    # Getting the type of 'y' (line 1012)
    y_34819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 7), 'y')
    # Obtaining the member 'ndim' of a type (line 1012)
    ndim_34820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1012, 7), y_34819, 'ndim')
    int_34821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1012, 17), 'int')
    # Applying the binary operator '!=' (line 1012)
    result_ne_34822 = python_operator(stypy.reporting.localization.Localization(__file__, 1012, 7), '!=', ndim_34820, int_34821)
    
    # Testing the type of an if condition (line 1012)
    if_condition_34823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1012, 4), result_ne_34822)
    # Assigning a type to the variable 'if_condition_34823' (line 1012)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1012, 4), 'if_condition_34823', if_condition_34823)
    # SSA begins for if statement (line 1012)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1013)
    # Processing the call arguments (line 1013)
    str_34825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1013, 25), 'str', '`y` must be 2 dimensional.')
    # Processing the call keyword arguments (line 1013)
    kwargs_34826 = {}
    # Getting the type of 'ValueError' (line 1013)
    ValueError_34824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1013)
    ValueError_call_result_34827 = invoke(stypy.reporting.localization.Localization(__file__, 1013, 14), ValueError_34824, *[str_34825], **kwargs_34826)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1013, 8), ValueError_call_result_34827, 'raise parameter', BaseException)
    # SSA join for if statement (line 1012)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_34828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1014, 15), 'int')
    # Getting the type of 'y' (line 1014)
    y_34829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 7), 'y')
    # Obtaining the member 'shape' of a type (line 1014)
    shape_34830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1014, 7), y_34829, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1014)
    getitem___34831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1014, 7), shape_34830, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1014)
    subscript_call_result_34832 = invoke(stypy.reporting.localization.Localization(__file__, 1014, 7), getitem___34831, int_34828)
    
    
    # Obtaining the type of the subscript
    int_34833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1014, 29), 'int')
    # Getting the type of 'x' (line 1014)
    x_34834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 21), 'x')
    # Obtaining the member 'shape' of a type (line 1014)
    shape_34835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1014, 21), x_34834, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1014)
    getitem___34836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1014, 21), shape_34835, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1014)
    subscript_call_result_34837 = invoke(stypy.reporting.localization.Localization(__file__, 1014, 21), getitem___34836, int_34833)
    
    # Applying the binary operator '!=' (line 1014)
    result_ne_34838 = python_operator(stypy.reporting.localization.Localization(__file__, 1014, 7), '!=', subscript_call_result_34832, subscript_call_result_34837)
    
    # Testing the type of an if condition (line 1014)
    if_condition_34839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1014, 4), result_ne_34838)
    # Assigning a type to the variable 'if_condition_34839' (line 1014)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1014, 4), 'if_condition_34839', if_condition_34839)
    # SSA begins for if statement (line 1014)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1015)
    # Processing the call arguments (line 1015)
    
    # Call to format(...): (line 1015)
    # Processing the call arguments (line 1015)
    
    # Obtaining the type of the subscript
    int_34843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1016, 50), 'int')
    # Getting the type of 'x' (line 1016)
    x_34844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 42), 'x', False)
    # Obtaining the member 'shape' of a type (line 1016)
    shape_34845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1016, 42), x_34844, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1016)
    getitem___34846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1016, 42), shape_34845, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1016)
    subscript_call_result_34847 = invoke(stypy.reporting.localization.Localization(__file__, 1016, 42), getitem___34846, int_34843)
    
    
    # Obtaining the type of the subscript
    int_34848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1016, 62), 'int')
    # Getting the type of 'y' (line 1016)
    y_34849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 54), 'y', False)
    # Obtaining the member 'shape' of a type (line 1016)
    shape_34850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1016, 54), y_34849, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1016)
    getitem___34851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1016, 54), shape_34850, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1016)
    subscript_call_result_34852 = invoke(stypy.reporting.localization.Localization(__file__, 1016, 54), getitem___34851, int_34848)
    
    # Processing the call keyword arguments (line 1015)
    kwargs_34853 = {}
    str_34841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1015, 25), 'str', '`y` is expected to have {} columns, but actually has {}.')
    # Obtaining the member 'format' of a type (line 1015)
    format_34842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1015, 25), str_34841, 'format')
    # Calling format(args, kwargs) (line 1015)
    format_call_result_34854 = invoke(stypy.reporting.localization.Localization(__file__, 1015, 25), format_34842, *[subscript_call_result_34847, subscript_call_result_34852], **kwargs_34853)
    
    # Processing the call keyword arguments (line 1015)
    kwargs_34855 = {}
    # Getting the type of 'ValueError' (line 1015)
    ValueError_34840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1015)
    ValueError_call_result_34856 = invoke(stypy.reporting.localization.Localization(__file__, 1015, 14), ValueError_34840, *[format_call_result_34854], **kwargs_34855)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1015, 8), ValueError_call_result_34856, 'raise parameter', BaseException)
    # SSA join for if statement (line 1014)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 1018)
    # Getting the type of 'p' (line 1018)
    p_34857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1018, 7), 'p')
    # Getting the type of 'None' (line 1018)
    None_34858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1018, 12), 'None')
    
    (may_be_34859, more_types_in_union_34860) = may_be_none(p_34857, None_34858)

    if may_be_34859:

        if more_types_in_union_34860:
            # Runtime conditional SSA (line 1018)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 1019):
        
        # Assigning a Call to a Name (line 1019):
        
        # Call to array(...): (line 1019)
        # Processing the call arguments (line 1019)
        
        # Obtaining an instance of the builtin type 'list' (line 1019)
        list_34863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1019, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1019)
        
        # Processing the call keyword arguments (line 1019)
        kwargs_34864 = {}
        # Getting the type of 'np' (line 1019)
        np_34861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 1019)
        array_34862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1019, 12), np_34861, 'array')
        # Calling array(args, kwargs) (line 1019)
        array_call_result_34865 = invoke(stypy.reporting.localization.Localization(__file__, 1019, 12), array_34862, *[list_34863], **kwargs_34864)
        
        # Assigning a type to the variable 'p' (line 1019)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1019, 8), 'p', array_call_result_34865)

        if more_types_in_union_34860:
            # Runtime conditional SSA for else branch (line 1018)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_34859) or more_types_in_union_34860):
        
        # Assigning a Call to a Name (line 1021):
        
        # Assigning a Call to a Name (line 1021):
        
        # Call to asarray(...): (line 1021)
        # Processing the call arguments (line 1021)
        # Getting the type of 'p' (line 1021)
        p_34868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1021, 23), 'p', False)
        # Processing the call keyword arguments (line 1021)
        # Getting the type of 'dtype' (line 1021)
        dtype_34869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1021, 32), 'dtype', False)
        keyword_34870 = dtype_34869
        kwargs_34871 = {'dtype': keyword_34870}
        # Getting the type of 'np' (line 1021)
        np_34866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1021, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 1021)
        asarray_34867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1021, 12), np_34866, 'asarray')
        # Calling asarray(args, kwargs) (line 1021)
        asarray_call_result_34872 = invoke(stypy.reporting.localization.Localization(__file__, 1021, 12), asarray_34867, *[p_34868], **kwargs_34871)
        
        # Assigning a type to the variable 'p' (line 1021)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1021, 8), 'p', asarray_call_result_34872)

        if (may_be_34859 and more_types_in_union_34860):
            # SSA join for if statement (line 1018)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'p' (line 1022)
    p_34873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1022, 7), 'p')
    # Obtaining the member 'ndim' of a type (line 1022)
    ndim_34874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1022, 7), p_34873, 'ndim')
    int_34875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1022, 17), 'int')
    # Applying the binary operator '!=' (line 1022)
    result_ne_34876 = python_operator(stypy.reporting.localization.Localization(__file__, 1022, 7), '!=', ndim_34874, int_34875)
    
    # Testing the type of an if condition (line 1022)
    if_condition_34877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1022, 4), result_ne_34876)
    # Assigning a type to the variable 'if_condition_34877' (line 1022)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1022, 4), 'if_condition_34877', if_condition_34877)
    # SSA begins for if statement (line 1022)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1023)
    # Processing the call arguments (line 1023)
    str_34879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 25), 'str', '`p` must be 1 dimensional.')
    # Processing the call keyword arguments (line 1023)
    kwargs_34880 = {}
    # Getting the type of 'ValueError' (line 1023)
    ValueError_34878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1023, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1023)
    ValueError_call_result_34881 = invoke(stypy.reporting.localization.Localization(__file__, 1023, 14), ValueError_34878, *[str_34879], **kwargs_34880)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1023, 8), ValueError_call_result_34881, 'raise parameter', BaseException)
    # SSA join for if statement (line 1022)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'tol' (line 1025)
    tol_34882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 7), 'tol')
    int_34883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1025, 13), 'int')
    # Getting the type of 'EPS' (line 1025)
    EPS_34884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 19), 'EPS')
    # Applying the binary operator '*' (line 1025)
    result_mul_34885 = python_operator(stypy.reporting.localization.Localization(__file__, 1025, 13), '*', int_34883, EPS_34884)
    
    # Applying the binary operator '<' (line 1025)
    result_lt_34886 = python_operator(stypy.reporting.localization.Localization(__file__, 1025, 7), '<', tol_34882, result_mul_34885)
    
    # Testing the type of an if condition (line 1025)
    if_condition_34887 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1025, 4), result_lt_34886)
    # Assigning a type to the variable 'if_condition_34887' (line 1025)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1025, 4), 'if_condition_34887', if_condition_34887)
    # SSA begins for if statement (line 1025)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 1026)
    # Processing the call arguments (line 1026)
    
    # Call to format(...): (line 1026)
    # Processing the call arguments (line 1026)
    int_34891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1026, 58), 'int')
    # Getting the type of 'EPS' (line 1026)
    EPS_34892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 64), 'EPS', False)
    # Applying the binary operator '*' (line 1026)
    result_mul_34893 = python_operator(stypy.reporting.localization.Localization(__file__, 1026, 58), '*', int_34891, EPS_34892)
    
    # Processing the call keyword arguments (line 1026)
    kwargs_34894 = {}
    str_34889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1026, 13), 'str', '`tol` is too low, setting to {:.2e}')
    # Obtaining the member 'format' of a type (line 1026)
    format_34890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1026, 13), str_34889, 'format')
    # Calling format(args, kwargs) (line 1026)
    format_call_result_34895 = invoke(stypy.reporting.localization.Localization(__file__, 1026, 13), format_34890, *[result_mul_34893], **kwargs_34894)
    
    # Processing the call keyword arguments (line 1026)
    kwargs_34896 = {}
    # Getting the type of 'warn' (line 1026)
    warn_34888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 1026)
    warn_call_result_34897 = invoke(stypy.reporting.localization.Localization(__file__, 1026, 8), warn_34888, *[format_call_result_34895], **kwargs_34896)
    
    
    # Assigning a BinOp to a Name (line 1027):
    
    # Assigning a BinOp to a Name (line 1027):
    int_34898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1027, 14), 'int')
    # Getting the type of 'EPS' (line 1027)
    EPS_34899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 20), 'EPS')
    # Applying the binary operator '*' (line 1027)
    result_mul_34900 = python_operator(stypy.reporting.localization.Localization(__file__, 1027, 14), '*', int_34898, EPS_34899)
    
    # Assigning a type to the variable 'tol' (line 1027)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1027, 8), 'tol', result_mul_34900)
    # SSA join for if statement (line 1025)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbose' (line 1029)
    verbose_34901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 7), 'verbose')
    
    # Obtaining an instance of the builtin type 'list' (line 1029)
    list_34902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1029)
    # Adding element type (line 1029)
    int_34903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1029, 22), list_34902, int_34903)
    # Adding element type (line 1029)
    int_34904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1029, 22), list_34902, int_34904)
    # Adding element type (line 1029)
    int_34905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1029, 22), list_34902, int_34905)
    
    # Applying the binary operator 'notin' (line 1029)
    result_contains_34906 = python_operator(stypy.reporting.localization.Localization(__file__, 1029, 7), 'notin', verbose_34901, list_34902)
    
    # Testing the type of an if condition (line 1029)
    if_condition_34907 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1029, 4), result_contains_34906)
    # Assigning a type to the variable 'if_condition_34907' (line 1029)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1029, 4), 'if_condition_34907', if_condition_34907)
    # SSA begins for if statement (line 1029)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1030)
    # Processing the call arguments (line 1030)
    str_34909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 25), 'str', '`verbose` must be in [0, 1, 2].')
    # Processing the call keyword arguments (line 1030)
    kwargs_34910 = {}
    # Getting the type of 'ValueError' (line 1030)
    ValueError_34908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1030)
    ValueError_call_result_34911 = invoke(stypy.reporting.localization.Localization(__file__, 1030, 14), ValueError_34908, *[str_34909], **kwargs_34910)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1030, 8), ValueError_call_result_34911, 'raise parameter', BaseException)
    # SSA join for if statement (line 1029)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 1032):
    
    # Assigning a Subscript to a Name (line 1032):
    
    # Obtaining the type of the subscript
    int_34912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 16), 'int')
    # Getting the type of 'y' (line 1032)
    y_34913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 8), 'y')
    # Obtaining the member 'shape' of a type (line 1032)
    shape_34914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1032, 8), y_34913, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1032)
    getitem___34915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1032, 8), shape_34914, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1032)
    subscript_call_result_34916 = invoke(stypy.reporting.localization.Localization(__file__, 1032, 8), getitem___34915, int_34912)
    
    # Assigning a type to the variable 'n' (line 1032)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1032, 4), 'n', subscript_call_result_34916)
    
    # Assigning a Subscript to a Name (line 1033):
    
    # Assigning a Subscript to a Name (line 1033):
    
    # Obtaining the type of the subscript
    int_34917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 16), 'int')
    # Getting the type of 'p' (line 1033)
    p_34918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'p')
    # Obtaining the member 'shape' of a type (line 1033)
    shape_34919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 8), p_34918, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1033)
    getitem___34920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 8), shape_34919, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1033)
    subscript_call_result_34921 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 8), getitem___34920, int_34917)
    
    # Assigning a type to the variable 'k' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 4), 'k', subscript_call_result_34921)
    
    # Type idiom detected: calculating its left and rigth part (line 1035)
    # Getting the type of 'S' (line 1035)
    S_34922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 4), 'S')
    # Getting the type of 'None' (line 1035)
    None_34923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 16), 'None')
    
    (may_be_34924, more_types_in_union_34925) = may_not_be_none(S_34922, None_34923)

    if may_be_34924:

        if more_types_in_union_34925:
            # Runtime conditional SSA (line 1035)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 1036):
        
        # Assigning a Call to a Name (line 1036):
        
        # Call to asarray(...): (line 1036)
        # Processing the call arguments (line 1036)
        # Getting the type of 'S' (line 1036)
        S_34928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 23), 'S', False)
        # Processing the call keyword arguments (line 1036)
        # Getting the type of 'dtype' (line 1036)
        dtype_34929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 32), 'dtype', False)
        keyword_34930 = dtype_34929
        kwargs_34931 = {'dtype': keyword_34930}
        # Getting the type of 'np' (line 1036)
        np_34926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 1036)
        asarray_34927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1036, 12), np_34926, 'asarray')
        # Calling asarray(args, kwargs) (line 1036)
        asarray_call_result_34932 = invoke(stypy.reporting.localization.Localization(__file__, 1036, 12), asarray_34927, *[S_34928], **kwargs_34931)
        
        # Assigning a type to the variable 'S' (line 1036)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1036, 8), 'S', asarray_call_result_34932)
        
        
        # Getting the type of 'S' (line 1037)
        S_34933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 11), 'S')
        # Obtaining the member 'shape' of a type (line 1037)
        shape_34934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 11), S_34933, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1037)
        tuple_34935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1037)
        # Adding element type (line 1037)
        # Getting the type of 'n' (line 1037)
        n_34936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 23), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1037, 23), tuple_34935, n_34936)
        # Adding element type (line 1037)
        # Getting the type of 'n' (line 1037)
        n_34937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 26), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1037, 23), tuple_34935, n_34937)
        
        # Applying the binary operator '!=' (line 1037)
        result_ne_34938 = python_operator(stypy.reporting.localization.Localization(__file__, 1037, 11), '!=', shape_34934, tuple_34935)
        
        # Testing the type of an if condition (line 1037)
        if_condition_34939 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1037, 8), result_ne_34938)
        # Assigning a type to the variable 'if_condition_34939' (line 1037)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 8), 'if_condition_34939', if_condition_34939)
        # SSA begins for if statement (line 1037)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 1038)
        # Processing the call arguments (line 1038)
        
        # Call to format(...): (line 1038)
        # Processing the call arguments (line 1038)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1039)
        tuple_34943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1039)
        # Adding element type (line 1039)
        # Getting the type of 'n' (line 1039)
        n_34944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 59), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1039, 59), tuple_34943, n_34944)
        # Adding element type (line 1039)
        # Getting the type of 'n' (line 1039)
        n_34945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 62), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1039, 59), tuple_34943, n_34945)
        
        # Getting the type of 'S' (line 1039)
        S_34946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 66), 'S', False)
        # Obtaining the member 'shape' of a type (line 1039)
        shape_34947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1039, 66), S_34946, 'shape')
        # Processing the call keyword arguments (line 1038)
        kwargs_34948 = {}
        str_34941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 29), 'str', '`S` is expected to have shape {}, but actually has {}')
        # Obtaining the member 'format' of a type (line 1038)
        format_34942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1038, 29), str_34941, 'format')
        # Calling format(args, kwargs) (line 1038)
        format_call_result_34949 = invoke(stypy.reporting.localization.Localization(__file__, 1038, 29), format_34942, *[tuple_34943, shape_34947], **kwargs_34948)
        
        # Processing the call keyword arguments (line 1038)
        kwargs_34950 = {}
        # Getting the type of 'ValueError' (line 1038)
        ValueError_34940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1038)
        ValueError_call_result_34951 = invoke(stypy.reporting.localization.Localization(__file__, 1038, 18), ValueError_34940, *[format_call_result_34949], **kwargs_34950)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1038, 12), ValueError_call_result_34951, 'raise parameter', BaseException)
        # SSA join for if statement (line 1037)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1042):
        
        # Assigning a BinOp to a Name (line 1042):
        
        # Call to identity(...): (line 1042)
        # Processing the call arguments (line 1042)
        # Getting the type of 'n' (line 1042)
        n_34954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 24), 'n', False)
        # Processing the call keyword arguments (line 1042)
        kwargs_34955 = {}
        # Getting the type of 'np' (line 1042)
        np_34952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 12), 'np', False)
        # Obtaining the member 'identity' of a type (line 1042)
        identity_34953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1042, 12), np_34952, 'identity')
        # Calling identity(args, kwargs) (line 1042)
        identity_call_result_34956 = invoke(stypy.reporting.localization.Localization(__file__, 1042, 12), identity_34953, *[n_34954], **kwargs_34955)
        
        
        # Call to dot(...): (line 1042)
        # Processing the call arguments (line 1042)
        
        # Call to pinv(...): (line 1042)
        # Processing the call arguments (line 1042)
        # Getting the type of 'S' (line 1042)
        S_34960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 41), 'S', False)
        # Processing the call keyword arguments (line 1042)
        kwargs_34961 = {}
        # Getting the type of 'pinv' (line 1042)
        pinv_34959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 36), 'pinv', False)
        # Calling pinv(args, kwargs) (line 1042)
        pinv_call_result_34962 = invoke(stypy.reporting.localization.Localization(__file__, 1042, 36), pinv_34959, *[S_34960], **kwargs_34961)
        
        # Getting the type of 'S' (line 1042)
        S_34963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 45), 'S', False)
        # Processing the call keyword arguments (line 1042)
        kwargs_34964 = {}
        # Getting the type of 'np' (line 1042)
        np_34957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 29), 'np', False)
        # Obtaining the member 'dot' of a type (line 1042)
        dot_34958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1042, 29), np_34957, 'dot')
        # Calling dot(args, kwargs) (line 1042)
        dot_call_result_34965 = invoke(stypy.reporting.localization.Localization(__file__, 1042, 29), dot_34958, *[pinv_call_result_34962, S_34963], **kwargs_34964)
        
        # Applying the binary operator '-' (line 1042)
        result_sub_34966 = python_operator(stypy.reporting.localization.Localization(__file__, 1042, 12), '-', identity_call_result_34956, dot_call_result_34965)
        
        # Assigning a type to the variable 'B' (line 1042)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1042, 8), 'B', result_sub_34966)
        
        # Assigning a Call to a Subscript (line 1044):
        
        # Assigning a Call to a Subscript (line 1044):
        
        # Call to dot(...): (line 1044)
        # Processing the call arguments (line 1044)
        # Getting the type of 'B' (line 1044)
        B_34969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 25), 'B', False)
        
        # Obtaining the type of the subscript
        slice_34970 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1044, 28), None, None, None)
        int_34971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 33), 'int')
        # Getting the type of 'y' (line 1044)
        y_34972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 28), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 1044)
        getitem___34973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 28), y_34972, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1044)
        subscript_call_result_34974 = invoke(stypy.reporting.localization.Localization(__file__, 1044, 28), getitem___34973, (slice_34970, int_34971))
        
        # Processing the call keyword arguments (line 1044)
        kwargs_34975 = {}
        # Getting the type of 'np' (line 1044)
        np_34967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 18), 'np', False)
        # Obtaining the member 'dot' of a type (line 1044)
        dot_34968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 18), np_34967, 'dot')
        # Calling dot(args, kwargs) (line 1044)
        dot_call_result_34976 = invoke(stypy.reporting.localization.Localization(__file__, 1044, 18), dot_34968, *[B_34969, subscript_call_result_34974], **kwargs_34975)
        
        # Getting the type of 'y' (line 1044)
        y_34977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 8), 'y')
        slice_34978 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1044, 8), None, None, None)
        int_34979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 13), 'int')
        # Storing an element on a container (line 1044)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1044, 8), y_34977, ((slice_34978, int_34979), dot_call_result_34976))
        
        # Assigning a Call to a Name (line 1047):
        
        # Assigning a Call to a Name (line 1047):
        
        # Call to pinv(...): (line 1047)
        # Processing the call arguments (line 1047)
        
        # Call to identity(...): (line 1047)
        # Processing the call arguments (line 1047)
        # Getting the type of 'n' (line 1047)
        n_34983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 29), 'n', False)
        # Processing the call keyword arguments (line 1047)
        kwargs_34984 = {}
        # Getting the type of 'np' (line 1047)
        np_34981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 17), 'np', False)
        # Obtaining the member 'identity' of a type (line 1047)
        identity_34982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1047, 17), np_34981, 'identity')
        # Calling identity(args, kwargs) (line 1047)
        identity_call_result_34985 = invoke(stypy.reporting.localization.Localization(__file__, 1047, 17), identity_34982, *[n_34983], **kwargs_34984)
        
        # Getting the type of 'S' (line 1047)
        S_34986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 34), 'S', False)
        # Applying the binary operator '-' (line 1047)
        result_sub_34987 = python_operator(stypy.reporting.localization.Localization(__file__, 1047, 17), '-', identity_call_result_34985, S_34986)
        
        # Processing the call keyword arguments (line 1047)
        kwargs_34988 = {}
        # Getting the type of 'pinv' (line 1047)
        pinv_34980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 12), 'pinv', False)
        # Calling pinv(args, kwargs) (line 1047)
        pinv_call_result_34989 = invoke(stypy.reporting.localization.Localization(__file__, 1047, 12), pinv_34980, *[result_sub_34987], **kwargs_34988)
        
        # Assigning a type to the variable 'D' (line 1047)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1047, 8), 'D', pinv_call_result_34989)

        if more_types_in_union_34925:
            # Runtime conditional SSA for else branch (line 1035)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_34924) or more_types_in_union_34925):
        
        # Assigning a Name to a Name (line 1049):
        
        # Assigning a Name to a Name (line 1049):
        # Getting the type of 'None' (line 1049)
        None_34990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 12), 'None')
        # Assigning a type to the variable 'B' (line 1049)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 8), 'B', None_34990)
        
        # Assigning a Name to a Name (line 1050):
        
        # Assigning a Name to a Name (line 1050):
        # Getting the type of 'None' (line 1050)
        None_34991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 12), 'None')
        # Assigning a type to the variable 'D' (line 1050)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1050, 8), 'D', None_34991)

        if (may_be_34924 and more_types_in_union_34925):
            # SSA join for if statement (line 1035)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 1052):
    
    # Assigning a Subscript to a Name (line 1052):
    
    # Obtaining the type of the subscript
    int_34992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 4), 'int')
    
    # Call to wrap_functions(...): (line 1052)
    # Processing the call arguments (line 1052)
    # Getting the type of 'fun' (line 1053)
    fun_34994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 8), 'fun', False)
    # Getting the type of 'bc' (line 1053)
    bc_34995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 13), 'bc', False)
    # Getting the type of 'fun_jac' (line 1053)
    fun_jac_34996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 17), 'fun_jac', False)
    # Getting the type of 'bc_jac' (line 1053)
    bc_jac_34997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 26), 'bc_jac', False)
    # Getting the type of 'k' (line 1053)
    k_34998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 34), 'k', False)
    # Getting the type of 'a' (line 1053)
    a_34999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 37), 'a', False)
    # Getting the type of 'S' (line 1053)
    S_35000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 40), 'S', False)
    # Getting the type of 'D' (line 1053)
    D_35001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 43), 'D', False)
    # Getting the type of 'dtype' (line 1053)
    dtype_35002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 46), 'dtype', False)
    # Processing the call keyword arguments (line 1052)
    kwargs_35003 = {}
    # Getting the type of 'wrap_functions' (line 1052)
    wrap_functions_34993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 63), 'wrap_functions', False)
    # Calling wrap_functions(args, kwargs) (line 1052)
    wrap_functions_call_result_35004 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 63), wrap_functions_34993, *[fun_34994, bc_34995, fun_jac_34996, bc_jac_34997, k_34998, a_34999, S_35000, D_35001, dtype_35002], **kwargs_35003)
    
    # Obtaining the member '__getitem__' of a type (line 1052)
    getitem___35005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 4), wrap_functions_call_result_35004, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1052)
    subscript_call_result_35006 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 4), getitem___35005, int_34992)
    
    # Assigning a type to the variable 'tuple_var_assignment_32419' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'tuple_var_assignment_32419', subscript_call_result_35006)
    
    # Assigning a Subscript to a Name (line 1052):
    
    # Obtaining the type of the subscript
    int_35007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 4), 'int')
    
    # Call to wrap_functions(...): (line 1052)
    # Processing the call arguments (line 1052)
    # Getting the type of 'fun' (line 1053)
    fun_35009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 8), 'fun', False)
    # Getting the type of 'bc' (line 1053)
    bc_35010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 13), 'bc', False)
    # Getting the type of 'fun_jac' (line 1053)
    fun_jac_35011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 17), 'fun_jac', False)
    # Getting the type of 'bc_jac' (line 1053)
    bc_jac_35012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 26), 'bc_jac', False)
    # Getting the type of 'k' (line 1053)
    k_35013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 34), 'k', False)
    # Getting the type of 'a' (line 1053)
    a_35014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 37), 'a', False)
    # Getting the type of 'S' (line 1053)
    S_35015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 40), 'S', False)
    # Getting the type of 'D' (line 1053)
    D_35016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 43), 'D', False)
    # Getting the type of 'dtype' (line 1053)
    dtype_35017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 46), 'dtype', False)
    # Processing the call keyword arguments (line 1052)
    kwargs_35018 = {}
    # Getting the type of 'wrap_functions' (line 1052)
    wrap_functions_35008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 63), 'wrap_functions', False)
    # Calling wrap_functions(args, kwargs) (line 1052)
    wrap_functions_call_result_35019 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 63), wrap_functions_35008, *[fun_35009, bc_35010, fun_jac_35011, bc_jac_35012, k_35013, a_35014, S_35015, D_35016, dtype_35017], **kwargs_35018)
    
    # Obtaining the member '__getitem__' of a type (line 1052)
    getitem___35020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 4), wrap_functions_call_result_35019, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1052)
    subscript_call_result_35021 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 4), getitem___35020, int_35007)
    
    # Assigning a type to the variable 'tuple_var_assignment_32420' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'tuple_var_assignment_32420', subscript_call_result_35021)
    
    # Assigning a Subscript to a Name (line 1052):
    
    # Obtaining the type of the subscript
    int_35022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 4), 'int')
    
    # Call to wrap_functions(...): (line 1052)
    # Processing the call arguments (line 1052)
    # Getting the type of 'fun' (line 1053)
    fun_35024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 8), 'fun', False)
    # Getting the type of 'bc' (line 1053)
    bc_35025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 13), 'bc', False)
    # Getting the type of 'fun_jac' (line 1053)
    fun_jac_35026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 17), 'fun_jac', False)
    # Getting the type of 'bc_jac' (line 1053)
    bc_jac_35027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 26), 'bc_jac', False)
    # Getting the type of 'k' (line 1053)
    k_35028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 34), 'k', False)
    # Getting the type of 'a' (line 1053)
    a_35029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 37), 'a', False)
    # Getting the type of 'S' (line 1053)
    S_35030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 40), 'S', False)
    # Getting the type of 'D' (line 1053)
    D_35031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 43), 'D', False)
    # Getting the type of 'dtype' (line 1053)
    dtype_35032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 46), 'dtype', False)
    # Processing the call keyword arguments (line 1052)
    kwargs_35033 = {}
    # Getting the type of 'wrap_functions' (line 1052)
    wrap_functions_35023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 63), 'wrap_functions', False)
    # Calling wrap_functions(args, kwargs) (line 1052)
    wrap_functions_call_result_35034 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 63), wrap_functions_35023, *[fun_35024, bc_35025, fun_jac_35026, bc_jac_35027, k_35028, a_35029, S_35030, D_35031, dtype_35032], **kwargs_35033)
    
    # Obtaining the member '__getitem__' of a type (line 1052)
    getitem___35035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 4), wrap_functions_call_result_35034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1052)
    subscript_call_result_35036 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 4), getitem___35035, int_35022)
    
    # Assigning a type to the variable 'tuple_var_assignment_32421' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'tuple_var_assignment_32421', subscript_call_result_35036)
    
    # Assigning a Subscript to a Name (line 1052):
    
    # Obtaining the type of the subscript
    int_35037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 4), 'int')
    
    # Call to wrap_functions(...): (line 1052)
    # Processing the call arguments (line 1052)
    # Getting the type of 'fun' (line 1053)
    fun_35039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 8), 'fun', False)
    # Getting the type of 'bc' (line 1053)
    bc_35040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 13), 'bc', False)
    # Getting the type of 'fun_jac' (line 1053)
    fun_jac_35041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 17), 'fun_jac', False)
    # Getting the type of 'bc_jac' (line 1053)
    bc_jac_35042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 26), 'bc_jac', False)
    # Getting the type of 'k' (line 1053)
    k_35043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 34), 'k', False)
    # Getting the type of 'a' (line 1053)
    a_35044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 37), 'a', False)
    # Getting the type of 'S' (line 1053)
    S_35045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 40), 'S', False)
    # Getting the type of 'D' (line 1053)
    D_35046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 43), 'D', False)
    # Getting the type of 'dtype' (line 1053)
    dtype_35047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 46), 'dtype', False)
    # Processing the call keyword arguments (line 1052)
    kwargs_35048 = {}
    # Getting the type of 'wrap_functions' (line 1052)
    wrap_functions_35038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 63), 'wrap_functions', False)
    # Calling wrap_functions(args, kwargs) (line 1052)
    wrap_functions_call_result_35049 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 63), wrap_functions_35038, *[fun_35039, bc_35040, fun_jac_35041, bc_jac_35042, k_35043, a_35044, S_35045, D_35046, dtype_35047], **kwargs_35048)
    
    # Obtaining the member '__getitem__' of a type (line 1052)
    getitem___35050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 4), wrap_functions_call_result_35049, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1052)
    subscript_call_result_35051 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 4), getitem___35050, int_35037)
    
    # Assigning a type to the variable 'tuple_var_assignment_32422' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'tuple_var_assignment_32422', subscript_call_result_35051)
    
    # Assigning a Name to a Name (line 1052):
    # Getting the type of 'tuple_var_assignment_32419' (line 1052)
    tuple_var_assignment_32419_35052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'tuple_var_assignment_32419')
    # Assigning a type to the variable 'fun_wrapped' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'fun_wrapped', tuple_var_assignment_32419_35052)
    
    # Assigning a Name to a Name (line 1052):
    # Getting the type of 'tuple_var_assignment_32420' (line 1052)
    tuple_var_assignment_32420_35053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'tuple_var_assignment_32420')
    # Assigning a type to the variable 'bc_wrapped' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 17), 'bc_wrapped', tuple_var_assignment_32420_35053)
    
    # Assigning a Name to a Name (line 1052):
    # Getting the type of 'tuple_var_assignment_32421' (line 1052)
    tuple_var_assignment_32421_35054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'tuple_var_assignment_32421')
    # Assigning a type to the variable 'fun_jac_wrapped' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 29), 'fun_jac_wrapped', tuple_var_assignment_32421_35054)
    
    # Assigning a Name to a Name (line 1052):
    # Getting the type of 'tuple_var_assignment_32422' (line 1052)
    tuple_var_assignment_32422_35055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'tuple_var_assignment_32422')
    # Assigning a type to the variable 'bc_jac_wrapped' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 46), 'bc_jac_wrapped', tuple_var_assignment_32422_35055)
    
    # Assigning a Call to a Name (line 1055):
    
    # Assigning a Call to a Name (line 1055):
    
    # Call to fun_wrapped(...): (line 1055)
    # Processing the call arguments (line 1055)
    # Getting the type of 'x' (line 1055)
    x_35057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 20), 'x', False)
    # Getting the type of 'y' (line 1055)
    y_35058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 23), 'y', False)
    # Getting the type of 'p' (line 1055)
    p_35059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 26), 'p', False)
    # Processing the call keyword arguments (line 1055)
    kwargs_35060 = {}
    # Getting the type of 'fun_wrapped' (line 1055)
    fun_wrapped_35056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 8), 'fun_wrapped', False)
    # Calling fun_wrapped(args, kwargs) (line 1055)
    fun_wrapped_call_result_35061 = invoke(stypy.reporting.localization.Localization(__file__, 1055, 8), fun_wrapped_35056, *[x_35057, y_35058, p_35059], **kwargs_35060)
    
    # Assigning a type to the variable 'f' (line 1055)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1055, 4), 'f', fun_wrapped_call_result_35061)
    
    
    # Getting the type of 'f' (line 1056)
    f_35062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 7), 'f')
    # Obtaining the member 'shape' of a type (line 1056)
    shape_35063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 7), f_35062, 'shape')
    # Getting the type of 'y' (line 1056)
    y_35064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 18), 'y')
    # Obtaining the member 'shape' of a type (line 1056)
    shape_35065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 18), y_35064, 'shape')
    # Applying the binary operator '!=' (line 1056)
    result_ne_35066 = python_operator(stypy.reporting.localization.Localization(__file__, 1056, 7), '!=', shape_35063, shape_35065)
    
    # Testing the type of an if condition (line 1056)
    if_condition_35067 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1056, 4), result_ne_35066)
    # Assigning a type to the variable 'if_condition_35067' (line 1056)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 4), 'if_condition_35067', if_condition_35067)
    # SSA begins for if statement (line 1056)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1057)
    # Processing the call arguments (line 1057)
    
    # Call to format(...): (line 1057)
    # Processing the call arguments (line 1057)
    # Getting the type of 'y' (line 1058)
    y_35071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 55), 'y', False)
    # Obtaining the member 'shape' of a type (line 1058)
    shape_35072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1058, 55), y_35071, 'shape')
    # Getting the type of 'f' (line 1058)
    f_35073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 64), 'f', False)
    # Obtaining the member 'shape' of a type (line 1058)
    shape_35074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1058, 64), f_35073, 'shape')
    # Processing the call keyword arguments (line 1057)
    kwargs_35075 = {}
    str_35069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 25), 'str', '`fun` return is expected to have shape {}, but actually has {}.')
    # Obtaining the member 'format' of a type (line 1057)
    format_35070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1057, 25), str_35069, 'format')
    # Calling format(args, kwargs) (line 1057)
    format_call_result_35076 = invoke(stypy.reporting.localization.Localization(__file__, 1057, 25), format_35070, *[shape_35072, shape_35074], **kwargs_35075)
    
    # Processing the call keyword arguments (line 1057)
    kwargs_35077 = {}
    # Getting the type of 'ValueError' (line 1057)
    ValueError_35068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1057)
    ValueError_call_result_35078 = invoke(stypy.reporting.localization.Localization(__file__, 1057, 14), ValueError_35068, *[format_call_result_35076], **kwargs_35077)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1057, 8), ValueError_call_result_35078, 'raise parameter', BaseException)
    # SSA join for if statement (line 1056)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1060):
    
    # Assigning a Call to a Name (line 1060):
    
    # Call to bc_wrapped(...): (line 1060)
    # Processing the call arguments (line 1060)
    
    # Obtaining the type of the subscript
    slice_35080 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1060, 24), None, None, None)
    int_35081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1060, 29), 'int')
    # Getting the type of 'y' (line 1060)
    y_35082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 24), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 1060)
    getitem___35083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1060, 24), y_35082, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1060)
    subscript_call_result_35084 = invoke(stypy.reporting.localization.Localization(__file__, 1060, 24), getitem___35083, (slice_35080, int_35081))
    
    
    # Obtaining the type of the subscript
    slice_35085 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1060, 33), None, None, None)
    int_35086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1060, 38), 'int')
    # Getting the type of 'y' (line 1060)
    y_35087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 33), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 1060)
    getitem___35088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1060, 33), y_35087, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1060)
    subscript_call_result_35089 = invoke(stypy.reporting.localization.Localization(__file__, 1060, 33), getitem___35088, (slice_35085, int_35086))
    
    # Getting the type of 'p' (line 1060)
    p_35090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 43), 'p', False)
    # Processing the call keyword arguments (line 1060)
    kwargs_35091 = {}
    # Getting the type of 'bc_wrapped' (line 1060)
    bc_wrapped_35079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 13), 'bc_wrapped', False)
    # Calling bc_wrapped(args, kwargs) (line 1060)
    bc_wrapped_call_result_35092 = invoke(stypy.reporting.localization.Localization(__file__, 1060, 13), bc_wrapped_35079, *[subscript_call_result_35084, subscript_call_result_35089, p_35090], **kwargs_35091)
    
    # Assigning a type to the variable 'bc_res' (line 1060)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1060, 4), 'bc_res', bc_wrapped_call_result_35092)
    
    
    # Getting the type of 'bc_res' (line 1061)
    bc_res_35093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 7), 'bc_res')
    # Obtaining the member 'shape' of a type (line 1061)
    shape_35094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1061, 7), bc_res_35093, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1061)
    tuple_35095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1061)
    # Adding element type (line 1061)
    # Getting the type of 'n' (line 1061)
    n_35096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 24), 'n')
    # Getting the type of 'k' (line 1061)
    k_35097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 28), 'k')
    # Applying the binary operator '+' (line 1061)
    result_add_35098 = python_operator(stypy.reporting.localization.Localization(__file__, 1061, 24), '+', n_35096, k_35097)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1061, 24), tuple_35095, result_add_35098)
    
    # Applying the binary operator '!=' (line 1061)
    result_ne_35099 = python_operator(stypy.reporting.localization.Localization(__file__, 1061, 7), '!=', shape_35094, tuple_35095)
    
    # Testing the type of an if condition (line 1061)
    if_condition_35100 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1061, 4), result_ne_35099)
    # Assigning a type to the variable 'if_condition_35100' (line 1061)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1061, 4), 'if_condition_35100', if_condition_35100)
    # SSA begins for if statement (line 1061)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1062)
    # Processing the call arguments (line 1062)
    
    # Call to format(...): (line 1062)
    # Processing the call arguments (line 1062)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1063)
    tuple_35104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1063, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1063)
    # Adding element type (line 1063)
    # Getting the type of 'n' (line 1063)
    n_35105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 56), 'n', False)
    # Getting the type of 'k' (line 1063)
    k_35106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 60), 'k', False)
    # Applying the binary operator '+' (line 1063)
    result_add_35107 = python_operator(stypy.reporting.localization.Localization(__file__, 1063, 56), '+', n_35105, k_35106)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1063, 56), tuple_35104, result_add_35107)
    
    # Getting the type of 'bc_res' (line 1063)
    bc_res_35108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 65), 'bc_res', False)
    # Obtaining the member 'shape' of a type (line 1063)
    shape_35109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1063, 65), bc_res_35108, 'shape')
    # Processing the call keyword arguments (line 1062)
    kwargs_35110 = {}
    str_35102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1062, 25), 'str', '`bc` return is expected to have shape {}, but actually has {}.')
    # Obtaining the member 'format' of a type (line 1062)
    format_35103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1062, 25), str_35102, 'format')
    # Calling format(args, kwargs) (line 1062)
    format_call_result_35111 = invoke(stypy.reporting.localization.Localization(__file__, 1062, 25), format_35103, *[tuple_35104, shape_35109], **kwargs_35110)
    
    # Processing the call keyword arguments (line 1062)
    kwargs_35112 = {}
    # Getting the type of 'ValueError' (line 1062)
    ValueError_35101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1062)
    ValueError_call_result_35113 = invoke(stypy.reporting.localization.Localization(__file__, 1062, 14), ValueError_35101, *[format_call_result_35111], **kwargs_35112)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1062, 8), ValueError_call_result_35113, 'raise parameter', BaseException)
    # SSA join for if statement (line 1061)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 1065):
    
    # Assigning a Num to a Name (line 1065):
    int_35114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1065, 13), 'int')
    # Assigning a type to the variable 'status' (line 1065)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1065, 4), 'status', int_35114)
    
    # Assigning a Num to a Name (line 1066):
    
    # Assigning a Num to a Name (line 1066):
    int_35115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1066, 16), 'int')
    # Assigning a type to the variable 'iteration' (line 1066)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1066, 4), 'iteration', int_35115)
    
    
    # Getting the type of 'verbose' (line 1067)
    verbose_35116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 7), 'verbose')
    int_35117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1067, 18), 'int')
    # Applying the binary operator '==' (line 1067)
    result_eq_35118 = python_operator(stypy.reporting.localization.Localization(__file__, 1067, 7), '==', verbose_35116, int_35117)
    
    # Testing the type of an if condition (line 1067)
    if_condition_35119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1067, 4), result_eq_35118)
    # Assigning a type to the variable 'if_condition_35119' (line 1067)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1067, 4), 'if_condition_35119', if_condition_35119)
    # SSA begins for if statement (line 1067)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print_iteration_header(...): (line 1068)
    # Processing the call keyword arguments (line 1068)
    kwargs_35121 = {}
    # Getting the type of 'print_iteration_header' (line 1068)
    print_iteration_header_35120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1068, 8), 'print_iteration_header', False)
    # Calling print_iteration_header(args, kwargs) (line 1068)
    print_iteration_header_call_result_35122 = invoke(stypy.reporting.localization.Localization(__file__, 1068, 8), print_iteration_header_35120, *[], **kwargs_35121)
    
    # SSA join for if statement (line 1067)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'True' (line 1070)
    True_35123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 10), 'True')
    # Testing the type of an if condition (line 1070)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1070, 4), True_35123)
    # SSA begins for while statement (line 1070)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Subscript to a Name (line 1071):
    
    # Assigning a Subscript to a Name (line 1071):
    
    # Obtaining the type of the subscript
    int_35124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1071, 20), 'int')
    # Getting the type of 'x' (line 1071)
    x_35125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1071, 12), 'x')
    # Obtaining the member 'shape' of a type (line 1071)
    shape_35126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1071, 12), x_35125, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1071)
    getitem___35127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1071, 12), shape_35126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1071)
    subscript_call_result_35128 = invoke(stypy.reporting.localization.Localization(__file__, 1071, 12), getitem___35127, int_35124)
    
    # Assigning a type to the variable 'm' (line 1071)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1071, 8), 'm', subscript_call_result_35128)
    
    # Assigning a Call to a Tuple (line 1073):
    
    # Assigning a Subscript to a Name (line 1073):
    
    # Obtaining the type of the subscript
    int_35129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1073, 8), 'int')
    
    # Call to prepare_sys(...): (line 1073)
    # Processing the call arguments (line 1073)
    # Getting the type of 'n' (line 1073)
    n_35131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 39), 'n', False)
    # Getting the type of 'm' (line 1073)
    m_35132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 42), 'm', False)
    # Getting the type of 'k' (line 1073)
    k_35133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 45), 'k', False)
    # Getting the type of 'fun_wrapped' (line 1073)
    fun_wrapped_35134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 48), 'fun_wrapped', False)
    # Getting the type of 'bc_wrapped' (line 1073)
    bc_wrapped_35135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 61), 'bc_wrapped', False)
    # Getting the type of 'fun_jac_wrapped' (line 1074)
    fun_jac_wrapped_35136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 39), 'fun_jac_wrapped', False)
    # Getting the type of 'bc_jac_wrapped' (line 1074)
    bc_jac_wrapped_35137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 56), 'bc_jac_wrapped', False)
    # Getting the type of 'x' (line 1074)
    x_35138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 72), 'x', False)
    # Getting the type of 'h' (line 1074)
    h_35139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 75), 'h', False)
    # Processing the call keyword arguments (line 1073)
    kwargs_35140 = {}
    # Getting the type of 'prepare_sys' (line 1073)
    prepare_sys_35130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 27), 'prepare_sys', False)
    # Calling prepare_sys(args, kwargs) (line 1073)
    prepare_sys_call_result_35141 = invoke(stypy.reporting.localization.Localization(__file__, 1073, 27), prepare_sys_35130, *[n_35131, m_35132, k_35133, fun_wrapped_35134, bc_wrapped_35135, fun_jac_wrapped_35136, bc_jac_wrapped_35137, x_35138, h_35139], **kwargs_35140)
    
    # Obtaining the member '__getitem__' of a type (line 1073)
    getitem___35142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 8), prepare_sys_call_result_35141, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1073)
    subscript_call_result_35143 = invoke(stypy.reporting.localization.Localization(__file__, 1073, 8), getitem___35142, int_35129)
    
    # Assigning a type to the variable 'tuple_var_assignment_32423' (line 1073)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'tuple_var_assignment_32423', subscript_call_result_35143)
    
    # Assigning a Subscript to a Name (line 1073):
    
    # Obtaining the type of the subscript
    int_35144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1073, 8), 'int')
    
    # Call to prepare_sys(...): (line 1073)
    # Processing the call arguments (line 1073)
    # Getting the type of 'n' (line 1073)
    n_35146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 39), 'n', False)
    # Getting the type of 'm' (line 1073)
    m_35147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 42), 'm', False)
    # Getting the type of 'k' (line 1073)
    k_35148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 45), 'k', False)
    # Getting the type of 'fun_wrapped' (line 1073)
    fun_wrapped_35149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 48), 'fun_wrapped', False)
    # Getting the type of 'bc_wrapped' (line 1073)
    bc_wrapped_35150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 61), 'bc_wrapped', False)
    # Getting the type of 'fun_jac_wrapped' (line 1074)
    fun_jac_wrapped_35151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 39), 'fun_jac_wrapped', False)
    # Getting the type of 'bc_jac_wrapped' (line 1074)
    bc_jac_wrapped_35152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 56), 'bc_jac_wrapped', False)
    # Getting the type of 'x' (line 1074)
    x_35153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 72), 'x', False)
    # Getting the type of 'h' (line 1074)
    h_35154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 75), 'h', False)
    # Processing the call keyword arguments (line 1073)
    kwargs_35155 = {}
    # Getting the type of 'prepare_sys' (line 1073)
    prepare_sys_35145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 27), 'prepare_sys', False)
    # Calling prepare_sys(args, kwargs) (line 1073)
    prepare_sys_call_result_35156 = invoke(stypy.reporting.localization.Localization(__file__, 1073, 27), prepare_sys_35145, *[n_35146, m_35147, k_35148, fun_wrapped_35149, bc_wrapped_35150, fun_jac_wrapped_35151, bc_jac_wrapped_35152, x_35153, h_35154], **kwargs_35155)
    
    # Obtaining the member '__getitem__' of a type (line 1073)
    getitem___35157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 8), prepare_sys_call_result_35156, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1073)
    subscript_call_result_35158 = invoke(stypy.reporting.localization.Localization(__file__, 1073, 8), getitem___35157, int_35144)
    
    # Assigning a type to the variable 'tuple_var_assignment_32424' (line 1073)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'tuple_var_assignment_32424', subscript_call_result_35158)
    
    # Assigning a Name to a Name (line 1073):
    # Getting the type of 'tuple_var_assignment_32423' (line 1073)
    tuple_var_assignment_32423_35159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'tuple_var_assignment_32423')
    # Assigning a type to the variable 'col_fun' (line 1073)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'col_fun', tuple_var_assignment_32423_35159)
    
    # Assigning a Name to a Name (line 1073):
    # Getting the type of 'tuple_var_assignment_32424' (line 1073)
    tuple_var_assignment_32424_35160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'tuple_var_assignment_32424')
    # Assigning a type to the variable 'jac_sys' (line 1073)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 17), 'jac_sys', tuple_var_assignment_32424_35160)
    
    # Assigning a Call to a Tuple (line 1075):
    
    # Assigning a Subscript to a Name (line 1075):
    
    # Obtaining the type of the subscript
    int_35161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 8), 'int')
    
    # Call to solve_newton(...): (line 1075)
    # Processing the call arguments (line 1075)
    # Getting the type of 'n' (line 1075)
    n_35163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 38), 'n', False)
    # Getting the type of 'm' (line 1075)
    m_35164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 41), 'm', False)
    # Getting the type of 'h' (line 1075)
    h_35165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 44), 'h', False)
    # Getting the type of 'col_fun' (line 1075)
    col_fun_35166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 47), 'col_fun', False)
    # Getting the type of 'bc_wrapped' (line 1075)
    bc_wrapped_35167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 56), 'bc_wrapped', False)
    # Getting the type of 'jac_sys' (line 1075)
    jac_sys_35168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 68), 'jac_sys', False)
    # Getting the type of 'y' (line 1076)
    y_35169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 38), 'y', False)
    # Getting the type of 'p' (line 1076)
    p_35170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 41), 'p', False)
    # Getting the type of 'B' (line 1076)
    B_35171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 44), 'B', False)
    # Getting the type of 'tol' (line 1076)
    tol_35172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 47), 'tol', False)
    # Processing the call keyword arguments (line 1075)
    kwargs_35173 = {}
    # Getting the type of 'solve_newton' (line 1075)
    solve_newton_35162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 25), 'solve_newton', False)
    # Calling solve_newton(args, kwargs) (line 1075)
    solve_newton_call_result_35174 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 25), solve_newton_35162, *[n_35163, m_35164, h_35165, col_fun_35166, bc_wrapped_35167, jac_sys_35168, y_35169, p_35170, B_35171, tol_35172], **kwargs_35173)
    
    # Obtaining the member '__getitem__' of a type (line 1075)
    getitem___35175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 8), solve_newton_call_result_35174, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1075)
    subscript_call_result_35176 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 8), getitem___35175, int_35161)
    
    # Assigning a type to the variable 'tuple_var_assignment_32425' (line 1075)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'tuple_var_assignment_32425', subscript_call_result_35176)
    
    # Assigning a Subscript to a Name (line 1075):
    
    # Obtaining the type of the subscript
    int_35177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 8), 'int')
    
    # Call to solve_newton(...): (line 1075)
    # Processing the call arguments (line 1075)
    # Getting the type of 'n' (line 1075)
    n_35179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 38), 'n', False)
    # Getting the type of 'm' (line 1075)
    m_35180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 41), 'm', False)
    # Getting the type of 'h' (line 1075)
    h_35181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 44), 'h', False)
    # Getting the type of 'col_fun' (line 1075)
    col_fun_35182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 47), 'col_fun', False)
    # Getting the type of 'bc_wrapped' (line 1075)
    bc_wrapped_35183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 56), 'bc_wrapped', False)
    # Getting the type of 'jac_sys' (line 1075)
    jac_sys_35184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 68), 'jac_sys', False)
    # Getting the type of 'y' (line 1076)
    y_35185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 38), 'y', False)
    # Getting the type of 'p' (line 1076)
    p_35186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 41), 'p', False)
    # Getting the type of 'B' (line 1076)
    B_35187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 44), 'B', False)
    # Getting the type of 'tol' (line 1076)
    tol_35188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 47), 'tol', False)
    # Processing the call keyword arguments (line 1075)
    kwargs_35189 = {}
    # Getting the type of 'solve_newton' (line 1075)
    solve_newton_35178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 25), 'solve_newton', False)
    # Calling solve_newton(args, kwargs) (line 1075)
    solve_newton_call_result_35190 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 25), solve_newton_35178, *[n_35179, m_35180, h_35181, col_fun_35182, bc_wrapped_35183, jac_sys_35184, y_35185, p_35186, B_35187, tol_35188], **kwargs_35189)
    
    # Obtaining the member '__getitem__' of a type (line 1075)
    getitem___35191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 8), solve_newton_call_result_35190, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1075)
    subscript_call_result_35192 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 8), getitem___35191, int_35177)
    
    # Assigning a type to the variable 'tuple_var_assignment_32426' (line 1075)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'tuple_var_assignment_32426', subscript_call_result_35192)
    
    # Assigning a Subscript to a Name (line 1075):
    
    # Obtaining the type of the subscript
    int_35193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 8), 'int')
    
    # Call to solve_newton(...): (line 1075)
    # Processing the call arguments (line 1075)
    # Getting the type of 'n' (line 1075)
    n_35195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 38), 'n', False)
    # Getting the type of 'm' (line 1075)
    m_35196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 41), 'm', False)
    # Getting the type of 'h' (line 1075)
    h_35197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 44), 'h', False)
    # Getting the type of 'col_fun' (line 1075)
    col_fun_35198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 47), 'col_fun', False)
    # Getting the type of 'bc_wrapped' (line 1075)
    bc_wrapped_35199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 56), 'bc_wrapped', False)
    # Getting the type of 'jac_sys' (line 1075)
    jac_sys_35200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 68), 'jac_sys', False)
    # Getting the type of 'y' (line 1076)
    y_35201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 38), 'y', False)
    # Getting the type of 'p' (line 1076)
    p_35202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 41), 'p', False)
    # Getting the type of 'B' (line 1076)
    B_35203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 44), 'B', False)
    # Getting the type of 'tol' (line 1076)
    tol_35204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 47), 'tol', False)
    # Processing the call keyword arguments (line 1075)
    kwargs_35205 = {}
    # Getting the type of 'solve_newton' (line 1075)
    solve_newton_35194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 25), 'solve_newton', False)
    # Calling solve_newton(args, kwargs) (line 1075)
    solve_newton_call_result_35206 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 25), solve_newton_35194, *[n_35195, m_35196, h_35197, col_fun_35198, bc_wrapped_35199, jac_sys_35200, y_35201, p_35202, B_35203, tol_35204], **kwargs_35205)
    
    # Obtaining the member '__getitem__' of a type (line 1075)
    getitem___35207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 8), solve_newton_call_result_35206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1075)
    subscript_call_result_35208 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 8), getitem___35207, int_35193)
    
    # Assigning a type to the variable 'tuple_var_assignment_32427' (line 1075)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'tuple_var_assignment_32427', subscript_call_result_35208)
    
    # Assigning a Name to a Name (line 1075):
    # Getting the type of 'tuple_var_assignment_32425' (line 1075)
    tuple_var_assignment_32425_35209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'tuple_var_assignment_32425')
    # Assigning a type to the variable 'y' (line 1075)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'y', tuple_var_assignment_32425_35209)
    
    # Assigning a Name to a Name (line 1075):
    # Getting the type of 'tuple_var_assignment_32426' (line 1075)
    tuple_var_assignment_32426_35210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'tuple_var_assignment_32426')
    # Assigning a type to the variable 'p' (line 1075)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 11), 'p', tuple_var_assignment_32426_35210)
    
    # Assigning a Name to a Name (line 1075):
    # Getting the type of 'tuple_var_assignment_32427' (line 1075)
    tuple_var_assignment_32427_35211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'tuple_var_assignment_32427')
    # Assigning a type to the variable 'singular' (line 1075)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 14), 'singular', tuple_var_assignment_32427_35211)
    
    # Getting the type of 'iteration' (line 1077)
    iteration_35212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 8), 'iteration')
    int_35213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1077, 21), 'int')
    # Applying the binary operator '+=' (line 1077)
    result_iadd_35214 = python_operator(stypy.reporting.localization.Localization(__file__, 1077, 8), '+=', iteration_35212, int_35213)
    # Assigning a type to the variable 'iteration' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 8), 'iteration', result_iadd_35214)
    
    
    # Assigning a Call to a Tuple (line 1079):
    
    # Assigning a Subscript to a Name (line 1079):
    
    # Obtaining the type of the subscript
    int_35215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1079, 8), 'int')
    
    # Call to collocation_fun(...): (line 1079)
    # Processing the call arguments (line 1079)
    # Getting the type of 'fun_wrapped' (line 1079)
    fun_wrapped_35217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 57), 'fun_wrapped', False)
    # Getting the type of 'y' (line 1079)
    y_35218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 70), 'y', False)
    # Getting the type of 'p' (line 1080)
    p_35219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 57), 'p', False)
    # Getting the type of 'x' (line 1080)
    x_35220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 60), 'x', False)
    # Getting the type of 'h' (line 1080)
    h_35221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 63), 'h', False)
    # Processing the call keyword arguments (line 1079)
    kwargs_35222 = {}
    # Getting the type of 'collocation_fun' (line 1079)
    collocation_fun_35216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 41), 'collocation_fun', False)
    # Calling collocation_fun(args, kwargs) (line 1079)
    collocation_fun_call_result_35223 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 41), collocation_fun_35216, *[fun_wrapped_35217, y_35218, p_35219, x_35220, h_35221], **kwargs_35222)
    
    # Obtaining the member '__getitem__' of a type (line 1079)
    getitem___35224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1079, 8), collocation_fun_call_result_35223, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1079)
    subscript_call_result_35225 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 8), getitem___35224, int_35215)
    
    # Assigning a type to the variable 'tuple_var_assignment_32428' (line 1079)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'tuple_var_assignment_32428', subscript_call_result_35225)
    
    # Assigning a Subscript to a Name (line 1079):
    
    # Obtaining the type of the subscript
    int_35226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1079, 8), 'int')
    
    # Call to collocation_fun(...): (line 1079)
    # Processing the call arguments (line 1079)
    # Getting the type of 'fun_wrapped' (line 1079)
    fun_wrapped_35228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 57), 'fun_wrapped', False)
    # Getting the type of 'y' (line 1079)
    y_35229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 70), 'y', False)
    # Getting the type of 'p' (line 1080)
    p_35230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 57), 'p', False)
    # Getting the type of 'x' (line 1080)
    x_35231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 60), 'x', False)
    # Getting the type of 'h' (line 1080)
    h_35232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 63), 'h', False)
    # Processing the call keyword arguments (line 1079)
    kwargs_35233 = {}
    # Getting the type of 'collocation_fun' (line 1079)
    collocation_fun_35227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 41), 'collocation_fun', False)
    # Calling collocation_fun(args, kwargs) (line 1079)
    collocation_fun_call_result_35234 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 41), collocation_fun_35227, *[fun_wrapped_35228, y_35229, p_35230, x_35231, h_35232], **kwargs_35233)
    
    # Obtaining the member '__getitem__' of a type (line 1079)
    getitem___35235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1079, 8), collocation_fun_call_result_35234, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1079)
    subscript_call_result_35236 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 8), getitem___35235, int_35226)
    
    # Assigning a type to the variable 'tuple_var_assignment_32429' (line 1079)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'tuple_var_assignment_32429', subscript_call_result_35236)
    
    # Assigning a Subscript to a Name (line 1079):
    
    # Obtaining the type of the subscript
    int_35237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1079, 8), 'int')
    
    # Call to collocation_fun(...): (line 1079)
    # Processing the call arguments (line 1079)
    # Getting the type of 'fun_wrapped' (line 1079)
    fun_wrapped_35239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 57), 'fun_wrapped', False)
    # Getting the type of 'y' (line 1079)
    y_35240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 70), 'y', False)
    # Getting the type of 'p' (line 1080)
    p_35241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 57), 'p', False)
    # Getting the type of 'x' (line 1080)
    x_35242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 60), 'x', False)
    # Getting the type of 'h' (line 1080)
    h_35243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 63), 'h', False)
    # Processing the call keyword arguments (line 1079)
    kwargs_35244 = {}
    # Getting the type of 'collocation_fun' (line 1079)
    collocation_fun_35238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 41), 'collocation_fun', False)
    # Calling collocation_fun(args, kwargs) (line 1079)
    collocation_fun_call_result_35245 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 41), collocation_fun_35238, *[fun_wrapped_35239, y_35240, p_35241, x_35242, h_35243], **kwargs_35244)
    
    # Obtaining the member '__getitem__' of a type (line 1079)
    getitem___35246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1079, 8), collocation_fun_call_result_35245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1079)
    subscript_call_result_35247 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 8), getitem___35246, int_35237)
    
    # Assigning a type to the variable 'tuple_var_assignment_32430' (line 1079)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'tuple_var_assignment_32430', subscript_call_result_35247)
    
    # Assigning a Subscript to a Name (line 1079):
    
    # Obtaining the type of the subscript
    int_35248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1079, 8), 'int')
    
    # Call to collocation_fun(...): (line 1079)
    # Processing the call arguments (line 1079)
    # Getting the type of 'fun_wrapped' (line 1079)
    fun_wrapped_35250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 57), 'fun_wrapped', False)
    # Getting the type of 'y' (line 1079)
    y_35251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 70), 'y', False)
    # Getting the type of 'p' (line 1080)
    p_35252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 57), 'p', False)
    # Getting the type of 'x' (line 1080)
    x_35253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 60), 'x', False)
    # Getting the type of 'h' (line 1080)
    h_35254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 63), 'h', False)
    # Processing the call keyword arguments (line 1079)
    kwargs_35255 = {}
    # Getting the type of 'collocation_fun' (line 1079)
    collocation_fun_35249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 41), 'collocation_fun', False)
    # Calling collocation_fun(args, kwargs) (line 1079)
    collocation_fun_call_result_35256 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 41), collocation_fun_35249, *[fun_wrapped_35250, y_35251, p_35252, x_35253, h_35254], **kwargs_35255)
    
    # Obtaining the member '__getitem__' of a type (line 1079)
    getitem___35257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1079, 8), collocation_fun_call_result_35256, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1079)
    subscript_call_result_35258 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 8), getitem___35257, int_35248)
    
    # Assigning a type to the variable 'tuple_var_assignment_32431' (line 1079)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'tuple_var_assignment_32431', subscript_call_result_35258)
    
    # Assigning a Name to a Name (line 1079):
    # Getting the type of 'tuple_var_assignment_32428' (line 1079)
    tuple_var_assignment_32428_35259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'tuple_var_assignment_32428')
    # Assigning a type to the variable 'col_res' (line 1079)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'col_res', tuple_var_assignment_32428_35259)
    
    # Assigning a Name to a Name (line 1079):
    # Getting the type of 'tuple_var_assignment_32429' (line 1079)
    tuple_var_assignment_32429_35260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'tuple_var_assignment_32429')
    # Assigning a type to the variable 'y_middle' (line 1079)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 17), 'y_middle', tuple_var_assignment_32429_35260)
    
    # Assigning a Name to a Name (line 1079):
    # Getting the type of 'tuple_var_assignment_32430' (line 1079)
    tuple_var_assignment_32430_35261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'tuple_var_assignment_32430')
    # Assigning a type to the variable 'f' (line 1079)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 27), 'f', tuple_var_assignment_32430_35261)
    
    # Assigning a Name to a Name (line 1079):
    # Getting the type of 'tuple_var_assignment_32431' (line 1079)
    tuple_var_assignment_32431_35262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'tuple_var_assignment_32431')
    # Assigning a type to the variable 'f_middle' (line 1079)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 30), 'f_middle', tuple_var_assignment_32431_35262)
    
    # Assigning a BinOp to a Name (line 1082):
    
    # Assigning a BinOp to a Name (line 1082):
    float_35263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1082, 19), 'float')
    # Getting the type of 'col_res' (line 1082)
    col_res_35264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1082, 25), 'col_res')
    # Applying the binary operator '*' (line 1082)
    result_mul_35265 = python_operator(stypy.reporting.localization.Localization(__file__, 1082, 19), '*', float_35263, col_res_35264)
    
    # Getting the type of 'h' (line 1082)
    h_35266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1082, 35), 'h')
    # Applying the binary operator 'div' (line 1082)
    result_div_35267 = python_operator(stypy.reporting.localization.Localization(__file__, 1082, 33), 'div', result_mul_35265, h_35266)
    
    # Assigning a type to the variable 'r_middle' (line 1082)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1082, 8), 'r_middle', result_div_35267)
    
    # Assigning a Call to a Name (line 1083):
    
    # Assigning a Call to a Name (line 1083):
    
    # Call to create_spline(...): (line 1083)
    # Processing the call arguments (line 1083)
    # Getting the type of 'y' (line 1083)
    y_35269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 28), 'y', False)
    # Getting the type of 'f' (line 1083)
    f_35270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 31), 'f', False)
    # Getting the type of 'x' (line 1083)
    x_35271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 34), 'x', False)
    # Getting the type of 'h' (line 1083)
    h_35272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 37), 'h', False)
    # Processing the call keyword arguments (line 1083)
    kwargs_35273 = {}
    # Getting the type of 'create_spline' (line 1083)
    create_spline_35268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 14), 'create_spline', False)
    # Calling create_spline(args, kwargs) (line 1083)
    create_spline_call_result_35274 = invoke(stypy.reporting.localization.Localization(__file__, 1083, 14), create_spline_35268, *[y_35269, f_35270, x_35271, h_35272], **kwargs_35273)
    
    # Assigning a type to the variable 'sol' (line 1083)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1083, 8), 'sol', create_spline_call_result_35274)
    
    # Assigning a Call to a Name (line 1084):
    
    # Assigning a Call to a Name (line 1084):
    
    # Call to estimate_rms_residuals(...): (line 1084)
    # Processing the call arguments (line 1084)
    # Getting the type of 'fun_wrapped' (line 1084)
    fun_wrapped_35276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 41), 'fun_wrapped', False)
    # Getting the type of 'sol' (line 1084)
    sol_35277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 54), 'sol', False)
    # Getting the type of 'x' (line 1084)
    x_35278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 59), 'x', False)
    # Getting the type of 'h' (line 1084)
    h_35279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 62), 'h', False)
    # Getting the type of 'p' (line 1084)
    p_35280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 65), 'p', False)
    # Getting the type of 'r_middle' (line 1085)
    r_middle_35281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 41), 'r_middle', False)
    # Getting the type of 'f_middle' (line 1085)
    f_middle_35282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 51), 'f_middle', False)
    # Processing the call keyword arguments (line 1084)
    kwargs_35283 = {}
    # Getting the type of 'estimate_rms_residuals' (line 1084)
    estimate_rms_residuals_35275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 18), 'estimate_rms_residuals', False)
    # Calling estimate_rms_residuals(args, kwargs) (line 1084)
    estimate_rms_residuals_call_result_35284 = invoke(stypy.reporting.localization.Localization(__file__, 1084, 18), estimate_rms_residuals_35275, *[fun_wrapped_35276, sol_35277, x_35278, h_35279, p_35280, r_middle_35281, f_middle_35282], **kwargs_35283)
    
    # Assigning a type to the variable 'rms_res' (line 1084)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1084, 8), 'rms_res', estimate_rms_residuals_call_result_35284)
    
    # Assigning a Call to a Name (line 1086):
    
    # Assigning a Call to a Name (line 1086):
    
    # Call to max(...): (line 1086)
    # Processing the call arguments (line 1086)
    # Getting the type of 'rms_res' (line 1086)
    rms_res_35287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 29), 'rms_res', False)
    # Processing the call keyword arguments (line 1086)
    kwargs_35288 = {}
    # Getting the type of 'np' (line 1086)
    np_35285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 22), 'np', False)
    # Obtaining the member 'max' of a type (line 1086)
    max_35286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1086, 22), np_35285, 'max')
    # Calling max(args, kwargs) (line 1086)
    max_call_result_35289 = invoke(stypy.reporting.localization.Localization(__file__, 1086, 22), max_35286, *[rms_res_35287], **kwargs_35288)
    
    # Assigning a type to the variable 'max_rms_res' (line 1086)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1086, 8), 'max_rms_res', max_call_result_35289)
    
    # Getting the type of 'singular' (line 1088)
    singular_35290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 11), 'singular')
    # Testing the type of an if condition (line 1088)
    if_condition_35291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1088, 8), singular_35290)
    # Assigning a type to the variable 'if_condition_35291' (line 1088)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1088, 8), 'if_condition_35291', if_condition_35291)
    # SSA begins for if statement (line 1088)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 1089):
    
    # Assigning a Num to a Name (line 1089):
    int_35292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 21), 'int')
    # Assigning a type to the variable 'status' (line 1089)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 12), 'status', int_35292)
    # SSA join for if statement (line 1088)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1092):
    
    # Assigning a Subscript to a Name (line 1092):
    
    # Obtaining the type of the subscript
    int_35293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, 8), 'int')
    
    # Call to nonzero(...): (line 1092)
    # Processing the call arguments (line 1092)
    
    # Getting the type of 'rms_res' (line 1092)
    rms_res_35296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 32), 'rms_res', False)
    # Getting the type of 'tol' (line 1092)
    tol_35297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 42), 'tol', False)
    # Applying the binary operator '>' (line 1092)
    result_gt_35298 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 32), '>', rms_res_35296, tol_35297)
    
    
    # Getting the type of 'rms_res' (line 1092)
    rms_res_35299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 50), 'rms_res', False)
    int_35300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, 60), 'int')
    # Getting the type of 'tol' (line 1092)
    tol_35301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 66), 'tol', False)
    # Applying the binary operator '*' (line 1092)
    result_mul_35302 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 60), '*', int_35300, tol_35301)
    
    # Applying the binary operator '<' (line 1092)
    result_lt_35303 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 50), '<', rms_res_35299, result_mul_35302)
    
    # Applying the binary operator '&' (line 1092)
    result_and__35304 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 31), '&', result_gt_35298, result_lt_35303)
    
    # Processing the call keyword arguments (line 1092)
    kwargs_35305 = {}
    # Getting the type of 'np' (line 1092)
    np_35294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 20), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 1092)
    nonzero_35295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1092, 20), np_35294, 'nonzero')
    # Calling nonzero(args, kwargs) (line 1092)
    nonzero_call_result_35306 = invoke(stypy.reporting.localization.Localization(__file__, 1092, 20), nonzero_35295, *[result_and__35304], **kwargs_35305)
    
    # Obtaining the member '__getitem__' of a type (line 1092)
    getitem___35307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1092, 8), nonzero_call_result_35306, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1092)
    subscript_call_result_35308 = invoke(stypy.reporting.localization.Localization(__file__, 1092, 8), getitem___35307, int_35293)
    
    # Assigning a type to the variable 'tuple_var_assignment_32432' (line 1092)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1092, 8), 'tuple_var_assignment_32432', subscript_call_result_35308)
    
    # Assigning a Name to a Name (line 1092):
    # Getting the type of 'tuple_var_assignment_32432' (line 1092)
    tuple_var_assignment_32432_35309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 8), 'tuple_var_assignment_32432')
    # Assigning a type to the variable 'insert_1' (line 1092)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1092, 8), 'insert_1', tuple_var_assignment_32432_35309)
    
    # Assigning a Call to a Tuple (line 1093):
    
    # Assigning a Subscript to a Name (line 1093):
    
    # Obtaining the type of the subscript
    int_35310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1093, 8), 'int')
    
    # Call to nonzero(...): (line 1093)
    # Processing the call arguments (line 1093)
    
    # Getting the type of 'rms_res' (line 1093)
    rms_res_35313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 31), 'rms_res', False)
    int_35314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1093, 42), 'int')
    # Getting the type of 'tol' (line 1093)
    tol_35315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 48), 'tol', False)
    # Applying the binary operator '*' (line 1093)
    result_mul_35316 = python_operator(stypy.reporting.localization.Localization(__file__, 1093, 42), '*', int_35314, tol_35315)
    
    # Applying the binary operator '>=' (line 1093)
    result_ge_35317 = python_operator(stypy.reporting.localization.Localization(__file__, 1093, 31), '>=', rms_res_35313, result_mul_35316)
    
    # Processing the call keyword arguments (line 1093)
    kwargs_35318 = {}
    # Getting the type of 'np' (line 1093)
    np_35311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 20), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 1093)
    nonzero_35312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1093, 20), np_35311, 'nonzero')
    # Calling nonzero(args, kwargs) (line 1093)
    nonzero_call_result_35319 = invoke(stypy.reporting.localization.Localization(__file__, 1093, 20), nonzero_35312, *[result_ge_35317], **kwargs_35318)
    
    # Obtaining the member '__getitem__' of a type (line 1093)
    getitem___35320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1093, 8), nonzero_call_result_35319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1093)
    subscript_call_result_35321 = invoke(stypy.reporting.localization.Localization(__file__, 1093, 8), getitem___35320, int_35310)
    
    # Assigning a type to the variable 'tuple_var_assignment_32433' (line 1093)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 8), 'tuple_var_assignment_32433', subscript_call_result_35321)
    
    # Assigning a Name to a Name (line 1093):
    # Getting the type of 'tuple_var_assignment_32433' (line 1093)
    tuple_var_assignment_32433_35322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 8), 'tuple_var_assignment_32433')
    # Assigning a type to the variable 'insert_2' (line 1093)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 8), 'insert_2', tuple_var_assignment_32433_35322)
    
    # Assigning a BinOp to a Name (line 1094):
    
    # Assigning a BinOp to a Name (line 1094):
    
    # Obtaining the type of the subscript
    int_35323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1094, 37), 'int')
    # Getting the type of 'insert_1' (line 1094)
    insert_1_35324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 22), 'insert_1')
    # Obtaining the member 'shape' of a type (line 1094)
    shape_35325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 22), insert_1_35324, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1094)
    getitem___35326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 22), shape_35325, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1094)
    subscript_call_result_35327 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 22), getitem___35326, int_35323)
    
    int_35328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1094, 42), 'int')
    
    # Obtaining the type of the subscript
    int_35329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1094, 61), 'int')
    # Getting the type of 'insert_2' (line 1094)
    insert_2_35330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 46), 'insert_2')
    # Obtaining the member 'shape' of a type (line 1094)
    shape_35331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 46), insert_2_35330, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1094)
    getitem___35332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 46), shape_35331, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1094)
    subscript_call_result_35333 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 46), getitem___35332, int_35329)
    
    # Applying the binary operator '*' (line 1094)
    result_mul_35334 = python_operator(stypy.reporting.localization.Localization(__file__, 1094, 42), '*', int_35328, subscript_call_result_35333)
    
    # Applying the binary operator '+' (line 1094)
    result_add_35335 = python_operator(stypy.reporting.localization.Localization(__file__, 1094, 22), '+', subscript_call_result_35327, result_mul_35334)
    
    # Assigning a type to the variable 'nodes_added' (line 1094)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1094, 8), 'nodes_added', result_add_35335)
    
    
    # Getting the type of 'm' (line 1096)
    m_35336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 11), 'm')
    # Getting the type of 'nodes_added' (line 1096)
    nodes_added_35337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 15), 'nodes_added')
    # Applying the binary operator '+' (line 1096)
    result_add_35338 = python_operator(stypy.reporting.localization.Localization(__file__, 1096, 11), '+', m_35336, nodes_added_35337)
    
    # Getting the type of 'max_nodes' (line 1096)
    max_nodes_35339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 29), 'max_nodes')
    # Applying the binary operator '>' (line 1096)
    result_gt_35340 = python_operator(stypy.reporting.localization.Localization(__file__, 1096, 11), '>', result_add_35338, max_nodes_35339)
    
    # Testing the type of an if condition (line 1096)
    if_condition_35341 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1096, 8), result_gt_35340)
    # Assigning a type to the variable 'if_condition_35341' (line 1096)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1096, 8), 'if_condition_35341', if_condition_35341)
    # SSA begins for if statement (line 1096)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 1097):
    
    # Assigning a Num to a Name (line 1097):
    int_35342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1097, 21), 'int')
    # Assigning a type to the variable 'status' (line 1097)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1097, 12), 'status', int_35342)
    
    
    # Getting the type of 'verbose' (line 1098)
    verbose_35343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 15), 'verbose')
    int_35344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1098, 26), 'int')
    # Applying the binary operator '==' (line 1098)
    result_eq_35345 = python_operator(stypy.reporting.localization.Localization(__file__, 1098, 15), '==', verbose_35343, int_35344)
    
    # Testing the type of an if condition (line 1098)
    if_condition_35346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1098, 12), result_eq_35345)
    # Assigning a type to the variable 'if_condition_35346' (line 1098)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1098, 12), 'if_condition_35346', if_condition_35346)
    # SSA begins for if statement (line 1098)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1099):
    
    # Assigning a Call to a Name (line 1099):
    
    # Call to format(...): (line 1099)
    # Processing the call arguments (line 1099)
    # Getting the type of 'nodes_added' (line 1099)
    nodes_added_35349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 44), 'nodes_added', False)
    # Processing the call keyword arguments (line 1099)
    kwargs_35350 = {}
    str_35347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 30), 'str', '({})')
    # Obtaining the member 'format' of a type (line 1099)
    format_35348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1099, 30), str_35347, 'format')
    # Calling format(args, kwargs) (line 1099)
    format_call_result_35351 = invoke(stypy.reporting.localization.Localization(__file__, 1099, 30), format_35348, *[nodes_added_35349], **kwargs_35350)
    
    # Assigning a type to the variable 'nodes_added' (line 1099)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1099, 16), 'nodes_added', format_call_result_35351)
    
    # Call to print_iteration_progress(...): (line 1100)
    # Processing the call arguments (line 1100)
    # Getting the type of 'iteration' (line 1100)
    iteration_35353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 41), 'iteration', False)
    # Getting the type of 'max_rms_res' (line 1100)
    max_rms_res_35354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 52), 'max_rms_res', False)
    # Getting the type of 'm' (line 1100)
    m_35355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 65), 'm', False)
    # Getting the type of 'nodes_added' (line 1101)
    nodes_added_35356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 41), 'nodes_added', False)
    # Processing the call keyword arguments (line 1100)
    kwargs_35357 = {}
    # Getting the type of 'print_iteration_progress' (line 1100)
    print_iteration_progress_35352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 16), 'print_iteration_progress', False)
    # Calling print_iteration_progress(args, kwargs) (line 1100)
    print_iteration_progress_call_result_35358 = invoke(stypy.reporting.localization.Localization(__file__, 1100, 16), print_iteration_progress_35352, *[iteration_35353, max_rms_res_35354, m_35355, nodes_added_35356], **kwargs_35357)
    
    # SSA join for if statement (line 1098)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1096)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbose' (line 1104)
    verbose_35359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 11), 'verbose')
    int_35360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 22), 'int')
    # Applying the binary operator '==' (line 1104)
    result_eq_35361 = python_operator(stypy.reporting.localization.Localization(__file__, 1104, 11), '==', verbose_35359, int_35360)
    
    # Testing the type of an if condition (line 1104)
    if_condition_35362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1104, 8), result_eq_35361)
    # Assigning a type to the variable 'if_condition_35362' (line 1104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1104, 8), 'if_condition_35362', if_condition_35362)
    # SSA begins for if statement (line 1104)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print_iteration_progress(...): (line 1105)
    # Processing the call arguments (line 1105)
    # Getting the type of 'iteration' (line 1105)
    iteration_35364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 37), 'iteration', False)
    # Getting the type of 'max_rms_res' (line 1105)
    max_rms_res_35365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 48), 'max_rms_res', False)
    # Getting the type of 'm' (line 1105)
    m_35366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 61), 'm', False)
    # Getting the type of 'nodes_added' (line 1105)
    nodes_added_35367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 64), 'nodes_added', False)
    # Processing the call keyword arguments (line 1105)
    kwargs_35368 = {}
    # Getting the type of 'print_iteration_progress' (line 1105)
    print_iteration_progress_35363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 12), 'print_iteration_progress', False)
    # Calling print_iteration_progress(args, kwargs) (line 1105)
    print_iteration_progress_call_result_35369 = invoke(stypy.reporting.localization.Localization(__file__, 1105, 12), print_iteration_progress_35363, *[iteration_35364, max_rms_res_35365, m_35366, nodes_added_35367], **kwargs_35368)
    
    # SSA join for if statement (line 1104)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'nodes_added' (line 1107)
    nodes_added_35370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 11), 'nodes_added')
    int_35371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1107, 25), 'int')
    # Applying the binary operator '>' (line 1107)
    result_gt_35372 = python_operator(stypy.reporting.localization.Localization(__file__, 1107, 11), '>', nodes_added_35370, int_35371)
    
    # Testing the type of an if condition (line 1107)
    if_condition_35373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1107, 8), result_gt_35372)
    # Assigning a type to the variable 'if_condition_35373' (line 1107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1107, 8), 'if_condition_35373', if_condition_35373)
    # SSA begins for if statement (line 1107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1108):
    
    # Assigning a Call to a Name (line 1108):
    
    # Call to modify_mesh(...): (line 1108)
    # Processing the call arguments (line 1108)
    # Getting the type of 'x' (line 1108)
    x_35375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 28), 'x', False)
    # Getting the type of 'insert_1' (line 1108)
    insert_1_35376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 31), 'insert_1', False)
    # Getting the type of 'insert_2' (line 1108)
    insert_2_35377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 41), 'insert_2', False)
    # Processing the call keyword arguments (line 1108)
    kwargs_35378 = {}
    # Getting the type of 'modify_mesh' (line 1108)
    modify_mesh_35374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 16), 'modify_mesh', False)
    # Calling modify_mesh(args, kwargs) (line 1108)
    modify_mesh_call_result_35379 = invoke(stypy.reporting.localization.Localization(__file__, 1108, 16), modify_mesh_35374, *[x_35375, insert_1_35376, insert_2_35377], **kwargs_35378)
    
    # Assigning a type to the variable 'x' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 12), 'x', modify_mesh_call_result_35379)
    
    # Assigning a Call to a Name (line 1109):
    
    # Assigning a Call to a Name (line 1109):
    
    # Call to diff(...): (line 1109)
    # Processing the call arguments (line 1109)
    # Getting the type of 'x' (line 1109)
    x_35382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 24), 'x', False)
    # Processing the call keyword arguments (line 1109)
    kwargs_35383 = {}
    # Getting the type of 'np' (line 1109)
    np_35380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 16), 'np', False)
    # Obtaining the member 'diff' of a type (line 1109)
    diff_35381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1109, 16), np_35380, 'diff')
    # Calling diff(args, kwargs) (line 1109)
    diff_call_result_35384 = invoke(stypy.reporting.localization.Localization(__file__, 1109, 16), diff_35381, *[x_35382], **kwargs_35383)
    
    # Assigning a type to the variable 'h' (line 1109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1109, 12), 'h', diff_call_result_35384)
    
    # Assigning a Call to a Name (line 1110):
    
    # Assigning a Call to a Name (line 1110):
    
    # Call to sol(...): (line 1110)
    # Processing the call arguments (line 1110)
    # Getting the type of 'x' (line 1110)
    x_35386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 20), 'x', False)
    # Processing the call keyword arguments (line 1110)
    kwargs_35387 = {}
    # Getting the type of 'sol' (line 1110)
    sol_35385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 16), 'sol', False)
    # Calling sol(args, kwargs) (line 1110)
    sol_call_result_35388 = invoke(stypy.reporting.localization.Localization(__file__, 1110, 16), sol_35385, *[x_35386], **kwargs_35387)
    
    # Assigning a type to the variable 'y' (line 1110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1110, 12), 'y', sol_call_result_35388)
    # SSA branch for the else part of an if statement (line 1107)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 1112):
    
    # Assigning a Num to a Name (line 1112):
    int_35389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1112, 21), 'int')
    # Assigning a type to the variable 'status' (line 1112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1112, 12), 'status', int_35389)
    # SSA join for if statement (line 1107)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 1070)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbose' (line 1115)
    verbose_35390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 7), 'verbose')
    int_35391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1115, 17), 'int')
    # Applying the binary operator '>' (line 1115)
    result_gt_35392 = python_operator(stypy.reporting.localization.Localization(__file__, 1115, 7), '>', verbose_35390, int_35391)
    
    # Testing the type of an if condition (line 1115)
    if_condition_35393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1115, 4), result_gt_35392)
    # Assigning a type to the variable 'if_condition_35393' (line 1115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1115, 4), 'if_condition_35393', if_condition_35393)
    # SSA begins for if statement (line 1115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'status' (line 1116)
    status_35394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 11), 'status')
    int_35395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1116, 21), 'int')
    # Applying the binary operator '==' (line 1116)
    result_eq_35396 = python_operator(stypy.reporting.localization.Localization(__file__, 1116, 11), '==', status_35394, int_35395)
    
    # Testing the type of an if condition (line 1116)
    if_condition_35397 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1116, 8), result_eq_35396)
    # Assigning a type to the variable 'if_condition_35397' (line 1116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1116, 8), 'if_condition_35397', if_condition_35397)
    # SSA begins for if statement (line 1116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 1117)
    # Processing the call arguments (line 1117)
    
    # Call to format(...): (line 1117)
    # Processing the call arguments (line 1117)
    # Getting the type of 'iteration' (line 1119)
    iteration_35401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 26), 'iteration', False)
    
    # Obtaining the type of the subscript
    int_35402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 45), 'int')
    # Getting the type of 'x' (line 1119)
    x_35403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 37), 'x', False)
    # Obtaining the member 'shape' of a type (line 1119)
    shape_35404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1119, 37), x_35403, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1119)
    getitem___35405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1119, 37), shape_35404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1119)
    subscript_call_result_35406 = invoke(stypy.reporting.localization.Localization(__file__, 1119, 37), getitem___35405, int_35402)
    
    # Getting the type of 'max_rms_res' (line 1119)
    max_rms_res_35407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 49), 'max_rms_res', False)
    # Processing the call keyword arguments (line 1117)
    kwargs_35408 = {}
    str_35399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 18), 'str', 'Solved in {} iterations, number of nodes {}, maximum relative residual {:.2e}.')
    # Obtaining the member 'format' of a type (line 1117)
    format_35400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1117, 18), str_35399, 'format')
    # Calling format(args, kwargs) (line 1117)
    format_call_result_35409 = invoke(stypy.reporting.localization.Localization(__file__, 1117, 18), format_35400, *[iteration_35401, subscript_call_result_35406, max_rms_res_35407], **kwargs_35408)
    
    # Processing the call keyword arguments (line 1117)
    kwargs_35410 = {}
    # Getting the type of 'print' (line 1117)
    print_35398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 12), 'print', False)
    # Calling print(args, kwargs) (line 1117)
    print_call_result_35411 = invoke(stypy.reporting.localization.Localization(__file__, 1117, 12), print_35398, *[format_call_result_35409], **kwargs_35410)
    
    # SSA branch for the else part of an if statement (line 1116)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'status' (line 1120)
    status_35412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1120, 13), 'status')
    int_35413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1120, 23), 'int')
    # Applying the binary operator '==' (line 1120)
    result_eq_35414 = python_operator(stypy.reporting.localization.Localization(__file__, 1120, 13), '==', status_35412, int_35413)
    
    # Testing the type of an if condition (line 1120)
    if_condition_35415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1120, 13), result_eq_35414)
    # Assigning a type to the variable 'if_condition_35415' (line 1120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1120, 13), 'if_condition_35415', if_condition_35415)
    # SSA begins for if statement (line 1120)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 1121)
    # Processing the call arguments (line 1121)
    
    # Call to format(...): (line 1121)
    # Processing the call arguments (line 1121)
    # Getting the type of 'iteration' (line 1123)
    iteration_35419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 26), 'iteration', False)
    # Getting the type of 'max_rms_res' (line 1123)
    max_rms_res_35420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 37), 'max_rms_res', False)
    # Processing the call keyword arguments (line 1121)
    kwargs_35421 = {}
    str_35417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1121, 18), 'str', 'Number of nodes is exceeded after iteration {}, maximum relative residual {:.2e}.')
    # Obtaining the member 'format' of a type (line 1121)
    format_35418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 18), str_35417, 'format')
    # Calling format(args, kwargs) (line 1121)
    format_call_result_35422 = invoke(stypy.reporting.localization.Localization(__file__, 1121, 18), format_35418, *[iteration_35419, max_rms_res_35420], **kwargs_35421)
    
    # Processing the call keyword arguments (line 1121)
    kwargs_35423 = {}
    # Getting the type of 'print' (line 1121)
    print_35416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 12), 'print', False)
    # Calling print(args, kwargs) (line 1121)
    print_call_result_35424 = invoke(stypy.reporting.localization.Localization(__file__, 1121, 12), print_35416, *[format_call_result_35422], **kwargs_35423)
    
    # SSA branch for the else part of an if statement (line 1120)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'status' (line 1124)
    status_35425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 13), 'status')
    int_35426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 23), 'int')
    # Applying the binary operator '==' (line 1124)
    result_eq_35427 = python_operator(stypy.reporting.localization.Localization(__file__, 1124, 13), '==', status_35425, int_35426)
    
    # Testing the type of an if condition (line 1124)
    if_condition_35428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1124, 13), result_eq_35427)
    # Assigning a type to the variable 'if_condition_35428' (line 1124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1124, 13), 'if_condition_35428', if_condition_35428)
    # SSA begins for if statement (line 1124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 1125)
    # Processing the call arguments (line 1125)
    
    # Call to format(...): (line 1125)
    # Processing the call arguments (line 1125)
    # Getting the type of 'iteration' (line 1127)
    iteration_35432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 26), 'iteration', False)
    # Getting the type of 'max_rms_res' (line 1127)
    max_rms_res_35433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 37), 'max_rms_res', False)
    # Processing the call keyword arguments (line 1125)
    kwargs_35434 = {}
    str_35430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 18), 'str', 'Singular Jacobian encountered when solving the collocation system on iteration {}, maximum relative residual {:.2e}.')
    # Obtaining the member 'format' of a type (line 1125)
    format_35431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1125, 18), str_35430, 'format')
    # Calling format(args, kwargs) (line 1125)
    format_call_result_35435 = invoke(stypy.reporting.localization.Localization(__file__, 1125, 18), format_35431, *[iteration_35432, max_rms_res_35433], **kwargs_35434)
    
    # Processing the call keyword arguments (line 1125)
    kwargs_35436 = {}
    # Getting the type of 'print' (line 1125)
    print_35429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 12), 'print', False)
    # Calling print(args, kwargs) (line 1125)
    print_call_result_35437 = invoke(stypy.reporting.localization.Localization(__file__, 1125, 12), print_35429, *[format_call_result_35435], **kwargs_35436)
    
    # SSA join for if statement (line 1124)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1120)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1116)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1115)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'p' (line 1129)
    p_35438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 7), 'p')
    # Obtaining the member 'size' of a type (line 1129)
    size_35439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1129, 7), p_35438, 'size')
    int_35440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1129, 17), 'int')
    # Applying the binary operator '==' (line 1129)
    result_eq_35441 = python_operator(stypy.reporting.localization.Localization(__file__, 1129, 7), '==', size_35439, int_35440)
    
    # Testing the type of an if condition (line 1129)
    if_condition_35442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1129, 4), result_eq_35441)
    # Assigning a type to the variable 'if_condition_35442' (line 1129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1129, 4), 'if_condition_35442', if_condition_35442)
    # SSA begins for if statement (line 1129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 1130):
    
    # Assigning a Name to a Name (line 1130):
    # Getting the type of 'None' (line 1130)
    None_35443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 12), 'None')
    # Assigning a type to the variable 'p' (line 1130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1130, 8), 'p', None_35443)
    # SSA join for if statement (line 1129)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to BVPResult(...): (line 1132)
    # Processing the call keyword arguments (line 1132)
    # Getting the type of 'sol' (line 1132)
    sol_35445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 25), 'sol', False)
    keyword_35446 = sol_35445
    # Getting the type of 'p' (line 1132)
    p_35447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 32), 'p', False)
    keyword_35448 = p_35447
    # Getting the type of 'x' (line 1132)
    x_35449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 37), 'x', False)
    keyword_35450 = x_35449
    # Getting the type of 'y' (line 1132)
    y_35451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 42), 'y', False)
    keyword_35452 = y_35451
    # Getting the type of 'f' (line 1132)
    f_35453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 48), 'f', False)
    keyword_35454 = f_35453
    # Getting the type of 'rms_res' (line 1132)
    rms_res_35455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 65), 'rms_res', False)
    keyword_35456 = rms_res_35455
    # Getting the type of 'iteration' (line 1133)
    iteration_35457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 27), 'iteration', False)
    keyword_35458 = iteration_35457
    # Getting the type of 'status' (line 1133)
    status_35459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 45), 'status', False)
    keyword_35460 = status_35459
    
    # Obtaining the type of the subscript
    # Getting the type of 'status' (line 1134)
    status_35461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 50), 'status', False)
    # Getting the type of 'TERMINATION_MESSAGES' (line 1134)
    TERMINATION_MESSAGES_35462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 29), 'TERMINATION_MESSAGES', False)
    # Obtaining the member '__getitem__' of a type (line 1134)
    getitem___35463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1134, 29), TERMINATION_MESSAGES_35462, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1134)
    subscript_call_result_35464 = invoke(stypy.reporting.localization.Localization(__file__, 1134, 29), getitem___35463, status_35461)
    
    keyword_35465 = subscript_call_result_35464
    
    # Getting the type of 'status' (line 1134)
    status_35466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 67), 'status', False)
    int_35467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1134, 77), 'int')
    # Applying the binary operator '==' (line 1134)
    result_eq_35468 = python_operator(stypy.reporting.localization.Localization(__file__, 1134, 67), '==', status_35466, int_35467)
    
    keyword_35469 = result_eq_35468
    kwargs_35470 = {'status': keyword_35460, 'niter': keyword_35458, 'success': keyword_35469, 'sol': keyword_35446, 'p': keyword_35448, 'rms_residuals': keyword_35456, 'x': keyword_35450, 'y': keyword_35452, 'yp': keyword_35454, 'message': keyword_35465}
    # Getting the type of 'BVPResult' (line 1132)
    BVPResult_35444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 11), 'BVPResult', False)
    # Calling BVPResult(args, kwargs) (line 1132)
    BVPResult_call_result_35471 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 11), BVPResult_35444, *[], **kwargs_35470)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 4), 'stypy_return_type', BVPResult_call_result_35471)
    
    # ################# End of 'solve_bvp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_bvp' in the type store
    # Getting the type of 'stypy_return_type' (line 714)
    stypy_return_type_35472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35472)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_bvp'
    return stypy_return_type_35472

# Assigning a type to the variable 'solve_bvp' (line 714)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 0), 'solve_bvp', solve_bvp)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
