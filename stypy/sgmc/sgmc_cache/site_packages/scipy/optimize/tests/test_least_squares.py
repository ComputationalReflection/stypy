
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division
2: 
3: from itertools import product
4: 
5: import numpy as np
6: from numpy.linalg import norm
7: from numpy.testing import (assert_, assert_allclose,
8:                            assert_equal)
9: from pytest import raises as assert_raises
10: from scipy._lib._numpy_compat import suppress_warnings
11: 
12: from scipy.sparse import issparse, lil_matrix
13: from scipy.sparse.linalg import aslinearoperator
14: 
15: from scipy.optimize import least_squares
16: from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
17: from scipy.optimize._lsq.common import EPS, make_strictly_feasible
18: 
19: 
20: def fun_trivial(x, a=0):
21:     return (x - a)**2 + 5.0
22: 
23: 
24: def jac_trivial(x, a=0.0):
25:     return 2 * (x - a)
26: 
27: 
28: def fun_2d_trivial(x):
29:     return np.array([x[0], x[1]])
30: 
31: 
32: def jac_2d_trivial(x):
33:     return np.identity(2)
34: 
35: 
36: def fun_rosenbrock(x):
37:     return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])
38: 
39: 
40: def jac_rosenbrock(x):
41:     return np.array([
42:         [-20 * x[0], 10],
43:         [-1, 0]
44:     ])
45: 
46: 
47: def jac_rosenbrock_bad_dim(x):
48:     return np.array([
49:         [-20 * x[0], 10],
50:         [-1, 0],
51:         [0.0, 0.0]
52:     ])
53: 
54: 
55: def fun_rosenbrock_cropped(x):
56:     return fun_rosenbrock(x)[0]
57: 
58: 
59: def jac_rosenbrock_cropped(x):
60:     return jac_rosenbrock(x)[0]
61: 
62: 
63: # When x is 1-d array, return is 2-d array.
64: def fun_wrong_dimensions(x):
65:     return np.array([x, x**2, x**3])
66: 
67: 
68: def jac_wrong_dimensions(x, a=0.0):
69:     return np.atleast_3d(jac_trivial(x, a=a))
70: 
71: 
72: def fun_bvp(x):
73:     n = int(np.sqrt(x.shape[0]))
74:     u = np.zeros((n + 2, n + 2))
75:     x = x.reshape((n, n))
76:     u[1:-1, 1:-1] = x
77:     y = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * x + x**3
78:     return y.ravel()
79: 
80: 
81: class BroydenTridiagonal(object):
82:     def __init__(self, n=100, mode='sparse'):
83:         np.random.seed(0)
84: 
85:         self.n = n
86: 
87:         self.x0 = -np.ones(n)
88:         self.lb = np.linspace(-2, -1.5, n)
89:         self.ub = np.linspace(-0.8, 0.0, n)
90: 
91:         self.lb += 0.1 * np.random.randn(n)
92:         self.ub += 0.1 * np.random.randn(n)
93: 
94:         self.x0 += 0.1 * np.random.randn(n)
95:         self.x0 = make_strictly_feasible(self.x0, self.lb, self.ub)
96: 
97:         if mode == 'sparse':
98:             self.sparsity = lil_matrix((n, n), dtype=int)
99:             i = np.arange(n)
100:             self.sparsity[i, i] = 1
101:             i = np.arange(1, n)
102:             self.sparsity[i, i - 1] = 1
103:             i = np.arange(n - 1)
104:             self.sparsity[i, i + 1] = 1
105: 
106:             self.jac = self._jac
107:         elif mode == 'operator':
108:             self.jac = lambda x: aslinearoperator(self._jac(x))
109:         elif mode == 'dense':
110:             self.sparsity = None
111:             self.jac = lambda x: self._jac(x).toarray()
112:         else:
113:             assert_(False)
114: 
115:     def fun(self, x):
116:         f = (3 - x) * x + 1
117:         f[1:] -= x[:-1]
118:         f[:-1] -= 2 * x[1:]
119:         return f
120: 
121:     def _jac(self, x):
122:         J = lil_matrix((self.n, self.n))
123:         i = np.arange(self.n)
124:         J[i, i] = 3 - 2 * x
125:         i = np.arange(1, self.n)
126:         J[i, i - 1] = -1
127:         i = np.arange(self.n - 1)
128:         J[i, i + 1] = -2
129:         return J
130: 
131: 
132: class ExponentialFittingProblem(object):
133:     '''Provide data and function for exponential fitting in the form
134:     y = a + exp(b * x) + noise.'''
135: 
136:     def __init__(self, a, b, noise, n_outliers=1, x_range=(-1, 1),
137:                  n_points=11, random_seed=None):
138:         np.random.seed(random_seed)
139:         self.m = n_points
140:         self.n = 2
141: 
142:         self.p0 = np.zeros(2)
143:         self.x = np.linspace(x_range[0], x_range[1], n_points)
144: 
145:         self.y = a + np.exp(b * self.x)
146:         self.y += noise * np.random.randn(self.m)
147: 
148:         outliers = np.random.randint(0, self.m, n_outliers)
149:         self.y[outliers] += 50 * noise * np.random.rand(n_outliers)
150: 
151:         self.p_opt = np.array([a, b])
152: 
153:     def fun(self, p):
154:         return p[0] + np.exp(p[1] * self.x) - self.y
155: 
156:     def jac(self, p):
157:         J = np.empty((self.m, self.n))
158:         J[:, 0] = 1
159:         J[:, 1] = self.x * np.exp(p[1] * self.x)
160:         return J
161: 
162: 
163: def cubic_soft_l1(z):
164:     rho = np.empty((3, z.size))
165: 
166:     t = 1 + z
167:     rho[0] = 3 * (t**(1/3) - 1)
168:     rho[1] = t ** (-2/3)
169:     rho[2] = -2/3 * t**(-5/3)
170: 
171:     return rho
172: 
173: 
174: LOSSES = list(IMPLEMENTED_LOSSES.keys()) + [cubic_soft_l1]
175: 
176: 
177: class BaseMixin(object):
178:     def test_basic(self):
179:         # Test that the basic calling sequence works.
180:         res = least_squares(fun_trivial, 2., method=self.method)
181:         assert_allclose(res.x, 0, atol=1e-4)
182:         assert_allclose(res.fun, fun_trivial(res.x))
183: 
184:     def test_args_kwargs(self):
185:         # Test that args and kwargs are passed correctly to the functions.
186:         a = 3.0
187:         for jac in ['2-point', '3-point', 'cs', jac_trivial]:
188:             with suppress_warnings() as sup:
189:                 sup.filter(UserWarning,
190:                            "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'")
191:                 res = least_squares(fun_trivial, 2.0, jac, args=(a,),
192:                                     method=self.method)
193:                 res1 = least_squares(fun_trivial, 2.0, jac, kwargs={'a': a},
194:                                     method=self.method)
195: 
196:             assert_allclose(res.x, a, rtol=1e-4)
197:             assert_allclose(res1.x, a, rtol=1e-4)
198: 
199:             assert_raises(TypeError, least_squares, fun_trivial, 2.0,
200:                           args=(3, 4,), method=self.method)
201:             assert_raises(TypeError, least_squares, fun_trivial, 2.0,
202:                           kwargs={'kaboom': 3}, method=self.method)
203: 
204:     def test_jac_options(self):
205:         for jac in ['2-point', '3-point', 'cs', jac_trivial]:
206:             with suppress_warnings() as sup:
207:                 sup.filter(UserWarning,
208:                            "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'")
209:                 res = least_squares(fun_trivial, 2.0, jac, method=self.method)
210:             assert_allclose(res.x, 0, atol=1e-4)
211: 
212:         assert_raises(ValueError, least_squares, fun_trivial, 2.0, jac='oops',
213:                       method=self.method)
214: 
215:     def test_nfev_options(self):
216:         for max_nfev in [None, 20]:
217:             res = least_squares(fun_trivial, 2.0, max_nfev=max_nfev,
218:                                 method=self.method)
219:             assert_allclose(res.x, 0, atol=1e-4)
220: 
221:     def test_x_scale_options(self):
222:         for x_scale in [1.0, np.array([0.5]), 'jac']:
223:             res = least_squares(fun_trivial, 2.0, x_scale=x_scale)
224:             assert_allclose(res.x, 0)
225:         assert_raises(ValueError, least_squares, fun_trivial,
226:                       2.0, x_scale='auto', method=self.method)
227:         assert_raises(ValueError, least_squares, fun_trivial,
228:                       2.0, x_scale=-1.0, method=self.method)
229:         assert_raises(ValueError, least_squares, fun_trivial,
230:                       2.0, x_scale=None, method=self.method)
231:         assert_raises(ValueError, least_squares, fun_trivial,
232:                       2.0, x_scale=1.0+2.0j, method=self.method)
233: 
234:     def test_diff_step(self):
235:         # res1 and res2 should be equivalent.
236:         # res2 and res3 should be different.
237:         res1 = least_squares(fun_trivial, 2.0, diff_step=1e-1,
238:                              method=self.method)
239:         res2 = least_squares(fun_trivial, 2.0, diff_step=-1e-1,
240:                              method=self.method)
241:         res3 = least_squares(fun_trivial, 2.0,
242:                              diff_step=None, method=self.method)
243:         assert_allclose(res1.x, 0, atol=1e-4)
244:         assert_allclose(res2.x, 0, atol=1e-4)
245:         assert_allclose(res3.x, 0, atol=1e-4)
246:         assert_equal(res1.x, res2.x)
247:         assert_equal(res1.nfev, res2.nfev)
248:         assert_(res2.nfev != res3.nfev)
249: 
250:     def test_incorrect_options_usage(self):
251:         assert_raises(TypeError, least_squares, fun_trivial, 2.0,
252:                       method=self.method, options={'no_such_option': 100})
253:         assert_raises(TypeError, least_squares, fun_trivial, 2.0,
254:                       method=self.method, options={'max_nfev': 100})
255: 
256:     def test_full_result(self):
257:         # MINPACK doesn't work very well with factor=100 on this problem,
258:         # thus using low 'atol'.
259:         res = least_squares(fun_trivial, 2.0, method=self.method)
260:         assert_allclose(res.x, 0, atol=1e-4)
261:         assert_allclose(res.cost, 12.5)
262:         assert_allclose(res.fun, 5)
263:         assert_allclose(res.jac, 0, atol=1e-4)
264:         assert_allclose(res.grad, 0, atol=1e-2)
265:         assert_allclose(res.optimality, 0, atol=1e-2)
266:         assert_equal(res.active_mask, 0)
267:         if self.method == 'lm':
268:             assert_(res.nfev < 30)
269:             assert_(res.njev is None)
270:         else:
271:             assert_(res.nfev < 10)
272:             assert_(res.njev < 10)
273:         assert_(res.status > 0)
274:         assert_(res.success)
275: 
276:     def test_full_result_single_fev(self):
277:         # MINPACK checks the number of nfev after the iteration,
278:         # so it's hard to tell what he is going to compute.
279:         if self.method == 'lm':
280:             return
281: 
282:         res = least_squares(fun_trivial, 2.0, method=self.method,
283:                             max_nfev=1)
284:         assert_equal(res.x, np.array([2]))
285:         assert_equal(res.cost, 40.5)
286:         assert_equal(res.fun, np.array([9]))
287:         assert_equal(res.jac, np.array([[4]]))
288:         assert_equal(res.grad, np.array([36]))
289:         assert_equal(res.optimality, 36)
290:         assert_equal(res.active_mask, np.array([0]))
291:         assert_equal(res.nfev, 1)
292:         assert_equal(res.njev, 1)
293:         assert_equal(res.status, 0)
294:         assert_equal(res.success, 0)
295: 
296:     def test_rosenbrock(self):
297:         x0 = [-2, 1]
298:         x_opt = [1, 1]
299:         for jac, x_scale, tr_solver in product(
300:                 ['2-point', '3-point', 'cs', jac_rosenbrock],
301:                 [1.0, np.array([1.0, 0.2]), 'jac'],
302:                 ['exact', 'lsmr']):
303:             with suppress_warnings() as sup:
304:                 sup.filter(UserWarning,
305:                            "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'")
306:                 res = least_squares(fun_rosenbrock, x0, jac, x_scale=x_scale,
307:                                     tr_solver=tr_solver, method=self.method)
308:             assert_allclose(res.x, x_opt)
309: 
310:     def test_rosenbrock_cropped(self):
311:         x0 = [-2, 1]
312:         if self.method == 'lm':
313:             assert_raises(ValueError, least_squares, fun_rosenbrock_cropped,
314:                           x0, method='lm')
315:         else:
316:             for jac, x_scale, tr_solver in product(
317:                     ['2-point', '3-point', 'cs', jac_rosenbrock_cropped],
318:                     [1.0, np.array([1.0, 0.2]), 'jac'],
319:                     ['exact', 'lsmr']):
320:                 res = least_squares(
321:                     fun_rosenbrock_cropped, x0, jac, x_scale=x_scale,
322:                     tr_solver=tr_solver, method=self.method)
323:                 assert_allclose(res.cost, 0, atol=1e-14)
324: 
325:     def test_fun_wrong_dimensions(self):
326:         assert_raises(ValueError, least_squares, fun_wrong_dimensions,
327:                       2.0, method=self.method)
328: 
329:     def test_jac_wrong_dimensions(self):
330:         assert_raises(ValueError, least_squares, fun_trivial,
331:                       2.0, jac_wrong_dimensions, method=self.method)
332: 
333:     def test_fun_and_jac_inconsistent_dimensions(self):
334:         x0 = [1, 2]
335:         assert_raises(ValueError, least_squares, fun_rosenbrock, x0,
336:                       jac_rosenbrock_bad_dim, method=self.method)
337: 
338:     def test_x0_multidimensional(self):
339:         x0 = np.ones(4).reshape(2, 2)
340:         assert_raises(ValueError, least_squares, fun_trivial, x0,
341:                       method=self.method)
342: 
343:     def test_x0_complex_scalar(self):
344:         x0 = 2.0 + 0.0*1j
345:         assert_raises(ValueError, least_squares, fun_trivial, x0,
346:                       method=self.method)
347: 
348:     def test_x0_complex_array(self):
349:         x0 = [1.0, 2.0 + 0.0*1j]
350:         assert_raises(ValueError, least_squares, fun_trivial, x0,
351:                       method=self.method)
352: 
353:     def test_bvp(self):
354:         # This test was introduced with fix #5556. It turned out that
355:         # dogbox solver had a bug with trust-region radius update, which
356:         # could block its progress and create an infinite loop. And this
357:         # discrete boundary value problem is the one which triggers it.
358:         n = 10
359:         x0 = np.ones(n**2)
360:         if self.method == 'lm':
361:             max_nfev = 5000  # To account for Jacobian estimation.
362:         else:
363:             max_nfev = 100
364:         res = least_squares(fun_bvp, x0, ftol=1e-2, method=self.method,
365:                             max_nfev=max_nfev)
366: 
367:         assert_(res.nfev < max_nfev)
368:         assert_(res.cost < 0.5)
369: 
370: 
371: class BoundsMixin(object):
372:     def test_inconsistent(self):
373:         assert_raises(ValueError, least_squares, fun_trivial, 2.0,
374:                       bounds=(10.0, 0.0), method=self.method)
375: 
376:     def test_infeasible(self):
377:         assert_raises(ValueError, least_squares, fun_trivial, 2.0,
378:                       bounds=(3., 4), method=self.method)
379: 
380:     def test_wrong_number(self):
381:         assert_raises(ValueError, least_squares, fun_trivial, 2.,
382:                       bounds=(1., 2, 3), method=self.method)
383: 
384:     def test_inconsistent_shape(self):
385:         assert_raises(ValueError, least_squares, fun_trivial, 2.0,
386:                       bounds=(1.0, [2.0, 3.0]), method=self.method)
387:         # 1-D array wont't be broadcasted
388:         assert_raises(ValueError, least_squares, fun_rosenbrock, [1.0, 2.0],
389:                       bounds=([0.0], [3.0, 4.0]), method=self.method)
390: 
391:     def test_in_bounds(self):
392:         for jac in ['2-point', '3-point', 'cs', jac_trivial]:
393:             res = least_squares(fun_trivial, 2.0, jac=jac,
394:                                 bounds=(-1.0, 3.0), method=self.method)
395:             assert_allclose(res.x, 0.0, atol=1e-4)
396:             assert_equal(res.active_mask, [0])
397:             assert_(-1 <= res.x <= 3)
398:             res = least_squares(fun_trivial, 2.0, jac=jac,
399:                                 bounds=(0.5, 3.0), method=self.method)
400:             assert_allclose(res.x, 0.5, atol=1e-4)
401:             assert_equal(res.active_mask, [-1])
402:             assert_(0.5 <= res.x <= 3)
403: 
404:     def test_bounds_shape(self):
405:         for jac in ['2-point', '3-point', 'cs', jac_2d_trivial]:
406:             x0 = [1.0, 1.0]
407:             res = least_squares(fun_2d_trivial, x0, jac=jac)
408:             assert_allclose(res.x, [0.0, 0.0])
409:             res = least_squares(fun_2d_trivial, x0, jac=jac,
410:                                 bounds=(0.5, [2.0, 2.0]), method=self.method)
411:             assert_allclose(res.x, [0.5, 0.5])
412:             res = least_squares(fun_2d_trivial, x0, jac=jac,
413:                                 bounds=([0.3, 0.2], 3.0), method=self.method)
414:             assert_allclose(res.x, [0.3, 0.2])
415:             res = least_squares(
416:                 fun_2d_trivial, x0, jac=jac, bounds=([-1, 0.5], [1.0, 3.0]),
417:                 method=self.method)
418:             assert_allclose(res.x, [0.0, 0.5], atol=1e-5)
419: 
420:     def test_rosenbrock_bounds(self):
421:         x0_1 = np.array([-2.0, 1.0])
422:         x0_2 = np.array([2.0, 2.0])
423:         x0_3 = np.array([-2.0, 2.0])
424:         x0_4 = np.array([0.0, 2.0])
425:         x0_5 = np.array([-1.2, 1.0])
426:         problems = [
427:             (x0_1, ([-np.inf, -1.5], np.inf)),
428:             (x0_2, ([-np.inf, 1.5], np.inf)),
429:             (x0_3, ([-np.inf, 1.5], np.inf)),
430:             (x0_4, ([-np.inf, 1.5], [1.0, np.inf])),
431:             (x0_2, ([1.0, 1.5], [3.0, 3.0])),
432:             (x0_5, ([-50.0, 0.0], [0.5, 100]))
433:         ]
434:         for x0, bounds in problems:
435:             for jac, x_scale, tr_solver in product(
436:                     ['2-point', '3-point', 'cs', jac_rosenbrock],
437:                     [1.0, [1.0, 0.5], 'jac'],
438:                     ['exact', 'lsmr']):
439:                 res = least_squares(fun_rosenbrock, x0, jac, bounds,
440:                                     x_scale=x_scale, tr_solver=tr_solver,
441:                                     method=self.method)
442:                 assert_allclose(res.optimality, 0.0, atol=1e-5)
443: 
444: 
445: class SparseMixin(object):
446:     def test_exact_tr_solver(self):
447:         p = BroydenTridiagonal()
448:         assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
449:                       tr_solver='exact', method=self.method)
450:         assert_raises(ValueError, least_squares, p.fun, p.x0,
451:                       tr_solver='exact', jac_sparsity=p.sparsity,
452:                       method=self.method)
453: 
454:     def test_equivalence(self):
455:         sparse = BroydenTridiagonal(mode='sparse')
456:         dense = BroydenTridiagonal(mode='dense')
457:         res_sparse = least_squares(
458:             sparse.fun, sparse.x0, jac=sparse.jac,
459:             method=self.method)
460:         res_dense = least_squares(
461:             dense.fun, dense.x0, jac=sparse.jac,
462:             method=self.method)
463:         assert_equal(res_sparse.nfev, res_dense.nfev)
464:         assert_allclose(res_sparse.x, res_dense.x, atol=1e-20)
465:         assert_allclose(res_sparse.cost, 0, atol=1e-20)
466:         assert_allclose(res_dense.cost, 0, atol=1e-20)
467: 
468:     def test_tr_options(self):
469:         p = BroydenTridiagonal()
470:         res = least_squares(p.fun, p.x0, p.jac, method=self.method,
471:                             tr_options={'btol': 1e-10})
472:         assert_allclose(res.cost, 0, atol=1e-20)
473: 
474:     def test_wrong_parameters(self):
475:         p = BroydenTridiagonal()
476:         assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
477:                       tr_solver='best', method=self.method)
478:         assert_raises(TypeError, least_squares, p.fun, p.x0, p.jac,
479:                       tr_solver='lsmr', tr_options={'tol': 1e-10})
480: 
481:     def test_solver_selection(self):
482:         sparse = BroydenTridiagonal(mode='sparse')
483:         dense = BroydenTridiagonal(mode='dense')
484:         res_sparse = least_squares(sparse.fun, sparse.x0, jac=sparse.jac,
485:                                    method=self.method)
486:         res_dense = least_squares(dense.fun, dense.x0, jac=dense.jac,
487:                                   method=self.method)
488:         assert_allclose(res_sparse.cost, 0, atol=1e-20)
489:         assert_allclose(res_dense.cost, 0, atol=1e-20)
490:         assert_(issparse(res_sparse.jac))
491:         assert_(isinstance(res_dense.jac, np.ndarray))
492: 
493:     def test_numerical_jac(self):
494:         p = BroydenTridiagonal()
495:         for jac in ['2-point', '3-point', 'cs']:
496:             res_dense = least_squares(p.fun, p.x0, jac, method=self.method)
497:             res_sparse = least_squares(
498:                 p.fun, p.x0, jac,method=self.method,
499:                 jac_sparsity=p.sparsity)
500:             assert_equal(res_dense.nfev, res_sparse.nfev)
501:             assert_allclose(res_dense.x, res_sparse.x, atol=1e-20)
502:             assert_allclose(res_dense.cost, 0, atol=1e-20)
503:             assert_allclose(res_sparse.cost, 0, atol=1e-20)
504: 
505:     def test_with_bounds(self):
506:         p = BroydenTridiagonal()
507:         for jac, jac_sparsity in product(
508:                 [p.jac, '2-point', '3-point', 'cs'], [None, p.sparsity]):
509:             res_1 = least_squares(
510:                 p.fun, p.x0, jac, bounds=(p.lb, np.inf),
511:                 method=self.method,jac_sparsity=jac_sparsity)
512:             res_2 = least_squares(
513:                 p.fun, p.x0, jac, bounds=(-np.inf, p.ub),
514:                 method=self.method, jac_sparsity=jac_sparsity)
515:             res_3 = least_squares(
516:                 p.fun, p.x0, jac, bounds=(p.lb, p.ub),
517:                 method=self.method, jac_sparsity=jac_sparsity)
518:             assert_allclose(res_1.optimality, 0, atol=1e-10)
519:             assert_allclose(res_2.optimality, 0, atol=1e-10)
520:             assert_allclose(res_3.optimality, 0, atol=1e-10)
521: 
522:     def test_wrong_jac_sparsity(self):
523:         p = BroydenTridiagonal()
524:         sparsity = p.sparsity[:-1]
525:         assert_raises(ValueError, least_squares, p.fun, p.x0,
526:                       jac_sparsity=sparsity, method=self.method)
527: 
528:     def test_linear_operator(self):
529:         p = BroydenTridiagonal(mode='operator')
530:         res = least_squares(p.fun, p.x0, p.jac, method=self.method)
531:         assert_allclose(res.cost, 0.0, atol=1e-20)
532:         assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
533:                       method=self.method, tr_solver='exact')
534: 
535:     def test_x_scale_jac_scale(self):
536:         p = BroydenTridiagonal()
537:         res = least_squares(p.fun, p.x0, p.jac, method=self.method,
538:                             x_scale='jac')
539:         assert_allclose(res.cost, 0.0, atol=1e-20)
540: 
541:         p = BroydenTridiagonal(mode='operator')
542:         assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
543:                       method=self.method, x_scale='jac')
544: 
545: 
546: class LossFunctionMixin(object):
547:     def test_options(self):
548:         for loss in LOSSES:
549:             res = least_squares(fun_trivial, 2.0, loss=loss,
550:                                 method=self.method)
551:             assert_allclose(res.x, 0, atol=1e-15)
552: 
553:         assert_raises(ValueError, least_squares, fun_trivial, 2.0,
554:                       loss='hinge', method=self.method)
555: 
556:     def test_fun(self):
557:         # Test that res.fun is actual residuals, and not modified by loss
558:         # function stuff.
559:         for loss in LOSSES:
560:             res = least_squares(fun_trivial, 2.0, loss=loss,
561:                                 method=self.method)
562:             assert_equal(res.fun, fun_trivial(res.x))
563: 
564:     def test_grad(self):
565:         # Test that res.grad is true gradient of loss function at the
566:         # solution. Use max_nfev = 1, to avoid reaching minimum.
567:         x = np.array([2.0])  # res.x will be this.
568: 
569:         res = least_squares(fun_trivial, x, jac_trivial, loss='linear',
570:                             max_nfev=1, method=self.method)
571:         assert_equal(res.grad, 2 * x * (x**2 + 5))
572: 
573:         res = least_squares(fun_trivial, x, jac_trivial, loss='huber',
574:                             max_nfev=1, method=self.method)
575:         assert_equal(res.grad, 2 * x)
576: 
577:         res = least_squares(fun_trivial, x, jac_trivial, loss='soft_l1',
578:                             max_nfev=1, method=self.method)
579:         assert_allclose(res.grad,
580:                         2 * x * (x**2 + 5) / (1 + (x**2 + 5)**2)**0.5)
581: 
582:         res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy',
583:                             max_nfev=1, method=self.method)
584:         assert_allclose(res.grad, 2 * x * (x**2 + 5) / (1 + (x**2 + 5)**2))
585: 
586:         res = least_squares(fun_trivial, x, jac_trivial, loss='arctan',
587:                             max_nfev=1, method=self.method)
588:         assert_allclose(res.grad, 2 * x * (x**2 + 5) / (1 + (x**2 + 5)**4))
589: 
590:         res = least_squares(fun_trivial, x, jac_trivial, loss=cubic_soft_l1,
591:                             max_nfev=1, method=self.method)
592:         assert_allclose(res.grad,
593:                         2 * x * (x**2 + 5) / (1 + (x**2 + 5)**2)**(2/3))
594: 
595:     def test_jac(self):
596:         # Test that res.jac.T.dot(res.jac) gives Gauss-Newton approximation
597:         # of Hessian. This approximation is computed by doubly differentiating
598:         # the cost function and dropping the part containing second derivative
599:         # of f. For a scalar function it is computed as
600:         # H = (rho' + 2 * rho'' * f**2) * f'**2, if the expression inside the
601:         # brackets is less than EPS it is replaced by EPS. Here we check
602:         # against the root of H.
603: 
604:         x = 2.0  # res.x will be this.
605:         f = x**2 + 5  # res.fun will be this.
606: 
607:         res = least_squares(fun_trivial, x, jac_trivial, loss='linear',
608:                             max_nfev=1, method=self.method)
609:         assert_equal(res.jac, 2 * x)
610: 
611:         # For `huber` loss the Jacobian correction is identically zero
612:         # in outlier region, in such cases it is modified to be equal EPS**0.5.
613:         res = least_squares(fun_trivial, x, jac_trivial, loss='huber',
614:                             max_nfev=1, method=self.method)
615:         assert_equal(res.jac, 2 * x * EPS**0.5)
616: 
617:         # Now let's apply `loss_scale` to turn the residual into an inlier.
618:         # The loss function becomes linear.
619:         res = least_squares(fun_trivial, x, jac_trivial, loss='huber',
620:                             f_scale=10, max_nfev=1)
621:         assert_equal(res.jac, 2 * x)
622: 
623:         # 'soft_l1' always gives a positive scaling.
624:         res = least_squares(fun_trivial, x, jac_trivial, loss='soft_l1',
625:                             max_nfev=1, method=self.method)
626:         assert_allclose(res.jac, 2 * x * (1 + f**2)**-0.75)
627: 
628:         # For 'cauchy' the correction term turns out to be negative, and it
629:         # replaced by EPS**0.5.
630:         res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy',
631:                             max_nfev=1, method=self.method)
632:         assert_allclose(res.jac, 2 * x * EPS**0.5)
633: 
634:         # Now use scaling to turn the residual to inlier.
635:         res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy',
636:                             f_scale=10, max_nfev=1, method=self.method)
637:         fs = f / 10
638:         assert_allclose(res.jac, 2 * x * (1 - fs**2)**0.5 / (1 + fs**2))
639: 
640:         # 'arctan' gives an outlier.
641:         res = least_squares(fun_trivial, x, jac_trivial, loss='arctan',
642:                             max_nfev=1, method=self.method)
643:         assert_allclose(res.jac, 2 * x * EPS**0.5)
644: 
645:         # Turn to inlier.
646:         res = least_squares(fun_trivial, x, jac_trivial, loss='arctan',
647:                             f_scale=20.0, max_nfev=1, method=self.method)
648:         fs = f / 20
649:         assert_allclose(res.jac, 2 * x * (1 - 3 * fs**4)**0.5 / (1 + fs**4))
650: 
651:         # cubic_soft_l1 will give an outlier.
652:         res = least_squares(fun_trivial, x, jac_trivial, loss=cubic_soft_l1,
653:                             max_nfev=1)
654:         assert_allclose(res.jac, 2 * x * EPS**0.5)
655: 
656:         # Turn to inlier.
657:         res = least_squares(fun_trivial, x, jac_trivial,
658:                             loss=cubic_soft_l1, f_scale=6, max_nfev=1)
659:         fs = f / 6
660:         assert_allclose(res.jac,
661:                         2 * x * (1 - fs**2 / 3)**0.5 * (1 + fs**2)**(-5/6))
662: 
663:     def test_robustness(self):
664:         for noise in [0.1, 1.0]:
665:             p = ExponentialFittingProblem(1, 0.1, noise, random_seed=0)
666: 
667:             for jac in ['2-point', '3-point', 'cs', p.jac]:
668:                 res_lsq = least_squares(p.fun, p.p0, jac=jac,
669:                                         method=self.method)
670:                 assert_allclose(res_lsq.optimality, 0, atol=1e-2)
671:                 for loss in LOSSES:
672:                     if loss == 'linear':
673:                         continue
674:                     res_robust = least_squares(
675:                         p.fun, p.p0, jac=jac, loss=loss, f_scale=noise,
676:                         method=self.method)
677:                     assert_allclose(res_robust.optimality, 0, atol=1e-2)
678:                     assert_(norm(res_robust.x - p.p_opt) <
679:                             norm(res_lsq.x - p.p_opt))
680: 
681: 
682: class TestDogbox(BaseMixin, BoundsMixin, SparseMixin, LossFunctionMixin):
683:     method = 'dogbox'
684: 
685: 
686: class TestTRF(BaseMixin, BoundsMixin, SparseMixin, LossFunctionMixin):
687:     method = 'trf'
688: 
689:     def test_lsmr_regularization(self):
690:         p = BroydenTridiagonal()
691:         for regularize in [True, False]:
692:             res = least_squares(p.fun, p.x0, p.jac, method='trf',
693:                                 tr_options={'regularize': regularize})
694:             assert_allclose(res.cost, 0, atol=1e-20)
695: 
696: 
697: class TestLM(BaseMixin):
698:     method = 'lm'
699: 
700:     def test_bounds_not_supported(self):
701:         assert_raises(ValueError, least_squares, fun_trivial,
702:                       2.0, bounds=(-3.0, 3.0), method='lm')
703: 
704:     def test_m_less_n_not_supported(self):
705:         x0 = [-2, 1]
706:         assert_raises(ValueError, least_squares, fun_rosenbrock_cropped, x0,
707:                       method='lm')
708: 
709:     def test_sparse_not_supported(self):
710:         p = BroydenTridiagonal()
711:         assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
712:                       method='lm')
713: 
714:     def test_jac_sparsity_not_supported(self):
715:         assert_raises(ValueError, least_squares, fun_trivial, 2.0,
716:                       jac_sparsity=[1], method='lm')
717: 
718:     def test_LinearOperator_not_supported(self):
719:         p = BroydenTridiagonal(mode="operator")
720:         assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
721:                       method='lm')
722: 
723:     def test_loss(self):
724:         res = least_squares(fun_trivial, 2.0, loss='linear', method='lm')
725:         assert_allclose(res.x, 0.0, atol=1e-4)
726: 
727:         assert_raises(ValueError, least_squares, fun_trivial, 2.0,
728:                       method='lm', loss='huber')
729: 
730: 
731: def test_basic():
732:     # test that 'method' arg is really optional
733:     res = least_squares(fun_trivial, 2.0)
734:     assert_allclose(res.x, 0, atol=1e-10)
735: 
736: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from itertools import product' statement (line 3)
try:
    from itertools import product

except:
    product = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'itertools', None, module_type_store, ['product'], [product])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205781 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_205781) is not StypyTypeError):

    if (import_205781 != 'pyd_module'):
        __import__(import_205781)
        sys_modules_205782 = sys.modules[import_205781]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_205782.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_205781)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.linalg import norm' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205783 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.linalg')

if (type(import_205783) is not StypyTypeError):

    if (import_205783 != 'pyd_module'):
        __import__(import_205783)
        sys_modules_205784 = sys.modules[import_205783]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.linalg', sys_modules_205784.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_205784, sys_modules_205784.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.linalg', import_205783)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_, assert_allclose, assert_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205785 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_205785) is not StypyTypeError):

    if (import_205785 != 'pyd_module'):
        __import__(import_205785)
        sys_modules_205786 = sys.modules[import_205785]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_205786.module_type_store, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_205786, sys_modules_205786.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'], [assert_, assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_205785)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from pytest import assert_raises' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205787 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest')

if (type(import_205787) is not StypyTypeError):

    if (import_205787 != 'pyd_module'):
        __import__(import_205787)
        sys_modules_205788 = sys.modules[import_205787]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', sys_modules_205788.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_205788, sys_modules_205788.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', import_205787)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205789 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._numpy_compat')

if (type(import_205789) is not StypyTypeError):

    if (import_205789 != 'pyd_module'):
        __import__(import_205789)
        sys_modules_205790 = sys.modules[import_205789]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._numpy_compat', sys_modules_205790.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_205790, sys_modules_205790.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._numpy_compat', import_205789)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.sparse import issparse, lil_matrix' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205791 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse')

if (type(import_205791) is not StypyTypeError):

    if (import_205791 != 'pyd_module'):
        __import__(import_205791)
        sys_modules_205792 = sys.modules[import_205791]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse', sys_modules_205792.module_type_store, module_type_store, ['issparse', 'lil_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_205792, sys_modules_205792.module_type_store, module_type_store)
    else:
        from scipy.sparse import issparse, lil_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse', None, module_type_store, ['issparse', 'lil_matrix'], [issparse, lil_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse', import_205791)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.sparse.linalg import aslinearoperator' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205793 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.linalg')

if (type(import_205793) is not StypyTypeError):

    if (import_205793 != 'pyd_module'):
        __import__(import_205793)
        sys_modules_205794 = sys.modules[import_205793]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.linalg', sys_modules_205794.module_type_store, module_type_store, ['aslinearoperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_205794, sys_modules_205794.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import aslinearoperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.linalg', None, module_type_store, ['aslinearoperator'], [aslinearoperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.linalg', import_205793)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.optimize import least_squares' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205795 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.optimize')

if (type(import_205795) is not StypyTypeError):

    if (import_205795 != 'pyd_module'):
        __import__(import_205795)
        sys_modules_205796 = sys.modules[import_205795]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.optimize', sys_modules_205796.module_type_store, module_type_store, ['least_squares'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_205796, sys_modules_205796.module_type_store, module_type_store)
    else:
        from scipy.optimize import least_squares

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.optimize', None, module_type_store, ['least_squares'], [least_squares])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.optimize', import_205795)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205797 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize._lsq.least_squares')

if (type(import_205797) is not StypyTypeError):

    if (import_205797 != 'pyd_module'):
        __import__(import_205797)
        sys_modules_205798 = sys.modules[import_205797]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize._lsq.least_squares', sys_modules_205798.module_type_store, module_type_store, ['IMPLEMENTED_LOSSES'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_205798, sys_modules_205798.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize._lsq.least_squares', None, module_type_store, ['IMPLEMENTED_LOSSES'], [IMPLEMENTED_LOSSES])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.least_squares' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize._lsq.least_squares', import_205797)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.optimize._lsq.common import EPS, make_strictly_feasible' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205799 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize._lsq.common')

if (type(import_205799) is not StypyTypeError):

    if (import_205799 != 'pyd_module'):
        __import__(import_205799)
        sys_modules_205800 = sys.modules[import_205799]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize._lsq.common', sys_modules_205800.module_type_store, module_type_store, ['EPS', 'make_strictly_feasible'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_205800, sys_modules_205800.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.common import EPS, make_strictly_feasible

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize._lsq.common', None, module_type_store, ['EPS', 'make_strictly_feasible'], [EPS, make_strictly_feasible])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.common' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize._lsq.common', import_205799)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')


@norecursion
def fun_trivial(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_205801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'int')
    defaults = [int_205801]
    # Create a new context for function 'fun_trivial'
    module_type_store = module_type_store.open_function_context('fun_trivial', 20, 0, False)
    
    # Passed parameters checking function
    fun_trivial.stypy_localization = localization
    fun_trivial.stypy_type_of_self = None
    fun_trivial.stypy_type_store = module_type_store
    fun_trivial.stypy_function_name = 'fun_trivial'
    fun_trivial.stypy_param_names_list = ['x', 'a']
    fun_trivial.stypy_varargs_param_name = None
    fun_trivial.stypy_kwargs_param_name = None
    fun_trivial.stypy_call_defaults = defaults
    fun_trivial.stypy_call_varargs = varargs
    fun_trivial.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fun_trivial', ['x', 'a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fun_trivial', localization, ['x', 'a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fun_trivial(...)' code ##################

    # Getting the type of 'x' (line 21)
    x_205802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'x')
    # Getting the type of 'a' (line 21)
    a_205803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'a')
    # Applying the binary operator '-' (line 21)
    result_sub_205804 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 12), '-', x_205802, a_205803)
    
    int_205805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'int')
    # Applying the binary operator '**' (line 21)
    result_pow_205806 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 11), '**', result_sub_205804, int_205805)
    
    float_205807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'float')
    # Applying the binary operator '+' (line 21)
    result_add_205808 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 11), '+', result_pow_205806, float_205807)
    
    # Assigning a type to the variable 'stypy_return_type' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type', result_add_205808)
    
    # ################# End of 'fun_trivial(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fun_trivial' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_205809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205809)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fun_trivial'
    return stypy_return_type_205809

# Assigning a type to the variable 'fun_trivial' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'fun_trivial', fun_trivial)

@norecursion
def jac_trivial(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_205810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'float')
    defaults = [float_205810]
    # Create a new context for function 'jac_trivial'
    module_type_store = module_type_store.open_function_context('jac_trivial', 24, 0, False)
    
    # Passed parameters checking function
    jac_trivial.stypy_localization = localization
    jac_trivial.stypy_type_of_self = None
    jac_trivial.stypy_type_store = module_type_store
    jac_trivial.stypy_function_name = 'jac_trivial'
    jac_trivial.stypy_param_names_list = ['x', 'a']
    jac_trivial.stypy_varargs_param_name = None
    jac_trivial.stypy_kwargs_param_name = None
    jac_trivial.stypy_call_defaults = defaults
    jac_trivial.stypy_call_varargs = varargs
    jac_trivial.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'jac_trivial', ['x', 'a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'jac_trivial', localization, ['x', 'a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'jac_trivial(...)' code ##################

    int_205811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'int')
    # Getting the type of 'x' (line 25)
    x_205812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'x')
    # Getting the type of 'a' (line 25)
    a_205813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'a')
    # Applying the binary operator '-' (line 25)
    result_sub_205814 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 16), '-', x_205812, a_205813)
    
    # Applying the binary operator '*' (line 25)
    result_mul_205815 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 11), '*', int_205811, result_sub_205814)
    
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type', result_mul_205815)
    
    # ################# End of 'jac_trivial(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'jac_trivial' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_205816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205816)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'jac_trivial'
    return stypy_return_type_205816

# Assigning a type to the variable 'jac_trivial' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'jac_trivial', jac_trivial)

@norecursion
def fun_2d_trivial(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fun_2d_trivial'
    module_type_store = module_type_store.open_function_context('fun_2d_trivial', 28, 0, False)
    
    # Passed parameters checking function
    fun_2d_trivial.stypy_localization = localization
    fun_2d_trivial.stypy_type_of_self = None
    fun_2d_trivial.stypy_type_store = module_type_store
    fun_2d_trivial.stypy_function_name = 'fun_2d_trivial'
    fun_2d_trivial.stypy_param_names_list = ['x']
    fun_2d_trivial.stypy_varargs_param_name = None
    fun_2d_trivial.stypy_kwargs_param_name = None
    fun_2d_trivial.stypy_call_defaults = defaults
    fun_2d_trivial.stypy_call_varargs = varargs
    fun_2d_trivial.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fun_2d_trivial', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fun_2d_trivial', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fun_2d_trivial(...)' code ##################

    
    # Call to array(...): (line 29)
    # Processing the call arguments (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_205819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    
    # Obtaining the type of the subscript
    int_205820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 23), 'int')
    # Getting the type of 'x' (line 29)
    x_205821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 21), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___205822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 21), x_205821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_205823 = invoke(stypy.reporting.localization.Localization(__file__, 29, 21), getitem___205822, int_205820)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 20), list_205819, subscript_call_result_205823)
    # Adding element type (line 29)
    
    # Obtaining the type of the subscript
    int_205824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'int')
    # Getting the type of 'x' (line 29)
    x_205825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 27), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___205826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 27), x_205825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_205827 = invoke(stypy.reporting.localization.Localization(__file__, 29, 27), getitem___205826, int_205824)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 20), list_205819, subscript_call_result_205827)
    
    # Processing the call keyword arguments (line 29)
    kwargs_205828 = {}
    # Getting the type of 'np' (line 29)
    np_205817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 29)
    array_205818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 11), np_205817, 'array')
    # Calling array(args, kwargs) (line 29)
    array_call_result_205829 = invoke(stypy.reporting.localization.Localization(__file__, 29, 11), array_205818, *[list_205819], **kwargs_205828)
    
    # Assigning a type to the variable 'stypy_return_type' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type', array_call_result_205829)
    
    # ################# End of 'fun_2d_trivial(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fun_2d_trivial' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_205830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205830)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fun_2d_trivial'
    return stypy_return_type_205830

# Assigning a type to the variable 'fun_2d_trivial' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'fun_2d_trivial', fun_2d_trivial)

@norecursion
def jac_2d_trivial(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'jac_2d_trivial'
    module_type_store = module_type_store.open_function_context('jac_2d_trivial', 32, 0, False)
    
    # Passed parameters checking function
    jac_2d_trivial.stypy_localization = localization
    jac_2d_trivial.stypy_type_of_self = None
    jac_2d_trivial.stypy_type_store = module_type_store
    jac_2d_trivial.stypy_function_name = 'jac_2d_trivial'
    jac_2d_trivial.stypy_param_names_list = ['x']
    jac_2d_trivial.stypy_varargs_param_name = None
    jac_2d_trivial.stypy_kwargs_param_name = None
    jac_2d_trivial.stypy_call_defaults = defaults
    jac_2d_trivial.stypy_call_varargs = varargs
    jac_2d_trivial.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'jac_2d_trivial', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'jac_2d_trivial', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'jac_2d_trivial(...)' code ##################

    
    # Call to identity(...): (line 33)
    # Processing the call arguments (line 33)
    int_205833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'int')
    # Processing the call keyword arguments (line 33)
    kwargs_205834 = {}
    # Getting the type of 'np' (line 33)
    np_205831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'np', False)
    # Obtaining the member 'identity' of a type (line 33)
    identity_205832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 11), np_205831, 'identity')
    # Calling identity(args, kwargs) (line 33)
    identity_call_result_205835 = invoke(stypy.reporting.localization.Localization(__file__, 33, 11), identity_205832, *[int_205833], **kwargs_205834)
    
    # Assigning a type to the variable 'stypy_return_type' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type', identity_call_result_205835)
    
    # ################# End of 'jac_2d_trivial(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'jac_2d_trivial' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_205836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205836)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'jac_2d_trivial'
    return stypy_return_type_205836

# Assigning a type to the variable 'jac_2d_trivial' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'jac_2d_trivial', jac_2d_trivial)

@norecursion
def fun_rosenbrock(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fun_rosenbrock'
    module_type_store = module_type_store.open_function_context('fun_rosenbrock', 36, 0, False)
    
    # Passed parameters checking function
    fun_rosenbrock.stypy_localization = localization
    fun_rosenbrock.stypy_type_of_self = None
    fun_rosenbrock.stypy_type_store = module_type_store
    fun_rosenbrock.stypy_function_name = 'fun_rosenbrock'
    fun_rosenbrock.stypy_param_names_list = ['x']
    fun_rosenbrock.stypy_varargs_param_name = None
    fun_rosenbrock.stypy_kwargs_param_name = None
    fun_rosenbrock.stypy_call_defaults = defaults
    fun_rosenbrock.stypy_call_varargs = varargs
    fun_rosenbrock.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fun_rosenbrock', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fun_rosenbrock', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fun_rosenbrock(...)' code ##################

    
    # Call to array(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_205839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    int_205840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'int')
    
    # Obtaining the type of the subscript
    int_205841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 29), 'int')
    # Getting the type of 'x' (line 37)
    x_205842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___205843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 27), x_205842, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_205844 = invoke(stypy.reporting.localization.Localization(__file__, 37, 27), getitem___205843, int_205841)
    
    
    # Obtaining the type of the subscript
    int_205845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 36), 'int')
    # Getting the type of 'x' (line 37)
    x_205846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 34), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___205847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 34), x_205846, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_205848 = invoke(stypy.reporting.localization.Localization(__file__, 37, 34), getitem___205847, int_205845)
    
    int_205849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 40), 'int')
    # Applying the binary operator '**' (line 37)
    result_pow_205850 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 34), '**', subscript_call_result_205848, int_205849)
    
    # Applying the binary operator '-' (line 37)
    result_sub_205851 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 27), '-', subscript_call_result_205844, result_pow_205850)
    
    # Applying the binary operator '*' (line 37)
    result_mul_205852 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 21), '*', int_205840, result_sub_205851)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 20), list_205839, result_mul_205852)
    # Adding element type (line 37)
    int_205853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 45), 'int')
    
    # Obtaining the type of the subscript
    int_205854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 51), 'int')
    # Getting the type of 'x' (line 37)
    x_205855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 49), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___205856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 49), x_205855, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_205857 = invoke(stypy.reporting.localization.Localization(__file__, 37, 49), getitem___205856, int_205854)
    
    # Applying the binary operator '-' (line 37)
    result_sub_205858 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 45), '-', int_205853, subscript_call_result_205857)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 20), list_205839, result_sub_205858)
    
    # Processing the call keyword arguments (line 37)
    kwargs_205859 = {}
    # Getting the type of 'np' (line 37)
    np_205837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 37)
    array_205838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 11), np_205837, 'array')
    # Calling array(args, kwargs) (line 37)
    array_call_result_205860 = invoke(stypy.reporting.localization.Localization(__file__, 37, 11), array_205838, *[list_205839], **kwargs_205859)
    
    # Assigning a type to the variable 'stypy_return_type' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type', array_call_result_205860)
    
    # ################# End of 'fun_rosenbrock(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fun_rosenbrock' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_205861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205861)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fun_rosenbrock'
    return stypy_return_type_205861

# Assigning a type to the variable 'fun_rosenbrock' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'fun_rosenbrock', fun_rosenbrock)

@norecursion
def jac_rosenbrock(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'jac_rosenbrock'
    module_type_store = module_type_store.open_function_context('jac_rosenbrock', 40, 0, False)
    
    # Passed parameters checking function
    jac_rosenbrock.stypy_localization = localization
    jac_rosenbrock.stypy_type_of_self = None
    jac_rosenbrock.stypy_type_store = module_type_store
    jac_rosenbrock.stypy_function_name = 'jac_rosenbrock'
    jac_rosenbrock.stypy_param_names_list = ['x']
    jac_rosenbrock.stypy_varargs_param_name = None
    jac_rosenbrock.stypy_kwargs_param_name = None
    jac_rosenbrock.stypy_call_defaults = defaults
    jac_rosenbrock.stypy_call_varargs = varargs
    jac_rosenbrock.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'jac_rosenbrock', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'jac_rosenbrock', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'jac_rosenbrock(...)' code ##################

    
    # Call to array(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_205864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    
    # Obtaining an instance of the builtin type 'list' (line 42)
    list_205865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 42)
    # Adding element type (line 42)
    int_205866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'int')
    
    # Obtaining the type of the subscript
    int_205867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 17), 'int')
    # Getting the type of 'x' (line 42)
    x_205868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___205869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 15), x_205868, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_205870 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), getitem___205869, int_205867)
    
    # Applying the binary operator '*' (line 42)
    result_mul_205871 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 9), '*', int_205866, subscript_call_result_205870)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 8), list_205865, result_mul_205871)
    # Adding element type (line 42)
    int_205872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 8), list_205865, int_205872)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), list_205864, list_205865)
    # Adding element type (line 41)
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_205873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    # Adding element type (line 43)
    int_205874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 8), list_205873, int_205874)
    # Adding element type (line 43)
    int_205875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 8), list_205873, int_205875)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), list_205864, list_205873)
    
    # Processing the call keyword arguments (line 41)
    kwargs_205876 = {}
    # Getting the type of 'np' (line 41)
    np_205862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 41)
    array_205863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 11), np_205862, 'array')
    # Calling array(args, kwargs) (line 41)
    array_call_result_205877 = invoke(stypy.reporting.localization.Localization(__file__, 41, 11), array_205863, *[list_205864], **kwargs_205876)
    
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type', array_call_result_205877)
    
    # ################# End of 'jac_rosenbrock(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'jac_rosenbrock' in the type store
    # Getting the type of 'stypy_return_type' (line 40)
    stypy_return_type_205878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205878)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'jac_rosenbrock'
    return stypy_return_type_205878

# Assigning a type to the variable 'jac_rosenbrock' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'jac_rosenbrock', jac_rosenbrock)

@norecursion
def jac_rosenbrock_bad_dim(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'jac_rosenbrock_bad_dim'
    module_type_store = module_type_store.open_function_context('jac_rosenbrock_bad_dim', 47, 0, False)
    
    # Passed parameters checking function
    jac_rosenbrock_bad_dim.stypy_localization = localization
    jac_rosenbrock_bad_dim.stypy_type_of_self = None
    jac_rosenbrock_bad_dim.stypy_type_store = module_type_store
    jac_rosenbrock_bad_dim.stypy_function_name = 'jac_rosenbrock_bad_dim'
    jac_rosenbrock_bad_dim.stypy_param_names_list = ['x']
    jac_rosenbrock_bad_dim.stypy_varargs_param_name = None
    jac_rosenbrock_bad_dim.stypy_kwargs_param_name = None
    jac_rosenbrock_bad_dim.stypy_call_defaults = defaults
    jac_rosenbrock_bad_dim.stypy_call_varargs = varargs
    jac_rosenbrock_bad_dim.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'jac_rosenbrock_bad_dim', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'jac_rosenbrock_bad_dim', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'jac_rosenbrock_bad_dim(...)' code ##################

    
    # Call to array(...): (line 48)
    # Processing the call arguments (line 48)
    
    # Obtaining an instance of the builtin type 'list' (line 48)
    list_205881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 48)
    # Adding element type (line 48)
    
    # Obtaining an instance of the builtin type 'list' (line 49)
    list_205882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 49)
    # Adding element type (line 49)
    int_205883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 9), 'int')
    
    # Obtaining the type of the subscript
    int_205884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 17), 'int')
    # Getting the type of 'x' (line 49)
    x_205885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___205886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 15), x_205885, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_205887 = invoke(stypy.reporting.localization.Localization(__file__, 49, 15), getitem___205886, int_205884)
    
    # Applying the binary operator '*' (line 49)
    result_mul_205888 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 9), '*', int_205883, subscript_call_result_205887)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 8), list_205882, result_mul_205888)
    # Adding element type (line 49)
    int_205889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 8), list_205882, int_205889)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 20), list_205881, list_205882)
    # Adding element type (line 48)
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_205890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    # Adding element type (line 50)
    int_205891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 8), list_205890, int_205891)
    # Adding element type (line 50)
    int_205892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 8), list_205890, int_205892)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 20), list_205881, list_205890)
    # Adding element type (line 48)
    
    # Obtaining an instance of the builtin type 'list' (line 51)
    list_205893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 51)
    # Adding element type (line 51)
    float_205894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), list_205893, float_205894)
    # Adding element type (line 51)
    float_205895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 14), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), list_205893, float_205895)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 20), list_205881, list_205893)
    
    # Processing the call keyword arguments (line 48)
    kwargs_205896 = {}
    # Getting the type of 'np' (line 48)
    np_205879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 48)
    array_205880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 11), np_205879, 'array')
    # Calling array(args, kwargs) (line 48)
    array_call_result_205897 = invoke(stypy.reporting.localization.Localization(__file__, 48, 11), array_205880, *[list_205881], **kwargs_205896)
    
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type', array_call_result_205897)
    
    # ################# End of 'jac_rosenbrock_bad_dim(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'jac_rosenbrock_bad_dim' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_205898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205898)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'jac_rosenbrock_bad_dim'
    return stypy_return_type_205898

# Assigning a type to the variable 'jac_rosenbrock_bad_dim' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'jac_rosenbrock_bad_dim', jac_rosenbrock_bad_dim)

@norecursion
def fun_rosenbrock_cropped(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fun_rosenbrock_cropped'
    module_type_store = module_type_store.open_function_context('fun_rosenbrock_cropped', 55, 0, False)
    
    # Passed parameters checking function
    fun_rosenbrock_cropped.stypy_localization = localization
    fun_rosenbrock_cropped.stypy_type_of_self = None
    fun_rosenbrock_cropped.stypy_type_store = module_type_store
    fun_rosenbrock_cropped.stypy_function_name = 'fun_rosenbrock_cropped'
    fun_rosenbrock_cropped.stypy_param_names_list = ['x']
    fun_rosenbrock_cropped.stypy_varargs_param_name = None
    fun_rosenbrock_cropped.stypy_kwargs_param_name = None
    fun_rosenbrock_cropped.stypy_call_defaults = defaults
    fun_rosenbrock_cropped.stypy_call_varargs = varargs
    fun_rosenbrock_cropped.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fun_rosenbrock_cropped', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fun_rosenbrock_cropped', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fun_rosenbrock_cropped(...)' code ##################

    
    # Obtaining the type of the subscript
    int_205899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 29), 'int')
    
    # Call to fun_rosenbrock(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'x' (line 56)
    x_205901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'x', False)
    # Processing the call keyword arguments (line 56)
    kwargs_205902 = {}
    # Getting the type of 'fun_rosenbrock' (line 56)
    fun_rosenbrock_205900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'fun_rosenbrock', False)
    # Calling fun_rosenbrock(args, kwargs) (line 56)
    fun_rosenbrock_call_result_205903 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), fun_rosenbrock_205900, *[x_205901], **kwargs_205902)
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___205904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), fun_rosenbrock_call_result_205903, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_205905 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), getitem___205904, int_205899)
    
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type', subscript_call_result_205905)
    
    # ################# End of 'fun_rosenbrock_cropped(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fun_rosenbrock_cropped' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_205906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205906)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fun_rosenbrock_cropped'
    return stypy_return_type_205906

# Assigning a type to the variable 'fun_rosenbrock_cropped' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'fun_rosenbrock_cropped', fun_rosenbrock_cropped)

@norecursion
def jac_rosenbrock_cropped(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'jac_rosenbrock_cropped'
    module_type_store = module_type_store.open_function_context('jac_rosenbrock_cropped', 59, 0, False)
    
    # Passed parameters checking function
    jac_rosenbrock_cropped.stypy_localization = localization
    jac_rosenbrock_cropped.stypy_type_of_self = None
    jac_rosenbrock_cropped.stypy_type_store = module_type_store
    jac_rosenbrock_cropped.stypy_function_name = 'jac_rosenbrock_cropped'
    jac_rosenbrock_cropped.stypy_param_names_list = ['x']
    jac_rosenbrock_cropped.stypy_varargs_param_name = None
    jac_rosenbrock_cropped.stypy_kwargs_param_name = None
    jac_rosenbrock_cropped.stypy_call_defaults = defaults
    jac_rosenbrock_cropped.stypy_call_varargs = varargs
    jac_rosenbrock_cropped.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'jac_rosenbrock_cropped', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'jac_rosenbrock_cropped', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'jac_rosenbrock_cropped(...)' code ##################

    
    # Obtaining the type of the subscript
    int_205907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 29), 'int')
    
    # Call to jac_rosenbrock(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'x' (line 60)
    x_205909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), 'x', False)
    # Processing the call keyword arguments (line 60)
    kwargs_205910 = {}
    # Getting the type of 'jac_rosenbrock' (line 60)
    jac_rosenbrock_205908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'jac_rosenbrock', False)
    # Calling jac_rosenbrock(args, kwargs) (line 60)
    jac_rosenbrock_call_result_205911 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), jac_rosenbrock_205908, *[x_205909], **kwargs_205910)
    
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___205912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 11), jac_rosenbrock_call_result_205911, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_205913 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), getitem___205912, int_205907)
    
    # Assigning a type to the variable 'stypy_return_type' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type', subscript_call_result_205913)
    
    # ################# End of 'jac_rosenbrock_cropped(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'jac_rosenbrock_cropped' in the type store
    # Getting the type of 'stypy_return_type' (line 59)
    stypy_return_type_205914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205914)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'jac_rosenbrock_cropped'
    return stypy_return_type_205914

# Assigning a type to the variable 'jac_rosenbrock_cropped' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'jac_rosenbrock_cropped', jac_rosenbrock_cropped)

@norecursion
def fun_wrong_dimensions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fun_wrong_dimensions'
    module_type_store = module_type_store.open_function_context('fun_wrong_dimensions', 64, 0, False)
    
    # Passed parameters checking function
    fun_wrong_dimensions.stypy_localization = localization
    fun_wrong_dimensions.stypy_type_of_self = None
    fun_wrong_dimensions.stypy_type_store = module_type_store
    fun_wrong_dimensions.stypy_function_name = 'fun_wrong_dimensions'
    fun_wrong_dimensions.stypy_param_names_list = ['x']
    fun_wrong_dimensions.stypy_varargs_param_name = None
    fun_wrong_dimensions.stypy_kwargs_param_name = None
    fun_wrong_dimensions.stypy_call_defaults = defaults
    fun_wrong_dimensions.stypy_call_varargs = varargs
    fun_wrong_dimensions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fun_wrong_dimensions', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fun_wrong_dimensions', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fun_wrong_dimensions(...)' code ##################

    
    # Call to array(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Obtaining an instance of the builtin type 'list' (line 65)
    list_205917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 65)
    # Adding element type (line 65)
    # Getting the type of 'x' (line 65)
    x_205918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 20), list_205917, x_205918)
    # Adding element type (line 65)
    # Getting the type of 'x' (line 65)
    x_205919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), 'x', False)
    int_205920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 27), 'int')
    # Applying the binary operator '**' (line 65)
    result_pow_205921 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 24), '**', x_205919, int_205920)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 20), list_205917, result_pow_205921)
    # Adding element type (line 65)
    # Getting the type of 'x' (line 65)
    x_205922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'x', False)
    int_205923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 33), 'int')
    # Applying the binary operator '**' (line 65)
    result_pow_205924 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 30), '**', x_205922, int_205923)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 20), list_205917, result_pow_205924)
    
    # Processing the call keyword arguments (line 65)
    kwargs_205925 = {}
    # Getting the type of 'np' (line 65)
    np_205915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 65)
    array_205916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), np_205915, 'array')
    # Calling array(args, kwargs) (line 65)
    array_call_result_205926 = invoke(stypy.reporting.localization.Localization(__file__, 65, 11), array_205916, *[list_205917], **kwargs_205925)
    
    # Assigning a type to the variable 'stypy_return_type' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type', array_call_result_205926)
    
    # ################# End of 'fun_wrong_dimensions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fun_wrong_dimensions' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_205927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205927)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fun_wrong_dimensions'
    return stypy_return_type_205927

# Assigning a type to the variable 'fun_wrong_dimensions' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'fun_wrong_dimensions', fun_wrong_dimensions)

@norecursion
def jac_wrong_dimensions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_205928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 30), 'float')
    defaults = [float_205928]
    # Create a new context for function 'jac_wrong_dimensions'
    module_type_store = module_type_store.open_function_context('jac_wrong_dimensions', 68, 0, False)
    
    # Passed parameters checking function
    jac_wrong_dimensions.stypy_localization = localization
    jac_wrong_dimensions.stypy_type_of_self = None
    jac_wrong_dimensions.stypy_type_store = module_type_store
    jac_wrong_dimensions.stypy_function_name = 'jac_wrong_dimensions'
    jac_wrong_dimensions.stypy_param_names_list = ['x', 'a']
    jac_wrong_dimensions.stypy_varargs_param_name = None
    jac_wrong_dimensions.stypy_kwargs_param_name = None
    jac_wrong_dimensions.stypy_call_defaults = defaults
    jac_wrong_dimensions.stypy_call_varargs = varargs
    jac_wrong_dimensions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'jac_wrong_dimensions', ['x', 'a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'jac_wrong_dimensions', localization, ['x', 'a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'jac_wrong_dimensions(...)' code ##################

    
    # Call to atleast_3d(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Call to jac_trivial(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'x' (line 69)
    x_205932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 37), 'x', False)
    # Processing the call keyword arguments (line 69)
    # Getting the type of 'a' (line 69)
    a_205933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 42), 'a', False)
    keyword_205934 = a_205933
    kwargs_205935 = {'a': keyword_205934}
    # Getting the type of 'jac_trivial' (line 69)
    jac_trivial_205931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'jac_trivial', False)
    # Calling jac_trivial(args, kwargs) (line 69)
    jac_trivial_call_result_205936 = invoke(stypy.reporting.localization.Localization(__file__, 69, 25), jac_trivial_205931, *[x_205932], **kwargs_205935)
    
    # Processing the call keyword arguments (line 69)
    kwargs_205937 = {}
    # Getting the type of 'np' (line 69)
    np_205929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'np', False)
    # Obtaining the member 'atleast_3d' of a type (line 69)
    atleast_3d_205930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 11), np_205929, 'atleast_3d')
    # Calling atleast_3d(args, kwargs) (line 69)
    atleast_3d_call_result_205938 = invoke(stypy.reporting.localization.Localization(__file__, 69, 11), atleast_3d_205930, *[jac_trivial_call_result_205936], **kwargs_205937)
    
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type', atleast_3d_call_result_205938)
    
    # ################# End of 'jac_wrong_dimensions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'jac_wrong_dimensions' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_205939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205939)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'jac_wrong_dimensions'
    return stypy_return_type_205939

# Assigning a type to the variable 'jac_wrong_dimensions' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'jac_wrong_dimensions', jac_wrong_dimensions)

@norecursion
def fun_bvp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fun_bvp'
    module_type_store = module_type_store.open_function_context('fun_bvp', 72, 0, False)
    
    # Passed parameters checking function
    fun_bvp.stypy_localization = localization
    fun_bvp.stypy_type_of_self = None
    fun_bvp.stypy_type_store = module_type_store
    fun_bvp.stypy_function_name = 'fun_bvp'
    fun_bvp.stypy_param_names_list = ['x']
    fun_bvp.stypy_varargs_param_name = None
    fun_bvp.stypy_kwargs_param_name = None
    fun_bvp.stypy_call_defaults = defaults
    fun_bvp.stypy_call_varargs = varargs
    fun_bvp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fun_bvp', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fun_bvp', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fun_bvp(...)' code ##################

    
    # Assigning a Call to a Name (line 73):
    
    # Call to int(...): (line 73)
    # Processing the call arguments (line 73)
    
    # Call to sqrt(...): (line 73)
    # Processing the call arguments (line 73)
    
    # Obtaining the type of the subscript
    int_205943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'int')
    # Getting the type of 'x' (line 73)
    x_205944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'x', False)
    # Obtaining the member 'shape' of a type (line 73)
    shape_205945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 20), x_205944, 'shape')
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___205946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 20), shape_205945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
    subscript_call_result_205947 = invoke(stypy.reporting.localization.Localization(__file__, 73, 20), getitem___205946, int_205943)
    
    # Processing the call keyword arguments (line 73)
    kwargs_205948 = {}
    # Getting the type of 'np' (line 73)
    np_205941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 73)
    sqrt_205942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 12), np_205941, 'sqrt')
    # Calling sqrt(args, kwargs) (line 73)
    sqrt_call_result_205949 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), sqrt_205942, *[subscript_call_result_205947], **kwargs_205948)
    
    # Processing the call keyword arguments (line 73)
    kwargs_205950 = {}
    # Getting the type of 'int' (line 73)
    int_205940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'int', False)
    # Calling int(args, kwargs) (line 73)
    int_call_result_205951 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), int_205940, *[sqrt_call_result_205949], **kwargs_205950)
    
    # Assigning a type to the variable 'n' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'n', int_call_result_205951)
    
    # Assigning a Call to a Name (line 74):
    
    # Call to zeros(...): (line 74)
    # Processing the call arguments (line 74)
    
    # Obtaining an instance of the builtin type 'tuple' (line 74)
    tuple_205954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 74)
    # Adding element type (line 74)
    # Getting the type of 'n' (line 74)
    n_205955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'n', False)
    int_205956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 22), 'int')
    # Applying the binary operator '+' (line 74)
    result_add_205957 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 18), '+', n_205955, int_205956)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 18), tuple_205954, result_add_205957)
    # Adding element type (line 74)
    # Getting the type of 'n' (line 74)
    n_205958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'n', False)
    int_205959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 29), 'int')
    # Applying the binary operator '+' (line 74)
    result_add_205960 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 25), '+', n_205958, int_205959)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 18), tuple_205954, result_add_205960)
    
    # Processing the call keyword arguments (line 74)
    kwargs_205961 = {}
    # Getting the type of 'np' (line 74)
    np_205952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 74)
    zeros_205953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), np_205952, 'zeros')
    # Calling zeros(args, kwargs) (line 74)
    zeros_call_result_205962 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), zeros_205953, *[tuple_205954], **kwargs_205961)
    
    # Assigning a type to the variable 'u' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'u', zeros_call_result_205962)
    
    # Assigning a Call to a Name (line 75):
    
    # Call to reshape(...): (line 75)
    # Processing the call arguments (line 75)
    
    # Obtaining an instance of the builtin type 'tuple' (line 75)
    tuple_205965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 75)
    # Adding element type (line 75)
    # Getting the type of 'n' (line 75)
    n_205966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 19), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 19), tuple_205965, n_205966)
    # Adding element type (line 75)
    # Getting the type of 'n' (line 75)
    n_205967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 19), tuple_205965, n_205967)
    
    # Processing the call keyword arguments (line 75)
    kwargs_205968 = {}
    # Getting the type of 'x' (line 75)
    x_205963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'x', False)
    # Obtaining the member 'reshape' of a type (line 75)
    reshape_205964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), x_205963, 'reshape')
    # Calling reshape(args, kwargs) (line 75)
    reshape_call_result_205969 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), reshape_205964, *[tuple_205965], **kwargs_205968)
    
    # Assigning a type to the variable 'x' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'x', reshape_call_result_205969)
    
    # Assigning a Name to a Subscript (line 76):
    # Getting the type of 'x' (line 76)
    x_205970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'x')
    # Getting the type of 'u' (line 76)
    u_205971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'u')
    int_205972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 6), 'int')
    int_205973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'int')
    slice_205974 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 76, 4), int_205972, int_205973, None)
    int_205975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 12), 'int')
    int_205976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 14), 'int')
    slice_205977 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 76, 4), int_205975, int_205976, None)
    # Storing an element on a container (line 76)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 4), u_205971, ((slice_205974, slice_205977), x_205970))
    
    # Assigning a BinOp to a Name (line 77):
    
    # Obtaining the type of the subscript
    int_205978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 11), 'int')
    slice_205979 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 8), None, int_205978, None)
    int_205980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 15), 'int')
    int_205981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 17), 'int')
    slice_205982 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 8), int_205980, int_205981, None)
    # Getting the type of 'u' (line 77)
    u_205983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'u')
    # Obtaining the member '__getitem__' of a type (line 77)
    getitem___205984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), u_205983, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 77)
    subscript_call_result_205985 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), getitem___205984, (slice_205979, slice_205982))
    
    
    # Obtaining the type of the subscript
    int_205986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'int')
    slice_205987 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 23), int_205986, None, None)
    int_205988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 29), 'int')
    int_205989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 31), 'int')
    slice_205990 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 23), int_205988, int_205989, None)
    # Getting the type of 'u' (line 77)
    u_205991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 23), 'u')
    # Obtaining the member '__getitem__' of a type (line 77)
    getitem___205992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 23), u_205991, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 77)
    subscript_call_result_205993 = invoke(stypy.reporting.localization.Localization(__file__, 77, 23), getitem___205992, (slice_205987, slice_205990))
    
    # Applying the binary operator '+' (line 77)
    result_add_205994 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 8), '+', subscript_call_result_205985, subscript_call_result_205993)
    
    
    # Obtaining the type of the subscript
    int_205995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 39), 'int')
    int_205996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 41), 'int')
    slice_205997 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 37), int_205995, int_205996, None)
    int_205998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 46), 'int')
    slice_205999 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 37), None, int_205998, None)
    # Getting the type of 'u' (line 77)
    u_206000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 37), 'u')
    # Obtaining the member '__getitem__' of a type (line 77)
    getitem___206001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 37), u_206000, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 77)
    subscript_call_result_206002 = invoke(stypy.reporting.localization.Localization(__file__, 77, 37), getitem___206001, (slice_205997, slice_205999))
    
    # Applying the binary operator '+' (line 77)
    result_add_206003 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 35), '+', result_add_205994, subscript_call_result_206002)
    
    
    # Obtaining the type of the subscript
    int_206004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 54), 'int')
    int_206005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 56), 'int')
    slice_206006 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 52), int_206004, int_206005, None)
    int_206007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 60), 'int')
    slice_206008 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 52), int_206007, None, None)
    # Getting the type of 'u' (line 77)
    u_206009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 52), 'u')
    # Obtaining the member '__getitem__' of a type (line 77)
    getitem___206010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 52), u_206009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 77)
    subscript_call_result_206011 = invoke(stypy.reporting.localization.Localization(__file__, 77, 52), getitem___206010, (slice_206006, slice_206008))
    
    # Applying the binary operator '+' (line 77)
    result_add_206012 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 50), '+', result_add_206003, subscript_call_result_206011)
    
    int_206013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 66), 'int')
    # Getting the type of 'x' (line 77)
    x_206014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 70), 'x')
    # Applying the binary operator '*' (line 77)
    result_mul_206015 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 66), '*', int_206013, x_206014)
    
    # Applying the binary operator '-' (line 77)
    result_sub_206016 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 64), '-', result_add_206012, result_mul_206015)
    
    # Getting the type of 'x' (line 77)
    x_206017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 74), 'x')
    int_206018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 77), 'int')
    # Applying the binary operator '**' (line 77)
    result_pow_206019 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 74), '**', x_206017, int_206018)
    
    # Applying the binary operator '+' (line 77)
    result_add_206020 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 72), '+', result_sub_206016, result_pow_206019)
    
    # Assigning a type to the variable 'y' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'y', result_add_206020)
    
    # Call to ravel(...): (line 78)
    # Processing the call keyword arguments (line 78)
    kwargs_206023 = {}
    # Getting the type of 'y' (line 78)
    y_206021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'y', False)
    # Obtaining the member 'ravel' of a type (line 78)
    ravel_206022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 11), y_206021, 'ravel')
    # Calling ravel(args, kwargs) (line 78)
    ravel_call_result_206024 = invoke(stypy.reporting.localization.Localization(__file__, 78, 11), ravel_206022, *[], **kwargs_206023)
    
    # Assigning a type to the variable 'stypy_return_type' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type', ravel_call_result_206024)
    
    # ################# End of 'fun_bvp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fun_bvp' in the type store
    # Getting the type of 'stypy_return_type' (line 72)
    stypy_return_type_206025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_206025)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fun_bvp'
    return stypy_return_type_206025

# Assigning a type to the variable 'fun_bvp' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'fun_bvp', fun_bvp)
# Declaration of the 'BroydenTridiagonal' class

class BroydenTridiagonal(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_206026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 25), 'int')
        str_206027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 35), 'str', 'sparse')
        defaults = [int_206026, str_206027]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BroydenTridiagonal.__init__', ['n', 'mode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['n', 'mode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to seed(...): (line 83)
        # Processing the call arguments (line 83)
        int_206031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 23), 'int')
        # Processing the call keyword arguments (line 83)
        kwargs_206032 = {}
        # Getting the type of 'np' (line 83)
        np_206028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 83)
        random_206029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), np_206028, 'random')
        # Obtaining the member 'seed' of a type (line 83)
        seed_206030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), random_206029, 'seed')
        # Calling seed(args, kwargs) (line 83)
        seed_call_result_206033 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), seed_206030, *[int_206031], **kwargs_206032)
        
        
        # Assigning a Name to a Attribute (line 85):
        # Getting the type of 'n' (line 85)
        n_206034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'n')
        # Getting the type of 'self' (line 85)
        self_206035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self')
        # Setting the type of the member 'n' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_206035, 'n', n_206034)
        
        # Assigning a UnaryOp to a Attribute (line 87):
        
        
        # Call to ones(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'n' (line 87)
        n_206038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 27), 'n', False)
        # Processing the call keyword arguments (line 87)
        kwargs_206039 = {}
        # Getting the type of 'np' (line 87)
        np_206036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'np', False)
        # Obtaining the member 'ones' of a type (line 87)
        ones_206037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 19), np_206036, 'ones')
        # Calling ones(args, kwargs) (line 87)
        ones_call_result_206040 = invoke(stypy.reporting.localization.Localization(__file__, 87, 19), ones_206037, *[n_206038], **kwargs_206039)
        
        # Applying the 'usub' unary operator (line 87)
        result___neg___206041 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 18), 'usub', ones_call_result_206040)
        
        # Getting the type of 'self' (line 87)
        self_206042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self')
        # Setting the type of the member 'x0' of a type (line 87)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_206042, 'x0', result___neg___206041)
        
        # Assigning a Call to a Attribute (line 88):
        
        # Call to linspace(...): (line 88)
        # Processing the call arguments (line 88)
        int_206045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 30), 'int')
        float_206046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 34), 'float')
        # Getting the type of 'n' (line 88)
        n_206047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 40), 'n', False)
        # Processing the call keyword arguments (line 88)
        kwargs_206048 = {}
        # Getting the type of 'np' (line 88)
        np_206043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'np', False)
        # Obtaining the member 'linspace' of a type (line 88)
        linspace_206044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 18), np_206043, 'linspace')
        # Calling linspace(args, kwargs) (line 88)
        linspace_call_result_206049 = invoke(stypy.reporting.localization.Localization(__file__, 88, 18), linspace_206044, *[int_206045, float_206046, n_206047], **kwargs_206048)
        
        # Getting the type of 'self' (line 88)
        self_206050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self')
        # Setting the type of the member 'lb' of a type (line 88)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_206050, 'lb', linspace_call_result_206049)
        
        # Assigning a Call to a Attribute (line 89):
        
        # Call to linspace(...): (line 89)
        # Processing the call arguments (line 89)
        float_206053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 30), 'float')
        float_206054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 36), 'float')
        # Getting the type of 'n' (line 89)
        n_206055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 41), 'n', False)
        # Processing the call keyword arguments (line 89)
        kwargs_206056 = {}
        # Getting the type of 'np' (line 89)
        np_206051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'np', False)
        # Obtaining the member 'linspace' of a type (line 89)
        linspace_206052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 18), np_206051, 'linspace')
        # Calling linspace(args, kwargs) (line 89)
        linspace_call_result_206057 = invoke(stypy.reporting.localization.Localization(__file__, 89, 18), linspace_206052, *[float_206053, float_206054, n_206055], **kwargs_206056)
        
        # Getting the type of 'self' (line 89)
        self_206058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self')
        # Setting the type of the member 'ub' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_206058, 'ub', linspace_call_result_206057)
        
        # Getting the type of 'self' (line 91)
        self_206059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Obtaining the member 'lb' of a type (line 91)
        lb_206060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_206059, 'lb')
        float_206061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'float')
        
        # Call to randn(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'n' (line 91)
        n_206065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 41), 'n', False)
        # Processing the call keyword arguments (line 91)
        kwargs_206066 = {}
        # Getting the type of 'np' (line 91)
        np_206062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'np', False)
        # Obtaining the member 'random' of a type (line 91)
        random_206063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 25), np_206062, 'random')
        # Obtaining the member 'randn' of a type (line 91)
        randn_206064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 25), random_206063, 'randn')
        # Calling randn(args, kwargs) (line 91)
        randn_call_result_206067 = invoke(stypy.reporting.localization.Localization(__file__, 91, 25), randn_206064, *[n_206065], **kwargs_206066)
        
        # Applying the binary operator '*' (line 91)
        result_mul_206068 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 19), '*', float_206061, randn_call_result_206067)
        
        # Applying the binary operator '+=' (line 91)
        result_iadd_206069 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 8), '+=', lb_206060, result_mul_206068)
        # Getting the type of 'self' (line 91)
        self_206070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member 'lb' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_206070, 'lb', result_iadd_206069)
        
        
        # Getting the type of 'self' (line 92)
        self_206071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self')
        # Obtaining the member 'ub' of a type (line 92)
        ub_206072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_206071, 'ub')
        float_206073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 19), 'float')
        
        # Call to randn(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'n' (line 92)
        n_206077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'n', False)
        # Processing the call keyword arguments (line 92)
        kwargs_206078 = {}
        # Getting the type of 'np' (line 92)
        np_206074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'np', False)
        # Obtaining the member 'random' of a type (line 92)
        random_206075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 25), np_206074, 'random')
        # Obtaining the member 'randn' of a type (line 92)
        randn_206076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 25), random_206075, 'randn')
        # Calling randn(args, kwargs) (line 92)
        randn_call_result_206079 = invoke(stypy.reporting.localization.Localization(__file__, 92, 25), randn_206076, *[n_206077], **kwargs_206078)
        
        # Applying the binary operator '*' (line 92)
        result_mul_206080 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 19), '*', float_206073, randn_call_result_206079)
        
        # Applying the binary operator '+=' (line 92)
        result_iadd_206081 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 8), '+=', ub_206072, result_mul_206080)
        # Getting the type of 'self' (line 92)
        self_206082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self')
        # Setting the type of the member 'ub' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_206082, 'ub', result_iadd_206081)
        
        
        # Getting the type of 'self' (line 94)
        self_206083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Obtaining the member 'x0' of a type (line 94)
        x0_206084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_206083, 'x0')
        float_206085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 19), 'float')
        
        # Call to randn(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'n' (line 94)
        n_206089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 41), 'n', False)
        # Processing the call keyword arguments (line 94)
        kwargs_206090 = {}
        # Getting the type of 'np' (line 94)
        np_206086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'np', False)
        # Obtaining the member 'random' of a type (line 94)
        random_206087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 25), np_206086, 'random')
        # Obtaining the member 'randn' of a type (line 94)
        randn_206088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 25), random_206087, 'randn')
        # Calling randn(args, kwargs) (line 94)
        randn_call_result_206091 = invoke(stypy.reporting.localization.Localization(__file__, 94, 25), randn_206088, *[n_206089], **kwargs_206090)
        
        # Applying the binary operator '*' (line 94)
        result_mul_206092 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 19), '*', float_206085, randn_call_result_206091)
        
        # Applying the binary operator '+=' (line 94)
        result_iadd_206093 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 8), '+=', x0_206084, result_mul_206092)
        # Getting the type of 'self' (line 94)
        self_206094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Setting the type of the member 'x0' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_206094, 'x0', result_iadd_206093)
        
        
        # Assigning a Call to a Attribute (line 95):
        
        # Call to make_strictly_feasible(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'self' (line 95)
        self_206096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 41), 'self', False)
        # Obtaining the member 'x0' of a type (line 95)
        x0_206097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 41), self_206096, 'x0')
        # Getting the type of 'self' (line 95)
        self_206098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 50), 'self', False)
        # Obtaining the member 'lb' of a type (line 95)
        lb_206099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 50), self_206098, 'lb')
        # Getting the type of 'self' (line 95)
        self_206100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 59), 'self', False)
        # Obtaining the member 'ub' of a type (line 95)
        ub_206101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 59), self_206100, 'ub')
        # Processing the call keyword arguments (line 95)
        kwargs_206102 = {}
        # Getting the type of 'make_strictly_feasible' (line 95)
        make_strictly_feasible_206095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'make_strictly_feasible', False)
        # Calling make_strictly_feasible(args, kwargs) (line 95)
        make_strictly_feasible_call_result_206103 = invoke(stypy.reporting.localization.Localization(__file__, 95, 18), make_strictly_feasible_206095, *[x0_206097, lb_206099, ub_206101], **kwargs_206102)
        
        # Getting the type of 'self' (line 95)
        self_206104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Setting the type of the member 'x0' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_206104, 'x0', make_strictly_feasible_call_result_206103)
        
        
        # Getting the type of 'mode' (line 97)
        mode_206105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'mode')
        str_206106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 19), 'str', 'sparse')
        # Applying the binary operator '==' (line 97)
        result_eq_206107 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), '==', mode_206105, str_206106)
        
        # Testing the type of an if condition (line 97)
        if_condition_206108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), result_eq_206107)
        # Assigning a type to the variable 'if_condition_206108' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_206108', if_condition_206108)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 98):
        
        # Call to lil_matrix(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Obtaining an instance of the builtin type 'tuple' (line 98)
        tuple_206110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 98)
        # Adding element type (line 98)
        # Getting the type of 'n' (line 98)
        n_206111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 40), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 40), tuple_206110, n_206111)
        # Adding element type (line 98)
        # Getting the type of 'n' (line 98)
        n_206112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 43), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 40), tuple_206110, n_206112)
        
        # Processing the call keyword arguments (line 98)
        # Getting the type of 'int' (line 98)
        int_206113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 53), 'int', False)
        keyword_206114 = int_206113
        kwargs_206115 = {'dtype': keyword_206114}
        # Getting the type of 'lil_matrix' (line 98)
        lil_matrix_206109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'lil_matrix', False)
        # Calling lil_matrix(args, kwargs) (line 98)
        lil_matrix_call_result_206116 = invoke(stypy.reporting.localization.Localization(__file__, 98, 28), lil_matrix_206109, *[tuple_206110], **kwargs_206115)
        
        # Getting the type of 'self' (line 98)
        self_206117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'self')
        # Setting the type of the member 'sparsity' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), self_206117, 'sparsity', lil_matrix_call_result_206116)
        
        # Assigning a Call to a Name (line 99):
        
        # Call to arange(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'n' (line 99)
        n_206120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'n', False)
        # Processing the call keyword arguments (line 99)
        kwargs_206121 = {}
        # Getting the type of 'np' (line 99)
        np_206118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'np', False)
        # Obtaining the member 'arange' of a type (line 99)
        arange_206119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), np_206118, 'arange')
        # Calling arange(args, kwargs) (line 99)
        arange_call_result_206122 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), arange_206119, *[n_206120], **kwargs_206121)
        
        # Assigning a type to the variable 'i' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'i', arange_call_result_206122)
        
        # Assigning a Num to a Subscript (line 100):
        int_206123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 34), 'int')
        # Getting the type of 'self' (line 100)
        self_206124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self')
        # Obtaining the member 'sparsity' of a type (line 100)
        sparsity_206125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_206124, 'sparsity')
        
        # Obtaining an instance of the builtin type 'tuple' (line 100)
        tuple_206126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 100)
        # Adding element type (line 100)
        # Getting the type of 'i' (line 100)
        i_206127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 26), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 26), tuple_206126, i_206127)
        # Adding element type (line 100)
        # Getting the type of 'i' (line 100)
        i_206128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 29), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 26), tuple_206126, i_206128)
        
        # Storing an element on a container (line 100)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 12), sparsity_206125, (tuple_206126, int_206123))
        
        # Assigning a Call to a Name (line 101):
        
        # Call to arange(...): (line 101)
        # Processing the call arguments (line 101)
        int_206131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'int')
        # Getting the type of 'n' (line 101)
        n_206132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 29), 'n', False)
        # Processing the call keyword arguments (line 101)
        kwargs_206133 = {}
        # Getting the type of 'np' (line 101)
        np_206129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'np', False)
        # Obtaining the member 'arange' of a type (line 101)
        arange_206130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 16), np_206129, 'arange')
        # Calling arange(args, kwargs) (line 101)
        arange_call_result_206134 = invoke(stypy.reporting.localization.Localization(__file__, 101, 16), arange_206130, *[int_206131, n_206132], **kwargs_206133)
        
        # Assigning a type to the variable 'i' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'i', arange_call_result_206134)
        
        # Assigning a Num to a Subscript (line 102):
        int_206135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 38), 'int')
        # Getting the type of 'self' (line 102)
        self_206136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self')
        # Obtaining the member 'sparsity' of a type (line 102)
        sparsity_206137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), self_206136, 'sparsity')
        
        # Obtaining an instance of the builtin type 'tuple' (line 102)
        tuple_206138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 102)
        # Adding element type (line 102)
        # Getting the type of 'i' (line 102)
        i_206139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 26), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 26), tuple_206138, i_206139)
        # Adding element type (line 102)
        # Getting the type of 'i' (line 102)
        i_206140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 'i')
        int_206141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 33), 'int')
        # Applying the binary operator '-' (line 102)
        result_sub_206142 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 29), '-', i_206140, int_206141)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 26), tuple_206138, result_sub_206142)
        
        # Storing an element on a container (line 102)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 12), sparsity_206137, (tuple_206138, int_206135))
        
        # Assigning a Call to a Name (line 103):
        
        # Call to arange(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'n' (line 103)
        n_206145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'n', False)
        int_206146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 30), 'int')
        # Applying the binary operator '-' (line 103)
        result_sub_206147 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 26), '-', n_206145, int_206146)
        
        # Processing the call keyword arguments (line 103)
        kwargs_206148 = {}
        # Getting the type of 'np' (line 103)
        np_206143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'np', False)
        # Obtaining the member 'arange' of a type (line 103)
        arange_206144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), np_206143, 'arange')
        # Calling arange(args, kwargs) (line 103)
        arange_call_result_206149 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), arange_206144, *[result_sub_206147], **kwargs_206148)
        
        # Assigning a type to the variable 'i' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'i', arange_call_result_206149)
        
        # Assigning a Num to a Subscript (line 104):
        int_206150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 38), 'int')
        # Getting the type of 'self' (line 104)
        self_206151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'self')
        # Obtaining the member 'sparsity' of a type (line 104)
        sparsity_206152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), self_206151, 'sparsity')
        
        # Obtaining an instance of the builtin type 'tuple' (line 104)
        tuple_206153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 104)
        # Adding element type (line 104)
        # Getting the type of 'i' (line 104)
        i_206154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 26), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 26), tuple_206153, i_206154)
        # Adding element type (line 104)
        # Getting the type of 'i' (line 104)
        i_206155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'i')
        int_206156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 33), 'int')
        # Applying the binary operator '+' (line 104)
        result_add_206157 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 29), '+', i_206155, int_206156)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 26), tuple_206153, result_add_206157)
        
        # Storing an element on a container (line 104)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 12), sparsity_206152, (tuple_206153, int_206150))
        
        # Assigning a Attribute to a Attribute (line 106):
        # Getting the type of 'self' (line 106)
        self_206158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'self')
        # Obtaining the member '_jac' of a type (line 106)
        _jac_206159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 23), self_206158, '_jac')
        # Getting the type of 'self' (line 106)
        self_206160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'self')
        # Setting the type of the member 'jac' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), self_206160, 'jac', _jac_206159)
        # SSA branch for the else part of an if statement (line 97)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mode' (line 107)
        mode_206161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'mode')
        str_206162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 21), 'str', 'operator')
        # Applying the binary operator '==' (line 107)
        result_eq_206163 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 13), '==', mode_206161, str_206162)
        
        # Testing the type of an if condition (line 107)
        if_condition_206164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 13), result_eq_206163)
        # Assigning a type to the variable 'if_condition_206164' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'if_condition_206164', if_condition_206164)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Lambda to a Attribute (line 108):

        @norecursion
        def _stypy_temp_lambda_63(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_63'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_63', 108, 23, True)
            # Passed parameters checking function
            _stypy_temp_lambda_63.stypy_localization = localization
            _stypy_temp_lambda_63.stypy_type_of_self = None
            _stypy_temp_lambda_63.stypy_type_store = module_type_store
            _stypy_temp_lambda_63.stypy_function_name = '_stypy_temp_lambda_63'
            _stypy_temp_lambda_63.stypy_param_names_list = ['x']
            _stypy_temp_lambda_63.stypy_varargs_param_name = None
            _stypy_temp_lambda_63.stypy_kwargs_param_name = None
            _stypy_temp_lambda_63.stypy_call_defaults = defaults
            _stypy_temp_lambda_63.stypy_call_varargs = varargs
            _stypy_temp_lambda_63.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_63', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_63', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to aslinearoperator(...): (line 108)
            # Processing the call arguments (line 108)
            
            # Call to _jac(...): (line 108)
            # Processing the call arguments (line 108)
            # Getting the type of 'x' (line 108)
            x_206168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 60), 'x', False)
            # Processing the call keyword arguments (line 108)
            kwargs_206169 = {}
            # Getting the type of 'self' (line 108)
            self_206166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 50), 'self', False)
            # Obtaining the member '_jac' of a type (line 108)
            _jac_206167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 50), self_206166, '_jac')
            # Calling _jac(args, kwargs) (line 108)
            _jac_call_result_206170 = invoke(stypy.reporting.localization.Localization(__file__, 108, 50), _jac_206167, *[x_206168], **kwargs_206169)
            
            # Processing the call keyword arguments (line 108)
            kwargs_206171 = {}
            # Getting the type of 'aslinearoperator' (line 108)
            aslinearoperator_206165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 33), 'aslinearoperator', False)
            # Calling aslinearoperator(args, kwargs) (line 108)
            aslinearoperator_call_result_206172 = invoke(stypy.reporting.localization.Localization(__file__, 108, 33), aslinearoperator_206165, *[_jac_call_result_206170], **kwargs_206171)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'stypy_return_type', aslinearoperator_call_result_206172)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_63' in the type store
            # Getting the type of 'stypy_return_type' (line 108)
            stypy_return_type_206173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_206173)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_63'
            return stypy_return_type_206173

        # Assigning a type to the variable '_stypy_temp_lambda_63' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), '_stypy_temp_lambda_63', _stypy_temp_lambda_63)
        # Getting the type of '_stypy_temp_lambda_63' (line 108)
        _stypy_temp_lambda_63_206174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), '_stypy_temp_lambda_63')
        # Getting the type of 'self' (line 108)
        self_206175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'self')
        # Setting the type of the member 'jac' of a type (line 108)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), self_206175, 'jac', _stypy_temp_lambda_63_206174)
        # SSA branch for the else part of an if statement (line 107)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mode' (line 109)
        mode_206176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'mode')
        str_206177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 21), 'str', 'dense')
        # Applying the binary operator '==' (line 109)
        result_eq_206178 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 13), '==', mode_206176, str_206177)
        
        # Testing the type of an if condition (line 109)
        if_condition_206179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 13), result_eq_206178)
        # Assigning a type to the variable 'if_condition_206179' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'if_condition_206179', if_condition_206179)
        # SSA begins for if statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 110):
        # Getting the type of 'None' (line 110)
        None_206180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 28), 'None')
        # Getting the type of 'self' (line 110)
        self_206181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'self')
        # Setting the type of the member 'sparsity' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), self_206181, 'sparsity', None_206180)
        
        # Assigning a Lambda to a Attribute (line 111):

        @norecursion
        def _stypy_temp_lambda_64(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_64'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_64', 111, 23, True)
            # Passed parameters checking function
            _stypy_temp_lambda_64.stypy_localization = localization
            _stypy_temp_lambda_64.stypy_type_of_self = None
            _stypy_temp_lambda_64.stypy_type_store = module_type_store
            _stypy_temp_lambda_64.stypy_function_name = '_stypy_temp_lambda_64'
            _stypy_temp_lambda_64.stypy_param_names_list = ['x']
            _stypy_temp_lambda_64.stypy_varargs_param_name = None
            _stypy_temp_lambda_64.stypy_kwargs_param_name = None
            _stypy_temp_lambda_64.stypy_call_defaults = defaults
            _stypy_temp_lambda_64.stypy_call_varargs = varargs
            _stypy_temp_lambda_64.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_64', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_64', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to toarray(...): (line 111)
            # Processing the call keyword arguments (line 111)
            kwargs_206188 = {}
            
            # Call to _jac(...): (line 111)
            # Processing the call arguments (line 111)
            # Getting the type of 'x' (line 111)
            x_206184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 43), 'x', False)
            # Processing the call keyword arguments (line 111)
            kwargs_206185 = {}
            # Getting the type of 'self' (line 111)
            self_206182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 33), 'self', False)
            # Obtaining the member '_jac' of a type (line 111)
            _jac_206183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 33), self_206182, '_jac')
            # Calling _jac(args, kwargs) (line 111)
            _jac_call_result_206186 = invoke(stypy.reporting.localization.Localization(__file__, 111, 33), _jac_206183, *[x_206184], **kwargs_206185)
            
            # Obtaining the member 'toarray' of a type (line 111)
            toarray_206187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 33), _jac_call_result_206186, 'toarray')
            # Calling toarray(args, kwargs) (line 111)
            toarray_call_result_206189 = invoke(stypy.reporting.localization.Localization(__file__, 111, 33), toarray_206187, *[], **kwargs_206188)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'stypy_return_type', toarray_call_result_206189)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_64' in the type store
            # Getting the type of 'stypy_return_type' (line 111)
            stypy_return_type_206190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_206190)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_64'
            return stypy_return_type_206190

        # Assigning a type to the variable '_stypy_temp_lambda_64' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), '_stypy_temp_lambda_64', _stypy_temp_lambda_64)
        # Getting the type of '_stypy_temp_lambda_64' (line 111)
        _stypy_temp_lambda_64_206191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), '_stypy_temp_lambda_64')
        # Getting the type of 'self' (line 111)
        self_206192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'self')
        # Setting the type of the member 'jac' of a type (line 111)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), self_206192, 'jac', _stypy_temp_lambda_64_206191)
        # SSA branch for the else part of an if statement (line 109)
        module_type_store.open_ssa_branch('else')
        
        # Call to assert_(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'False' (line 113)
        False_206194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'False', False)
        # Processing the call keyword arguments (line 113)
        kwargs_206195 = {}
        # Getting the type of 'assert_' (line 113)
        assert__206193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 113)
        assert__call_result_206196 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), assert__206193, *[False_206194], **kwargs_206195)
        
        # SSA join for if statement (line 109)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def fun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun'
        module_type_store = module_type_store.open_function_context('fun', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BroydenTridiagonal.fun.__dict__.__setitem__('stypy_localization', localization)
        BroydenTridiagonal.fun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BroydenTridiagonal.fun.__dict__.__setitem__('stypy_type_store', module_type_store)
        BroydenTridiagonal.fun.__dict__.__setitem__('stypy_function_name', 'BroydenTridiagonal.fun')
        BroydenTridiagonal.fun.__dict__.__setitem__('stypy_param_names_list', ['x'])
        BroydenTridiagonal.fun.__dict__.__setitem__('stypy_varargs_param_name', None)
        BroydenTridiagonal.fun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BroydenTridiagonal.fun.__dict__.__setitem__('stypy_call_defaults', defaults)
        BroydenTridiagonal.fun.__dict__.__setitem__('stypy_call_varargs', varargs)
        BroydenTridiagonal.fun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BroydenTridiagonal.fun.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BroydenTridiagonal.fun', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun(...)' code ##################

        
        # Assigning a BinOp to a Name (line 116):
        int_206197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 13), 'int')
        # Getting the type of 'x' (line 116)
        x_206198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'x')
        # Applying the binary operator '-' (line 116)
        result_sub_206199 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 13), '-', int_206197, x_206198)
        
        # Getting the type of 'x' (line 116)
        x_206200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'x')
        # Applying the binary operator '*' (line 116)
        result_mul_206201 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 12), '*', result_sub_206199, x_206200)
        
        int_206202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 26), 'int')
        # Applying the binary operator '+' (line 116)
        result_add_206203 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 12), '+', result_mul_206201, int_206202)
        
        # Assigning a type to the variable 'f' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'f', result_add_206203)
        
        # Getting the type of 'f' (line 117)
        f_206204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'f')
        
        # Obtaining the type of the subscript
        int_206205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 10), 'int')
        slice_206206 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 117, 8), int_206205, None, None)
        # Getting the type of 'f' (line 117)
        f_206207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'f')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___206208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), f_206207, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_206209 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), getitem___206208, slice_206206)
        
        
        # Obtaining the type of the subscript
        int_206210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 20), 'int')
        slice_206211 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 117, 17), None, int_206210, None)
        # Getting the type of 'x' (line 117)
        x_206212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'x')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___206213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 17), x_206212, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_206214 = invoke(stypy.reporting.localization.Localization(__file__, 117, 17), getitem___206213, slice_206211)
        
        # Applying the binary operator '-=' (line 117)
        result_isub_206215 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 8), '-=', subscript_call_result_206209, subscript_call_result_206214)
        # Getting the type of 'f' (line 117)
        f_206216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'f')
        int_206217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 10), 'int')
        slice_206218 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 117, 8), int_206217, None, None)
        # Storing an element on a container (line 117)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 8), f_206216, (slice_206218, result_isub_206215))
        
        
        # Getting the type of 'f' (line 118)
        f_206219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'f')
        
        # Obtaining the type of the subscript
        int_206220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 11), 'int')
        slice_206221 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 118, 8), None, int_206220, None)
        # Getting the type of 'f' (line 118)
        f_206222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'f')
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___206223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), f_206222, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_206224 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), getitem___206223, slice_206221)
        
        int_206225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 18), 'int')
        
        # Obtaining the type of the subscript
        int_206226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 24), 'int')
        slice_206227 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 118, 22), int_206226, None, None)
        # Getting the type of 'x' (line 118)
        x_206228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), 'x')
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___206229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 22), x_206228, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_206230 = invoke(stypy.reporting.localization.Localization(__file__, 118, 22), getitem___206229, slice_206227)
        
        # Applying the binary operator '*' (line 118)
        result_mul_206231 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 18), '*', int_206225, subscript_call_result_206230)
        
        # Applying the binary operator '-=' (line 118)
        result_isub_206232 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 8), '-=', subscript_call_result_206224, result_mul_206231)
        # Getting the type of 'f' (line 118)
        f_206233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'f')
        int_206234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 11), 'int')
        slice_206235 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 118, 8), None, int_206234, None)
        # Storing an element on a container (line 118)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 8), f_206233, (slice_206235, result_isub_206232))
        
        # Getting the type of 'f' (line 119)
        f_206236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'f')
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'stypy_return_type', f_206236)
        
        # ################# End of 'fun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_206237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206237)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun'
        return stypy_return_type_206237


    @norecursion
    def _jac(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_jac'
        module_type_store = module_type_store.open_function_context('_jac', 121, 4, False)
        # Assigning a type to the variable 'self' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BroydenTridiagonal._jac.__dict__.__setitem__('stypy_localization', localization)
        BroydenTridiagonal._jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BroydenTridiagonal._jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        BroydenTridiagonal._jac.__dict__.__setitem__('stypy_function_name', 'BroydenTridiagonal._jac')
        BroydenTridiagonal._jac.__dict__.__setitem__('stypy_param_names_list', ['x'])
        BroydenTridiagonal._jac.__dict__.__setitem__('stypy_varargs_param_name', None)
        BroydenTridiagonal._jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BroydenTridiagonal._jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        BroydenTridiagonal._jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        BroydenTridiagonal._jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BroydenTridiagonal._jac.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BroydenTridiagonal._jac', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_jac', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_jac(...)' code ##################

        
        # Assigning a Call to a Name (line 122):
        
        # Call to lil_matrix(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining an instance of the builtin type 'tuple' (line 122)
        tuple_206239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 122)
        # Adding element type (line 122)
        # Getting the type of 'self' (line 122)
        self_206240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'self', False)
        # Obtaining the member 'n' of a type (line 122)
        n_206241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 24), self_206240, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 24), tuple_206239, n_206241)
        # Adding element type (line 122)
        # Getting the type of 'self' (line 122)
        self_206242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 32), 'self', False)
        # Obtaining the member 'n' of a type (line 122)
        n_206243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 32), self_206242, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 24), tuple_206239, n_206243)
        
        # Processing the call keyword arguments (line 122)
        kwargs_206244 = {}
        # Getting the type of 'lil_matrix' (line 122)
        lil_matrix_206238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'lil_matrix', False)
        # Calling lil_matrix(args, kwargs) (line 122)
        lil_matrix_call_result_206245 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), lil_matrix_206238, *[tuple_206239], **kwargs_206244)
        
        # Assigning a type to the variable 'J' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'J', lil_matrix_call_result_206245)
        
        # Assigning a Call to a Name (line 123):
        
        # Call to arange(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'self' (line 123)
        self_206248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 'self', False)
        # Obtaining the member 'n' of a type (line 123)
        n_206249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 22), self_206248, 'n')
        # Processing the call keyword arguments (line 123)
        kwargs_206250 = {}
        # Getting the type of 'np' (line 123)
        np_206246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 123)
        arange_206247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), np_206246, 'arange')
        # Calling arange(args, kwargs) (line 123)
        arange_call_result_206251 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), arange_206247, *[n_206249], **kwargs_206250)
        
        # Assigning a type to the variable 'i' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'i', arange_call_result_206251)
        
        # Assigning a BinOp to a Subscript (line 124):
        int_206252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 18), 'int')
        int_206253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 22), 'int')
        # Getting the type of 'x' (line 124)
        x_206254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 26), 'x')
        # Applying the binary operator '*' (line 124)
        result_mul_206255 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 22), '*', int_206253, x_206254)
        
        # Applying the binary operator '-' (line 124)
        result_sub_206256 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 18), '-', int_206252, result_mul_206255)
        
        # Getting the type of 'J' (line 124)
        J_206257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'J')
        
        # Obtaining an instance of the builtin type 'tuple' (line 124)
        tuple_206258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 124)
        # Adding element type (line 124)
        # Getting the type of 'i' (line 124)
        i_206259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 10), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 10), tuple_206258, i_206259)
        # Adding element type (line 124)
        # Getting the type of 'i' (line 124)
        i_206260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 10), tuple_206258, i_206260)
        
        # Storing an element on a container (line 124)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), J_206257, (tuple_206258, result_sub_206256))
        
        # Assigning a Call to a Name (line 125):
        
        # Call to arange(...): (line 125)
        # Processing the call arguments (line 125)
        int_206263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 22), 'int')
        # Getting the type of 'self' (line 125)
        self_206264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'self', False)
        # Obtaining the member 'n' of a type (line 125)
        n_206265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 25), self_206264, 'n')
        # Processing the call keyword arguments (line 125)
        kwargs_206266 = {}
        # Getting the type of 'np' (line 125)
        np_206261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 125)
        arange_206262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), np_206261, 'arange')
        # Calling arange(args, kwargs) (line 125)
        arange_call_result_206267 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), arange_206262, *[int_206263, n_206265], **kwargs_206266)
        
        # Assigning a type to the variable 'i' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'i', arange_call_result_206267)
        
        # Assigning a Num to a Subscript (line 126):
        int_206268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 22), 'int')
        # Getting the type of 'J' (line 126)
        J_206269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'J')
        
        # Obtaining an instance of the builtin type 'tuple' (line 126)
        tuple_206270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 126)
        # Adding element type (line 126)
        # Getting the type of 'i' (line 126)
        i_206271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 10), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 10), tuple_206270, i_206271)
        # Adding element type (line 126)
        # Getting the type of 'i' (line 126)
        i_206272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 13), 'i')
        int_206273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 17), 'int')
        # Applying the binary operator '-' (line 126)
        result_sub_206274 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 13), '-', i_206272, int_206273)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 10), tuple_206270, result_sub_206274)
        
        # Storing an element on a container (line 126)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 8), J_206269, (tuple_206270, int_206268))
        
        # Assigning a Call to a Name (line 127):
        
        # Call to arange(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'self' (line 127)
        self_206277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 22), 'self', False)
        # Obtaining the member 'n' of a type (line 127)
        n_206278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 22), self_206277, 'n')
        int_206279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 31), 'int')
        # Applying the binary operator '-' (line 127)
        result_sub_206280 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 22), '-', n_206278, int_206279)
        
        # Processing the call keyword arguments (line 127)
        kwargs_206281 = {}
        # Getting the type of 'np' (line 127)
        np_206275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 127)
        arange_206276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), np_206275, 'arange')
        # Calling arange(args, kwargs) (line 127)
        arange_call_result_206282 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), arange_206276, *[result_sub_206280], **kwargs_206281)
        
        # Assigning a type to the variable 'i' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'i', arange_call_result_206282)
        
        # Assigning a Num to a Subscript (line 128):
        int_206283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 22), 'int')
        # Getting the type of 'J' (line 128)
        J_206284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'J')
        
        # Obtaining an instance of the builtin type 'tuple' (line 128)
        tuple_206285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 128)
        # Adding element type (line 128)
        # Getting the type of 'i' (line 128)
        i_206286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 10), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 10), tuple_206285, i_206286)
        # Adding element type (line 128)
        # Getting the type of 'i' (line 128)
        i_206287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 13), 'i')
        int_206288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 17), 'int')
        # Applying the binary operator '+' (line 128)
        result_add_206289 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 13), '+', i_206287, int_206288)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 10), tuple_206285, result_add_206289)
        
        # Storing an element on a container (line 128)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 8), J_206284, (tuple_206285, int_206283))
        # Getting the type of 'J' (line 129)
        J_206290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'J')
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', J_206290)
        
        # ################# End of '_jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_jac' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_206291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206291)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_jac'
        return stypy_return_type_206291


# Assigning a type to the variable 'BroydenTridiagonal' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'BroydenTridiagonal', BroydenTridiagonal)
# Declaration of the 'ExponentialFittingProblem' class

class ExponentialFittingProblem(object, ):
    str_206292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, (-1)), 'str', 'Provide data and function for exponential fitting in the form\n    y = a + exp(b * x) + noise.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_206293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 47), 'int')
        
        # Obtaining an instance of the builtin type 'tuple' (line 136)
        tuple_206294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 136)
        # Adding element type (line 136)
        int_206295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 59), tuple_206294, int_206295)
        # Adding element type (line 136)
        int_206296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 59), tuple_206294, int_206296)
        
        int_206297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 26), 'int')
        # Getting the type of 'None' (line 137)
        None_206298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), 'None')
        defaults = [int_206293, tuple_206294, int_206297, None_206298]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ExponentialFittingProblem.__init__', ['a', 'b', 'noise', 'n_outliers', 'x_range', 'n_points', 'random_seed'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['a', 'b', 'noise', 'n_outliers', 'x_range', 'n_points', 'random_seed'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to seed(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'random_seed' (line 138)
        random_seed_206302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'random_seed', False)
        # Processing the call keyword arguments (line 138)
        kwargs_206303 = {}
        # Getting the type of 'np' (line 138)
        np_206299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 138)
        random_206300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), np_206299, 'random')
        # Obtaining the member 'seed' of a type (line 138)
        seed_206301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), random_206300, 'seed')
        # Calling seed(args, kwargs) (line 138)
        seed_call_result_206304 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), seed_206301, *[random_seed_206302], **kwargs_206303)
        
        
        # Assigning a Name to a Attribute (line 139):
        # Getting the type of 'n_points' (line 139)
        n_points_206305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 17), 'n_points')
        # Getting the type of 'self' (line 139)
        self_206306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self')
        # Setting the type of the member 'm' of a type (line 139)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_206306, 'm', n_points_206305)
        
        # Assigning a Num to a Attribute (line 140):
        int_206307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 17), 'int')
        # Getting the type of 'self' (line 140)
        self_206308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'self')
        # Setting the type of the member 'n' of a type (line 140)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), self_206308, 'n', int_206307)
        
        # Assigning a Call to a Attribute (line 142):
        
        # Call to zeros(...): (line 142)
        # Processing the call arguments (line 142)
        int_206311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 27), 'int')
        # Processing the call keyword arguments (line 142)
        kwargs_206312 = {}
        # Getting the type of 'np' (line 142)
        np_206309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'np', False)
        # Obtaining the member 'zeros' of a type (line 142)
        zeros_206310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 18), np_206309, 'zeros')
        # Calling zeros(args, kwargs) (line 142)
        zeros_call_result_206313 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), zeros_206310, *[int_206311], **kwargs_206312)
        
        # Getting the type of 'self' (line 142)
        self_206314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self')
        # Setting the type of the member 'p0' of a type (line 142)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_206314, 'p0', zeros_call_result_206313)
        
        # Assigning a Call to a Attribute (line 143):
        
        # Call to linspace(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Obtaining the type of the subscript
        int_206317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 37), 'int')
        # Getting the type of 'x_range' (line 143)
        x_range_206318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'x_range', False)
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___206319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 29), x_range_206318, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_206320 = invoke(stypy.reporting.localization.Localization(__file__, 143, 29), getitem___206319, int_206317)
        
        
        # Obtaining the type of the subscript
        int_206321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 49), 'int')
        # Getting the type of 'x_range' (line 143)
        x_range_206322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 41), 'x_range', False)
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___206323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 41), x_range_206322, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_206324 = invoke(stypy.reporting.localization.Localization(__file__, 143, 41), getitem___206323, int_206321)
        
        # Getting the type of 'n_points' (line 143)
        n_points_206325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 53), 'n_points', False)
        # Processing the call keyword arguments (line 143)
        kwargs_206326 = {}
        # Getting the type of 'np' (line 143)
        np_206315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 17), 'np', False)
        # Obtaining the member 'linspace' of a type (line 143)
        linspace_206316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 17), np_206315, 'linspace')
        # Calling linspace(args, kwargs) (line 143)
        linspace_call_result_206327 = invoke(stypy.reporting.localization.Localization(__file__, 143, 17), linspace_206316, *[subscript_call_result_206320, subscript_call_result_206324, n_points_206325], **kwargs_206326)
        
        # Getting the type of 'self' (line 143)
        self_206328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self')
        # Setting the type of the member 'x' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_206328, 'x', linspace_call_result_206327)
        
        # Assigning a BinOp to a Attribute (line 145):
        # Getting the type of 'a' (line 145)
        a_206329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 17), 'a')
        
        # Call to exp(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'b' (line 145)
        b_206332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 28), 'b', False)
        # Getting the type of 'self' (line 145)
        self_206333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 32), 'self', False)
        # Obtaining the member 'x' of a type (line 145)
        x_206334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 32), self_206333, 'x')
        # Applying the binary operator '*' (line 145)
        result_mul_206335 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 28), '*', b_206332, x_206334)
        
        # Processing the call keyword arguments (line 145)
        kwargs_206336 = {}
        # Getting the type of 'np' (line 145)
        np_206330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 21), 'np', False)
        # Obtaining the member 'exp' of a type (line 145)
        exp_206331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 21), np_206330, 'exp')
        # Calling exp(args, kwargs) (line 145)
        exp_call_result_206337 = invoke(stypy.reporting.localization.Localization(__file__, 145, 21), exp_206331, *[result_mul_206335], **kwargs_206336)
        
        # Applying the binary operator '+' (line 145)
        result_add_206338 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 17), '+', a_206329, exp_call_result_206337)
        
        # Getting the type of 'self' (line 145)
        self_206339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self')
        # Setting the type of the member 'y' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_206339, 'y', result_add_206338)
        
        # Getting the type of 'self' (line 146)
        self_206340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self')
        # Obtaining the member 'y' of a type (line 146)
        y_206341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_206340, 'y')
        # Getting the type of 'noise' (line 146)
        noise_206342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'noise')
        
        # Call to randn(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'self' (line 146)
        self_206346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 42), 'self', False)
        # Obtaining the member 'm' of a type (line 146)
        m_206347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 42), self_206346, 'm')
        # Processing the call keyword arguments (line 146)
        kwargs_206348 = {}
        # Getting the type of 'np' (line 146)
        np_206343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'np', False)
        # Obtaining the member 'random' of a type (line 146)
        random_206344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 26), np_206343, 'random')
        # Obtaining the member 'randn' of a type (line 146)
        randn_206345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 26), random_206344, 'randn')
        # Calling randn(args, kwargs) (line 146)
        randn_call_result_206349 = invoke(stypy.reporting.localization.Localization(__file__, 146, 26), randn_206345, *[m_206347], **kwargs_206348)
        
        # Applying the binary operator '*' (line 146)
        result_mul_206350 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 18), '*', noise_206342, randn_call_result_206349)
        
        # Applying the binary operator '+=' (line 146)
        result_iadd_206351 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 8), '+=', y_206341, result_mul_206350)
        # Getting the type of 'self' (line 146)
        self_206352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self')
        # Setting the type of the member 'y' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_206352, 'y', result_iadd_206351)
        
        
        # Assigning a Call to a Name (line 148):
        
        # Call to randint(...): (line 148)
        # Processing the call arguments (line 148)
        int_206356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 37), 'int')
        # Getting the type of 'self' (line 148)
        self_206357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 40), 'self', False)
        # Obtaining the member 'm' of a type (line 148)
        m_206358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 40), self_206357, 'm')
        # Getting the type of 'n_outliers' (line 148)
        n_outliers_206359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 48), 'n_outliers', False)
        # Processing the call keyword arguments (line 148)
        kwargs_206360 = {}
        # Getting the type of 'np' (line 148)
        np_206353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'np', False)
        # Obtaining the member 'random' of a type (line 148)
        random_206354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 19), np_206353, 'random')
        # Obtaining the member 'randint' of a type (line 148)
        randint_206355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 19), random_206354, 'randint')
        # Calling randint(args, kwargs) (line 148)
        randint_call_result_206361 = invoke(stypy.reporting.localization.Localization(__file__, 148, 19), randint_206355, *[int_206356, m_206358, n_outliers_206359], **kwargs_206360)
        
        # Assigning a type to the variable 'outliers' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'outliers', randint_call_result_206361)
        
        # Getting the type of 'self' (line 149)
        self_206362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self')
        # Obtaining the member 'y' of a type (line 149)
        y_206363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_206362, 'y')
        
        # Obtaining the type of the subscript
        # Getting the type of 'outliers' (line 149)
        outliers_206364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'outliers')
        # Getting the type of 'self' (line 149)
        self_206365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self')
        # Obtaining the member 'y' of a type (line 149)
        y_206366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_206365, 'y')
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___206367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), y_206366, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_206368 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), getitem___206367, outliers_206364)
        
        int_206369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 28), 'int')
        # Getting the type of 'noise' (line 149)
        noise_206370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 33), 'noise')
        # Applying the binary operator '*' (line 149)
        result_mul_206371 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 28), '*', int_206369, noise_206370)
        
        
        # Call to rand(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'n_outliers' (line 149)
        n_outliers_206375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 56), 'n_outliers', False)
        # Processing the call keyword arguments (line 149)
        kwargs_206376 = {}
        # Getting the type of 'np' (line 149)
        np_206372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 41), 'np', False)
        # Obtaining the member 'random' of a type (line 149)
        random_206373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 41), np_206372, 'random')
        # Obtaining the member 'rand' of a type (line 149)
        rand_206374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 41), random_206373, 'rand')
        # Calling rand(args, kwargs) (line 149)
        rand_call_result_206377 = invoke(stypy.reporting.localization.Localization(__file__, 149, 41), rand_206374, *[n_outliers_206375], **kwargs_206376)
        
        # Applying the binary operator '*' (line 149)
        result_mul_206378 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 39), '*', result_mul_206371, rand_call_result_206377)
        
        # Applying the binary operator '+=' (line 149)
        result_iadd_206379 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 8), '+=', subscript_call_result_206368, result_mul_206378)
        # Getting the type of 'self' (line 149)
        self_206380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self')
        # Obtaining the member 'y' of a type (line 149)
        y_206381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_206380, 'y')
        # Getting the type of 'outliers' (line 149)
        outliers_206382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'outliers')
        # Storing an element on a container (line 149)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 8), y_206381, (outliers_206382, result_iadd_206379))
        
        
        # Assigning a Call to a Attribute (line 151):
        
        # Call to array(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_206385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        # Getting the type of 'a' (line 151)
        a_206386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 31), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 30), list_206385, a_206386)
        # Adding element type (line 151)
        # Getting the type of 'b' (line 151)
        b_206387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 30), list_206385, b_206387)
        
        # Processing the call keyword arguments (line 151)
        kwargs_206388 = {}
        # Getting the type of 'np' (line 151)
        np_206383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'np', False)
        # Obtaining the member 'array' of a type (line 151)
        array_206384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 21), np_206383, 'array')
        # Calling array(args, kwargs) (line 151)
        array_call_result_206389 = invoke(stypy.reporting.localization.Localization(__file__, 151, 21), array_206384, *[list_206385], **kwargs_206388)
        
        # Getting the type of 'self' (line 151)
        self_206390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'self')
        # Setting the type of the member 'p_opt' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), self_206390, 'p_opt', array_call_result_206389)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def fun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun'
        module_type_store = module_type_store.open_function_context('fun', 153, 4, False)
        # Assigning a type to the variable 'self' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ExponentialFittingProblem.fun.__dict__.__setitem__('stypy_localization', localization)
        ExponentialFittingProblem.fun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ExponentialFittingProblem.fun.__dict__.__setitem__('stypy_type_store', module_type_store)
        ExponentialFittingProblem.fun.__dict__.__setitem__('stypy_function_name', 'ExponentialFittingProblem.fun')
        ExponentialFittingProblem.fun.__dict__.__setitem__('stypy_param_names_list', ['p'])
        ExponentialFittingProblem.fun.__dict__.__setitem__('stypy_varargs_param_name', None)
        ExponentialFittingProblem.fun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ExponentialFittingProblem.fun.__dict__.__setitem__('stypy_call_defaults', defaults)
        ExponentialFittingProblem.fun.__dict__.__setitem__('stypy_call_varargs', varargs)
        ExponentialFittingProblem.fun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ExponentialFittingProblem.fun.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ExponentialFittingProblem.fun', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun(...)' code ##################

        
        # Obtaining the type of the subscript
        int_206391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 17), 'int')
        # Getting the type of 'p' (line 154)
        p_206392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'p')
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___206393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 15), p_206392, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_206394 = invoke(stypy.reporting.localization.Localization(__file__, 154, 15), getitem___206393, int_206391)
        
        
        # Call to exp(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Obtaining the type of the subscript
        int_206397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 31), 'int')
        # Getting the type of 'p' (line 154)
        p_206398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___206399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 29), p_206398, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_206400 = invoke(stypy.reporting.localization.Localization(__file__, 154, 29), getitem___206399, int_206397)
        
        # Getting the type of 'self' (line 154)
        self_206401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 36), 'self', False)
        # Obtaining the member 'x' of a type (line 154)
        x_206402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 36), self_206401, 'x')
        # Applying the binary operator '*' (line 154)
        result_mul_206403 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 29), '*', subscript_call_result_206400, x_206402)
        
        # Processing the call keyword arguments (line 154)
        kwargs_206404 = {}
        # Getting the type of 'np' (line 154)
        np_206395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'np', False)
        # Obtaining the member 'exp' of a type (line 154)
        exp_206396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 22), np_206395, 'exp')
        # Calling exp(args, kwargs) (line 154)
        exp_call_result_206405 = invoke(stypy.reporting.localization.Localization(__file__, 154, 22), exp_206396, *[result_mul_206403], **kwargs_206404)
        
        # Applying the binary operator '+' (line 154)
        result_add_206406 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 15), '+', subscript_call_result_206394, exp_call_result_206405)
        
        # Getting the type of 'self' (line 154)
        self_206407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 46), 'self')
        # Obtaining the member 'y' of a type (line 154)
        y_206408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 46), self_206407, 'y')
        # Applying the binary operator '-' (line 154)
        result_sub_206409 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 44), '-', result_add_206406, y_206408)
        
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', result_sub_206409)
        
        # ################# End of 'fun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_206410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206410)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun'
        return stypy_return_type_206410


    @norecursion
    def jac(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac'
        module_type_store = module_type_store.open_function_context('jac', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ExponentialFittingProblem.jac.__dict__.__setitem__('stypy_localization', localization)
        ExponentialFittingProblem.jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ExponentialFittingProblem.jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        ExponentialFittingProblem.jac.__dict__.__setitem__('stypy_function_name', 'ExponentialFittingProblem.jac')
        ExponentialFittingProblem.jac.__dict__.__setitem__('stypy_param_names_list', ['p'])
        ExponentialFittingProblem.jac.__dict__.__setitem__('stypy_varargs_param_name', None)
        ExponentialFittingProblem.jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ExponentialFittingProblem.jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        ExponentialFittingProblem.jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        ExponentialFittingProblem.jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ExponentialFittingProblem.jac.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ExponentialFittingProblem.jac', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac(...)' code ##################

        
        # Assigning a Call to a Name (line 157):
        
        # Call to empty(...): (line 157)
        # Processing the call arguments (line 157)
        
        # Obtaining an instance of the builtin type 'tuple' (line 157)
        tuple_206413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 157)
        # Adding element type (line 157)
        # Getting the type of 'self' (line 157)
        self_206414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 22), 'self', False)
        # Obtaining the member 'm' of a type (line 157)
        m_206415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 22), self_206414, 'm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 22), tuple_206413, m_206415)
        # Adding element type (line 157)
        # Getting the type of 'self' (line 157)
        self_206416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 'self', False)
        # Obtaining the member 'n' of a type (line 157)
        n_206417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 30), self_206416, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 22), tuple_206413, n_206417)
        
        # Processing the call keyword arguments (line 157)
        kwargs_206418 = {}
        # Getting the type of 'np' (line 157)
        np_206411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'np', False)
        # Obtaining the member 'empty' of a type (line 157)
        empty_206412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), np_206411, 'empty')
        # Calling empty(args, kwargs) (line 157)
        empty_call_result_206419 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), empty_206412, *[tuple_206413], **kwargs_206418)
        
        # Assigning a type to the variable 'J' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'J', empty_call_result_206419)
        
        # Assigning a Num to a Subscript (line 158):
        int_206420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 18), 'int')
        # Getting the type of 'J' (line 158)
        J_206421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'J')
        slice_206422 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 158, 8), None, None, None)
        int_206423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 13), 'int')
        # Storing an element on a container (line 158)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 8), J_206421, ((slice_206422, int_206423), int_206420))
        
        # Assigning a BinOp to a Subscript (line 159):
        # Getting the type of 'self' (line 159)
        self_206424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 18), 'self')
        # Obtaining the member 'x' of a type (line 159)
        x_206425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 18), self_206424, 'x')
        
        # Call to exp(...): (line 159)
        # Processing the call arguments (line 159)
        
        # Obtaining the type of the subscript
        int_206428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 36), 'int')
        # Getting the type of 'p' (line 159)
        p_206429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___206430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 34), p_206429, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_206431 = invoke(stypy.reporting.localization.Localization(__file__, 159, 34), getitem___206430, int_206428)
        
        # Getting the type of 'self' (line 159)
        self_206432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 41), 'self', False)
        # Obtaining the member 'x' of a type (line 159)
        x_206433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 41), self_206432, 'x')
        # Applying the binary operator '*' (line 159)
        result_mul_206434 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 34), '*', subscript_call_result_206431, x_206433)
        
        # Processing the call keyword arguments (line 159)
        kwargs_206435 = {}
        # Getting the type of 'np' (line 159)
        np_206426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 27), 'np', False)
        # Obtaining the member 'exp' of a type (line 159)
        exp_206427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 27), np_206426, 'exp')
        # Calling exp(args, kwargs) (line 159)
        exp_call_result_206436 = invoke(stypy.reporting.localization.Localization(__file__, 159, 27), exp_206427, *[result_mul_206434], **kwargs_206435)
        
        # Applying the binary operator '*' (line 159)
        result_mul_206437 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 18), '*', x_206425, exp_call_result_206436)
        
        # Getting the type of 'J' (line 159)
        J_206438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'J')
        slice_206439 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 159, 8), None, None, None)
        int_206440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 13), 'int')
        # Storing an element on a container (line 159)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 8), J_206438, ((slice_206439, int_206440), result_mul_206437))
        # Getting the type of 'J' (line 160)
        J_206441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'J')
        # Assigning a type to the variable 'stypy_return_type' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'stypy_return_type', J_206441)
        
        # ################# End of 'jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac' in the type store
        # Getting the type of 'stypy_return_type' (line 156)
        stypy_return_type_206442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206442)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac'
        return stypy_return_type_206442


# Assigning a type to the variable 'ExponentialFittingProblem' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'ExponentialFittingProblem', ExponentialFittingProblem)

@norecursion
def cubic_soft_l1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cubic_soft_l1'
    module_type_store = module_type_store.open_function_context('cubic_soft_l1', 163, 0, False)
    
    # Passed parameters checking function
    cubic_soft_l1.stypy_localization = localization
    cubic_soft_l1.stypy_type_of_self = None
    cubic_soft_l1.stypy_type_store = module_type_store
    cubic_soft_l1.stypy_function_name = 'cubic_soft_l1'
    cubic_soft_l1.stypy_param_names_list = ['z']
    cubic_soft_l1.stypy_varargs_param_name = None
    cubic_soft_l1.stypy_kwargs_param_name = None
    cubic_soft_l1.stypy_call_defaults = defaults
    cubic_soft_l1.stypy_call_varargs = varargs
    cubic_soft_l1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cubic_soft_l1', ['z'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cubic_soft_l1', localization, ['z'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cubic_soft_l1(...)' code ##################

    
    # Assigning a Call to a Name (line 164):
    
    # Call to empty(...): (line 164)
    # Processing the call arguments (line 164)
    
    # Obtaining an instance of the builtin type 'tuple' (line 164)
    tuple_206445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 164)
    # Adding element type (line 164)
    int_206446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 20), tuple_206445, int_206446)
    # Adding element type (line 164)
    # Getting the type of 'z' (line 164)
    z_206447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'z', False)
    # Obtaining the member 'size' of a type (line 164)
    size_206448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 23), z_206447, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 20), tuple_206445, size_206448)
    
    # Processing the call keyword arguments (line 164)
    kwargs_206449 = {}
    # Getting the type of 'np' (line 164)
    np_206443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 164)
    empty_206444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 10), np_206443, 'empty')
    # Calling empty(args, kwargs) (line 164)
    empty_call_result_206450 = invoke(stypy.reporting.localization.Localization(__file__, 164, 10), empty_206444, *[tuple_206445], **kwargs_206449)
    
    # Assigning a type to the variable 'rho' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'rho', empty_call_result_206450)
    
    # Assigning a BinOp to a Name (line 166):
    int_206451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
    # Getting the type of 'z' (line 166)
    z_206452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'z')
    # Applying the binary operator '+' (line 166)
    result_add_206453 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 8), '+', int_206451, z_206452)
    
    # Assigning a type to the variable 't' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 't', result_add_206453)
    
    # Assigning a BinOp to a Subscript (line 167):
    int_206454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 13), 'int')
    # Getting the type of 't' (line 167)
    t_206455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 't')
    int_206456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 22), 'int')
    int_206457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 24), 'int')
    # Applying the binary operator 'div' (line 167)
    result_div_206458 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 22), 'div', int_206456, int_206457)
    
    # Applying the binary operator '**' (line 167)
    result_pow_206459 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 18), '**', t_206455, result_div_206458)
    
    int_206460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 29), 'int')
    # Applying the binary operator '-' (line 167)
    result_sub_206461 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 18), '-', result_pow_206459, int_206460)
    
    # Applying the binary operator '*' (line 167)
    result_mul_206462 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 13), '*', int_206454, result_sub_206461)
    
    # Getting the type of 'rho' (line 167)
    rho_206463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'rho')
    int_206464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 8), 'int')
    # Storing an element on a container (line 167)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 4), rho_206463, (int_206464, result_mul_206462))
    
    # Assigning a BinOp to a Subscript (line 168):
    # Getting the type of 't' (line 168)
    t_206465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 13), 't')
    int_206466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 19), 'int')
    int_206467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 22), 'int')
    # Applying the binary operator 'div' (line 168)
    result_div_206468 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 19), 'div', int_206466, int_206467)
    
    # Applying the binary operator '**' (line 168)
    result_pow_206469 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 13), '**', t_206465, result_div_206468)
    
    # Getting the type of 'rho' (line 168)
    rho_206470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'rho')
    int_206471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 8), 'int')
    # Storing an element on a container (line 168)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 4), rho_206470, (int_206471, result_pow_206469))
    
    # Assigning a BinOp to a Subscript (line 169):
    int_206472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 13), 'int')
    int_206473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 16), 'int')
    # Applying the binary operator 'div' (line 169)
    result_div_206474 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 13), 'div', int_206472, int_206473)
    
    # Getting the type of 't' (line 169)
    t_206475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 't')
    int_206476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 24), 'int')
    int_206477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 27), 'int')
    # Applying the binary operator 'div' (line 169)
    result_div_206478 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 24), 'div', int_206476, int_206477)
    
    # Applying the binary operator '**' (line 169)
    result_pow_206479 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 20), '**', t_206475, result_div_206478)
    
    # Applying the binary operator '*' (line 169)
    result_mul_206480 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 18), '*', result_div_206474, result_pow_206479)
    
    # Getting the type of 'rho' (line 169)
    rho_206481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'rho')
    int_206482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'int')
    # Storing an element on a container (line 169)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 4), rho_206481, (int_206482, result_mul_206480))
    # Getting the type of 'rho' (line 171)
    rho_206483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'rho')
    # Assigning a type to the variable 'stypy_return_type' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type', rho_206483)
    
    # ################# End of 'cubic_soft_l1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cubic_soft_l1' in the type store
    # Getting the type of 'stypy_return_type' (line 163)
    stypy_return_type_206484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_206484)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cubic_soft_l1'
    return stypy_return_type_206484

# Assigning a type to the variable 'cubic_soft_l1' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'cubic_soft_l1', cubic_soft_l1)

# Assigning a BinOp to a Name (line 174):

# Call to list(...): (line 174)
# Processing the call arguments (line 174)

# Call to keys(...): (line 174)
# Processing the call keyword arguments (line 174)
kwargs_206488 = {}
# Getting the type of 'IMPLEMENTED_LOSSES' (line 174)
IMPLEMENTED_LOSSES_206486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 14), 'IMPLEMENTED_LOSSES', False)
# Obtaining the member 'keys' of a type (line 174)
keys_206487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 14), IMPLEMENTED_LOSSES_206486, 'keys')
# Calling keys(args, kwargs) (line 174)
keys_call_result_206489 = invoke(stypy.reporting.localization.Localization(__file__, 174, 14), keys_206487, *[], **kwargs_206488)

# Processing the call keyword arguments (line 174)
kwargs_206490 = {}
# Getting the type of 'list' (line 174)
list_206485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 9), 'list', False)
# Calling list(args, kwargs) (line 174)
list_call_result_206491 = invoke(stypy.reporting.localization.Localization(__file__, 174, 9), list_206485, *[keys_call_result_206489], **kwargs_206490)


# Obtaining an instance of the builtin type 'list' (line 174)
list_206492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 43), 'list')
# Adding type elements to the builtin type 'list' instance (line 174)
# Adding element type (line 174)
# Getting the type of 'cubic_soft_l1' (line 174)
cubic_soft_l1_206493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 44), 'cubic_soft_l1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 43), list_206492, cubic_soft_l1_206493)

# Applying the binary operator '+' (line 174)
result_add_206494 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 9), '+', list_call_result_206491, list_206492)

# Assigning a type to the variable 'LOSSES' (line 174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'LOSSES', result_add_206494)
# Declaration of the 'BaseMixin' class

class BaseMixin(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_basic.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_basic.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_basic')
        BaseMixin.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 180):
        
        # Call to least_squares(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'fun_trivial' (line 180)
        fun_trivial_206496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'fun_trivial', False)
        float_206497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 41), 'float')
        # Processing the call keyword arguments (line 180)
        # Getting the type of 'self' (line 180)
        self_206498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 52), 'self', False)
        # Obtaining the member 'method' of a type (line 180)
        method_206499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 52), self_206498, 'method')
        keyword_206500 = method_206499
        kwargs_206501 = {'method': keyword_206500}
        # Getting the type of 'least_squares' (line 180)
        least_squares_206495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 180)
        least_squares_call_result_206502 = invoke(stypy.reporting.localization.Localization(__file__, 180, 14), least_squares_206495, *[fun_trivial_206496, float_206497], **kwargs_206501)
        
        # Assigning a type to the variable 'res' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'res', least_squares_call_result_206502)
        
        # Call to assert_allclose(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'res' (line 181)
        res_206504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 181)
        x_206505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 24), res_206504, 'x')
        int_206506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 31), 'int')
        # Processing the call keyword arguments (line 181)
        float_206507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 39), 'float')
        keyword_206508 = float_206507
        kwargs_206509 = {'atol': keyword_206508}
        # Getting the type of 'assert_allclose' (line 181)
        assert_allclose_206503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 181)
        assert_allclose_call_result_206510 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), assert_allclose_206503, *[x_206505, int_206506], **kwargs_206509)
        
        
        # Call to assert_allclose(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'res' (line 182)
        res_206512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 24), 'res', False)
        # Obtaining the member 'fun' of a type (line 182)
        fun_206513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 24), res_206512, 'fun')
        
        # Call to fun_trivial(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'res' (line 182)
        res_206515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 45), 'res', False)
        # Obtaining the member 'x' of a type (line 182)
        x_206516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 45), res_206515, 'x')
        # Processing the call keyword arguments (line 182)
        kwargs_206517 = {}
        # Getting the type of 'fun_trivial' (line 182)
        fun_trivial_206514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 33), 'fun_trivial', False)
        # Calling fun_trivial(args, kwargs) (line 182)
        fun_trivial_call_result_206518 = invoke(stypy.reporting.localization.Localization(__file__, 182, 33), fun_trivial_206514, *[x_206516], **kwargs_206517)
        
        # Processing the call keyword arguments (line 182)
        kwargs_206519 = {}
        # Getting the type of 'assert_allclose' (line 182)
        assert_allclose_206511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 182)
        assert_allclose_call_result_206520 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), assert_allclose_206511, *[fun_206513, fun_trivial_call_result_206518], **kwargs_206519)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_206521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206521)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_206521


    @norecursion
    def test_args_kwargs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_args_kwargs'
        module_type_store = module_type_store.open_function_context('test_args_kwargs', 184, 4, False)
        # Assigning a type to the variable 'self' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_args_kwargs.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_args_kwargs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_args_kwargs.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_args_kwargs.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_args_kwargs')
        BaseMixin.test_args_kwargs.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_args_kwargs.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_args_kwargs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_args_kwargs.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_args_kwargs.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_args_kwargs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_args_kwargs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_args_kwargs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_args_kwargs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_args_kwargs(...)' code ##################

        
        # Assigning a Num to a Name (line 186):
        float_206522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 12), 'float')
        # Assigning a type to the variable 'a' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'a', float_206522)
        
        
        # Obtaining an instance of the builtin type 'list' (line 187)
        list_206523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 187)
        # Adding element type (line 187)
        str_206524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 20), 'str', '2-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 19), list_206523, str_206524)
        # Adding element type (line 187)
        str_206525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 31), 'str', '3-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 19), list_206523, str_206525)
        # Adding element type (line 187)
        str_206526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 42), 'str', 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 19), list_206523, str_206526)
        # Adding element type (line 187)
        # Getting the type of 'jac_trivial' (line 187)
        jac_trivial_206527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 48), 'jac_trivial')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 19), list_206523, jac_trivial_206527)
        
        # Testing the type of a for loop iterable (line 187)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 187, 8), list_206523)
        # Getting the type of the for loop variable (line 187)
        for_loop_var_206528 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 187, 8), list_206523)
        # Assigning a type to the variable 'jac' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'jac', for_loop_var_206528)
        # SSA begins for a for statement (line 187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to suppress_warnings(...): (line 188)
        # Processing the call keyword arguments (line 188)
        kwargs_206530 = {}
        # Getting the type of 'suppress_warnings' (line 188)
        suppress_warnings_206529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 17), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 188)
        suppress_warnings_call_result_206531 = invoke(stypy.reporting.localization.Localization(__file__, 188, 17), suppress_warnings_206529, *[], **kwargs_206530)
        
        with_206532 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 188, 17), suppress_warnings_call_result_206531, 'with parameter', '__enter__', '__exit__')

        if with_206532:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 188)
            enter___206533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 17), suppress_warnings_call_result_206531, '__enter__')
            with_enter_206534 = invoke(stypy.reporting.localization.Localization(__file__, 188, 17), enter___206533)
            # Assigning a type to the variable 'sup' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 17), 'sup', with_enter_206534)
            
            # Call to filter(...): (line 189)
            # Processing the call arguments (line 189)
            # Getting the type of 'UserWarning' (line 189)
            UserWarning_206537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 27), 'UserWarning', False)
            str_206538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 27), 'str', "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'")
            # Processing the call keyword arguments (line 189)
            kwargs_206539 = {}
            # Getting the type of 'sup' (line 189)
            sup_206535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'sup', False)
            # Obtaining the member 'filter' of a type (line 189)
            filter_206536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 16), sup_206535, 'filter')
            # Calling filter(args, kwargs) (line 189)
            filter_call_result_206540 = invoke(stypy.reporting.localization.Localization(__file__, 189, 16), filter_206536, *[UserWarning_206537, str_206538], **kwargs_206539)
            
            
            # Assigning a Call to a Name (line 191):
            
            # Call to least_squares(...): (line 191)
            # Processing the call arguments (line 191)
            # Getting the type of 'fun_trivial' (line 191)
            fun_trivial_206542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 36), 'fun_trivial', False)
            float_206543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 49), 'float')
            # Getting the type of 'jac' (line 191)
            jac_206544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 54), 'jac', False)
            # Processing the call keyword arguments (line 191)
            
            # Obtaining an instance of the builtin type 'tuple' (line 191)
            tuple_206545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 65), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 191)
            # Adding element type (line 191)
            # Getting the type of 'a' (line 191)
            a_206546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 65), 'a', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 65), tuple_206545, a_206546)
            
            keyword_206547 = tuple_206545
            # Getting the type of 'self' (line 192)
            self_206548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 43), 'self', False)
            # Obtaining the member 'method' of a type (line 192)
            method_206549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 43), self_206548, 'method')
            keyword_206550 = method_206549
            kwargs_206551 = {'args': keyword_206547, 'method': keyword_206550}
            # Getting the type of 'least_squares' (line 191)
            least_squares_206541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 22), 'least_squares', False)
            # Calling least_squares(args, kwargs) (line 191)
            least_squares_call_result_206552 = invoke(stypy.reporting.localization.Localization(__file__, 191, 22), least_squares_206541, *[fun_trivial_206542, float_206543, jac_206544], **kwargs_206551)
            
            # Assigning a type to the variable 'res' (line 191)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'res', least_squares_call_result_206552)
            
            # Assigning a Call to a Name (line 193):
            
            # Call to least_squares(...): (line 193)
            # Processing the call arguments (line 193)
            # Getting the type of 'fun_trivial' (line 193)
            fun_trivial_206554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 37), 'fun_trivial', False)
            float_206555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 50), 'float')
            # Getting the type of 'jac' (line 193)
            jac_206556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 55), 'jac', False)
            # Processing the call keyword arguments (line 193)
            
            # Obtaining an instance of the builtin type 'dict' (line 193)
            dict_206557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 67), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 193)
            # Adding element type (key, value) (line 193)
            str_206558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 68), 'str', 'a')
            # Getting the type of 'a' (line 193)
            a_206559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 73), 'a', False)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 67), dict_206557, (str_206558, a_206559))
            
            keyword_206560 = dict_206557
            # Getting the type of 'self' (line 194)
            self_206561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 43), 'self', False)
            # Obtaining the member 'method' of a type (line 194)
            method_206562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 43), self_206561, 'method')
            keyword_206563 = method_206562
            kwargs_206564 = {'method': keyword_206563, 'kwargs': keyword_206560}
            # Getting the type of 'least_squares' (line 193)
            least_squares_206553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'least_squares', False)
            # Calling least_squares(args, kwargs) (line 193)
            least_squares_call_result_206565 = invoke(stypy.reporting.localization.Localization(__file__, 193, 23), least_squares_206553, *[fun_trivial_206554, float_206555, jac_206556], **kwargs_206564)
            
            # Assigning a type to the variable 'res1' (line 193)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'res1', least_squares_call_result_206565)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 188)
            exit___206566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 17), suppress_warnings_call_result_206531, '__exit__')
            with_exit_206567 = invoke(stypy.reporting.localization.Localization(__file__, 188, 17), exit___206566, None, None, None)

        
        # Call to assert_allclose(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'res' (line 196)
        res_206569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 196)
        x_206570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 28), res_206569, 'x')
        # Getting the type of 'a' (line 196)
        a_206571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 35), 'a', False)
        # Processing the call keyword arguments (line 196)
        float_206572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 43), 'float')
        keyword_206573 = float_206572
        kwargs_206574 = {'rtol': keyword_206573}
        # Getting the type of 'assert_allclose' (line 196)
        assert_allclose_206568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 196)
        assert_allclose_call_result_206575 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), assert_allclose_206568, *[x_206570, a_206571], **kwargs_206574)
        
        
        # Call to assert_allclose(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'res1' (line 197)
        res1_206577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 28), 'res1', False)
        # Obtaining the member 'x' of a type (line 197)
        x_206578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 28), res1_206577, 'x')
        # Getting the type of 'a' (line 197)
        a_206579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 36), 'a', False)
        # Processing the call keyword arguments (line 197)
        float_206580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 44), 'float')
        keyword_206581 = float_206580
        kwargs_206582 = {'rtol': keyword_206581}
        # Getting the type of 'assert_allclose' (line 197)
        assert_allclose_206576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 197)
        assert_allclose_call_result_206583 = invoke(stypy.reporting.localization.Localization(__file__, 197, 12), assert_allclose_206576, *[x_206578, a_206579], **kwargs_206582)
        
        
        # Call to assert_raises(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'TypeError' (line 199)
        TypeError_206585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 26), 'TypeError', False)
        # Getting the type of 'least_squares' (line 199)
        least_squares_206586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 37), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 199)
        fun_trivial_206587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 52), 'fun_trivial', False)
        float_206588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 65), 'float')
        # Processing the call keyword arguments (line 199)
        
        # Obtaining an instance of the builtin type 'tuple' (line 200)
        tuple_206589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 200)
        # Adding element type (line 200)
        int_206590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 32), tuple_206589, int_206590)
        # Adding element type (line 200)
        int_206591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 32), tuple_206589, int_206591)
        
        keyword_206592 = tuple_206589
        # Getting the type of 'self' (line 200)
        self_206593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 200)
        method_206594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 47), self_206593, 'method')
        keyword_206595 = method_206594
        kwargs_206596 = {'args': keyword_206592, 'method': keyword_206595}
        # Getting the type of 'assert_raises' (line 199)
        assert_raises_206584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 199)
        assert_raises_call_result_206597 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), assert_raises_206584, *[TypeError_206585, least_squares_206586, fun_trivial_206587, float_206588], **kwargs_206596)
        
        
        # Call to assert_raises(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'TypeError' (line 201)
        TypeError_206599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 26), 'TypeError', False)
        # Getting the type of 'least_squares' (line 201)
        least_squares_206600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 37), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 201)
        fun_trivial_206601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 52), 'fun_trivial', False)
        float_206602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 65), 'float')
        # Processing the call keyword arguments (line 201)
        
        # Obtaining an instance of the builtin type 'dict' (line 202)
        dict_206603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 33), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 202)
        # Adding element type (key, value) (line 202)
        str_206604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 34), 'str', 'kaboom')
        int_206605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 44), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 33), dict_206603, (str_206604, int_206605))
        
        keyword_206606 = dict_206603
        # Getting the type of 'self' (line 202)
        self_206607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 55), 'self', False)
        # Obtaining the member 'method' of a type (line 202)
        method_206608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 55), self_206607, 'method')
        keyword_206609 = method_206608
        kwargs_206610 = {'method': keyword_206609, 'kwargs': keyword_206606}
        # Getting the type of 'assert_raises' (line 201)
        assert_raises_206598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 201)
        assert_raises_call_result_206611 = invoke(stypy.reporting.localization.Localization(__file__, 201, 12), assert_raises_206598, *[TypeError_206599, least_squares_206600, fun_trivial_206601, float_206602], **kwargs_206610)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_args_kwargs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_args_kwargs' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_206612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206612)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_args_kwargs'
        return stypy_return_type_206612


    @norecursion
    def test_jac_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_jac_options'
        module_type_store = module_type_store.open_function_context('test_jac_options', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_jac_options.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_jac_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_jac_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_jac_options.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_jac_options')
        BaseMixin.test_jac_options.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_jac_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_jac_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_jac_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_jac_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_jac_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_jac_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_jac_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_jac_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_jac_options(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 205)
        list_206613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 205)
        # Adding element type (line 205)
        str_206614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 20), 'str', '2-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 19), list_206613, str_206614)
        # Adding element type (line 205)
        str_206615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 31), 'str', '3-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 19), list_206613, str_206615)
        # Adding element type (line 205)
        str_206616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 42), 'str', 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 19), list_206613, str_206616)
        # Adding element type (line 205)
        # Getting the type of 'jac_trivial' (line 205)
        jac_trivial_206617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 48), 'jac_trivial')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 19), list_206613, jac_trivial_206617)
        
        # Testing the type of a for loop iterable (line 205)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 205, 8), list_206613)
        # Getting the type of the for loop variable (line 205)
        for_loop_var_206618 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 205, 8), list_206613)
        # Assigning a type to the variable 'jac' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'jac', for_loop_var_206618)
        # SSA begins for a for statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to suppress_warnings(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_206620 = {}
        # Getting the type of 'suppress_warnings' (line 206)
        suppress_warnings_206619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 206)
        suppress_warnings_call_result_206621 = invoke(stypy.reporting.localization.Localization(__file__, 206, 17), suppress_warnings_206619, *[], **kwargs_206620)
        
        with_206622 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 206, 17), suppress_warnings_call_result_206621, 'with parameter', '__enter__', '__exit__')

        if with_206622:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 206)
            enter___206623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 17), suppress_warnings_call_result_206621, '__enter__')
            with_enter_206624 = invoke(stypy.reporting.localization.Localization(__file__, 206, 17), enter___206623)
            # Assigning a type to the variable 'sup' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'sup', with_enter_206624)
            
            # Call to filter(...): (line 207)
            # Processing the call arguments (line 207)
            # Getting the type of 'UserWarning' (line 207)
            UserWarning_206627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 27), 'UserWarning', False)
            str_206628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 27), 'str', "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'")
            # Processing the call keyword arguments (line 207)
            kwargs_206629 = {}
            # Getting the type of 'sup' (line 207)
            sup_206625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'sup', False)
            # Obtaining the member 'filter' of a type (line 207)
            filter_206626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), sup_206625, 'filter')
            # Calling filter(args, kwargs) (line 207)
            filter_call_result_206630 = invoke(stypy.reporting.localization.Localization(__file__, 207, 16), filter_206626, *[UserWarning_206627, str_206628], **kwargs_206629)
            
            
            # Assigning a Call to a Name (line 209):
            
            # Call to least_squares(...): (line 209)
            # Processing the call arguments (line 209)
            # Getting the type of 'fun_trivial' (line 209)
            fun_trivial_206632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 36), 'fun_trivial', False)
            float_206633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 49), 'float')
            # Getting the type of 'jac' (line 209)
            jac_206634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 54), 'jac', False)
            # Processing the call keyword arguments (line 209)
            # Getting the type of 'self' (line 209)
            self_206635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 66), 'self', False)
            # Obtaining the member 'method' of a type (line 209)
            method_206636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 66), self_206635, 'method')
            keyword_206637 = method_206636
            kwargs_206638 = {'method': keyword_206637}
            # Getting the type of 'least_squares' (line 209)
            least_squares_206631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 22), 'least_squares', False)
            # Calling least_squares(args, kwargs) (line 209)
            least_squares_call_result_206639 = invoke(stypy.reporting.localization.Localization(__file__, 209, 22), least_squares_206631, *[fun_trivial_206632, float_206633, jac_206634], **kwargs_206638)
            
            # Assigning a type to the variable 'res' (line 209)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'res', least_squares_call_result_206639)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 206)
            exit___206640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 17), suppress_warnings_call_result_206621, '__exit__')
            with_exit_206641 = invoke(stypy.reporting.localization.Localization(__file__, 206, 17), exit___206640, None, None, None)

        
        # Call to assert_allclose(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'res' (line 210)
        res_206643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 210)
        x_206644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 28), res_206643, 'x')
        int_206645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 35), 'int')
        # Processing the call keyword arguments (line 210)
        float_206646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 43), 'float')
        keyword_206647 = float_206646
        kwargs_206648 = {'atol': keyword_206647}
        # Getting the type of 'assert_allclose' (line 210)
        assert_allclose_206642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 210)
        assert_allclose_call_result_206649 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), assert_allclose_206642, *[x_206644, int_206645], **kwargs_206648)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_raises(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'ValueError' (line 212)
        ValueError_206651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 212)
        least_squares_206652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 212)
        fun_trivial_206653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 49), 'fun_trivial', False)
        float_206654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 62), 'float')
        # Processing the call keyword arguments (line 212)
        str_206655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 71), 'str', 'oops')
        keyword_206656 = str_206655
        # Getting the type of 'self' (line 213)
        self_206657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 29), 'self', False)
        # Obtaining the member 'method' of a type (line 213)
        method_206658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 29), self_206657, 'method')
        keyword_206659 = method_206658
        kwargs_206660 = {'jac': keyword_206656, 'method': keyword_206659}
        # Getting the type of 'assert_raises' (line 212)
        assert_raises_206650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 212)
        assert_raises_call_result_206661 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), assert_raises_206650, *[ValueError_206651, least_squares_206652, fun_trivial_206653, float_206654], **kwargs_206660)
        
        
        # ################# End of 'test_jac_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_jac_options' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_206662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206662)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_jac_options'
        return stypy_return_type_206662


    @norecursion
    def test_nfev_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_nfev_options'
        module_type_store = module_type_store.open_function_context('test_nfev_options', 215, 4, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_nfev_options.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_nfev_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_nfev_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_nfev_options.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_nfev_options')
        BaseMixin.test_nfev_options.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_nfev_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_nfev_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_nfev_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_nfev_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_nfev_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_nfev_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_nfev_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_nfev_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_nfev_options(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 216)
        list_206663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 216)
        # Adding element type (line 216)
        # Getting the type of 'None' (line 216)
        None_206664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 25), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 24), list_206663, None_206664)
        # Adding element type (line 216)
        int_206665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 24), list_206663, int_206665)
        
        # Testing the type of a for loop iterable (line 216)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 216, 8), list_206663)
        # Getting the type of the for loop variable (line 216)
        for_loop_var_206666 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 216, 8), list_206663)
        # Assigning a type to the variable 'max_nfev' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'max_nfev', for_loop_var_206666)
        # SSA begins for a for statement (line 216)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 217):
        
        # Call to least_squares(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'fun_trivial' (line 217)
        fun_trivial_206668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 32), 'fun_trivial', False)
        float_206669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 45), 'float')
        # Processing the call keyword arguments (line 217)
        # Getting the type of 'max_nfev' (line 217)
        max_nfev_206670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 59), 'max_nfev', False)
        keyword_206671 = max_nfev_206670
        # Getting the type of 'self' (line 218)
        self_206672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 39), 'self', False)
        # Obtaining the member 'method' of a type (line 218)
        method_206673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 39), self_206672, 'method')
        keyword_206674 = method_206673
        kwargs_206675 = {'max_nfev': keyword_206671, 'method': keyword_206674}
        # Getting the type of 'least_squares' (line 217)
        least_squares_206667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 217)
        least_squares_call_result_206676 = invoke(stypy.reporting.localization.Localization(__file__, 217, 18), least_squares_206667, *[fun_trivial_206668, float_206669], **kwargs_206675)
        
        # Assigning a type to the variable 'res' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'res', least_squares_call_result_206676)
        
        # Call to assert_allclose(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'res' (line 219)
        res_206678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 219)
        x_206679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), res_206678, 'x')
        int_206680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 35), 'int')
        # Processing the call keyword arguments (line 219)
        float_206681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 43), 'float')
        keyword_206682 = float_206681
        kwargs_206683 = {'atol': keyword_206682}
        # Getting the type of 'assert_allclose' (line 219)
        assert_allclose_206677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 219)
        assert_allclose_call_result_206684 = invoke(stypy.reporting.localization.Localization(__file__, 219, 12), assert_allclose_206677, *[x_206679, int_206680], **kwargs_206683)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_nfev_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_nfev_options' in the type store
        # Getting the type of 'stypy_return_type' (line 215)
        stypy_return_type_206685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206685)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_nfev_options'
        return stypy_return_type_206685


    @norecursion
    def test_x_scale_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_scale_options'
        module_type_store = module_type_store.open_function_context('test_x_scale_options', 221, 4, False)
        # Assigning a type to the variable 'self' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_x_scale_options.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_x_scale_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_x_scale_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_x_scale_options.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_x_scale_options')
        BaseMixin.test_x_scale_options.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_x_scale_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_x_scale_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_x_scale_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_x_scale_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_x_scale_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_x_scale_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_x_scale_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_scale_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_scale_options(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_206686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        # Adding element type (line 222)
        float_206687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 23), list_206686, float_206687)
        # Adding element type (line 222)
        
        # Call to array(...): (line 222)
        # Processing the call arguments (line 222)
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_206690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        # Adding element type (line 222)
        float_206691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 38), list_206690, float_206691)
        
        # Processing the call keyword arguments (line 222)
        kwargs_206692 = {}
        # Getting the type of 'np' (line 222)
        np_206688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 29), 'np', False)
        # Obtaining the member 'array' of a type (line 222)
        array_206689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 29), np_206688, 'array')
        # Calling array(args, kwargs) (line 222)
        array_call_result_206693 = invoke(stypy.reporting.localization.Localization(__file__, 222, 29), array_206689, *[list_206690], **kwargs_206692)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 23), list_206686, array_call_result_206693)
        # Adding element type (line 222)
        str_206694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 46), 'str', 'jac')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 23), list_206686, str_206694)
        
        # Testing the type of a for loop iterable (line 222)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 222, 8), list_206686)
        # Getting the type of the for loop variable (line 222)
        for_loop_var_206695 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 222, 8), list_206686)
        # Assigning a type to the variable 'x_scale' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'x_scale', for_loop_var_206695)
        # SSA begins for a for statement (line 222)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 223):
        
        # Call to least_squares(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'fun_trivial' (line 223)
        fun_trivial_206697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 32), 'fun_trivial', False)
        float_206698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 45), 'float')
        # Processing the call keyword arguments (line 223)
        # Getting the type of 'x_scale' (line 223)
        x_scale_206699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 58), 'x_scale', False)
        keyword_206700 = x_scale_206699
        kwargs_206701 = {'x_scale': keyword_206700}
        # Getting the type of 'least_squares' (line 223)
        least_squares_206696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 18), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 223)
        least_squares_call_result_206702 = invoke(stypy.reporting.localization.Localization(__file__, 223, 18), least_squares_206696, *[fun_trivial_206697, float_206698], **kwargs_206701)
        
        # Assigning a type to the variable 'res' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'res', least_squares_call_result_206702)
        
        # Call to assert_allclose(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'res' (line 224)
        res_206704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 224)
        x_206705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 28), res_206704, 'x')
        int_206706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 35), 'int')
        # Processing the call keyword arguments (line 224)
        kwargs_206707 = {}
        # Getting the type of 'assert_allclose' (line 224)
        assert_allclose_206703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 224)
        assert_allclose_call_result_206708 = invoke(stypy.reporting.localization.Localization(__file__, 224, 12), assert_allclose_206703, *[x_206705, int_206706], **kwargs_206707)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_raises(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'ValueError' (line 225)
        ValueError_206710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 225)
        least_squares_206711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 225)
        fun_trivial_206712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 49), 'fun_trivial', False)
        float_206713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 22), 'float')
        # Processing the call keyword arguments (line 225)
        str_206714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 35), 'str', 'auto')
        keyword_206715 = str_206714
        # Getting the type of 'self' (line 226)
        self_206716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 50), 'self', False)
        # Obtaining the member 'method' of a type (line 226)
        method_206717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 50), self_206716, 'method')
        keyword_206718 = method_206717
        kwargs_206719 = {'x_scale': keyword_206715, 'method': keyword_206718}
        # Getting the type of 'assert_raises' (line 225)
        assert_raises_206709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 225)
        assert_raises_call_result_206720 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), assert_raises_206709, *[ValueError_206710, least_squares_206711, fun_trivial_206712, float_206713], **kwargs_206719)
        
        
        # Call to assert_raises(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'ValueError' (line 227)
        ValueError_206722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 227)
        least_squares_206723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 227)
        fun_trivial_206724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 49), 'fun_trivial', False)
        float_206725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 22), 'float')
        # Processing the call keyword arguments (line 227)
        float_206726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 35), 'float')
        keyword_206727 = float_206726
        # Getting the type of 'self' (line 228)
        self_206728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 48), 'self', False)
        # Obtaining the member 'method' of a type (line 228)
        method_206729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 48), self_206728, 'method')
        keyword_206730 = method_206729
        kwargs_206731 = {'x_scale': keyword_206727, 'method': keyword_206730}
        # Getting the type of 'assert_raises' (line 227)
        assert_raises_206721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 227)
        assert_raises_call_result_206732 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), assert_raises_206721, *[ValueError_206722, least_squares_206723, fun_trivial_206724, float_206725], **kwargs_206731)
        
        
        # Call to assert_raises(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'ValueError' (line 229)
        ValueError_206734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 229)
        least_squares_206735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 229)
        fun_trivial_206736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 49), 'fun_trivial', False)
        float_206737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 22), 'float')
        # Processing the call keyword arguments (line 229)
        # Getting the type of 'None' (line 230)
        None_206738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 35), 'None', False)
        keyword_206739 = None_206738
        # Getting the type of 'self' (line 230)
        self_206740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 48), 'self', False)
        # Obtaining the member 'method' of a type (line 230)
        method_206741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 48), self_206740, 'method')
        keyword_206742 = method_206741
        kwargs_206743 = {'x_scale': keyword_206739, 'method': keyword_206742}
        # Getting the type of 'assert_raises' (line 229)
        assert_raises_206733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 229)
        assert_raises_call_result_206744 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), assert_raises_206733, *[ValueError_206734, least_squares_206735, fun_trivial_206736, float_206737], **kwargs_206743)
        
        
        # Call to assert_raises(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'ValueError' (line 231)
        ValueError_206746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 231)
        least_squares_206747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 231)
        fun_trivial_206748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 49), 'fun_trivial', False)
        float_206749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 22), 'float')
        # Processing the call keyword arguments (line 231)
        float_206750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 35), 'float')
        complex_206751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 39), 'complex')
        # Applying the binary operator '+' (line 232)
        result_add_206752 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 35), '+', float_206750, complex_206751)
        
        keyword_206753 = result_add_206752
        # Getting the type of 'self' (line 232)
        self_206754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 52), 'self', False)
        # Obtaining the member 'method' of a type (line 232)
        method_206755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 52), self_206754, 'method')
        keyword_206756 = method_206755
        kwargs_206757 = {'x_scale': keyword_206753, 'method': keyword_206756}
        # Getting the type of 'assert_raises' (line 231)
        assert_raises_206745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 231)
        assert_raises_call_result_206758 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), assert_raises_206745, *[ValueError_206746, least_squares_206747, fun_trivial_206748, float_206749], **kwargs_206757)
        
        
        # ################# End of 'test_x_scale_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_scale_options' in the type store
        # Getting the type of 'stypy_return_type' (line 221)
        stypy_return_type_206759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206759)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_scale_options'
        return stypy_return_type_206759


    @norecursion
    def test_diff_step(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_diff_step'
        module_type_store = module_type_store.open_function_context('test_diff_step', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_diff_step.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_diff_step.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_diff_step.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_diff_step.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_diff_step')
        BaseMixin.test_diff_step.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_diff_step.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_diff_step.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_diff_step.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_diff_step.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_diff_step.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_diff_step.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_diff_step', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_diff_step', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_diff_step(...)' code ##################

        
        # Assigning a Call to a Name (line 237):
        
        # Call to least_squares(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'fun_trivial' (line 237)
        fun_trivial_206761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 29), 'fun_trivial', False)
        float_206762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 42), 'float')
        # Processing the call keyword arguments (line 237)
        float_206763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 57), 'float')
        keyword_206764 = float_206763
        # Getting the type of 'self' (line 238)
        self_206765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 36), 'self', False)
        # Obtaining the member 'method' of a type (line 238)
        method_206766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 36), self_206765, 'method')
        keyword_206767 = method_206766
        kwargs_206768 = {'method': keyword_206767, 'diff_step': keyword_206764}
        # Getting the type of 'least_squares' (line 237)
        least_squares_206760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 237)
        least_squares_call_result_206769 = invoke(stypy.reporting.localization.Localization(__file__, 237, 15), least_squares_206760, *[fun_trivial_206761, float_206762], **kwargs_206768)
        
        # Assigning a type to the variable 'res1' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'res1', least_squares_call_result_206769)
        
        # Assigning a Call to a Name (line 239):
        
        # Call to least_squares(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'fun_trivial' (line 239)
        fun_trivial_206771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 29), 'fun_trivial', False)
        float_206772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 42), 'float')
        # Processing the call keyword arguments (line 239)
        float_206773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 57), 'float')
        keyword_206774 = float_206773
        # Getting the type of 'self' (line 240)
        self_206775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 36), 'self', False)
        # Obtaining the member 'method' of a type (line 240)
        method_206776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 36), self_206775, 'method')
        keyword_206777 = method_206776
        kwargs_206778 = {'method': keyword_206777, 'diff_step': keyword_206774}
        # Getting the type of 'least_squares' (line 239)
        least_squares_206770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 15), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 239)
        least_squares_call_result_206779 = invoke(stypy.reporting.localization.Localization(__file__, 239, 15), least_squares_206770, *[fun_trivial_206771, float_206772], **kwargs_206778)
        
        # Assigning a type to the variable 'res2' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'res2', least_squares_call_result_206779)
        
        # Assigning a Call to a Name (line 241):
        
        # Call to least_squares(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'fun_trivial' (line 241)
        fun_trivial_206781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 29), 'fun_trivial', False)
        float_206782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 42), 'float')
        # Processing the call keyword arguments (line 241)
        # Getting the type of 'None' (line 242)
        None_206783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 39), 'None', False)
        keyword_206784 = None_206783
        # Getting the type of 'self' (line 242)
        self_206785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 52), 'self', False)
        # Obtaining the member 'method' of a type (line 242)
        method_206786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 52), self_206785, 'method')
        keyword_206787 = method_206786
        kwargs_206788 = {'method': keyword_206787, 'diff_step': keyword_206784}
        # Getting the type of 'least_squares' (line 241)
        least_squares_206780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 241)
        least_squares_call_result_206789 = invoke(stypy.reporting.localization.Localization(__file__, 241, 15), least_squares_206780, *[fun_trivial_206781, float_206782], **kwargs_206788)
        
        # Assigning a type to the variable 'res3' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'res3', least_squares_call_result_206789)
        
        # Call to assert_allclose(...): (line 243)
        # Processing the call arguments (line 243)
        # Getting the type of 'res1' (line 243)
        res1_206791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 24), 'res1', False)
        # Obtaining the member 'x' of a type (line 243)
        x_206792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 24), res1_206791, 'x')
        int_206793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 32), 'int')
        # Processing the call keyword arguments (line 243)
        float_206794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 40), 'float')
        keyword_206795 = float_206794
        kwargs_206796 = {'atol': keyword_206795}
        # Getting the type of 'assert_allclose' (line 243)
        assert_allclose_206790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 243)
        assert_allclose_call_result_206797 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), assert_allclose_206790, *[x_206792, int_206793], **kwargs_206796)
        
        
        # Call to assert_allclose(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'res2' (line 244)
        res2_206799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'res2', False)
        # Obtaining the member 'x' of a type (line 244)
        x_206800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), res2_206799, 'x')
        int_206801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 32), 'int')
        # Processing the call keyword arguments (line 244)
        float_206802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 40), 'float')
        keyword_206803 = float_206802
        kwargs_206804 = {'atol': keyword_206803}
        # Getting the type of 'assert_allclose' (line 244)
        assert_allclose_206798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 244)
        assert_allclose_call_result_206805 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), assert_allclose_206798, *[x_206800, int_206801], **kwargs_206804)
        
        
        # Call to assert_allclose(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'res3' (line 245)
        res3_206807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 24), 'res3', False)
        # Obtaining the member 'x' of a type (line 245)
        x_206808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 24), res3_206807, 'x')
        int_206809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 32), 'int')
        # Processing the call keyword arguments (line 245)
        float_206810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 40), 'float')
        keyword_206811 = float_206810
        kwargs_206812 = {'atol': keyword_206811}
        # Getting the type of 'assert_allclose' (line 245)
        assert_allclose_206806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 245)
        assert_allclose_call_result_206813 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), assert_allclose_206806, *[x_206808, int_206809], **kwargs_206812)
        
        
        # Call to assert_equal(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'res1' (line 246)
        res1_206815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 21), 'res1', False)
        # Obtaining the member 'x' of a type (line 246)
        x_206816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 21), res1_206815, 'x')
        # Getting the type of 'res2' (line 246)
        res2_206817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 29), 'res2', False)
        # Obtaining the member 'x' of a type (line 246)
        x_206818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 29), res2_206817, 'x')
        # Processing the call keyword arguments (line 246)
        kwargs_206819 = {}
        # Getting the type of 'assert_equal' (line 246)
        assert_equal_206814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 246)
        assert_equal_call_result_206820 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), assert_equal_206814, *[x_206816, x_206818], **kwargs_206819)
        
        
        # Call to assert_equal(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'res1' (line 247)
        res1_206822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 21), 'res1', False)
        # Obtaining the member 'nfev' of a type (line 247)
        nfev_206823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 21), res1_206822, 'nfev')
        # Getting the type of 'res2' (line 247)
        res2_206824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 32), 'res2', False)
        # Obtaining the member 'nfev' of a type (line 247)
        nfev_206825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 32), res2_206824, 'nfev')
        # Processing the call keyword arguments (line 247)
        kwargs_206826 = {}
        # Getting the type of 'assert_equal' (line 247)
        assert_equal_206821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 247)
        assert_equal_call_result_206827 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), assert_equal_206821, *[nfev_206823, nfev_206825], **kwargs_206826)
        
        
        # Call to assert_(...): (line 248)
        # Processing the call arguments (line 248)
        
        # Getting the type of 'res2' (line 248)
        res2_206829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'res2', False)
        # Obtaining the member 'nfev' of a type (line 248)
        nfev_206830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 16), res2_206829, 'nfev')
        # Getting the type of 'res3' (line 248)
        res3_206831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 29), 'res3', False)
        # Obtaining the member 'nfev' of a type (line 248)
        nfev_206832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 29), res3_206831, 'nfev')
        # Applying the binary operator '!=' (line 248)
        result_ne_206833 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 16), '!=', nfev_206830, nfev_206832)
        
        # Processing the call keyword arguments (line 248)
        kwargs_206834 = {}
        # Getting the type of 'assert_' (line 248)
        assert__206828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 248)
        assert__call_result_206835 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), assert__206828, *[result_ne_206833], **kwargs_206834)
        
        
        # ################# End of 'test_diff_step(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_diff_step' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_206836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206836)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_diff_step'
        return stypy_return_type_206836


    @norecursion
    def test_incorrect_options_usage(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_incorrect_options_usage'
        module_type_store = module_type_store.open_function_context('test_incorrect_options_usage', 250, 4, False)
        # Assigning a type to the variable 'self' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_incorrect_options_usage.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_incorrect_options_usage.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_incorrect_options_usage.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_incorrect_options_usage.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_incorrect_options_usage')
        BaseMixin.test_incorrect_options_usage.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_incorrect_options_usage.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_incorrect_options_usage.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_incorrect_options_usage.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_incorrect_options_usage.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_incorrect_options_usage.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_incorrect_options_usage.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_incorrect_options_usage', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_incorrect_options_usage', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_incorrect_options_usage(...)' code ##################

        
        # Call to assert_raises(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'TypeError' (line 251)
        TypeError_206838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 22), 'TypeError', False)
        # Getting the type of 'least_squares' (line 251)
        least_squares_206839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 33), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 251)
        fun_trivial_206840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 48), 'fun_trivial', False)
        float_206841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 61), 'float')
        # Processing the call keyword arguments (line 251)
        # Getting the type of 'self' (line 252)
        self_206842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 29), 'self', False)
        # Obtaining the member 'method' of a type (line 252)
        method_206843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 29), self_206842, 'method')
        keyword_206844 = method_206843
        
        # Obtaining an instance of the builtin type 'dict' (line 252)
        dict_206845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 50), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 252)
        # Adding element type (key, value) (line 252)
        str_206846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 51), 'str', 'no_such_option')
        int_206847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 69), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 50), dict_206845, (str_206846, int_206847))
        
        keyword_206848 = dict_206845
        kwargs_206849 = {'method': keyword_206844, 'options': keyword_206848}
        # Getting the type of 'assert_raises' (line 251)
        assert_raises_206837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 251)
        assert_raises_call_result_206850 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), assert_raises_206837, *[TypeError_206838, least_squares_206839, fun_trivial_206840, float_206841], **kwargs_206849)
        
        
        # Call to assert_raises(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'TypeError' (line 253)
        TypeError_206852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 22), 'TypeError', False)
        # Getting the type of 'least_squares' (line 253)
        least_squares_206853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 33), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 253)
        fun_trivial_206854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 48), 'fun_trivial', False)
        float_206855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 61), 'float')
        # Processing the call keyword arguments (line 253)
        # Getting the type of 'self' (line 254)
        self_206856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 29), 'self', False)
        # Obtaining the member 'method' of a type (line 254)
        method_206857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 29), self_206856, 'method')
        keyword_206858 = method_206857
        
        # Obtaining an instance of the builtin type 'dict' (line 254)
        dict_206859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 50), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 254)
        # Adding element type (key, value) (line 254)
        str_206860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 51), 'str', 'max_nfev')
        int_206861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 63), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 50), dict_206859, (str_206860, int_206861))
        
        keyword_206862 = dict_206859
        kwargs_206863 = {'method': keyword_206858, 'options': keyword_206862}
        # Getting the type of 'assert_raises' (line 253)
        assert_raises_206851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 253)
        assert_raises_call_result_206864 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), assert_raises_206851, *[TypeError_206852, least_squares_206853, fun_trivial_206854, float_206855], **kwargs_206863)
        
        
        # ################# End of 'test_incorrect_options_usage(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_incorrect_options_usage' in the type store
        # Getting the type of 'stypy_return_type' (line 250)
        stypy_return_type_206865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206865)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_incorrect_options_usage'
        return stypy_return_type_206865


    @norecursion
    def test_full_result(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_full_result'
        module_type_store = module_type_store.open_function_context('test_full_result', 256, 4, False)
        # Assigning a type to the variable 'self' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_full_result')
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_full_result.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_full_result', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_full_result', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_full_result(...)' code ##################

        
        # Assigning a Call to a Name (line 259):
        
        # Call to least_squares(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'fun_trivial' (line 259)
        fun_trivial_206867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'fun_trivial', False)
        float_206868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 41), 'float')
        # Processing the call keyword arguments (line 259)
        # Getting the type of 'self' (line 259)
        self_206869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 53), 'self', False)
        # Obtaining the member 'method' of a type (line 259)
        method_206870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 53), self_206869, 'method')
        keyword_206871 = method_206870
        kwargs_206872 = {'method': keyword_206871}
        # Getting the type of 'least_squares' (line 259)
        least_squares_206866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 259)
        least_squares_call_result_206873 = invoke(stypy.reporting.localization.Localization(__file__, 259, 14), least_squares_206866, *[fun_trivial_206867, float_206868], **kwargs_206872)
        
        # Assigning a type to the variable 'res' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'res', least_squares_call_result_206873)
        
        # Call to assert_allclose(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'res' (line 260)
        res_206875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 260)
        x_206876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 24), res_206875, 'x')
        int_206877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 31), 'int')
        # Processing the call keyword arguments (line 260)
        float_206878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 39), 'float')
        keyword_206879 = float_206878
        kwargs_206880 = {'atol': keyword_206879}
        # Getting the type of 'assert_allclose' (line 260)
        assert_allclose_206874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 260)
        assert_allclose_call_result_206881 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), assert_allclose_206874, *[x_206876, int_206877], **kwargs_206880)
        
        
        # Call to assert_allclose(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'res' (line 261)
        res_206883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 24), 'res', False)
        # Obtaining the member 'cost' of a type (line 261)
        cost_206884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 24), res_206883, 'cost')
        float_206885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 34), 'float')
        # Processing the call keyword arguments (line 261)
        kwargs_206886 = {}
        # Getting the type of 'assert_allclose' (line 261)
        assert_allclose_206882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 261)
        assert_allclose_call_result_206887 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), assert_allclose_206882, *[cost_206884, float_206885], **kwargs_206886)
        
        
        # Call to assert_allclose(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'res' (line 262)
        res_206889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'res', False)
        # Obtaining the member 'fun' of a type (line 262)
        fun_206890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 24), res_206889, 'fun')
        int_206891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 33), 'int')
        # Processing the call keyword arguments (line 262)
        kwargs_206892 = {}
        # Getting the type of 'assert_allclose' (line 262)
        assert_allclose_206888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 262)
        assert_allclose_call_result_206893 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), assert_allclose_206888, *[fun_206890, int_206891], **kwargs_206892)
        
        
        # Call to assert_allclose(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'res' (line 263)
        res_206895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 24), 'res', False)
        # Obtaining the member 'jac' of a type (line 263)
        jac_206896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 24), res_206895, 'jac')
        int_206897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 33), 'int')
        # Processing the call keyword arguments (line 263)
        float_206898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 41), 'float')
        keyword_206899 = float_206898
        kwargs_206900 = {'atol': keyword_206899}
        # Getting the type of 'assert_allclose' (line 263)
        assert_allclose_206894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 263)
        assert_allclose_call_result_206901 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), assert_allclose_206894, *[jac_206896, int_206897], **kwargs_206900)
        
        
        # Call to assert_allclose(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'res' (line 264)
        res_206903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'res', False)
        # Obtaining the member 'grad' of a type (line 264)
        grad_206904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 24), res_206903, 'grad')
        int_206905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 34), 'int')
        # Processing the call keyword arguments (line 264)
        float_206906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 42), 'float')
        keyword_206907 = float_206906
        kwargs_206908 = {'atol': keyword_206907}
        # Getting the type of 'assert_allclose' (line 264)
        assert_allclose_206902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 264)
        assert_allclose_call_result_206909 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), assert_allclose_206902, *[grad_206904, int_206905], **kwargs_206908)
        
        
        # Call to assert_allclose(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'res' (line 265)
        res_206911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 24), 'res', False)
        # Obtaining the member 'optimality' of a type (line 265)
        optimality_206912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 24), res_206911, 'optimality')
        int_206913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 40), 'int')
        # Processing the call keyword arguments (line 265)
        float_206914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 48), 'float')
        keyword_206915 = float_206914
        kwargs_206916 = {'atol': keyword_206915}
        # Getting the type of 'assert_allclose' (line 265)
        assert_allclose_206910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 265)
        assert_allclose_call_result_206917 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), assert_allclose_206910, *[optimality_206912, int_206913], **kwargs_206916)
        
        
        # Call to assert_equal(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'res' (line 266)
        res_206919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 21), 'res', False)
        # Obtaining the member 'active_mask' of a type (line 266)
        active_mask_206920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 21), res_206919, 'active_mask')
        int_206921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 38), 'int')
        # Processing the call keyword arguments (line 266)
        kwargs_206922 = {}
        # Getting the type of 'assert_equal' (line 266)
        assert_equal_206918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 266)
        assert_equal_call_result_206923 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), assert_equal_206918, *[active_mask_206920, int_206921], **kwargs_206922)
        
        
        
        # Getting the type of 'self' (line 267)
        self_206924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 11), 'self')
        # Obtaining the member 'method' of a type (line 267)
        method_206925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 11), self_206924, 'method')
        str_206926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 26), 'str', 'lm')
        # Applying the binary operator '==' (line 267)
        result_eq_206927 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 11), '==', method_206925, str_206926)
        
        # Testing the type of an if condition (line 267)
        if_condition_206928 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 8), result_eq_206927)
        # Assigning a type to the variable 'if_condition_206928' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'if_condition_206928', if_condition_206928)
        # SSA begins for if statement (line 267)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_(...): (line 268)
        # Processing the call arguments (line 268)
        
        # Getting the type of 'res' (line 268)
        res_206930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'res', False)
        # Obtaining the member 'nfev' of a type (line 268)
        nfev_206931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 20), res_206930, 'nfev')
        int_206932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 31), 'int')
        # Applying the binary operator '<' (line 268)
        result_lt_206933 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 20), '<', nfev_206931, int_206932)
        
        # Processing the call keyword arguments (line 268)
        kwargs_206934 = {}
        # Getting the type of 'assert_' (line 268)
        assert__206929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 268)
        assert__call_result_206935 = invoke(stypy.reporting.localization.Localization(__file__, 268, 12), assert__206929, *[result_lt_206933], **kwargs_206934)
        
        
        # Call to assert_(...): (line 269)
        # Processing the call arguments (line 269)
        
        # Getting the type of 'res' (line 269)
        res_206937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), 'res', False)
        # Obtaining the member 'njev' of a type (line 269)
        njev_206938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 20), res_206937, 'njev')
        # Getting the type of 'None' (line 269)
        None_206939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 32), 'None', False)
        # Applying the binary operator 'is' (line 269)
        result_is__206940 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 20), 'is', njev_206938, None_206939)
        
        # Processing the call keyword arguments (line 269)
        kwargs_206941 = {}
        # Getting the type of 'assert_' (line 269)
        assert__206936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 269)
        assert__call_result_206942 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), assert__206936, *[result_is__206940], **kwargs_206941)
        
        # SSA branch for the else part of an if statement (line 267)
        module_type_store.open_ssa_branch('else')
        
        # Call to assert_(...): (line 271)
        # Processing the call arguments (line 271)
        
        # Getting the type of 'res' (line 271)
        res_206944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 20), 'res', False)
        # Obtaining the member 'nfev' of a type (line 271)
        nfev_206945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 20), res_206944, 'nfev')
        int_206946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 31), 'int')
        # Applying the binary operator '<' (line 271)
        result_lt_206947 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 20), '<', nfev_206945, int_206946)
        
        # Processing the call keyword arguments (line 271)
        kwargs_206948 = {}
        # Getting the type of 'assert_' (line 271)
        assert__206943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 271)
        assert__call_result_206949 = invoke(stypy.reporting.localization.Localization(__file__, 271, 12), assert__206943, *[result_lt_206947], **kwargs_206948)
        
        
        # Call to assert_(...): (line 272)
        # Processing the call arguments (line 272)
        
        # Getting the type of 'res' (line 272)
        res_206951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 'res', False)
        # Obtaining the member 'njev' of a type (line 272)
        njev_206952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 20), res_206951, 'njev')
        int_206953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 31), 'int')
        # Applying the binary operator '<' (line 272)
        result_lt_206954 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 20), '<', njev_206952, int_206953)
        
        # Processing the call keyword arguments (line 272)
        kwargs_206955 = {}
        # Getting the type of 'assert_' (line 272)
        assert__206950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 272)
        assert__call_result_206956 = invoke(stypy.reporting.localization.Localization(__file__, 272, 12), assert__206950, *[result_lt_206954], **kwargs_206955)
        
        # SSA join for if statement (line 267)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 273)
        # Processing the call arguments (line 273)
        
        # Getting the type of 'res' (line 273)
        res_206958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'res', False)
        # Obtaining the member 'status' of a type (line 273)
        status_206959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), res_206958, 'status')
        int_206960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 29), 'int')
        # Applying the binary operator '>' (line 273)
        result_gt_206961 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 16), '>', status_206959, int_206960)
        
        # Processing the call keyword arguments (line 273)
        kwargs_206962 = {}
        # Getting the type of 'assert_' (line 273)
        assert__206957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 273)
        assert__call_result_206963 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), assert__206957, *[result_gt_206961], **kwargs_206962)
        
        
        # Call to assert_(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'res' (line 274)
        res_206965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'res', False)
        # Obtaining the member 'success' of a type (line 274)
        success_206966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 16), res_206965, 'success')
        # Processing the call keyword arguments (line 274)
        kwargs_206967 = {}
        # Getting the type of 'assert_' (line 274)
        assert__206964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 274)
        assert__call_result_206968 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), assert__206964, *[success_206966], **kwargs_206967)
        
        
        # ################# End of 'test_full_result(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_full_result' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_206969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206969)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_full_result'
        return stypy_return_type_206969


    @norecursion
    def test_full_result_single_fev(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_full_result_single_fev'
        module_type_store = module_type_store.open_function_context('test_full_result_single_fev', 276, 4, False)
        # Assigning a type to the variable 'self' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_full_result_single_fev.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_full_result_single_fev.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_full_result_single_fev.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_full_result_single_fev.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_full_result_single_fev')
        BaseMixin.test_full_result_single_fev.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_full_result_single_fev.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_full_result_single_fev.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_full_result_single_fev.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_full_result_single_fev.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_full_result_single_fev.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_full_result_single_fev.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_full_result_single_fev', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_full_result_single_fev', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_full_result_single_fev(...)' code ##################

        
        
        # Getting the type of 'self' (line 279)
        self_206970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), 'self')
        # Obtaining the member 'method' of a type (line 279)
        method_206971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 11), self_206970, 'method')
        str_206972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 26), 'str', 'lm')
        # Applying the binary operator '==' (line 279)
        result_eq_206973 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 11), '==', method_206971, str_206972)
        
        # Testing the type of an if condition (line 279)
        if_condition_206974 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), result_eq_206973)
        # Assigning a type to the variable 'if_condition_206974' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_206974', if_condition_206974)
        # SSA begins for if statement (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 279)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 282):
        
        # Call to least_squares(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'fun_trivial' (line 282)
        fun_trivial_206976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 28), 'fun_trivial', False)
        float_206977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 41), 'float')
        # Processing the call keyword arguments (line 282)
        # Getting the type of 'self' (line 282)
        self_206978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 53), 'self', False)
        # Obtaining the member 'method' of a type (line 282)
        method_206979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 53), self_206978, 'method')
        keyword_206980 = method_206979
        int_206981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 37), 'int')
        keyword_206982 = int_206981
        kwargs_206983 = {'max_nfev': keyword_206982, 'method': keyword_206980}
        # Getting the type of 'least_squares' (line 282)
        least_squares_206975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 282)
        least_squares_call_result_206984 = invoke(stypy.reporting.localization.Localization(__file__, 282, 14), least_squares_206975, *[fun_trivial_206976, float_206977], **kwargs_206983)
        
        # Assigning a type to the variable 'res' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'res', least_squares_call_result_206984)
        
        # Call to assert_equal(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'res' (line 284)
        res_206986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 21), 'res', False)
        # Obtaining the member 'x' of a type (line 284)
        x_206987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 21), res_206986, 'x')
        
        # Call to array(...): (line 284)
        # Processing the call arguments (line 284)
        
        # Obtaining an instance of the builtin type 'list' (line 284)
        list_206990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 284)
        # Adding element type (line 284)
        int_206991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 37), list_206990, int_206991)
        
        # Processing the call keyword arguments (line 284)
        kwargs_206992 = {}
        # Getting the type of 'np' (line 284)
        np_206988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 28), 'np', False)
        # Obtaining the member 'array' of a type (line 284)
        array_206989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 28), np_206988, 'array')
        # Calling array(args, kwargs) (line 284)
        array_call_result_206993 = invoke(stypy.reporting.localization.Localization(__file__, 284, 28), array_206989, *[list_206990], **kwargs_206992)
        
        # Processing the call keyword arguments (line 284)
        kwargs_206994 = {}
        # Getting the type of 'assert_equal' (line 284)
        assert_equal_206985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 284)
        assert_equal_call_result_206995 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), assert_equal_206985, *[x_206987, array_call_result_206993], **kwargs_206994)
        
        
        # Call to assert_equal(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'res' (line 285)
        res_206997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 21), 'res', False)
        # Obtaining the member 'cost' of a type (line 285)
        cost_206998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 21), res_206997, 'cost')
        float_206999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 31), 'float')
        # Processing the call keyword arguments (line 285)
        kwargs_207000 = {}
        # Getting the type of 'assert_equal' (line 285)
        assert_equal_206996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 285)
        assert_equal_call_result_207001 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), assert_equal_206996, *[cost_206998, float_206999], **kwargs_207000)
        
        
        # Call to assert_equal(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'res' (line 286)
        res_207003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 21), 'res', False)
        # Obtaining the member 'fun' of a type (line 286)
        fun_207004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 21), res_207003, 'fun')
        
        # Call to array(...): (line 286)
        # Processing the call arguments (line 286)
        
        # Obtaining an instance of the builtin type 'list' (line 286)
        list_207007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 286)
        # Adding element type (line 286)
        int_207008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 39), list_207007, int_207008)
        
        # Processing the call keyword arguments (line 286)
        kwargs_207009 = {}
        # Getting the type of 'np' (line 286)
        np_207005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 30), 'np', False)
        # Obtaining the member 'array' of a type (line 286)
        array_207006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 30), np_207005, 'array')
        # Calling array(args, kwargs) (line 286)
        array_call_result_207010 = invoke(stypy.reporting.localization.Localization(__file__, 286, 30), array_207006, *[list_207007], **kwargs_207009)
        
        # Processing the call keyword arguments (line 286)
        kwargs_207011 = {}
        # Getting the type of 'assert_equal' (line 286)
        assert_equal_207002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 286)
        assert_equal_call_result_207012 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), assert_equal_207002, *[fun_207004, array_call_result_207010], **kwargs_207011)
        
        
        # Call to assert_equal(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'res' (line 287)
        res_207014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 21), 'res', False)
        # Obtaining the member 'jac' of a type (line 287)
        jac_207015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 21), res_207014, 'jac')
        
        # Call to array(...): (line 287)
        # Processing the call arguments (line 287)
        
        # Obtaining an instance of the builtin type 'list' (line 287)
        list_207018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 287)
        # Adding element type (line 287)
        
        # Obtaining an instance of the builtin type 'list' (line 287)
        list_207019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 287)
        # Adding element type (line 287)
        int_207020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 40), list_207019, int_207020)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 39), list_207018, list_207019)
        
        # Processing the call keyword arguments (line 287)
        kwargs_207021 = {}
        # Getting the type of 'np' (line 287)
        np_207016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 30), 'np', False)
        # Obtaining the member 'array' of a type (line 287)
        array_207017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 30), np_207016, 'array')
        # Calling array(args, kwargs) (line 287)
        array_call_result_207022 = invoke(stypy.reporting.localization.Localization(__file__, 287, 30), array_207017, *[list_207018], **kwargs_207021)
        
        # Processing the call keyword arguments (line 287)
        kwargs_207023 = {}
        # Getting the type of 'assert_equal' (line 287)
        assert_equal_207013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 287)
        assert_equal_call_result_207024 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), assert_equal_207013, *[jac_207015, array_call_result_207022], **kwargs_207023)
        
        
        # Call to assert_equal(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'res' (line 288)
        res_207026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'res', False)
        # Obtaining the member 'grad' of a type (line 288)
        grad_207027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 21), res_207026, 'grad')
        
        # Call to array(...): (line 288)
        # Processing the call arguments (line 288)
        
        # Obtaining an instance of the builtin type 'list' (line 288)
        list_207030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 288)
        # Adding element type (line 288)
        int_207031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 40), list_207030, int_207031)
        
        # Processing the call keyword arguments (line 288)
        kwargs_207032 = {}
        # Getting the type of 'np' (line 288)
        np_207028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 31), 'np', False)
        # Obtaining the member 'array' of a type (line 288)
        array_207029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 31), np_207028, 'array')
        # Calling array(args, kwargs) (line 288)
        array_call_result_207033 = invoke(stypy.reporting.localization.Localization(__file__, 288, 31), array_207029, *[list_207030], **kwargs_207032)
        
        # Processing the call keyword arguments (line 288)
        kwargs_207034 = {}
        # Getting the type of 'assert_equal' (line 288)
        assert_equal_207025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 288)
        assert_equal_call_result_207035 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), assert_equal_207025, *[grad_207027, array_call_result_207033], **kwargs_207034)
        
        
        # Call to assert_equal(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'res' (line 289)
        res_207037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 21), 'res', False)
        # Obtaining the member 'optimality' of a type (line 289)
        optimality_207038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 21), res_207037, 'optimality')
        int_207039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 37), 'int')
        # Processing the call keyword arguments (line 289)
        kwargs_207040 = {}
        # Getting the type of 'assert_equal' (line 289)
        assert_equal_207036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 289)
        assert_equal_call_result_207041 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), assert_equal_207036, *[optimality_207038, int_207039], **kwargs_207040)
        
        
        # Call to assert_equal(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'res' (line 290)
        res_207043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 21), 'res', False)
        # Obtaining the member 'active_mask' of a type (line 290)
        active_mask_207044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 21), res_207043, 'active_mask')
        
        # Call to array(...): (line 290)
        # Processing the call arguments (line 290)
        
        # Obtaining an instance of the builtin type 'list' (line 290)
        list_207047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 290)
        # Adding element type (line 290)
        int_207048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 47), list_207047, int_207048)
        
        # Processing the call keyword arguments (line 290)
        kwargs_207049 = {}
        # Getting the type of 'np' (line 290)
        np_207045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 38), 'np', False)
        # Obtaining the member 'array' of a type (line 290)
        array_207046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 38), np_207045, 'array')
        # Calling array(args, kwargs) (line 290)
        array_call_result_207050 = invoke(stypy.reporting.localization.Localization(__file__, 290, 38), array_207046, *[list_207047], **kwargs_207049)
        
        # Processing the call keyword arguments (line 290)
        kwargs_207051 = {}
        # Getting the type of 'assert_equal' (line 290)
        assert_equal_207042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 290)
        assert_equal_call_result_207052 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), assert_equal_207042, *[active_mask_207044, array_call_result_207050], **kwargs_207051)
        
        
        # Call to assert_equal(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'res' (line 291)
        res_207054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 21), 'res', False)
        # Obtaining the member 'nfev' of a type (line 291)
        nfev_207055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 21), res_207054, 'nfev')
        int_207056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 31), 'int')
        # Processing the call keyword arguments (line 291)
        kwargs_207057 = {}
        # Getting the type of 'assert_equal' (line 291)
        assert_equal_207053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 291)
        assert_equal_call_result_207058 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), assert_equal_207053, *[nfev_207055, int_207056], **kwargs_207057)
        
        
        # Call to assert_equal(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'res' (line 292)
        res_207060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 21), 'res', False)
        # Obtaining the member 'njev' of a type (line 292)
        njev_207061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 21), res_207060, 'njev')
        int_207062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 31), 'int')
        # Processing the call keyword arguments (line 292)
        kwargs_207063 = {}
        # Getting the type of 'assert_equal' (line 292)
        assert_equal_207059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 292)
        assert_equal_call_result_207064 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), assert_equal_207059, *[njev_207061, int_207062], **kwargs_207063)
        
        
        # Call to assert_equal(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'res' (line 293)
        res_207066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 21), 'res', False)
        # Obtaining the member 'status' of a type (line 293)
        status_207067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 21), res_207066, 'status')
        int_207068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 33), 'int')
        # Processing the call keyword arguments (line 293)
        kwargs_207069 = {}
        # Getting the type of 'assert_equal' (line 293)
        assert_equal_207065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 293)
        assert_equal_call_result_207070 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), assert_equal_207065, *[status_207067, int_207068], **kwargs_207069)
        
        
        # Call to assert_equal(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'res' (line 294)
        res_207072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'res', False)
        # Obtaining the member 'success' of a type (line 294)
        success_207073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 21), res_207072, 'success')
        int_207074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 34), 'int')
        # Processing the call keyword arguments (line 294)
        kwargs_207075 = {}
        # Getting the type of 'assert_equal' (line 294)
        assert_equal_207071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 294)
        assert_equal_call_result_207076 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), assert_equal_207071, *[success_207073, int_207074], **kwargs_207075)
        
        
        # ################# End of 'test_full_result_single_fev(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_full_result_single_fev' in the type store
        # Getting the type of 'stypy_return_type' (line 276)
        stypy_return_type_207077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207077)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_full_result_single_fev'
        return stypy_return_type_207077


    @norecursion
    def test_rosenbrock(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_rosenbrock'
        module_type_store = module_type_store.open_function_context('test_rosenbrock', 296, 4, False)
        # Assigning a type to the variable 'self' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_rosenbrock.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_rosenbrock.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_rosenbrock.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_rosenbrock.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_rosenbrock')
        BaseMixin.test_rosenbrock.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_rosenbrock.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_rosenbrock.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_rosenbrock.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_rosenbrock.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_rosenbrock.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_rosenbrock.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_rosenbrock', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_rosenbrock', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_rosenbrock(...)' code ##################

        
        # Assigning a List to a Name (line 297):
        
        # Obtaining an instance of the builtin type 'list' (line 297)
        list_207078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 297)
        # Adding element type (line 297)
        int_207079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 13), list_207078, int_207079)
        # Adding element type (line 297)
        int_207080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 13), list_207078, int_207080)
        
        # Assigning a type to the variable 'x0' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'x0', list_207078)
        
        # Assigning a List to a Name (line 298):
        
        # Obtaining an instance of the builtin type 'list' (line 298)
        list_207081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 298)
        # Adding element type (line 298)
        int_207082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 16), list_207081, int_207082)
        # Adding element type (line 298)
        int_207083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 16), list_207081, int_207083)
        
        # Assigning a type to the variable 'x_opt' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'x_opt', list_207081)
        
        
        # Call to product(...): (line 299)
        # Processing the call arguments (line 299)
        
        # Obtaining an instance of the builtin type 'list' (line 300)
        list_207085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 300)
        # Adding element type (line 300)
        str_207086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 17), 'str', '2-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 16), list_207085, str_207086)
        # Adding element type (line 300)
        str_207087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 28), 'str', '3-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 16), list_207085, str_207087)
        # Adding element type (line 300)
        str_207088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 39), 'str', 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 16), list_207085, str_207088)
        # Adding element type (line 300)
        # Getting the type of 'jac_rosenbrock' (line 300)
        jac_rosenbrock_207089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 45), 'jac_rosenbrock', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 16), list_207085, jac_rosenbrock_207089)
        
        
        # Obtaining an instance of the builtin type 'list' (line 301)
        list_207090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 301)
        # Adding element type (line 301)
        float_207091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 16), list_207090, float_207091)
        # Adding element type (line 301)
        
        # Call to array(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Obtaining an instance of the builtin type 'list' (line 301)
        list_207094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 301)
        # Adding element type (line 301)
        float_207095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 31), list_207094, float_207095)
        # Adding element type (line 301)
        float_207096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 31), list_207094, float_207096)
        
        # Processing the call keyword arguments (line 301)
        kwargs_207097 = {}
        # Getting the type of 'np' (line 301)
        np_207092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 301)
        array_207093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 22), np_207092, 'array')
        # Calling array(args, kwargs) (line 301)
        array_call_result_207098 = invoke(stypy.reporting.localization.Localization(__file__, 301, 22), array_207093, *[list_207094], **kwargs_207097)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 16), list_207090, array_call_result_207098)
        # Adding element type (line 301)
        str_207099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 44), 'str', 'jac')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 16), list_207090, str_207099)
        
        
        # Obtaining an instance of the builtin type 'list' (line 302)
        list_207100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 302)
        # Adding element type (line 302)
        str_207101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 17), 'str', 'exact')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 16), list_207100, str_207101)
        # Adding element type (line 302)
        str_207102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 26), 'str', 'lsmr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 16), list_207100, str_207102)
        
        # Processing the call keyword arguments (line 299)
        kwargs_207103 = {}
        # Getting the type of 'product' (line 299)
        product_207084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 39), 'product', False)
        # Calling product(args, kwargs) (line 299)
        product_call_result_207104 = invoke(stypy.reporting.localization.Localization(__file__, 299, 39), product_207084, *[list_207085, list_207090, list_207100], **kwargs_207103)
        
        # Testing the type of a for loop iterable (line 299)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 299, 8), product_call_result_207104)
        # Getting the type of the for loop variable (line 299)
        for_loop_var_207105 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 299, 8), product_call_result_207104)
        # Assigning a type to the variable 'jac' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'jac', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 8), for_loop_var_207105))
        # Assigning a type to the variable 'x_scale' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'x_scale', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 8), for_loop_var_207105))
        # Assigning a type to the variable 'tr_solver' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'tr_solver', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 8), for_loop_var_207105))
        # SSA begins for a for statement (line 299)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to suppress_warnings(...): (line 303)
        # Processing the call keyword arguments (line 303)
        kwargs_207107 = {}
        # Getting the type of 'suppress_warnings' (line 303)
        suppress_warnings_207106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 17), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 303)
        suppress_warnings_call_result_207108 = invoke(stypy.reporting.localization.Localization(__file__, 303, 17), suppress_warnings_207106, *[], **kwargs_207107)
        
        with_207109 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 303, 17), suppress_warnings_call_result_207108, 'with parameter', '__enter__', '__exit__')

        if with_207109:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 303)
            enter___207110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 17), suppress_warnings_call_result_207108, '__enter__')
            with_enter_207111 = invoke(stypy.reporting.localization.Localization(__file__, 303, 17), enter___207110)
            # Assigning a type to the variable 'sup' (line 303)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 17), 'sup', with_enter_207111)
            
            # Call to filter(...): (line 304)
            # Processing the call arguments (line 304)
            # Getting the type of 'UserWarning' (line 304)
            UserWarning_207114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'UserWarning', False)
            str_207115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 27), 'str', "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'")
            # Processing the call keyword arguments (line 304)
            kwargs_207116 = {}
            # Getting the type of 'sup' (line 304)
            sup_207112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'sup', False)
            # Obtaining the member 'filter' of a type (line 304)
            filter_207113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 16), sup_207112, 'filter')
            # Calling filter(args, kwargs) (line 304)
            filter_call_result_207117 = invoke(stypy.reporting.localization.Localization(__file__, 304, 16), filter_207113, *[UserWarning_207114, str_207115], **kwargs_207116)
            
            
            # Assigning a Call to a Name (line 306):
            
            # Call to least_squares(...): (line 306)
            # Processing the call arguments (line 306)
            # Getting the type of 'fun_rosenbrock' (line 306)
            fun_rosenbrock_207119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 36), 'fun_rosenbrock', False)
            # Getting the type of 'x0' (line 306)
            x0_207120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 52), 'x0', False)
            # Getting the type of 'jac' (line 306)
            jac_207121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 56), 'jac', False)
            # Processing the call keyword arguments (line 306)
            # Getting the type of 'x_scale' (line 306)
            x_scale_207122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 69), 'x_scale', False)
            keyword_207123 = x_scale_207122
            # Getting the type of 'tr_solver' (line 307)
            tr_solver_207124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 46), 'tr_solver', False)
            keyword_207125 = tr_solver_207124
            # Getting the type of 'self' (line 307)
            self_207126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 64), 'self', False)
            # Obtaining the member 'method' of a type (line 307)
            method_207127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 64), self_207126, 'method')
            keyword_207128 = method_207127
            kwargs_207129 = {'tr_solver': keyword_207125, 'x_scale': keyword_207123, 'method': keyword_207128}
            # Getting the type of 'least_squares' (line 306)
            least_squares_207118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 22), 'least_squares', False)
            # Calling least_squares(args, kwargs) (line 306)
            least_squares_call_result_207130 = invoke(stypy.reporting.localization.Localization(__file__, 306, 22), least_squares_207118, *[fun_rosenbrock_207119, x0_207120, jac_207121], **kwargs_207129)
            
            # Assigning a type to the variable 'res' (line 306)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'res', least_squares_call_result_207130)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 303)
            exit___207131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 17), suppress_warnings_call_result_207108, '__exit__')
            with_exit_207132 = invoke(stypy.reporting.localization.Localization(__file__, 303, 17), exit___207131, None, None, None)

        
        # Call to assert_allclose(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'res' (line 308)
        res_207134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 308)
        x_207135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 28), res_207134, 'x')
        # Getting the type of 'x_opt' (line 308)
        x_opt_207136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 35), 'x_opt', False)
        # Processing the call keyword arguments (line 308)
        kwargs_207137 = {}
        # Getting the type of 'assert_allclose' (line 308)
        assert_allclose_207133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 308)
        assert_allclose_call_result_207138 = invoke(stypy.reporting.localization.Localization(__file__, 308, 12), assert_allclose_207133, *[x_207135, x_opt_207136], **kwargs_207137)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_rosenbrock(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_rosenbrock' in the type store
        # Getting the type of 'stypy_return_type' (line 296)
        stypy_return_type_207139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207139)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_rosenbrock'
        return stypy_return_type_207139


    @norecursion
    def test_rosenbrock_cropped(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_rosenbrock_cropped'
        module_type_store = module_type_store.open_function_context('test_rosenbrock_cropped', 310, 4, False)
        # Assigning a type to the variable 'self' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_rosenbrock_cropped.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_rosenbrock_cropped.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_rosenbrock_cropped.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_rosenbrock_cropped.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_rosenbrock_cropped')
        BaseMixin.test_rosenbrock_cropped.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_rosenbrock_cropped.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_rosenbrock_cropped.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_rosenbrock_cropped.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_rosenbrock_cropped.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_rosenbrock_cropped.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_rosenbrock_cropped.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_rosenbrock_cropped', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_rosenbrock_cropped', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_rosenbrock_cropped(...)' code ##################

        
        # Assigning a List to a Name (line 311):
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_207140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        # Adding element type (line 311)
        int_207141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 13), list_207140, int_207141)
        # Adding element type (line 311)
        int_207142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 13), list_207140, int_207142)
        
        # Assigning a type to the variable 'x0' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'x0', list_207140)
        
        
        # Getting the type of 'self' (line 312)
        self_207143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 11), 'self')
        # Obtaining the member 'method' of a type (line 312)
        method_207144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 11), self_207143, 'method')
        str_207145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 26), 'str', 'lm')
        # Applying the binary operator '==' (line 312)
        result_eq_207146 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 11), '==', method_207144, str_207145)
        
        # Testing the type of an if condition (line 312)
        if_condition_207147 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 8), result_eq_207146)
        # Assigning a type to the variable 'if_condition_207147' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'if_condition_207147', if_condition_207147)
        # SSA begins for if statement (line 312)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_raises(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'ValueError' (line 313)
        ValueError_207149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 26), 'ValueError', False)
        # Getting the type of 'least_squares' (line 313)
        least_squares_207150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 38), 'least_squares', False)
        # Getting the type of 'fun_rosenbrock_cropped' (line 313)
        fun_rosenbrock_cropped_207151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 53), 'fun_rosenbrock_cropped', False)
        # Getting the type of 'x0' (line 314)
        x0_207152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), 'x0', False)
        # Processing the call keyword arguments (line 313)
        str_207153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 37), 'str', 'lm')
        keyword_207154 = str_207153
        kwargs_207155 = {'method': keyword_207154}
        # Getting the type of 'assert_raises' (line 313)
        assert_raises_207148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 313)
        assert_raises_call_result_207156 = invoke(stypy.reporting.localization.Localization(__file__, 313, 12), assert_raises_207148, *[ValueError_207149, least_squares_207150, fun_rosenbrock_cropped_207151, x0_207152], **kwargs_207155)
        
        # SSA branch for the else part of an if statement (line 312)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to product(...): (line 316)
        # Processing the call arguments (line 316)
        
        # Obtaining an instance of the builtin type 'list' (line 317)
        list_207158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 317)
        # Adding element type (line 317)
        str_207159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 21), 'str', '2-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 20), list_207158, str_207159)
        # Adding element type (line 317)
        str_207160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 32), 'str', '3-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 20), list_207158, str_207160)
        # Adding element type (line 317)
        str_207161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 43), 'str', 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 20), list_207158, str_207161)
        # Adding element type (line 317)
        # Getting the type of 'jac_rosenbrock_cropped' (line 317)
        jac_rosenbrock_cropped_207162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 49), 'jac_rosenbrock_cropped', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 20), list_207158, jac_rosenbrock_cropped_207162)
        
        
        # Obtaining an instance of the builtin type 'list' (line 318)
        list_207163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 318)
        # Adding element type (line 318)
        float_207164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 20), list_207163, float_207164)
        # Adding element type (line 318)
        
        # Call to array(...): (line 318)
        # Processing the call arguments (line 318)
        
        # Obtaining an instance of the builtin type 'list' (line 318)
        list_207167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 318)
        # Adding element type (line 318)
        float_207168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 35), list_207167, float_207168)
        # Adding element type (line 318)
        float_207169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 35), list_207167, float_207169)
        
        # Processing the call keyword arguments (line 318)
        kwargs_207170 = {}
        # Getting the type of 'np' (line 318)
        np_207165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 26), 'np', False)
        # Obtaining the member 'array' of a type (line 318)
        array_207166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 26), np_207165, 'array')
        # Calling array(args, kwargs) (line 318)
        array_call_result_207171 = invoke(stypy.reporting.localization.Localization(__file__, 318, 26), array_207166, *[list_207167], **kwargs_207170)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 20), list_207163, array_call_result_207171)
        # Adding element type (line 318)
        str_207172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 48), 'str', 'jac')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 20), list_207163, str_207172)
        
        
        # Obtaining an instance of the builtin type 'list' (line 319)
        list_207173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 319)
        # Adding element type (line 319)
        str_207174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 21), 'str', 'exact')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 20), list_207173, str_207174)
        # Adding element type (line 319)
        str_207175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 30), 'str', 'lsmr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 20), list_207173, str_207175)
        
        # Processing the call keyword arguments (line 316)
        kwargs_207176 = {}
        # Getting the type of 'product' (line 316)
        product_207157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 43), 'product', False)
        # Calling product(args, kwargs) (line 316)
        product_call_result_207177 = invoke(stypy.reporting.localization.Localization(__file__, 316, 43), product_207157, *[list_207158, list_207163, list_207173], **kwargs_207176)
        
        # Testing the type of a for loop iterable (line 316)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 316, 12), product_call_result_207177)
        # Getting the type of the for loop variable (line 316)
        for_loop_var_207178 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 316, 12), product_call_result_207177)
        # Assigning a type to the variable 'jac' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'jac', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 12), for_loop_var_207178))
        # Assigning a type to the variable 'x_scale' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'x_scale', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 12), for_loop_var_207178))
        # Assigning a type to the variable 'tr_solver' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'tr_solver', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 12), for_loop_var_207178))
        # SSA begins for a for statement (line 316)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 320):
        
        # Call to least_squares(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'fun_rosenbrock_cropped' (line 321)
        fun_rosenbrock_cropped_207180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 20), 'fun_rosenbrock_cropped', False)
        # Getting the type of 'x0' (line 321)
        x0_207181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 44), 'x0', False)
        # Getting the type of 'jac' (line 321)
        jac_207182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 48), 'jac', False)
        # Processing the call keyword arguments (line 320)
        # Getting the type of 'x_scale' (line 321)
        x_scale_207183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 61), 'x_scale', False)
        keyword_207184 = x_scale_207183
        # Getting the type of 'tr_solver' (line 322)
        tr_solver_207185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 30), 'tr_solver', False)
        keyword_207186 = tr_solver_207185
        # Getting the type of 'self' (line 322)
        self_207187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 48), 'self', False)
        # Obtaining the member 'method' of a type (line 322)
        method_207188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 48), self_207187, 'method')
        keyword_207189 = method_207188
        kwargs_207190 = {'tr_solver': keyword_207186, 'x_scale': keyword_207184, 'method': keyword_207189}
        # Getting the type of 'least_squares' (line 320)
        least_squares_207179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 22), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 320)
        least_squares_call_result_207191 = invoke(stypy.reporting.localization.Localization(__file__, 320, 22), least_squares_207179, *[fun_rosenbrock_cropped_207180, x0_207181, jac_207182], **kwargs_207190)
        
        # Assigning a type to the variable 'res' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'res', least_squares_call_result_207191)
        
        # Call to assert_allclose(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'res' (line 323)
        res_207193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 32), 'res', False)
        # Obtaining the member 'cost' of a type (line 323)
        cost_207194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 32), res_207193, 'cost')
        int_207195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 42), 'int')
        # Processing the call keyword arguments (line 323)
        float_207196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 50), 'float')
        keyword_207197 = float_207196
        kwargs_207198 = {'atol': keyword_207197}
        # Getting the type of 'assert_allclose' (line 323)
        assert_allclose_207192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 323)
        assert_allclose_call_result_207199 = invoke(stypy.reporting.localization.Localization(__file__, 323, 16), assert_allclose_207192, *[cost_207194, int_207195], **kwargs_207198)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 312)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_rosenbrock_cropped(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_rosenbrock_cropped' in the type store
        # Getting the type of 'stypy_return_type' (line 310)
        stypy_return_type_207200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207200)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_rosenbrock_cropped'
        return stypy_return_type_207200


    @norecursion
    def test_fun_wrong_dimensions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_fun_wrong_dimensions'
        module_type_store = module_type_store.open_function_context('test_fun_wrong_dimensions', 325, 4, False)
        # Assigning a type to the variable 'self' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_fun_wrong_dimensions.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_fun_wrong_dimensions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_fun_wrong_dimensions.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_fun_wrong_dimensions.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_fun_wrong_dimensions')
        BaseMixin.test_fun_wrong_dimensions.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_fun_wrong_dimensions.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_fun_wrong_dimensions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_fun_wrong_dimensions.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_fun_wrong_dimensions.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_fun_wrong_dimensions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_fun_wrong_dimensions.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_fun_wrong_dimensions', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_fun_wrong_dimensions', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_fun_wrong_dimensions(...)' code ##################

        
        # Call to assert_raises(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'ValueError' (line 326)
        ValueError_207202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 326)
        least_squares_207203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 34), 'least_squares', False)
        # Getting the type of 'fun_wrong_dimensions' (line 326)
        fun_wrong_dimensions_207204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 49), 'fun_wrong_dimensions', False)
        float_207205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 22), 'float')
        # Processing the call keyword arguments (line 326)
        # Getting the type of 'self' (line 327)
        self_207206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 34), 'self', False)
        # Obtaining the member 'method' of a type (line 327)
        method_207207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 34), self_207206, 'method')
        keyword_207208 = method_207207
        kwargs_207209 = {'method': keyword_207208}
        # Getting the type of 'assert_raises' (line 326)
        assert_raises_207201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 326)
        assert_raises_call_result_207210 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), assert_raises_207201, *[ValueError_207202, least_squares_207203, fun_wrong_dimensions_207204, float_207205], **kwargs_207209)
        
        
        # ################# End of 'test_fun_wrong_dimensions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_fun_wrong_dimensions' in the type store
        # Getting the type of 'stypy_return_type' (line 325)
        stypy_return_type_207211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207211)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_fun_wrong_dimensions'
        return stypy_return_type_207211


    @norecursion
    def test_jac_wrong_dimensions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_jac_wrong_dimensions'
        module_type_store = module_type_store.open_function_context('test_jac_wrong_dimensions', 329, 4, False)
        # Assigning a type to the variable 'self' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_jac_wrong_dimensions.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_jac_wrong_dimensions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_jac_wrong_dimensions.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_jac_wrong_dimensions.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_jac_wrong_dimensions')
        BaseMixin.test_jac_wrong_dimensions.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_jac_wrong_dimensions.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_jac_wrong_dimensions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_jac_wrong_dimensions.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_jac_wrong_dimensions.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_jac_wrong_dimensions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_jac_wrong_dimensions.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_jac_wrong_dimensions', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_jac_wrong_dimensions', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_jac_wrong_dimensions(...)' code ##################

        
        # Call to assert_raises(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'ValueError' (line 330)
        ValueError_207213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 330)
        least_squares_207214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 330)
        fun_trivial_207215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 49), 'fun_trivial', False)
        float_207216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 22), 'float')
        # Getting the type of 'jac_wrong_dimensions' (line 331)
        jac_wrong_dimensions_207217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 27), 'jac_wrong_dimensions', False)
        # Processing the call keyword arguments (line 330)
        # Getting the type of 'self' (line 331)
        self_207218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 56), 'self', False)
        # Obtaining the member 'method' of a type (line 331)
        method_207219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 56), self_207218, 'method')
        keyword_207220 = method_207219
        kwargs_207221 = {'method': keyword_207220}
        # Getting the type of 'assert_raises' (line 330)
        assert_raises_207212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 330)
        assert_raises_call_result_207222 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), assert_raises_207212, *[ValueError_207213, least_squares_207214, fun_trivial_207215, float_207216, jac_wrong_dimensions_207217], **kwargs_207221)
        
        
        # ################# End of 'test_jac_wrong_dimensions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_jac_wrong_dimensions' in the type store
        # Getting the type of 'stypy_return_type' (line 329)
        stypy_return_type_207223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207223)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_jac_wrong_dimensions'
        return stypy_return_type_207223


    @norecursion
    def test_fun_and_jac_inconsistent_dimensions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_fun_and_jac_inconsistent_dimensions'
        module_type_store = module_type_store.open_function_context('test_fun_and_jac_inconsistent_dimensions', 333, 4, False)
        # Assigning a type to the variable 'self' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_fun_and_jac_inconsistent_dimensions.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_fun_and_jac_inconsistent_dimensions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_fun_and_jac_inconsistent_dimensions.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_fun_and_jac_inconsistent_dimensions.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_fun_and_jac_inconsistent_dimensions')
        BaseMixin.test_fun_and_jac_inconsistent_dimensions.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_fun_and_jac_inconsistent_dimensions.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_fun_and_jac_inconsistent_dimensions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_fun_and_jac_inconsistent_dimensions.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_fun_and_jac_inconsistent_dimensions.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_fun_and_jac_inconsistent_dimensions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_fun_and_jac_inconsistent_dimensions.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_fun_and_jac_inconsistent_dimensions', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_fun_and_jac_inconsistent_dimensions', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_fun_and_jac_inconsistent_dimensions(...)' code ##################

        
        # Assigning a List to a Name (line 334):
        
        # Obtaining an instance of the builtin type 'list' (line 334)
        list_207224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 334)
        # Adding element type (line 334)
        int_207225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 13), list_207224, int_207225)
        # Adding element type (line 334)
        int_207226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 13), list_207224, int_207226)
        
        # Assigning a type to the variable 'x0' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'x0', list_207224)
        
        # Call to assert_raises(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'ValueError' (line 335)
        ValueError_207228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 335)
        least_squares_207229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 34), 'least_squares', False)
        # Getting the type of 'fun_rosenbrock' (line 335)
        fun_rosenbrock_207230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 49), 'fun_rosenbrock', False)
        # Getting the type of 'x0' (line 335)
        x0_207231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 65), 'x0', False)
        # Getting the type of 'jac_rosenbrock_bad_dim' (line 336)
        jac_rosenbrock_bad_dim_207232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 22), 'jac_rosenbrock_bad_dim', False)
        # Processing the call keyword arguments (line 335)
        # Getting the type of 'self' (line 336)
        self_207233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 53), 'self', False)
        # Obtaining the member 'method' of a type (line 336)
        method_207234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 53), self_207233, 'method')
        keyword_207235 = method_207234
        kwargs_207236 = {'method': keyword_207235}
        # Getting the type of 'assert_raises' (line 335)
        assert_raises_207227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 335)
        assert_raises_call_result_207237 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), assert_raises_207227, *[ValueError_207228, least_squares_207229, fun_rosenbrock_207230, x0_207231, jac_rosenbrock_bad_dim_207232], **kwargs_207236)
        
        
        # ################# End of 'test_fun_and_jac_inconsistent_dimensions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_fun_and_jac_inconsistent_dimensions' in the type store
        # Getting the type of 'stypy_return_type' (line 333)
        stypy_return_type_207238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207238)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_fun_and_jac_inconsistent_dimensions'
        return stypy_return_type_207238


    @norecursion
    def test_x0_multidimensional(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x0_multidimensional'
        module_type_store = module_type_store.open_function_context('test_x0_multidimensional', 338, 4, False)
        # Assigning a type to the variable 'self' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_x0_multidimensional.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_x0_multidimensional.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_x0_multidimensional.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_x0_multidimensional.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_x0_multidimensional')
        BaseMixin.test_x0_multidimensional.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_x0_multidimensional.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_x0_multidimensional.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_x0_multidimensional.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_x0_multidimensional.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_x0_multidimensional.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_x0_multidimensional.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_x0_multidimensional', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x0_multidimensional', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x0_multidimensional(...)' code ##################

        
        # Assigning a Call to a Name (line 339):
        
        # Call to reshape(...): (line 339)
        # Processing the call arguments (line 339)
        int_207245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 32), 'int')
        int_207246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 35), 'int')
        # Processing the call keyword arguments (line 339)
        kwargs_207247 = {}
        
        # Call to ones(...): (line 339)
        # Processing the call arguments (line 339)
        int_207241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 21), 'int')
        # Processing the call keyword arguments (line 339)
        kwargs_207242 = {}
        # Getting the type of 'np' (line 339)
        np_207239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 13), 'np', False)
        # Obtaining the member 'ones' of a type (line 339)
        ones_207240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 13), np_207239, 'ones')
        # Calling ones(args, kwargs) (line 339)
        ones_call_result_207243 = invoke(stypy.reporting.localization.Localization(__file__, 339, 13), ones_207240, *[int_207241], **kwargs_207242)
        
        # Obtaining the member 'reshape' of a type (line 339)
        reshape_207244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 13), ones_call_result_207243, 'reshape')
        # Calling reshape(args, kwargs) (line 339)
        reshape_call_result_207248 = invoke(stypy.reporting.localization.Localization(__file__, 339, 13), reshape_207244, *[int_207245, int_207246], **kwargs_207247)
        
        # Assigning a type to the variable 'x0' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'x0', reshape_call_result_207248)
        
        # Call to assert_raises(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'ValueError' (line 340)
        ValueError_207250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 340)
        least_squares_207251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 340)
        fun_trivial_207252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 49), 'fun_trivial', False)
        # Getting the type of 'x0' (line 340)
        x0_207253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 62), 'x0', False)
        # Processing the call keyword arguments (line 340)
        # Getting the type of 'self' (line 341)
        self_207254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 29), 'self', False)
        # Obtaining the member 'method' of a type (line 341)
        method_207255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 29), self_207254, 'method')
        keyword_207256 = method_207255
        kwargs_207257 = {'method': keyword_207256}
        # Getting the type of 'assert_raises' (line 340)
        assert_raises_207249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 340)
        assert_raises_call_result_207258 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), assert_raises_207249, *[ValueError_207250, least_squares_207251, fun_trivial_207252, x0_207253], **kwargs_207257)
        
        
        # ################# End of 'test_x0_multidimensional(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x0_multidimensional' in the type store
        # Getting the type of 'stypy_return_type' (line 338)
        stypy_return_type_207259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207259)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x0_multidimensional'
        return stypy_return_type_207259


    @norecursion
    def test_x0_complex_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x0_complex_scalar'
        module_type_store = module_type_store.open_function_context('test_x0_complex_scalar', 343, 4, False)
        # Assigning a type to the variable 'self' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_x0_complex_scalar.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_x0_complex_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_x0_complex_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_x0_complex_scalar.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_x0_complex_scalar')
        BaseMixin.test_x0_complex_scalar.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_x0_complex_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_x0_complex_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_x0_complex_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_x0_complex_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_x0_complex_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_x0_complex_scalar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_x0_complex_scalar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x0_complex_scalar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x0_complex_scalar(...)' code ##################

        
        # Assigning a BinOp to a Name (line 344):
        float_207260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 13), 'float')
        float_207261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 19), 'float')
        complex_207262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 23), 'complex')
        # Applying the binary operator '*' (line 344)
        result_mul_207263 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 19), '*', float_207261, complex_207262)
        
        # Applying the binary operator '+' (line 344)
        result_add_207264 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 13), '+', float_207260, result_mul_207263)
        
        # Assigning a type to the variable 'x0' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'x0', result_add_207264)
        
        # Call to assert_raises(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'ValueError' (line 345)
        ValueError_207266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 345)
        least_squares_207267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 345)
        fun_trivial_207268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 49), 'fun_trivial', False)
        # Getting the type of 'x0' (line 345)
        x0_207269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 62), 'x0', False)
        # Processing the call keyword arguments (line 345)
        # Getting the type of 'self' (line 346)
        self_207270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 29), 'self', False)
        # Obtaining the member 'method' of a type (line 346)
        method_207271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 29), self_207270, 'method')
        keyword_207272 = method_207271
        kwargs_207273 = {'method': keyword_207272}
        # Getting the type of 'assert_raises' (line 345)
        assert_raises_207265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 345)
        assert_raises_call_result_207274 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), assert_raises_207265, *[ValueError_207266, least_squares_207267, fun_trivial_207268, x0_207269], **kwargs_207273)
        
        
        # ################# End of 'test_x0_complex_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x0_complex_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 343)
        stypy_return_type_207275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207275)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x0_complex_scalar'
        return stypy_return_type_207275


    @norecursion
    def test_x0_complex_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x0_complex_array'
        module_type_store = module_type_store.open_function_context('test_x0_complex_array', 348, 4, False)
        # Assigning a type to the variable 'self' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_x0_complex_array.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_x0_complex_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_x0_complex_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_x0_complex_array.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_x0_complex_array')
        BaseMixin.test_x0_complex_array.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_x0_complex_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_x0_complex_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_x0_complex_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_x0_complex_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_x0_complex_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_x0_complex_array.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_x0_complex_array', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x0_complex_array', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x0_complex_array(...)' code ##################

        
        # Assigning a List to a Name (line 349):
        
        # Obtaining an instance of the builtin type 'list' (line 349)
        list_207276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 349)
        # Adding element type (line 349)
        float_207277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 13), list_207276, float_207277)
        # Adding element type (line 349)
        float_207278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 19), 'float')
        float_207279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 25), 'float')
        complex_207280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 29), 'complex')
        # Applying the binary operator '*' (line 349)
        result_mul_207281 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 25), '*', float_207279, complex_207280)
        
        # Applying the binary operator '+' (line 349)
        result_add_207282 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 19), '+', float_207278, result_mul_207281)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 13), list_207276, result_add_207282)
        
        # Assigning a type to the variable 'x0' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'x0', list_207276)
        
        # Call to assert_raises(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'ValueError' (line 350)
        ValueError_207284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 350)
        least_squares_207285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 350)
        fun_trivial_207286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 49), 'fun_trivial', False)
        # Getting the type of 'x0' (line 350)
        x0_207287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 62), 'x0', False)
        # Processing the call keyword arguments (line 350)
        # Getting the type of 'self' (line 351)
        self_207288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 29), 'self', False)
        # Obtaining the member 'method' of a type (line 351)
        method_207289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 29), self_207288, 'method')
        keyword_207290 = method_207289
        kwargs_207291 = {'method': keyword_207290}
        # Getting the type of 'assert_raises' (line 350)
        assert_raises_207283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 350)
        assert_raises_call_result_207292 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), assert_raises_207283, *[ValueError_207284, least_squares_207285, fun_trivial_207286, x0_207287], **kwargs_207291)
        
        
        # ################# End of 'test_x0_complex_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x0_complex_array' in the type store
        # Getting the type of 'stypy_return_type' (line 348)
        stypy_return_type_207293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207293)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x0_complex_array'
        return stypy_return_type_207293


    @norecursion
    def test_bvp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bvp'
        module_type_store = module_type_store.open_function_context('test_bvp', 353, 4, False)
        # Assigning a type to the variable 'self' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseMixin.test_bvp.__dict__.__setitem__('stypy_localization', localization)
        BaseMixin.test_bvp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseMixin.test_bvp.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseMixin.test_bvp.__dict__.__setitem__('stypy_function_name', 'BaseMixin.test_bvp')
        BaseMixin.test_bvp.__dict__.__setitem__('stypy_param_names_list', [])
        BaseMixin.test_bvp.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseMixin.test_bvp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseMixin.test_bvp.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseMixin.test_bvp.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseMixin.test_bvp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseMixin.test_bvp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.test_bvp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bvp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bvp(...)' code ##################

        
        # Assigning a Num to a Name (line 358):
        int_207294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 12), 'int')
        # Assigning a type to the variable 'n' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'n', int_207294)
        
        # Assigning a Call to a Name (line 359):
        
        # Call to ones(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'n' (line 359)
        n_207297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 21), 'n', False)
        int_207298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 24), 'int')
        # Applying the binary operator '**' (line 359)
        result_pow_207299 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 21), '**', n_207297, int_207298)
        
        # Processing the call keyword arguments (line 359)
        kwargs_207300 = {}
        # Getting the type of 'np' (line 359)
        np_207295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 13), 'np', False)
        # Obtaining the member 'ones' of a type (line 359)
        ones_207296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), np_207295, 'ones')
        # Calling ones(args, kwargs) (line 359)
        ones_call_result_207301 = invoke(stypy.reporting.localization.Localization(__file__, 359, 13), ones_207296, *[result_pow_207299], **kwargs_207300)
        
        # Assigning a type to the variable 'x0' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'x0', ones_call_result_207301)
        
        
        # Getting the type of 'self' (line 360)
        self_207302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 11), 'self')
        # Obtaining the member 'method' of a type (line 360)
        method_207303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 11), self_207302, 'method')
        str_207304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 26), 'str', 'lm')
        # Applying the binary operator '==' (line 360)
        result_eq_207305 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 11), '==', method_207303, str_207304)
        
        # Testing the type of an if condition (line 360)
        if_condition_207306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 8), result_eq_207305)
        # Assigning a type to the variable 'if_condition_207306' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'if_condition_207306', if_condition_207306)
        # SSA begins for if statement (line 360)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 361):
        int_207307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 23), 'int')
        # Assigning a type to the variable 'max_nfev' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'max_nfev', int_207307)
        # SSA branch for the else part of an if statement (line 360)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 363):
        int_207308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 23), 'int')
        # Assigning a type to the variable 'max_nfev' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'max_nfev', int_207308)
        # SSA join for if statement (line 360)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 364):
        
        # Call to least_squares(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'fun_bvp' (line 364)
        fun_bvp_207310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 28), 'fun_bvp', False)
        # Getting the type of 'x0' (line 364)
        x0_207311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 37), 'x0', False)
        # Processing the call keyword arguments (line 364)
        float_207312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 46), 'float')
        keyword_207313 = float_207312
        # Getting the type of 'self' (line 364)
        self_207314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 59), 'self', False)
        # Obtaining the member 'method' of a type (line 364)
        method_207315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 59), self_207314, 'method')
        keyword_207316 = method_207315
        # Getting the type of 'max_nfev' (line 365)
        max_nfev_207317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 37), 'max_nfev', False)
        keyword_207318 = max_nfev_207317
        kwargs_207319 = {'max_nfev': keyword_207318, 'method': keyword_207316, 'ftol': keyword_207313}
        # Getting the type of 'least_squares' (line 364)
        least_squares_207309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 364)
        least_squares_call_result_207320 = invoke(stypy.reporting.localization.Localization(__file__, 364, 14), least_squares_207309, *[fun_bvp_207310, x0_207311], **kwargs_207319)
        
        # Assigning a type to the variable 'res' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'res', least_squares_call_result_207320)
        
        # Call to assert_(...): (line 367)
        # Processing the call arguments (line 367)
        
        # Getting the type of 'res' (line 367)
        res_207322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'res', False)
        # Obtaining the member 'nfev' of a type (line 367)
        nfev_207323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 16), res_207322, 'nfev')
        # Getting the type of 'max_nfev' (line 367)
        max_nfev_207324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 27), 'max_nfev', False)
        # Applying the binary operator '<' (line 367)
        result_lt_207325 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 16), '<', nfev_207323, max_nfev_207324)
        
        # Processing the call keyword arguments (line 367)
        kwargs_207326 = {}
        # Getting the type of 'assert_' (line 367)
        assert__207321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 367)
        assert__call_result_207327 = invoke(stypy.reporting.localization.Localization(__file__, 367, 8), assert__207321, *[result_lt_207325], **kwargs_207326)
        
        
        # Call to assert_(...): (line 368)
        # Processing the call arguments (line 368)
        
        # Getting the type of 'res' (line 368)
        res_207329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'res', False)
        # Obtaining the member 'cost' of a type (line 368)
        cost_207330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 16), res_207329, 'cost')
        float_207331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 27), 'float')
        # Applying the binary operator '<' (line 368)
        result_lt_207332 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 16), '<', cost_207330, float_207331)
        
        # Processing the call keyword arguments (line 368)
        kwargs_207333 = {}
        # Getting the type of 'assert_' (line 368)
        assert__207328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 368)
        assert__call_result_207334 = invoke(stypy.reporting.localization.Localization(__file__, 368, 8), assert__207328, *[result_lt_207332], **kwargs_207333)
        
        
        # ################# End of 'test_bvp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bvp' in the type store
        # Getting the type of 'stypy_return_type' (line 353)
        stypy_return_type_207335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207335)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bvp'
        return stypy_return_type_207335


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 177, 0, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseMixin.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BaseMixin' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'BaseMixin', BaseMixin)
# Declaration of the 'BoundsMixin' class

class BoundsMixin(object, ):

    @norecursion
    def test_inconsistent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_inconsistent'
        module_type_store = module_type_store.open_function_context('test_inconsistent', 372, 4, False)
        # Assigning a type to the variable 'self' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BoundsMixin.test_inconsistent.__dict__.__setitem__('stypy_localization', localization)
        BoundsMixin.test_inconsistent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BoundsMixin.test_inconsistent.__dict__.__setitem__('stypy_type_store', module_type_store)
        BoundsMixin.test_inconsistent.__dict__.__setitem__('stypy_function_name', 'BoundsMixin.test_inconsistent')
        BoundsMixin.test_inconsistent.__dict__.__setitem__('stypy_param_names_list', [])
        BoundsMixin.test_inconsistent.__dict__.__setitem__('stypy_varargs_param_name', None)
        BoundsMixin.test_inconsistent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BoundsMixin.test_inconsistent.__dict__.__setitem__('stypy_call_defaults', defaults)
        BoundsMixin.test_inconsistent.__dict__.__setitem__('stypy_call_varargs', varargs)
        BoundsMixin.test_inconsistent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BoundsMixin.test_inconsistent.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BoundsMixin.test_inconsistent', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_inconsistent', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_inconsistent(...)' code ##################

        
        # Call to assert_raises(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'ValueError' (line 373)
        ValueError_207337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 373)
        least_squares_207338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 373)
        fun_trivial_207339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 49), 'fun_trivial', False)
        float_207340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 62), 'float')
        # Processing the call keyword arguments (line 373)
        
        # Obtaining an instance of the builtin type 'tuple' (line 374)
        tuple_207341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 374)
        # Adding element type (line 374)
        float_207342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 30), tuple_207341, float_207342)
        # Adding element type (line 374)
        float_207343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 30), tuple_207341, float_207343)
        
        keyword_207344 = tuple_207341
        # Getting the type of 'self' (line 374)
        self_207345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 49), 'self', False)
        # Obtaining the member 'method' of a type (line 374)
        method_207346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 49), self_207345, 'method')
        keyword_207347 = method_207346
        kwargs_207348 = {'bounds': keyword_207344, 'method': keyword_207347}
        # Getting the type of 'assert_raises' (line 373)
        assert_raises_207336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 373)
        assert_raises_call_result_207349 = invoke(stypy.reporting.localization.Localization(__file__, 373, 8), assert_raises_207336, *[ValueError_207337, least_squares_207338, fun_trivial_207339, float_207340], **kwargs_207348)
        
        
        # ################# End of 'test_inconsistent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_inconsistent' in the type store
        # Getting the type of 'stypy_return_type' (line 372)
        stypy_return_type_207350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207350)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_inconsistent'
        return stypy_return_type_207350


    @norecursion
    def test_infeasible(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_infeasible'
        module_type_store = module_type_store.open_function_context('test_infeasible', 376, 4, False)
        # Assigning a type to the variable 'self' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BoundsMixin.test_infeasible.__dict__.__setitem__('stypy_localization', localization)
        BoundsMixin.test_infeasible.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BoundsMixin.test_infeasible.__dict__.__setitem__('stypy_type_store', module_type_store)
        BoundsMixin.test_infeasible.__dict__.__setitem__('stypy_function_name', 'BoundsMixin.test_infeasible')
        BoundsMixin.test_infeasible.__dict__.__setitem__('stypy_param_names_list', [])
        BoundsMixin.test_infeasible.__dict__.__setitem__('stypy_varargs_param_name', None)
        BoundsMixin.test_infeasible.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BoundsMixin.test_infeasible.__dict__.__setitem__('stypy_call_defaults', defaults)
        BoundsMixin.test_infeasible.__dict__.__setitem__('stypy_call_varargs', varargs)
        BoundsMixin.test_infeasible.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BoundsMixin.test_infeasible.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BoundsMixin.test_infeasible', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_infeasible', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_infeasible(...)' code ##################

        
        # Call to assert_raises(...): (line 377)
        # Processing the call arguments (line 377)
        # Getting the type of 'ValueError' (line 377)
        ValueError_207352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 377)
        least_squares_207353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 377)
        fun_trivial_207354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 49), 'fun_trivial', False)
        float_207355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 62), 'float')
        # Processing the call keyword arguments (line 377)
        
        # Obtaining an instance of the builtin type 'tuple' (line 378)
        tuple_207356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 378)
        # Adding element type (line 378)
        float_207357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 30), tuple_207356, float_207357)
        # Adding element type (line 378)
        int_207358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 30), tuple_207356, int_207358)
        
        keyword_207359 = tuple_207356
        # Getting the type of 'self' (line 378)
        self_207360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 45), 'self', False)
        # Obtaining the member 'method' of a type (line 378)
        method_207361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 45), self_207360, 'method')
        keyword_207362 = method_207361
        kwargs_207363 = {'bounds': keyword_207359, 'method': keyword_207362}
        # Getting the type of 'assert_raises' (line 377)
        assert_raises_207351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 377)
        assert_raises_call_result_207364 = invoke(stypy.reporting.localization.Localization(__file__, 377, 8), assert_raises_207351, *[ValueError_207352, least_squares_207353, fun_trivial_207354, float_207355], **kwargs_207363)
        
        
        # ################# End of 'test_infeasible(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_infeasible' in the type store
        # Getting the type of 'stypy_return_type' (line 376)
        stypy_return_type_207365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207365)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_infeasible'
        return stypy_return_type_207365


    @norecursion
    def test_wrong_number(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_wrong_number'
        module_type_store = module_type_store.open_function_context('test_wrong_number', 380, 4, False)
        # Assigning a type to the variable 'self' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BoundsMixin.test_wrong_number.__dict__.__setitem__('stypy_localization', localization)
        BoundsMixin.test_wrong_number.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BoundsMixin.test_wrong_number.__dict__.__setitem__('stypy_type_store', module_type_store)
        BoundsMixin.test_wrong_number.__dict__.__setitem__('stypy_function_name', 'BoundsMixin.test_wrong_number')
        BoundsMixin.test_wrong_number.__dict__.__setitem__('stypy_param_names_list', [])
        BoundsMixin.test_wrong_number.__dict__.__setitem__('stypy_varargs_param_name', None)
        BoundsMixin.test_wrong_number.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BoundsMixin.test_wrong_number.__dict__.__setitem__('stypy_call_defaults', defaults)
        BoundsMixin.test_wrong_number.__dict__.__setitem__('stypy_call_varargs', varargs)
        BoundsMixin.test_wrong_number.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BoundsMixin.test_wrong_number.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BoundsMixin.test_wrong_number', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_wrong_number', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_wrong_number(...)' code ##################

        
        # Call to assert_raises(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'ValueError' (line 381)
        ValueError_207367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 381)
        least_squares_207368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 381)
        fun_trivial_207369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 49), 'fun_trivial', False)
        float_207370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 62), 'float')
        # Processing the call keyword arguments (line 381)
        
        # Obtaining an instance of the builtin type 'tuple' (line 382)
        tuple_207371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 382)
        # Adding element type (line 382)
        float_207372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 30), tuple_207371, float_207372)
        # Adding element type (line 382)
        int_207373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 30), tuple_207371, int_207373)
        # Adding element type (line 382)
        int_207374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 30), tuple_207371, int_207374)
        
        keyword_207375 = tuple_207371
        # Getting the type of 'self' (line 382)
        self_207376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 48), 'self', False)
        # Obtaining the member 'method' of a type (line 382)
        method_207377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 48), self_207376, 'method')
        keyword_207378 = method_207377
        kwargs_207379 = {'bounds': keyword_207375, 'method': keyword_207378}
        # Getting the type of 'assert_raises' (line 381)
        assert_raises_207366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 381)
        assert_raises_call_result_207380 = invoke(stypy.reporting.localization.Localization(__file__, 381, 8), assert_raises_207366, *[ValueError_207367, least_squares_207368, fun_trivial_207369, float_207370], **kwargs_207379)
        
        
        # ################# End of 'test_wrong_number(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_wrong_number' in the type store
        # Getting the type of 'stypy_return_type' (line 380)
        stypy_return_type_207381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207381)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_wrong_number'
        return stypy_return_type_207381


    @norecursion
    def test_inconsistent_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_inconsistent_shape'
        module_type_store = module_type_store.open_function_context('test_inconsistent_shape', 384, 4, False)
        # Assigning a type to the variable 'self' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BoundsMixin.test_inconsistent_shape.__dict__.__setitem__('stypy_localization', localization)
        BoundsMixin.test_inconsistent_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BoundsMixin.test_inconsistent_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        BoundsMixin.test_inconsistent_shape.__dict__.__setitem__('stypy_function_name', 'BoundsMixin.test_inconsistent_shape')
        BoundsMixin.test_inconsistent_shape.__dict__.__setitem__('stypy_param_names_list', [])
        BoundsMixin.test_inconsistent_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        BoundsMixin.test_inconsistent_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BoundsMixin.test_inconsistent_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        BoundsMixin.test_inconsistent_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        BoundsMixin.test_inconsistent_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BoundsMixin.test_inconsistent_shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BoundsMixin.test_inconsistent_shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_inconsistent_shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_inconsistent_shape(...)' code ##################

        
        # Call to assert_raises(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'ValueError' (line 385)
        ValueError_207383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 385)
        least_squares_207384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 385)
        fun_trivial_207385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 49), 'fun_trivial', False)
        float_207386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 62), 'float')
        # Processing the call keyword arguments (line 385)
        
        # Obtaining an instance of the builtin type 'tuple' (line 386)
        tuple_207387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 386)
        # Adding element type (line 386)
        float_207388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 30), tuple_207387, float_207388)
        # Adding element type (line 386)
        
        # Obtaining an instance of the builtin type 'list' (line 386)
        list_207389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 386)
        # Adding element type (line 386)
        float_207390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 35), list_207389, float_207390)
        # Adding element type (line 386)
        float_207391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 35), list_207389, float_207391)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 30), tuple_207387, list_207389)
        
        keyword_207392 = tuple_207387
        # Getting the type of 'self' (line 386)
        self_207393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 55), 'self', False)
        # Obtaining the member 'method' of a type (line 386)
        method_207394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 55), self_207393, 'method')
        keyword_207395 = method_207394
        kwargs_207396 = {'bounds': keyword_207392, 'method': keyword_207395}
        # Getting the type of 'assert_raises' (line 385)
        assert_raises_207382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 385)
        assert_raises_call_result_207397 = invoke(stypy.reporting.localization.Localization(__file__, 385, 8), assert_raises_207382, *[ValueError_207383, least_squares_207384, fun_trivial_207385, float_207386], **kwargs_207396)
        
        
        # Call to assert_raises(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'ValueError' (line 388)
        ValueError_207399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 388)
        least_squares_207400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 34), 'least_squares', False)
        # Getting the type of 'fun_rosenbrock' (line 388)
        fun_rosenbrock_207401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 49), 'fun_rosenbrock', False)
        
        # Obtaining an instance of the builtin type 'list' (line 388)
        list_207402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 65), 'list')
        # Adding type elements to the builtin type 'list' instance (line 388)
        # Adding element type (line 388)
        float_207403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 65), list_207402, float_207403)
        # Adding element type (line 388)
        float_207404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 71), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 65), list_207402, float_207404)
        
        # Processing the call keyword arguments (line 388)
        
        # Obtaining an instance of the builtin type 'tuple' (line 389)
        tuple_207405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 389)
        # Adding element type (line 389)
        
        # Obtaining an instance of the builtin type 'list' (line 389)
        list_207406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 389)
        # Adding element type (line 389)
        float_207407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 30), list_207406, float_207407)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 30), tuple_207405, list_207406)
        # Adding element type (line 389)
        
        # Obtaining an instance of the builtin type 'list' (line 389)
        list_207408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 389)
        # Adding element type (line 389)
        float_207409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 37), list_207408, float_207409)
        # Adding element type (line 389)
        float_207410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 37), list_207408, float_207410)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 30), tuple_207405, list_207408)
        
        keyword_207411 = tuple_207405
        # Getting the type of 'self' (line 389)
        self_207412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 57), 'self', False)
        # Obtaining the member 'method' of a type (line 389)
        method_207413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 57), self_207412, 'method')
        keyword_207414 = method_207413
        kwargs_207415 = {'bounds': keyword_207411, 'method': keyword_207414}
        # Getting the type of 'assert_raises' (line 388)
        assert_raises_207398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 388)
        assert_raises_call_result_207416 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), assert_raises_207398, *[ValueError_207399, least_squares_207400, fun_rosenbrock_207401, list_207402], **kwargs_207415)
        
        
        # ################# End of 'test_inconsistent_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_inconsistent_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 384)
        stypy_return_type_207417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207417)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_inconsistent_shape'
        return stypy_return_type_207417


    @norecursion
    def test_in_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_in_bounds'
        module_type_store = module_type_store.open_function_context('test_in_bounds', 391, 4, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BoundsMixin.test_in_bounds.__dict__.__setitem__('stypy_localization', localization)
        BoundsMixin.test_in_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BoundsMixin.test_in_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        BoundsMixin.test_in_bounds.__dict__.__setitem__('stypy_function_name', 'BoundsMixin.test_in_bounds')
        BoundsMixin.test_in_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        BoundsMixin.test_in_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        BoundsMixin.test_in_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BoundsMixin.test_in_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        BoundsMixin.test_in_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        BoundsMixin.test_in_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BoundsMixin.test_in_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BoundsMixin.test_in_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_in_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_in_bounds(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 392)
        list_207418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 392)
        # Adding element type (line 392)
        str_207419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 20), 'str', '2-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 19), list_207418, str_207419)
        # Adding element type (line 392)
        str_207420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 31), 'str', '3-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 19), list_207418, str_207420)
        # Adding element type (line 392)
        str_207421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 42), 'str', 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 19), list_207418, str_207421)
        # Adding element type (line 392)
        # Getting the type of 'jac_trivial' (line 392)
        jac_trivial_207422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 48), 'jac_trivial')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 19), list_207418, jac_trivial_207422)
        
        # Testing the type of a for loop iterable (line 392)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 392, 8), list_207418)
        # Getting the type of the for loop variable (line 392)
        for_loop_var_207423 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 392, 8), list_207418)
        # Assigning a type to the variable 'jac' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'jac', for_loop_var_207423)
        # SSA begins for a for statement (line 392)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 393):
        
        # Call to least_squares(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'fun_trivial' (line 393)
        fun_trivial_207425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 32), 'fun_trivial', False)
        float_207426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 45), 'float')
        # Processing the call keyword arguments (line 393)
        # Getting the type of 'jac' (line 393)
        jac_207427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 54), 'jac', False)
        keyword_207428 = jac_207427
        
        # Obtaining an instance of the builtin type 'tuple' (line 394)
        tuple_207429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 394)
        # Adding element type (line 394)
        float_207430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 40), tuple_207429, float_207430)
        # Adding element type (line 394)
        float_207431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 40), tuple_207429, float_207431)
        
        keyword_207432 = tuple_207429
        # Getting the type of 'self' (line 394)
        self_207433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 59), 'self', False)
        # Obtaining the member 'method' of a type (line 394)
        method_207434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 59), self_207433, 'method')
        keyword_207435 = method_207434
        kwargs_207436 = {'method': keyword_207435, 'jac': keyword_207428, 'bounds': keyword_207432}
        # Getting the type of 'least_squares' (line 393)
        least_squares_207424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 18), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 393)
        least_squares_call_result_207437 = invoke(stypy.reporting.localization.Localization(__file__, 393, 18), least_squares_207424, *[fun_trivial_207425, float_207426], **kwargs_207436)
        
        # Assigning a type to the variable 'res' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'res', least_squares_call_result_207437)
        
        # Call to assert_allclose(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'res' (line 395)
        res_207439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 395)
        x_207440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 28), res_207439, 'x')
        float_207441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 35), 'float')
        # Processing the call keyword arguments (line 395)
        float_207442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 45), 'float')
        keyword_207443 = float_207442
        kwargs_207444 = {'atol': keyword_207443}
        # Getting the type of 'assert_allclose' (line 395)
        assert_allclose_207438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 395)
        assert_allclose_call_result_207445 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), assert_allclose_207438, *[x_207440, float_207441], **kwargs_207444)
        
        
        # Call to assert_equal(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'res' (line 396)
        res_207447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 25), 'res', False)
        # Obtaining the member 'active_mask' of a type (line 396)
        active_mask_207448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 25), res_207447, 'active_mask')
        
        # Obtaining an instance of the builtin type 'list' (line 396)
        list_207449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 396)
        # Adding element type (line 396)
        int_207450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 42), list_207449, int_207450)
        
        # Processing the call keyword arguments (line 396)
        kwargs_207451 = {}
        # Getting the type of 'assert_equal' (line 396)
        assert_equal_207446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 396)
        assert_equal_call_result_207452 = invoke(stypy.reporting.localization.Localization(__file__, 396, 12), assert_equal_207446, *[active_mask_207448, list_207449], **kwargs_207451)
        
        
        # Call to assert_(...): (line 397)
        # Processing the call arguments (line 397)
        
        int_207454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 20), 'int')
        # Getting the type of 'res' (line 397)
        res_207455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 26), 'res', False)
        # Obtaining the member 'x' of a type (line 397)
        x_207456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 26), res_207455, 'x')
        # Applying the binary operator '<=' (line 397)
        result_le_207457 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 20), '<=', int_207454, x_207456)
        int_207458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 35), 'int')
        # Applying the binary operator '<=' (line 397)
        result_le_207459 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 20), '<=', x_207456, int_207458)
        # Applying the binary operator '&' (line 397)
        result_and__207460 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 20), '&', result_le_207457, result_le_207459)
        
        # Processing the call keyword arguments (line 397)
        kwargs_207461 = {}
        # Getting the type of 'assert_' (line 397)
        assert__207453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 397)
        assert__call_result_207462 = invoke(stypy.reporting.localization.Localization(__file__, 397, 12), assert__207453, *[result_and__207460], **kwargs_207461)
        
        
        # Assigning a Call to a Name (line 398):
        
        # Call to least_squares(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'fun_trivial' (line 398)
        fun_trivial_207464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 32), 'fun_trivial', False)
        float_207465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 45), 'float')
        # Processing the call keyword arguments (line 398)
        # Getting the type of 'jac' (line 398)
        jac_207466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 54), 'jac', False)
        keyword_207467 = jac_207466
        
        # Obtaining an instance of the builtin type 'tuple' (line 399)
        tuple_207468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 399)
        # Adding element type (line 399)
        float_207469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 40), tuple_207468, float_207469)
        # Adding element type (line 399)
        float_207470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 40), tuple_207468, float_207470)
        
        keyword_207471 = tuple_207468
        # Getting the type of 'self' (line 399)
        self_207472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 58), 'self', False)
        # Obtaining the member 'method' of a type (line 399)
        method_207473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 58), self_207472, 'method')
        keyword_207474 = method_207473
        kwargs_207475 = {'method': keyword_207474, 'jac': keyword_207467, 'bounds': keyword_207471}
        # Getting the type of 'least_squares' (line 398)
        least_squares_207463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 18), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 398)
        least_squares_call_result_207476 = invoke(stypy.reporting.localization.Localization(__file__, 398, 18), least_squares_207463, *[fun_trivial_207464, float_207465], **kwargs_207475)
        
        # Assigning a type to the variable 'res' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'res', least_squares_call_result_207476)
        
        # Call to assert_allclose(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'res' (line 400)
        res_207478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 400)
        x_207479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 28), res_207478, 'x')
        float_207480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 35), 'float')
        # Processing the call keyword arguments (line 400)
        float_207481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 45), 'float')
        keyword_207482 = float_207481
        kwargs_207483 = {'atol': keyword_207482}
        # Getting the type of 'assert_allclose' (line 400)
        assert_allclose_207477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 400)
        assert_allclose_call_result_207484 = invoke(stypy.reporting.localization.Localization(__file__, 400, 12), assert_allclose_207477, *[x_207479, float_207480], **kwargs_207483)
        
        
        # Call to assert_equal(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'res' (line 401)
        res_207486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 25), 'res', False)
        # Obtaining the member 'active_mask' of a type (line 401)
        active_mask_207487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 25), res_207486, 'active_mask')
        
        # Obtaining an instance of the builtin type 'list' (line 401)
        list_207488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 401)
        # Adding element type (line 401)
        int_207489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 42), list_207488, int_207489)
        
        # Processing the call keyword arguments (line 401)
        kwargs_207490 = {}
        # Getting the type of 'assert_equal' (line 401)
        assert_equal_207485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 401)
        assert_equal_call_result_207491 = invoke(stypy.reporting.localization.Localization(__file__, 401, 12), assert_equal_207485, *[active_mask_207487, list_207488], **kwargs_207490)
        
        
        # Call to assert_(...): (line 402)
        # Processing the call arguments (line 402)
        
        float_207493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 20), 'float')
        # Getting the type of 'res' (line 402)
        res_207494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 27), 'res', False)
        # Obtaining the member 'x' of a type (line 402)
        x_207495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 27), res_207494, 'x')
        # Applying the binary operator '<=' (line 402)
        result_le_207496 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 20), '<=', float_207493, x_207495)
        int_207497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 36), 'int')
        # Applying the binary operator '<=' (line 402)
        result_le_207498 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 20), '<=', x_207495, int_207497)
        # Applying the binary operator '&' (line 402)
        result_and__207499 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 20), '&', result_le_207496, result_le_207498)
        
        # Processing the call keyword arguments (line 402)
        kwargs_207500 = {}
        # Getting the type of 'assert_' (line 402)
        assert__207492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 402)
        assert__call_result_207501 = invoke(stypy.reporting.localization.Localization(__file__, 402, 12), assert__207492, *[result_and__207499], **kwargs_207500)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_in_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_in_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 391)
        stypy_return_type_207502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207502)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_in_bounds'
        return stypy_return_type_207502


    @norecursion
    def test_bounds_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bounds_shape'
        module_type_store = module_type_store.open_function_context('test_bounds_shape', 404, 4, False)
        # Assigning a type to the variable 'self' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BoundsMixin.test_bounds_shape.__dict__.__setitem__('stypy_localization', localization)
        BoundsMixin.test_bounds_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BoundsMixin.test_bounds_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        BoundsMixin.test_bounds_shape.__dict__.__setitem__('stypy_function_name', 'BoundsMixin.test_bounds_shape')
        BoundsMixin.test_bounds_shape.__dict__.__setitem__('stypy_param_names_list', [])
        BoundsMixin.test_bounds_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        BoundsMixin.test_bounds_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BoundsMixin.test_bounds_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        BoundsMixin.test_bounds_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        BoundsMixin.test_bounds_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BoundsMixin.test_bounds_shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BoundsMixin.test_bounds_shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bounds_shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bounds_shape(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 405)
        list_207503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 405)
        # Adding element type (line 405)
        str_207504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 20), 'str', '2-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 19), list_207503, str_207504)
        # Adding element type (line 405)
        str_207505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 31), 'str', '3-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 19), list_207503, str_207505)
        # Adding element type (line 405)
        str_207506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 42), 'str', 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 19), list_207503, str_207506)
        # Adding element type (line 405)
        # Getting the type of 'jac_2d_trivial' (line 405)
        jac_2d_trivial_207507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 48), 'jac_2d_trivial')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 19), list_207503, jac_2d_trivial_207507)
        
        # Testing the type of a for loop iterable (line 405)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 405, 8), list_207503)
        # Getting the type of the for loop variable (line 405)
        for_loop_var_207508 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 405, 8), list_207503)
        # Assigning a type to the variable 'jac' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'jac', for_loop_var_207508)
        # SSA begins for a for statement (line 405)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 406):
        
        # Obtaining an instance of the builtin type 'list' (line 406)
        list_207509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 406)
        # Adding element type (line 406)
        float_207510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 17), list_207509, float_207510)
        # Adding element type (line 406)
        float_207511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 17), list_207509, float_207511)
        
        # Assigning a type to the variable 'x0' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'x0', list_207509)
        
        # Assigning a Call to a Name (line 407):
        
        # Call to least_squares(...): (line 407)
        # Processing the call arguments (line 407)
        # Getting the type of 'fun_2d_trivial' (line 407)
        fun_2d_trivial_207513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 32), 'fun_2d_trivial', False)
        # Getting the type of 'x0' (line 407)
        x0_207514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 48), 'x0', False)
        # Processing the call keyword arguments (line 407)
        # Getting the type of 'jac' (line 407)
        jac_207515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 56), 'jac', False)
        keyword_207516 = jac_207515
        kwargs_207517 = {'jac': keyword_207516}
        # Getting the type of 'least_squares' (line 407)
        least_squares_207512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 18), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 407)
        least_squares_call_result_207518 = invoke(stypy.reporting.localization.Localization(__file__, 407, 18), least_squares_207512, *[fun_2d_trivial_207513, x0_207514], **kwargs_207517)
        
        # Assigning a type to the variable 'res' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'res', least_squares_call_result_207518)
        
        # Call to assert_allclose(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'res' (line 408)
        res_207520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 408)
        x_207521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 28), res_207520, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 408)
        list_207522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 408)
        # Adding element type (line 408)
        float_207523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 35), list_207522, float_207523)
        # Adding element type (line 408)
        float_207524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 35), list_207522, float_207524)
        
        # Processing the call keyword arguments (line 408)
        kwargs_207525 = {}
        # Getting the type of 'assert_allclose' (line 408)
        assert_allclose_207519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 408)
        assert_allclose_call_result_207526 = invoke(stypy.reporting.localization.Localization(__file__, 408, 12), assert_allclose_207519, *[x_207521, list_207522], **kwargs_207525)
        
        
        # Assigning a Call to a Name (line 409):
        
        # Call to least_squares(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'fun_2d_trivial' (line 409)
        fun_2d_trivial_207528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 32), 'fun_2d_trivial', False)
        # Getting the type of 'x0' (line 409)
        x0_207529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 48), 'x0', False)
        # Processing the call keyword arguments (line 409)
        # Getting the type of 'jac' (line 409)
        jac_207530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 56), 'jac', False)
        keyword_207531 = jac_207530
        
        # Obtaining an instance of the builtin type 'tuple' (line 410)
        tuple_207532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 410)
        # Adding element type (line 410)
        float_207533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 40), tuple_207532, float_207533)
        # Adding element type (line 410)
        
        # Obtaining an instance of the builtin type 'list' (line 410)
        list_207534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 410)
        # Adding element type (line 410)
        float_207535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 45), list_207534, float_207535)
        # Adding element type (line 410)
        float_207536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 45), list_207534, float_207536)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 40), tuple_207532, list_207534)
        
        keyword_207537 = tuple_207532
        # Getting the type of 'self' (line 410)
        self_207538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 65), 'self', False)
        # Obtaining the member 'method' of a type (line 410)
        method_207539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 65), self_207538, 'method')
        keyword_207540 = method_207539
        kwargs_207541 = {'method': keyword_207540, 'jac': keyword_207531, 'bounds': keyword_207537}
        # Getting the type of 'least_squares' (line 409)
        least_squares_207527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 18), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 409)
        least_squares_call_result_207542 = invoke(stypy.reporting.localization.Localization(__file__, 409, 18), least_squares_207527, *[fun_2d_trivial_207528, x0_207529], **kwargs_207541)
        
        # Assigning a type to the variable 'res' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'res', least_squares_call_result_207542)
        
        # Call to assert_allclose(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'res' (line 411)
        res_207544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 411)
        x_207545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 28), res_207544, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 411)
        list_207546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 411)
        # Adding element type (line 411)
        float_207547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 35), list_207546, float_207547)
        # Adding element type (line 411)
        float_207548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 35), list_207546, float_207548)
        
        # Processing the call keyword arguments (line 411)
        kwargs_207549 = {}
        # Getting the type of 'assert_allclose' (line 411)
        assert_allclose_207543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 411)
        assert_allclose_call_result_207550 = invoke(stypy.reporting.localization.Localization(__file__, 411, 12), assert_allclose_207543, *[x_207545, list_207546], **kwargs_207549)
        
        
        # Assigning a Call to a Name (line 412):
        
        # Call to least_squares(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'fun_2d_trivial' (line 412)
        fun_2d_trivial_207552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 32), 'fun_2d_trivial', False)
        # Getting the type of 'x0' (line 412)
        x0_207553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 48), 'x0', False)
        # Processing the call keyword arguments (line 412)
        # Getting the type of 'jac' (line 412)
        jac_207554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 56), 'jac', False)
        keyword_207555 = jac_207554
        
        # Obtaining an instance of the builtin type 'tuple' (line 413)
        tuple_207556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 413)
        # Adding element type (line 413)
        
        # Obtaining an instance of the builtin type 'list' (line 413)
        list_207557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 413)
        # Adding element type (line 413)
        float_207558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 40), list_207557, float_207558)
        # Adding element type (line 413)
        float_207559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 40), list_207557, float_207559)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 40), tuple_207556, list_207557)
        # Adding element type (line 413)
        float_207560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 40), tuple_207556, float_207560)
        
        keyword_207561 = tuple_207556
        # Getting the type of 'self' (line 413)
        self_207562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 65), 'self', False)
        # Obtaining the member 'method' of a type (line 413)
        method_207563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 65), self_207562, 'method')
        keyword_207564 = method_207563
        kwargs_207565 = {'method': keyword_207564, 'jac': keyword_207555, 'bounds': keyword_207561}
        # Getting the type of 'least_squares' (line 412)
        least_squares_207551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 18), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 412)
        least_squares_call_result_207566 = invoke(stypy.reporting.localization.Localization(__file__, 412, 18), least_squares_207551, *[fun_2d_trivial_207552, x0_207553], **kwargs_207565)
        
        # Assigning a type to the variable 'res' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'res', least_squares_call_result_207566)
        
        # Call to assert_allclose(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'res' (line 414)
        res_207568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 414)
        x_207569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 28), res_207568, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 414)
        list_207570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 414)
        # Adding element type (line 414)
        float_207571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 35), list_207570, float_207571)
        # Adding element type (line 414)
        float_207572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 35), list_207570, float_207572)
        
        # Processing the call keyword arguments (line 414)
        kwargs_207573 = {}
        # Getting the type of 'assert_allclose' (line 414)
        assert_allclose_207567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 414)
        assert_allclose_call_result_207574 = invoke(stypy.reporting.localization.Localization(__file__, 414, 12), assert_allclose_207567, *[x_207569, list_207570], **kwargs_207573)
        
        
        # Assigning a Call to a Name (line 415):
        
        # Call to least_squares(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'fun_2d_trivial' (line 416)
        fun_2d_trivial_207576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 16), 'fun_2d_trivial', False)
        # Getting the type of 'x0' (line 416)
        x0_207577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 32), 'x0', False)
        # Processing the call keyword arguments (line 415)
        # Getting the type of 'jac' (line 416)
        jac_207578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 40), 'jac', False)
        keyword_207579 = jac_207578
        
        # Obtaining an instance of the builtin type 'tuple' (line 416)
        tuple_207580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 416)
        # Adding element type (line 416)
        
        # Obtaining an instance of the builtin type 'list' (line 416)
        list_207581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 416)
        # Adding element type (line 416)
        int_207582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 53), list_207581, int_207582)
        # Adding element type (line 416)
        float_207583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 53), list_207581, float_207583)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 53), tuple_207580, list_207581)
        # Adding element type (line 416)
        
        # Obtaining an instance of the builtin type 'list' (line 416)
        list_207584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 416)
        # Adding element type (line 416)
        float_207585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 64), list_207584, float_207585)
        # Adding element type (line 416)
        float_207586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 64), list_207584, float_207586)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 53), tuple_207580, list_207584)
        
        keyword_207587 = tuple_207580
        # Getting the type of 'self' (line 417)
        self_207588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 23), 'self', False)
        # Obtaining the member 'method' of a type (line 417)
        method_207589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 23), self_207588, 'method')
        keyword_207590 = method_207589
        kwargs_207591 = {'method': keyword_207590, 'jac': keyword_207579, 'bounds': keyword_207587}
        # Getting the type of 'least_squares' (line 415)
        least_squares_207575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 18), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 415)
        least_squares_call_result_207592 = invoke(stypy.reporting.localization.Localization(__file__, 415, 18), least_squares_207575, *[fun_2d_trivial_207576, x0_207577], **kwargs_207591)
        
        # Assigning a type to the variable 'res' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'res', least_squares_call_result_207592)
        
        # Call to assert_allclose(...): (line 418)
        # Processing the call arguments (line 418)
        # Getting the type of 'res' (line 418)
        res_207594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 418)
        x_207595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 28), res_207594, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 418)
        list_207596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 418)
        # Adding element type (line 418)
        float_207597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 35), list_207596, float_207597)
        # Adding element type (line 418)
        float_207598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 35), list_207596, float_207598)
        
        # Processing the call keyword arguments (line 418)
        float_207599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 52), 'float')
        keyword_207600 = float_207599
        kwargs_207601 = {'atol': keyword_207600}
        # Getting the type of 'assert_allclose' (line 418)
        assert_allclose_207593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 418)
        assert_allclose_call_result_207602 = invoke(stypy.reporting.localization.Localization(__file__, 418, 12), assert_allclose_207593, *[x_207595, list_207596], **kwargs_207601)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_bounds_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bounds_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 404)
        stypy_return_type_207603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207603)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bounds_shape'
        return stypy_return_type_207603


    @norecursion
    def test_rosenbrock_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_rosenbrock_bounds'
        module_type_store = module_type_store.open_function_context('test_rosenbrock_bounds', 420, 4, False)
        # Assigning a type to the variable 'self' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BoundsMixin.test_rosenbrock_bounds.__dict__.__setitem__('stypy_localization', localization)
        BoundsMixin.test_rosenbrock_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BoundsMixin.test_rosenbrock_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        BoundsMixin.test_rosenbrock_bounds.__dict__.__setitem__('stypy_function_name', 'BoundsMixin.test_rosenbrock_bounds')
        BoundsMixin.test_rosenbrock_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        BoundsMixin.test_rosenbrock_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        BoundsMixin.test_rosenbrock_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BoundsMixin.test_rosenbrock_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        BoundsMixin.test_rosenbrock_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        BoundsMixin.test_rosenbrock_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BoundsMixin.test_rosenbrock_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BoundsMixin.test_rosenbrock_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_rosenbrock_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_rosenbrock_bounds(...)' code ##################

        
        # Assigning a Call to a Name (line 421):
        
        # Call to array(...): (line 421)
        # Processing the call arguments (line 421)
        
        # Obtaining an instance of the builtin type 'list' (line 421)
        list_207606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 421)
        # Adding element type (line 421)
        float_207607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 24), list_207606, float_207607)
        # Adding element type (line 421)
        float_207608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 24), list_207606, float_207608)
        
        # Processing the call keyword arguments (line 421)
        kwargs_207609 = {}
        # Getting the type of 'np' (line 421)
        np_207604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 421)
        array_207605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 15), np_207604, 'array')
        # Calling array(args, kwargs) (line 421)
        array_call_result_207610 = invoke(stypy.reporting.localization.Localization(__file__, 421, 15), array_207605, *[list_207606], **kwargs_207609)
        
        # Assigning a type to the variable 'x0_1' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'x0_1', array_call_result_207610)
        
        # Assigning a Call to a Name (line 422):
        
        # Call to array(...): (line 422)
        # Processing the call arguments (line 422)
        
        # Obtaining an instance of the builtin type 'list' (line 422)
        list_207613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 422)
        # Adding element type (line 422)
        float_207614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 24), list_207613, float_207614)
        # Adding element type (line 422)
        float_207615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 24), list_207613, float_207615)
        
        # Processing the call keyword arguments (line 422)
        kwargs_207616 = {}
        # Getting the type of 'np' (line 422)
        np_207611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 422)
        array_207612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 15), np_207611, 'array')
        # Calling array(args, kwargs) (line 422)
        array_call_result_207617 = invoke(stypy.reporting.localization.Localization(__file__, 422, 15), array_207612, *[list_207613], **kwargs_207616)
        
        # Assigning a type to the variable 'x0_2' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'x0_2', array_call_result_207617)
        
        # Assigning a Call to a Name (line 423):
        
        # Call to array(...): (line 423)
        # Processing the call arguments (line 423)
        
        # Obtaining an instance of the builtin type 'list' (line 423)
        list_207620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 423)
        # Adding element type (line 423)
        float_207621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 24), list_207620, float_207621)
        # Adding element type (line 423)
        float_207622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 24), list_207620, float_207622)
        
        # Processing the call keyword arguments (line 423)
        kwargs_207623 = {}
        # Getting the type of 'np' (line 423)
        np_207618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 423)
        array_207619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 15), np_207618, 'array')
        # Calling array(args, kwargs) (line 423)
        array_call_result_207624 = invoke(stypy.reporting.localization.Localization(__file__, 423, 15), array_207619, *[list_207620], **kwargs_207623)
        
        # Assigning a type to the variable 'x0_3' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'x0_3', array_call_result_207624)
        
        # Assigning a Call to a Name (line 424):
        
        # Call to array(...): (line 424)
        # Processing the call arguments (line 424)
        
        # Obtaining an instance of the builtin type 'list' (line 424)
        list_207627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 424)
        # Adding element type (line 424)
        float_207628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 24), list_207627, float_207628)
        # Adding element type (line 424)
        float_207629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 24), list_207627, float_207629)
        
        # Processing the call keyword arguments (line 424)
        kwargs_207630 = {}
        # Getting the type of 'np' (line 424)
        np_207625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 424)
        array_207626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 15), np_207625, 'array')
        # Calling array(args, kwargs) (line 424)
        array_call_result_207631 = invoke(stypy.reporting.localization.Localization(__file__, 424, 15), array_207626, *[list_207627], **kwargs_207630)
        
        # Assigning a type to the variable 'x0_4' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'x0_4', array_call_result_207631)
        
        # Assigning a Call to a Name (line 425):
        
        # Call to array(...): (line 425)
        # Processing the call arguments (line 425)
        
        # Obtaining an instance of the builtin type 'list' (line 425)
        list_207634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 425)
        # Adding element type (line 425)
        float_207635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 24), list_207634, float_207635)
        # Adding element type (line 425)
        float_207636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 24), list_207634, float_207636)
        
        # Processing the call keyword arguments (line 425)
        kwargs_207637 = {}
        # Getting the type of 'np' (line 425)
        np_207632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 425)
        array_207633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 15), np_207632, 'array')
        # Calling array(args, kwargs) (line 425)
        array_call_result_207638 = invoke(stypy.reporting.localization.Localization(__file__, 425, 15), array_207633, *[list_207634], **kwargs_207637)
        
        # Assigning a type to the variable 'x0_5' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'x0_5', array_call_result_207638)
        
        # Assigning a List to a Name (line 426):
        
        # Obtaining an instance of the builtin type 'list' (line 426)
        list_207639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 426)
        # Adding element type (line 426)
        
        # Obtaining an instance of the builtin type 'tuple' (line 427)
        tuple_207640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 427)
        # Adding element type (line 427)
        # Getting the type of 'x0_1' (line 427)
        x0_1_207641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 13), 'x0_1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 13), tuple_207640, x0_1_207641)
        # Adding element type (line 427)
        
        # Obtaining an instance of the builtin type 'tuple' (line 427)
        tuple_207642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 427)
        # Adding element type (line 427)
        
        # Obtaining an instance of the builtin type 'list' (line 427)
        list_207643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 427)
        # Adding element type (line 427)
        
        # Getting the type of 'np' (line 427)
        np_207644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 22), 'np')
        # Obtaining the member 'inf' of a type (line 427)
        inf_207645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 22), np_207644, 'inf')
        # Applying the 'usub' unary operator (line 427)
        result___neg___207646 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 21), 'usub', inf_207645)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 20), list_207643, result___neg___207646)
        # Adding element type (line 427)
        float_207647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 20), list_207643, float_207647)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 20), tuple_207642, list_207643)
        # Adding element type (line 427)
        # Getting the type of 'np' (line 427)
        np_207648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 37), 'np')
        # Obtaining the member 'inf' of a type (line 427)
        inf_207649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 37), np_207648, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 20), tuple_207642, inf_207649)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 13), tuple_207640, tuple_207642)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 19), list_207639, tuple_207640)
        # Adding element type (line 426)
        
        # Obtaining an instance of the builtin type 'tuple' (line 428)
        tuple_207650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 428)
        # Adding element type (line 428)
        # Getting the type of 'x0_2' (line 428)
        x0_2_207651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 13), 'x0_2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 13), tuple_207650, x0_2_207651)
        # Adding element type (line 428)
        
        # Obtaining an instance of the builtin type 'tuple' (line 428)
        tuple_207652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 428)
        # Adding element type (line 428)
        
        # Obtaining an instance of the builtin type 'list' (line 428)
        list_207653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 428)
        # Adding element type (line 428)
        
        # Getting the type of 'np' (line 428)
        np_207654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 22), 'np')
        # Obtaining the member 'inf' of a type (line 428)
        inf_207655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 22), np_207654, 'inf')
        # Applying the 'usub' unary operator (line 428)
        result___neg___207656 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 21), 'usub', inf_207655)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 20), list_207653, result___neg___207656)
        # Adding element type (line 428)
        float_207657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 20), list_207653, float_207657)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 20), tuple_207652, list_207653)
        # Adding element type (line 428)
        # Getting the type of 'np' (line 428)
        np_207658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 36), 'np')
        # Obtaining the member 'inf' of a type (line 428)
        inf_207659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 36), np_207658, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 20), tuple_207652, inf_207659)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 13), tuple_207650, tuple_207652)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 19), list_207639, tuple_207650)
        # Adding element type (line 426)
        
        # Obtaining an instance of the builtin type 'tuple' (line 429)
        tuple_207660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 429)
        # Adding element type (line 429)
        # Getting the type of 'x0_3' (line 429)
        x0_3_207661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 13), 'x0_3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 13), tuple_207660, x0_3_207661)
        # Adding element type (line 429)
        
        # Obtaining an instance of the builtin type 'tuple' (line 429)
        tuple_207662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 429)
        # Adding element type (line 429)
        
        # Obtaining an instance of the builtin type 'list' (line 429)
        list_207663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 429)
        # Adding element type (line 429)
        
        # Getting the type of 'np' (line 429)
        np_207664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 22), 'np')
        # Obtaining the member 'inf' of a type (line 429)
        inf_207665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 22), np_207664, 'inf')
        # Applying the 'usub' unary operator (line 429)
        result___neg___207666 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 21), 'usub', inf_207665)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 20), list_207663, result___neg___207666)
        # Adding element type (line 429)
        float_207667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 20), list_207663, float_207667)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 20), tuple_207662, list_207663)
        # Adding element type (line 429)
        # Getting the type of 'np' (line 429)
        np_207668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 36), 'np')
        # Obtaining the member 'inf' of a type (line 429)
        inf_207669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 36), np_207668, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 20), tuple_207662, inf_207669)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 13), tuple_207660, tuple_207662)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 19), list_207639, tuple_207660)
        # Adding element type (line 426)
        
        # Obtaining an instance of the builtin type 'tuple' (line 430)
        tuple_207670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 430)
        # Adding element type (line 430)
        # Getting the type of 'x0_4' (line 430)
        x0_4_207671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 13), 'x0_4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 13), tuple_207670, x0_4_207671)
        # Adding element type (line 430)
        
        # Obtaining an instance of the builtin type 'tuple' (line 430)
        tuple_207672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 430)
        # Adding element type (line 430)
        
        # Obtaining an instance of the builtin type 'list' (line 430)
        list_207673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 430)
        # Adding element type (line 430)
        
        # Getting the type of 'np' (line 430)
        np_207674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 22), 'np')
        # Obtaining the member 'inf' of a type (line 430)
        inf_207675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 22), np_207674, 'inf')
        # Applying the 'usub' unary operator (line 430)
        result___neg___207676 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 21), 'usub', inf_207675)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 20), list_207673, result___neg___207676)
        # Adding element type (line 430)
        float_207677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 20), list_207673, float_207677)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 20), tuple_207672, list_207673)
        # Adding element type (line 430)
        
        # Obtaining an instance of the builtin type 'list' (line 430)
        list_207678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 430)
        # Adding element type (line 430)
        float_207679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 36), list_207678, float_207679)
        # Adding element type (line 430)
        # Getting the type of 'np' (line 430)
        np_207680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 42), 'np')
        # Obtaining the member 'inf' of a type (line 430)
        inf_207681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 42), np_207680, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 36), list_207678, inf_207681)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 20), tuple_207672, list_207678)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 13), tuple_207670, tuple_207672)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 19), list_207639, tuple_207670)
        # Adding element type (line 426)
        
        # Obtaining an instance of the builtin type 'tuple' (line 431)
        tuple_207682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 431)
        # Adding element type (line 431)
        # Getting the type of 'x0_2' (line 431)
        x0_2_207683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 13), 'x0_2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 13), tuple_207682, x0_2_207683)
        # Adding element type (line 431)
        
        # Obtaining an instance of the builtin type 'tuple' (line 431)
        tuple_207684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 431)
        # Adding element type (line 431)
        
        # Obtaining an instance of the builtin type 'list' (line 431)
        list_207685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 431)
        # Adding element type (line 431)
        float_207686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 20), list_207685, float_207686)
        # Adding element type (line 431)
        float_207687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 20), list_207685, float_207687)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 20), tuple_207684, list_207685)
        # Adding element type (line 431)
        
        # Obtaining an instance of the builtin type 'list' (line 431)
        list_207688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 431)
        # Adding element type (line 431)
        float_207689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 32), list_207688, float_207689)
        # Adding element type (line 431)
        float_207690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 32), list_207688, float_207690)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 20), tuple_207684, list_207688)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 13), tuple_207682, tuple_207684)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 19), list_207639, tuple_207682)
        # Adding element type (line 426)
        
        # Obtaining an instance of the builtin type 'tuple' (line 432)
        tuple_207691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 432)
        # Adding element type (line 432)
        # Getting the type of 'x0_5' (line 432)
        x0_5_207692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 13), 'x0_5')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 13), tuple_207691, x0_5_207692)
        # Adding element type (line 432)
        
        # Obtaining an instance of the builtin type 'tuple' (line 432)
        tuple_207693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 432)
        # Adding element type (line 432)
        
        # Obtaining an instance of the builtin type 'list' (line 432)
        list_207694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 432)
        # Adding element type (line 432)
        float_207695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 20), list_207694, float_207695)
        # Adding element type (line 432)
        float_207696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 20), list_207694, float_207696)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 20), tuple_207693, list_207694)
        # Adding element type (line 432)
        
        # Obtaining an instance of the builtin type 'list' (line 432)
        list_207697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 432)
        # Adding element type (line 432)
        float_207698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 34), list_207697, float_207698)
        # Adding element type (line 432)
        int_207699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 34), list_207697, int_207699)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 20), tuple_207693, list_207697)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 13), tuple_207691, tuple_207693)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 19), list_207639, tuple_207691)
        
        # Assigning a type to the variable 'problems' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'problems', list_207639)
        
        # Getting the type of 'problems' (line 434)
        problems_207700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 26), 'problems')
        # Testing the type of a for loop iterable (line 434)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 434, 8), problems_207700)
        # Getting the type of the for loop variable (line 434)
        for_loop_var_207701 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 434, 8), problems_207700)
        # Assigning a type to the variable 'x0' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'x0', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 8), for_loop_var_207701))
        # Assigning a type to the variable 'bounds' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'bounds', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 8), for_loop_var_207701))
        # SSA begins for a for statement (line 434)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to product(...): (line 435)
        # Processing the call arguments (line 435)
        
        # Obtaining an instance of the builtin type 'list' (line 436)
        list_207703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 436)
        # Adding element type (line 436)
        str_207704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 21), 'str', '2-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 20), list_207703, str_207704)
        # Adding element type (line 436)
        str_207705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 32), 'str', '3-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 20), list_207703, str_207705)
        # Adding element type (line 436)
        str_207706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 43), 'str', 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 20), list_207703, str_207706)
        # Adding element type (line 436)
        # Getting the type of 'jac_rosenbrock' (line 436)
        jac_rosenbrock_207707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 49), 'jac_rosenbrock', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 20), list_207703, jac_rosenbrock_207707)
        
        
        # Obtaining an instance of the builtin type 'list' (line 437)
        list_207708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 437)
        # Adding element type (line 437)
        float_207709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 20), list_207708, float_207709)
        # Adding element type (line 437)
        
        # Obtaining an instance of the builtin type 'list' (line 437)
        list_207710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 437)
        # Adding element type (line 437)
        float_207711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 26), list_207710, float_207711)
        # Adding element type (line 437)
        float_207712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 26), list_207710, float_207712)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 20), list_207708, list_207710)
        # Adding element type (line 437)
        str_207713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 38), 'str', 'jac')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 20), list_207708, str_207713)
        
        
        # Obtaining an instance of the builtin type 'list' (line 438)
        list_207714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 438)
        # Adding element type (line 438)
        str_207715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 21), 'str', 'exact')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 20), list_207714, str_207715)
        # Adding element type (line 438)
        str_207716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 30), 'str', 'lsmr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 20), list_207714, str_207716)
        
        # Processing the call keyword arguments (line 435)
        kwargs_207717 = {}
        # Getting the type of 'product' (line 435)
        product_207702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 43), 'product', False)
        # Calling product(args, kwargs) (line 435)
        product_call_result_207718 = invoke(stypy.reporting.localization.Localization(__file__, 435, 43), product_207702, *[list_207703, list_207708, list_207714], **kwargs_207717)
        
        # Testing the type of a for loop iterable (line 435)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 435, 12), product_call_result_207718)
        # Getting the type of the for loop variable (line 435)
        for_loop_var_207719 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 435, 12), product_call_result_207718)
        # Assigning a type to the variable 'jac' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'jac', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 12), for_loop_var_207719))
        # Assigning a type to the variable 'x_scale' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'x_scale', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 12), for_loop_var_207719))
        # Assigning a type to the variable 'tr_solver' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'tr_solver', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 12), for_loop_var_207719))
        # SSA begins for a for statement (line 435)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 439):
        
        # Call to least_squares(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'fun_rosenbrock' (line 439)
        fun_rosenbrock_207721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 36), 'fun_rosenbrock', False)
        # Getting the type of 'x0' (line 439)
        x0_207722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 52), 'x0', False)
        # Getting the type of 'jac' (line 439)
        jac_207723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 56), 'jac', False)
        # Getting the type of 'bounds' (line 439)
        bounds_207724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 61), 'bounds', False)
        # Processing the call keyword arguments (line 439)
        # Getting the type of 'x_scale' (line 440)
        x_scale_207725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 44), 'x_scale', False)
        keyword_207726 = x_scale_207725
        # Getting the type of 'tr_solver' (line 440)
        tr_solver_207727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 63), 'tr_solver', False)
        keyword_207728 = tr_solver_207727
        # Getting the type of 'self' (line 441)
        self_207729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 43), 'self', False)
        # Obtaining the member 'method' of a type (line 441)
        method_207730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 43), self_207729, 'method')
        keyword_207731 = method_207730
        kwargs_207732 = {'tr_solver': keyword_207728, 'x_scale': keyword_207726, 'method': keyword_207731}
        # Getting the type of 'least_squares' (line 439)
        least_squares_207720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 22), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 439)
        least_squares_call_result_207733 = invoke(stypy.reporting.localization.Localization(__file__, 439, 22), least_squares_207720, *[fun_rosenbrock_207721, x0_207722, jac_207723, bounds_207724], **kwargs_207732)
        
        # Assigning a type to the variable 'res' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'res', least_squares_call_result_207733)
        
        # Call to assert_allclose(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 'res' (line 442)
        res_207735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 32), 'res', False)
        # Obtaining the member 'optimality' of a type (line 442)
        optimality_207736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 32), res_207735, 'optimality')
        float_207737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 48), 'float')
        # Processing the call keyword arguments (line 442)
        float_207738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 58), 'float')
        keyword_207739 = float_207738
        kwargs_207740 = {'atol': keyword_207739}
        # Getting the type of 'assert_allclose' (line 442)
        assert_allclose_207734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 442)
        assert_allclose_call_result_207741 = invoke(stypy.reporting.localization.Localization(__file__, 442, 16), assert_allclose_207734, *[optimality_207736, float_207737], **kwargs_207740)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_rosenbrock_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_rosenbrock_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 420)
        stypy_return_type_207742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207742)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_rosenbrock_bounds'
        return stypy_return_type_207742


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 371, 0, False)
        # Assigning a type to the variable 'self' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BoundsMixin.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BoundsMixin' (line 371)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), 'BoundsMixin', BoundsMixin)
# Declaration of the 'SparseMixin' class

class SparseMixin(object, ):

    @norecursion
    def test_exact_tr_solver(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_exact_tr_solver'
        module_type_store = module_type_store.open_function_context('test_exact_tr_solver', 446, 4, False)
        # Assigning a type to the variable 'self' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SparseMixin.test_exact_tr_solver.__dict__.__setitem__('stypy_localization', localization)
        SparseMixin.test_exact_tr_solver.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SparseMixin.test_exact_tr_solver.__dict__.__setitem__('stypy_type_store', module_type_store)
        SparseMixin.test_exact_tr_solver.__dict__.__setitem__('stypy_function_name', 'SparseMixin.test_exact_tr_solver')
        SparseMixin.test_exact_tr_solver.__dict__.__setitem__('stypy_param_names_list', [])
        SparseMixin.test_exact_tr_solver.__dict__.__setitem__('stypy_varargs_param_name', None)
        SparseMixin.test_exact_tr_solver.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SparseMixin.test_exact_tr_solver.__dict__.__setitem__('stypy_call_defaults', defaults)
        SparseMixin.test_exact_tr_solver.__dict__.__setitem__('stypy_call_varargs', varargs)
        SparseMixin.test_exact_tr_solver.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SparseMixin.test_exact_tr_solver.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.test_exact_tr_solver', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_exact_tr_solver', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_exact_tr_solver(...)' code ##################

        
        # Assigning a Call to a Name (line 447):
        
        # Call to BroydenTridiagonal(...): (line 447)
        # Processing the call keyword arguments (line 447)
        kwargs_207744 = {}
        # Getting the type of 'BroydenTridiagonal' (line 447)
        BroydenTridiagonal_207743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 447)
        BroydenTridiagonal_call_result_207745 = invoke(stypy.reporting.localization.Localization(__file__, 447, 12), BroydenTridiagonal_207743, *[], **kwargs_207744)
        
        # Assigning a type to the variable 'p' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'p', BroydenTridiagonal_call_result_207745)
        
        # Call to assert_raises(...): (line 448)
        # Processing the call arguments (line 448)
        # Getting the type of 'ValueError' (line 448)
        ValueError_207747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 448)
        least_squares_207748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 34), 'least_squares', False)
        # Getting the type of 'p' (line 448)
        p_207749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 49), 'p', False)
        # Obtaining the member 'fun' of a type (line 448)
        fun_207750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 49), p_207749, 'fun')
        # Getting the type of 'p' (line 448)
        p_207751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 56), 'p', False)
        # Obtaining the member 'x0' of a type (line 448)
        x0_207752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 56), p_207751, 'x0')
        # Getting the type of 'p' (line 448)
        p_207753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 62), 'p', False)
        # Obtaining the member 'jac' of a type (line 448)
        jac_207754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 62), p_207753, 'jac')
        # Processing the call keyword arguments (line 448)
        str_207755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 32), 'str', 'exact')
        keyword_207756 = str_207755
        # Getting the type of 'self' (line 449)
        self_207757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 48), 'self', False)
        # Obtaining the member 'method' of a type (line 449)
        method_207758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 48), self_207757, 'method')
        keyword_207759 = method_207758
        kwargs_207760 = {'tr_solver': keyword_207756, 'method': keyword_207759}
        # Getting the type of 'assert_raises' (line 448)
        assert_raises_207746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 448)
        assert_raises_call_result_207761 = invoke(stypy.reporting.localization.Localization(__file__, 448, 8), assert_raises_207746, *[ValueError_207747, least_squares_207748, fun_207750, x0_207752, jac_207754], **kwargs_207760)
        
        
        # Call to assert_raises(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'ValueError' (line 450)
        ValueError_207763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 450)
        least_squares_207764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 34), 'least_squares', False)
        # Getting the type of 'p' (line 450)
        p_207765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 49), 'p', False)
        # Obtaining the member 'fun' of a type (line 450)
        fun_207766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 49), p_207765, 'fun')
        # Getting the type of 'p' (line 450)
        p_207767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 56), 'p', False)
        # Obtaining the member 'x0' of a type (line 450)
        x0_207768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 56), p_207767, 'x0')
        # Processing the call keyword arguments (line 450)
        str_207769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 32), 'str', 'exact')
        keyword_207770 = str_207769
        # Getting the type of 'p' (line 451)
        p_207771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 54), 'p', False)
        # Obtaining the member 'sparsity' of a type (line 451)
        sparsity_207772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 54), p_207771, 'sparsity')
        keyword_207773 = sparsity_207772
        # Getting the type of 'self' (line 452)
        self_207774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 29), 'self', False)
        # Obtaining the member 'method' of a type (line 452)
        method_207775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 29), self_207774, 'method')
        keyword_207776 = method_207775
        kwargs_207777 = {'tr_solver': keyword_207770, 'jac_sparsity': keyword_207773, 'method': keyword_207776}
        # Getting the type of 'assert_raises' (line 450)
        assert_raises_207762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 450)
        assert_raises_call_result_207778 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), assert_raises_207762, *[ValueError_207763, least_squares_207764, fun_207766, x0_207768], **kwargs_207777)
        
        
        # ################# End of 'test_exact_tr_solver(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_exact_tr_solver' in the type store
        # Getting the type of 'stypy_return_type' (line 446)
        stypy_return_type_207779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207779)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_exact_tr_solver'
        return stypy_return_type_207779


    @norecursion
    def test_equivalence(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_equivalence'
        module_type_store = module_type_store.open_function_context('test_equivalence', 454, 4, False)
        # Assigning a type to the variable 'self' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SparseMixin.test_equivalence.__dict__.__setitem__('stypy_localization', localization)
        SparseMixin.test_equivalence.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SparseMixin.test_equivalence.__dict__.__setitem__('stypy_type_store', module_type_store)
        SparseMixin.test_equivalence.__dict__.__setitem__('stypy_function_name', 'SparseMixin.test_equivalence')
        SparseMixin.test_equivalence.__dict__.__setitem__('stypy_param_names_list', [])
        SparseMixin.test_equivalence.__dict__.__setitem__('stypy_varargs_param_name', None)
        SparseMixin.test_equivalence.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SparseMixin.test_equivalence.__dict__.__setitem__('stypy_call_defaults', defaults)
        SparseMixin.test_equivalence.__dict__.__setitem__('stypy_call_varargs', varargs)
        SparseMixin.test_equivalence.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SparseMixin.test_equivalence.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.test_equivalence', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_equivalence', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_equivalence(...)' code ##################

        
        # Assigning a Call to a Name (line 455):
        
        # Call to BroydenTridiagonal(...): (line 455)
        # Processing the call keyword arguments (line 455)
        str_207781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 41), 'str', 'sparse')
        keyword_207782 = str_207781
        kwargs_207783 = {'mode': keyword_207782}
        # Getting the type of 'BroydenTridiagonal' (line 455)
        BroydenTridiagonal_207780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 17), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 455)
        BroydenTridiagonal_call_result_207784 = invoke(stypy.reporting.localization.Localization(__file__, 455, 17), BroydenTridiagonal_207780, *[], **kwargs_207783)
        
        # Assigning a type to the variable 'sparse' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'sparse', BroydenTridiagonal_call_result_207784)
        
        # Assigning a Call to a Name (line 456):
        
        # Call to BroydenTridiagonal(...): (line 456)
        # Processing the call keyword arguments (line 456)
        str_207786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 40), 'str', 'dense')
        keyword_207787 = str_207786
        kwargs_207788 = {'mode': keyword_207787}
        # Getting the type of 'BroydenTridiagonal' (line 456)
        BroydenTridiagonal_207785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 456)
        BroydenTridiagonal_call_result_207789 = invoke(stypy.reporting.localization.Localization(__file__, 456, 16), BroydenTridiagonal_207785, *[], **kwargs_207788)
        
        # Assigning a type to the variable 'dense' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'dense', BroydenTridiagonal_call_result_207789)
        
        # Assigning a Call to a Name (line 457):
        
        # Call to least_squares(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'sparse' (line 458)
        sparse_207791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'sparse', False)
        # Obtaining the member 'fun' of a type (line 458)
        fun_207792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 12), sparse_207791, 'fun')
        # Getting the type of 'sparse' (line 458)
        sparse_207793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 24), 'sparse', False)
        # Obtaining the member 'x0' of a type (line 458)
        x0_207794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 24), sparse_207793, 'x0')
        # Processing the call keyword arguments (line 457)
        # Getting the type of 'sparse' (line 458)
        sparse_207795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 39), 'sparse', False)
        # Obtaining the member 'jac' of a type (line 458)
        jac_207796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 39), sparse_207795, 'jac')
        keyword_207797 = jac_207796
        # Getting the type of 'self' (line 459)
        self_207798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 19), 'self', False)
        # Obtaining the member 'method' of a type (line 459)
        method_207799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 19), self_207798, 'method')
        keyword_207800 = method_207799
        kwargs_207801 = {'jac': keyword_207797, 'method': keyword_207800}
        # Getting the type of 'least_squares' (line 457)
        least_squares_207790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 21), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 457)
        least_squares_call_result_207802 = invoke(stypy.reporting.localization.Localization(__file__, 457, 21), least_squares_207790, *[fun_207792, x0_207794], **kwargs_207801)
        
        # Assigning a type to the variable 'res_sparse' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'res_sparse', least_squares_call_result_207802)
        
        # Assigning a Call to a Name (line 460):
        
        # Call to least_squares(...): (line 460)
        # Processing the call arguments (line 460)
        # Getting the type of 'dense' (line 461)
        dense_207804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'dense', False)
        # Obtaining the member 'fun' of a type (line 461)
        fun_207805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 12), dense_207804, 'fun')
        # Getting the type of 'dense' (line 461)
        dense_207806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 23), 'dense', False)
        # Obtaining the member 'x0' of a type (line 461)
        x0_207807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 23), dense_207806, 'x0')
        # Processing the call keyword arguments (line 460)
        # Getting the type of 'sparse' (line 461)
        sparse_207808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 37), 'sparse', False)
        # Obtaining the member 'jac' of a type (line 461)
        jac_207809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 37), sparse_207808, 'jac')
        keyword_207810 = jac_207809
        # Getting the type of 'self' (line 462)
        self_207811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 19), 'self', False)
        # Obtaining the member 'method' of a type (line 462)
        method_207812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 19), self_207811, 'method')
        keyword_207813 = method_207812
        kwargs_207814 = {'jac': keyword_207810, 'method': keyword_207813}
        # Getting the type of 'least_squares' (line 460)
        least_squares_207803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 20), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 460)
        least_squares_call_result_207815 = invoke(stypy.reporting.localization.Localization(__file__, 460, 20), least_squares_207803, *[fun_207805, x0_207807], **kwargs_207814)
        
        # Assigning a type to the variable 'res_dense' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'res_dense', least_squares_call_result_207815)
        
        # Call to assert_equal(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'res_sparse' (line 463)
        res_sparse_207817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 21), 'res_sparse', False)
        # Obtaining the member 'nfev' of a type (line 463)
        nfev_207818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 21), res_sparse_207817, 'nfev')
        # Getting the type of 'res_dense' (line 463)
        res_dense_207819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 38), 'res_dense', False)
        # Obtaining the member 'nfev' of a type (line 463)
        nfev_207820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 38), res_dense_207819, 'nfev')
        # Processing the call keyword arguments (line 463)
        kwargs_207821 = {}
        # Getting the type of 'assert_equal' (line 463)
        assert_equal_207816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 463)
        assert_equal_call_result_207822 = invoke(stypy.reporting.localization.Localization(__file__, 463, 8), assert_equal_207816, *[nfev_207818, nfev_207820], **kwargs_207821)
        
        
        # Call to assert_allclose(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'res_sparse' (line 464)
        res_sparse_207824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 24), 'res_sparse', False)
        # Obtaining the member 'x' of a type (line 464)
        x_207825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 24), res_sparse_207824, 'x')
        # Getting the type of 'res_dense' (line 464)
        res_dense_207826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 38), 'res_dense', False)
        # Obtaining the member 'x' of a type (line 464)
        x_207827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 38), res_dense_207826, 'x')
        # Processing the call keyword arguments (line 464)
        float_207828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 56), 'float')
        keyword_207829 = float_207828
        kwargs_207830 = {'atol': keyword_207829}
        # Getting the type of 'assert_allclose' (line 464)
        assert_allclose_207823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 464)
        assert_allclose_call_result_207831 = invoke(stypy.reporting.localization.Localization(__file__, 464, 8), assert_allclose_207823, *[x_207825, x_207827], **kwargs_207830)
        
        
        # Call to assert_allclose(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'res_sparse' (line 465)
        res_sparse_207833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 24), 'res_sparse', False)
        # Obtaining the member 'cost' of a type (line 465)
        cost_207834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 24), res_sparse_207833, 'cost')
        int_207835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 41), 'int')
        # Processing the call keyword arguments (line 465)
        float_207836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 49), 'float')
        keyword_207837 = float_207836
        kwargs_207838 = {'atol': keyword_207837}
        # Getting the type of 'assert_allclose' (line 465)
        assert_allclose_207832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 465)
        assert_allclose_call_result_207839 = invoke(stypy.reporting.localization.Localization(__file__, 465, 8), assert_allclose_207832, *[cost_207834, int_207835], **kwargs_207838)
        
        
        # Call to assert_allclose(...): (line 466)
        # Processing the call arguments (line 466)
        # Getting the type of 'res_dense' (line 466)
        res_dense_207841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 24), 'res_dense', False)
        # Obtaining the member 'cost' of a type (line 466)
        cost_207842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 24), res_dense_207841, 'cost')
        int_207843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 40), 'int')
        # Processing the call keyword arguments (line 466)
        float_207844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 48), 'float')
        keyword_207845 = float_207844
        kwargs_207846 = {'atol': keyword_207845}
        # Getting the type of 'assert_allclose' (line 466)
        assert_allclose_207840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 466)
        assert_allclose_call_result_207847 = invoke(stypy.reporting.localization.Localization(__file__, 466, 8), assert_allclose_207840, *[cost_207842, int_207843], **kwargs_207846)
        
        
        # ################# End of 'test_equivalence(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_equivalence' in the type store
        # Getting the type of 'stypy_return_type' (line 454)
        stypy_return_type_207848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207848)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_equivalence'
        return stypy_return_type_207848


    @norecursion
    def test_tr_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tr_options'
        module_type_store = module_type_store.open_function_context('test_tr_options', 468, 4, False)
        # Assigning a type to the variable 'self' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SparseMixin.test_tr_options.__dict__.__setitem__('stypy_localization', localization)
        SparseMixin.test_tr_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SparseMixin.test_tr_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        SparseMixin.test_tr_options.__dict__.__setitem__('stypy_function_name', 'SparseMixin.test_tr_options')
        SparseMixin.test_tr_options.__dict__.__setitem__('stypy_param_names_list', [])
        SparseMixin.test_tr_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        SparseMixin.test_tr_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SparseMixin.test_tr_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        SparseMixin.test_tr_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        SparseMixin.test_tr_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SparseMixin.test_tr_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.test_tr_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tr_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tr_options(...)' code ##################

        
        # Assigning a Call to a Name (line 469):
        
        # Call to BroydenTridiagonal(...): (line 469)
        # Processing the call keyword arguments (line 469)
        kwargs_207850 = {}
        # Getting the type of 'BroydenTridiagonal' (line 469)
        BroydenTridiagonal_207849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 469)
        BroydenTridiagonal_call_result_207851 = invoke(stypy.reporting.localization.Localization(__file__, 469, 12), BroydenTridiagonal_207849, *[], **kwargs_207850)
        
        # Assigning a type to the variable 'p' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'p', BroydenTridiagonal_call_result_207851)
        
        # Assigning a Call to a Name (line 470):
        
        # Call to least_squares(...): (line 470)
        # Processing the call arguments (line 470)
        # Getting the type of 'p' (line 470)
        p_207853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 28), 'p', False)
        # Obtaining the member 'fun' of a type (line 470)
        fun_207854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 28), p_207853, 'fun')
        # Getting the type of 'p' (line 470)
        p_207855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 35), 'p', False)
        # Obtaining the member 'x0' of a type (line 470)
        x0_207856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 35), p_207855, 'x0')
        # Getting the type of 'p' (line 470)
        p_207857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 41), 'p', False)
        # Obtaining the member 'jac' of a type (line 470)
        jac_207858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 41), p_207857, 'jac')
        # Processing the call keyword arguments (line 470)
        # Getting the type of 'self' (line 470)
        self_207859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 55), 'self', False)
        # Obtaining the member 'method' of a type (line 470)
        method_207860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 55), self_207859, 'method')
        keyword_207861 = method_207860
        
        # Obtaining an instance of the builtin type 'dict' (line 471)
        dict_207862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 39), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 471)
        # Adding element type (key, value) (line 471)
        str_207863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 40), 'str', 'btol')
        float_207864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 48), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 39), dict_207862, (str_207863, float_207864))
        
        keyword_207865 = dict_207862
        kwargs_207866 = {'tr_options': keyword_207865, 'method': keyword_207861}
        # Getting the type of 'least_squares' (line 470)
        least_squares_207852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 470)
        least_squares_call_result_207867 = invoke(stypy.reporting.localization.Localization(__file__, 470, 14), least_squares_207852, *[fun_207854, x0_207856, jac_207858], **kwargs_207866)
        
        # Assigning a type to the variable 'res' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'res', least_squares_call_result_207867)
        
        # Call to assert_allclose(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'res' (line 472)
        res_207869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 24), 'res', False)
        # Obtaining the member 'cost' of a type (line 472)
        cost_207870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 24), res_207869, 'cost')
        int_207871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 34), 'int')
        # Processing the call keyword arguments (line 472)
        float_207872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 42), 'float')
        keyword_207873 = float_207872
        kwargs_207874 = {'atol': keyword_207873}
        # Getting the type of 'assert_allclose' (line 472)
        assert_allclose_207868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 472)
        assert_allclose_call_result_207875 = invoke(stypy.reporting.localization.Localization(__file__, 472, 8), assert_allclose_207868, *[cost_207870, int_207871], **kwargs_207874)
        
        
        # ################# End of 'test_tr_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tr_options' in the type store
        # Getting the type of 'stypy_return_type' (line 468)
        stypy_return_type_207876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207876)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tr_options'
        return stypy_return_type_207876


    @norecursion
    def test_wrong_parameters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_wrong_parameters'
        module_type_store = module_type_store.open_function_context('test_wrong_parameters', 474, 4, False)
        # Assigning a type to the variable 'self' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SparseMixin.test_wrong_parameters.__dict__.__setitem__('stypy_localization', localization)
        SparseMixin.test_wrong_parameters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SparseMixin.test_wrong_parameters.__dict__.__setitem__('stypy_type_store', module_type_store)
        SparseMixin.test_wrong_parameters.__dict__.__setitem__('stypy_function_name', 'SparseMixin.test_wrong_parameters')
        SparseMixin.test_wrong_parameters.__dict__.__setitem__('stypy_param_names_list', [])
        SparseMixin.test_wrong_parameters.__dict__.__setitem__('stypy_varargs_param_name', None)
        SparseMixin.test_wrong_parameters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SparseMixin.test_wrong_parameters.__dict__.__setitem__('stypy_call_defaults', defaults)
        SparseMixin.test_wrong_parameters.__dict__.__setitem__('stypy_call_varargs', varargs)
        SparseMixin.test_wrong_parameters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SparseMixin.test_wrong_parameters.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.test_wrong_parameters', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_wrong_parameters', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_wrong_parameters(...)' code ##################

        
        # Assigning a Call to a Name (line 475):
        
        # Call to BroydenTridiagonal(...): (line 475)
        # Processing the call keyword arguments (line 475)
        kwargs_207878 = {}
        # Getting the type of 'BroydenTridiagonal' (line 475)
        BroydenTridiagonal_207877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 475)
        BroydenTridiagonal_call_result_207879 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), BroydenTridiagonal_207877, *[], **kwargs_207878)
        
        # Assigning a type to the variable 'p' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'p', BroydenTridiagonal_call_result_207879)
        
        # Call to assert_raises(...): (line 476)
        # Processing the call arguments (line 476)
        # Getting the type of 'ValueError' (line 476)
        ValueError_207881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 476)
        least_squares_207882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 34), 'least_squares', False)
        # Getting the type of 'p' (line 476)
        p_207883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 49), 'p', False)
        # Obtaining the member 'fun' of a type (line 476)
        fun_207884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 49), p_207883, 'fun')
        # Getting the type of 'p' (line 476)
        p_207885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 56), 'p', False)
        # Obtaining the member 'x0' of a type (line 476)
        x0_207886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 56), p_207885, 'x0')
        # Getting the type of 'p' (line 476)
        p_207887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 62), 'p', False)
        # Obtaining the member 'jac' of a type (line 476)
        jac_207888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 62), p_207887, 'jac')
        # Processing the call keyword arguments (line 476)
        str_207889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 32), 'str', 'best')
        keyword_207890 = str_207889
        # Getting the type of 'self' (line 477)
        self_207891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 477)
        method_207892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 47), self_207891, 'method')
        keyword_207893 = method_207892
        kwargs_207894 = {'tr_solver': keyword_207890, 'method': keyword_207893}
        # Getting the type of 'assert_raises' (line 476)
        assert_raises_207880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 476)
        assert_raises_call_result_207895 = invoke(stypy.reporting.localization.Localization(__file__, 476, 8), assert_raises_207880, *[ValueError_207881, least_squares_207882, fun_207884, x0_207886, jac_207888], **kwargs_207894)
        
        
        # Call to assert_raises(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'TypeError' (line 478)
        TypeError_207897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 22), 'TypeError', False)
        # Getting the type of 'least_squares' (line 478)
        least_squares_207898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 33), 'least_squares', False)
        # Getting the type of 'p' (line 478)
        p_207899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 48), 'p', False)
        # Obtaining the member 'fun' of a type (line 478)
        fun_207900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 48), p_207899, 'fun')
        # Getting the type of 'p' (line 478)
        p_207901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 55), 'p', False)
        # Obtaining the member 'x0' of a type (line 478)
        x0_207902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 55), p_207901, 'x0')
        # Getting the type of 'p' (line 478)
        p_207903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 61), 'p', False)
        # Obtaining the member 'jac' of a type (line 478)
        jac_207904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 61), p_207903, 'jac')
        # Processing the call keyword arguments (line 478)
        str_207905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 32), 'str', 'lsmr')
        keyword_207906 = str_207905
        
        # Obtaining an instance of the builtin type 'dict' (line 479)
        dict_207907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 51), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 479)
        # Adding element type (key, value) (line 479)
        str_207908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 52), 'str', 'tol')
        float_207909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 59), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 51), dict_207907, (str_207908, float_207909))
        
        keyword_207910 = dict_207907
        kwargs_207911 = {'tr_solver': keyword_207906, 'tr_options': keyword_207910}
        # Getting the type of 'assert_raises' (line 478)
        assert_raises_207896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 478)
        assert_raises_call_result_207912 = invoke(stypy.reporting.localization.Localization(__file__, 478, 8), assert_raises_207896, *[TypeError_207897, least_squares_207898, fun_207900, x0_207902, jac_207904], **kwargs_207911)
        
        
        # ################# End of 'test_wrong_parameters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_wrong_parameters' in the type store
        # Getting the type of 'stypy_return_type' (line 474)
        stypy_return_type_207913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207913)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_wrong_parameters'
        return stypy_return_type_207913


    @norecursion
    def test_solver_selection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_solver_selection'
        module_type_store = module_type_store.open_function_context('test_solver_selection', 481, 4, False)
        # Assigning a type to the variable 'self' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SparseMixin.test_solver_selection.__dict__.__setitem__('stypy_localization', localization)
        SparseMixin.test_solver_selection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SparseMixin.test_solver_selection.__dict__.__setitem__('stypy_type_store', module_type_store)
        SparseMixin.test_solver_selection.__dict__.__setitem__('stypy_function_name', 'SparseMixin.test_solver_selection')
        SparseMixin.test_solver_selection.__dict__.__setitem__('stypy_param_names_list', [])
        SparseMixin.test_solver_selection.__dict__.__setitem__('stypy_varargs_param_name', None)
        SparseMixin.test_solver_selection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SparseMixin.test_solver_selection.__dict__.__setitem__('stypy_call_defaults', defaults)
        SparseMixin.test_solver_selection.__dict__.__setitem__('stypy_call_varargs', varargs)
        SparseMixin.test_solver_selection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SparseMixin.test_solver_selection.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.test_solver_selection', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_solver_selection', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_solver_selection(...)' code ##################

        
        # Assigning a Call to a Name (line 482):
        
        # Call to BroydenTridiagonal(...): (line 482)
        # Processing the call keyword arguments (line 482)
        str_207915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 41), 'str', 'sparse')
        keyword_207916 = str_207915
        kwargs_207917 = {'mode': keyword_207916}
        # Getting the type of 'BroydenTridiagonal' (line 482)
        BroydenTridiagonal_207914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 17), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 482)
        BroydenTridiagonal_call_result_207918 = invoke(stypy.reporting.localization.Localization(__file__, 482, 17), BroydenTridiagonal_207914, *[], **kwargs_207917)
        
        # Assigning a type to the variable 'sparse' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'sparse', BroydenTridiagonal_call_result_207918)
        
        # Assigning a Call to a Name (line 483):
        
        # Call to BroydenTridiagonal(...): (line 483)
        # Processing the call keyword arguments (line 483)
        str_207920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 40), 'str', 'dense')
        keyword_207921 = str_207920
        kwargs_207922 = {'mode': keyword_207921}
        # Getting the type of 'BroydenTridiagonal' (line 483)
        BroydenTridiagonal_207919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 16), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 483)
        BroydenTridiagonal_call_result_207923 = invoke(stypy.reporting.localization.Localization(__file__, 483, 16), BroydenTridiagonal_207919, *[], **kwargs_207922)
        
        # Assigning a type to the variable 'dense' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'dense', BroydenTridiagonal_call_result_207923)
        
        # Assigning a Call to a Name (line 484):
        
        # Call to least_squares(...): (line 484)
        # Processing the call arguments (line 484)
        # Getting the type of 'sparse' (line 484)
        sparse_207925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 35), 'sparse', False)
        # Obtaining the member 'fun' of a type (line 484)
        fun_207926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 35), sparse_207925, 'fun')
        # Getting the type of 'sparse' (line 484)
        sparse_207927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 47), 'sparse', False)
        # Obtaining the member 'x0' of a type (line 484)
        x0_207928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 47), sparse_207927, 'x0')
        # Processing the call keyword arguments (line 484)
        # Getting the type of 'sparse' (line 484)
        sparse_207929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 62), 'sparse', False)
        # Obtaining the member 'jac' of a type (line 484)
        jac_207930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 62), sparse_207929, 'jac')
        keyword_207931 = jac_207930
        # Getting the type of 'self' (line 485)
        self_207932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 42), 'self', False)
        # Obtaining the member 'method' of a type (line 485)
        method_207933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 42), self_207932, 'method')
        keyword_207934 = method_207933
        kwargs_207935 = {'jac': keyword_207931, 'method': keyword_207934}
        # Getting the type of 'least_squares' (line 484)
        least_squares_207924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 21), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 484)
        least_squares_call_result_207936 = invoke(stypy.reporting.localization.Localization(__file__, 484, 21), least_squares_207924, *[fun_207926, x0_207928], **kwargs_207935)
        
        # Assigning a type to the variable 'res_sparse' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'res_sparse', least_squares_call_result_207936)
        
        # Assigning a Call to a Name (line 486):
        
        # Call to least_squares(...): (line 486)
        # Processing the call arguments (line 486)
        # Getting the type of 'dense' (line 486)
        dense_207938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 34), 'dense', False)
        # Obtaining the member 'fun' of a type (line 486)
        fun_207939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 34), dense_207938, 'fun')
        # Getting the type of 'dense' (line 486)
        dense_207940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 45), 'dense', False)
        # Obtaining the member 'x0' of a type (line 486)
        x0_207941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 45), dense_207940, 'x0')
        # Processing the call keyword arguments (line 486)
        # Getting the type of 'dense' (line 486)
        dense_207942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 59), 'dense', False)
        # Obtaining the member 'jac' of a type (line 486)
        jac_207943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 59), dense_207942, 'jac')
        keyword_207944 = jac_207943
        # Getting the type of 'self' (line 487)
        self_207945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 41), 'self', False)
        # Obtaining the member 'method' of a type (line 487)
        method_207946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 41), self_207945, 'method')
        keyword_207947 = method_207946
        kwargs_207948 = {'jac': keyword_207944, 'method': keyword_207947}
        # Getting the type of 'least_squares' (line 486)
        least_squares_207937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 20), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 486)
        least_squares_call_result_207949 = invoke(stypy.reporting.localization.Localization(__file__, 486, 20), least_squares_207937, *[fun_207939, x0_207941], **kwargs_207948)
        
        # Assigning a type to the variable 'res_dense' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'res_dense', least_squares_call_result_207949)
        
        # Call to assert_allclose(...): (line 488)
        # Processing the call arguments (line 488)
        # Getting the type of 'res_sparse' (line 488)
        res_sparse_207951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 24), 'res_sparse', False)
        # Obtaining the member 'cost' of a type (line 488)
        cost_207952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 24), res_sparse_207951, 'cost')
        int_207953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 41), 'int')
        # Processing the call keyword arguments (line 488)
        float_207954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 49), 'float')
        keyword_207955 = float_207954
        kwargs_207956 = {'atol': keyword_207955}
        # Getting the type of 'assert_allclose' (line 488)
        assert_allclose_207950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 488)
        assert_allclose_call_result_207957 = invoke(stypy.reporting.localization.Localization(__file__, 488, 8), assert_allclose_207950, *[cost_207952, int_207953], **kwargs_207956)
        
        
        # Call to assert_allclose(...): (line 489)
        # Processing the call arguments (line 489)
        # Getting the type of 'res_dense' (line 489)
        res_dense_207959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 24), 'res_dense', False)
        # Obtaining the member 'cost' of a type (line 489)
        cost_207960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 24), res_dense_207959, 'cost')
        int_207961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 40), 'int')
        # Processing the call keyword arguments (line 489)
        float_207962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 48), 'float')
        keyword_207963 = float_207962
        kwargs_207964 = {'atol': keyword_207963}
        # Getting the type of 'assert_allclose' (line 489)
        assert_allclose_207958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 489)
        assert_allclose_call_result_207965 = invoke(stypy.reporting.localization.Localization(__file__, 489, 8), assert_allclose_207958, *[cost_207960, int_207961], **kwargs_207964)
        
        
        # Call to assert_(...): (line 490)
        # Processing the call arguments (line 490)
        
        # Call to issparse(...): (line 490)
        # Processing the call arguments (line 490)
        # Getting the type of 'res_sparse' (line 490)
        res_sparse_207968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 25), 'res_sparse', False)
        # Obtaining the member 'jac' of a type (line 490)
        jac_207969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 25), res_sparse_207968, 'jac')
        # Processing the call keyword arguments (line 490)
        kwargs_207970 = {}
        # Getting the type of 'issparse' (line 490)
        issparse_207967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 16), 'issparse', False)
        # Calling issparse(args, kwargs) (line 490)
        issparse_call_result_207971 = invoke(stypy.reporting.localization.Localization(__file__, 490, 16), issparse_207967, *[jac_207969], **kwargs_207970)
        
        # Processing the call keyword arguments (line 490)
        kwargs_207972 = {}
        # Getting the type of 'assert_' (line 490)
        assert__207966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 490)
        assert__call_result_207973 = invoke(stypy.reporting.localization.Localization(__file__, 490, 8), assert__207966, *[issparse_call_result_207971], **kwargs_207972)
        
        
        # Call to assert_(...): (line 491)
        # Processing the call arguments (line 491)
        
        # Call to isinstance(...): (line 491)
        # Processing the call arguments (line 491)
        # Getting the type of 'res_dense' (line 491)
        res_dense_207976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 27), 'res_dense', False)
        # Obtaining the member 'jac' of a type (line 491)
        jac_207977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 27), res_dense_207976, 'jac')
        # Getting the type of 'np' (line 491)
        np_207978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 42), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 491)
        ndarray_207979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 42), np_207978, 'ndarray')
        # Processing the call keyword arguments (line 491)
        kwargs_207980 = {}
        # Getting the type of 'isinstance' (line 491)
        isinstance_207975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 491)
        isinstance_call_result_207981 = invoke(stypy.reporting.localization.Localization(__file__, 491, 16), isinstance_207975, *[jac_207977, ndarray_207979], **kwargs_207980)
        
        # Processing the call keyword arguments (line 491)
        kwargs_207982 = {}
        # Getting the type of 'assert_' (line 491)
        assert__207974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 491)
        assert__call_result_207983 = invoke(stypy.reporting.localization.Localization(__file__, 491, 8), assert__207974, *[isinstance_call_result_207981], **kwargs_207982)
        
        
        # ################# End of 'test_solver_selection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_solver_selection' in the type store
        # Getting the type of 'stypy_return_type' (line 481)
        stypy_return_type_207984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207984)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_solver_selection'
        return stypy_return_type_207984


    @norecursion
    def test_numerical_jac(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_numerical_jac'
        module_type_store = module_type_store.open_function_context('test_numerical_jac', 493, 4, False)
        # Assigning a type to the variable 'self' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SparseMixin.test_numerical_jac.__dict__.__setitem__('stypy_localization', localization)
        SparseMixin.test_numerical_jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SparseMixin.test_numerical_jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        SparseMixin.test_numerical_jac.__dict__.__setitem__('stypy_function_name', 'SparseMixin.test_numerical_jac')
        SparseMixin.test_numerical_jac.__dict__.__setitem__('stypy_param_names_list', [])
        SparseMixin.test_numerical_jac.__dict__.__setitem__('stypy_varargs_param_name', None)
        SparseMixin.test_numerical_jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SparseMixin.test_numerical_jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        SparseMixin.test_numerical_jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        SparseMixin.test_numerical_jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SparseMixin.test_numerical_jac.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.test_numerical_jac', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_numerical_jac', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_numerical_jac(...)' code ##################

        
        # Assigning a Call to a Name (line 494):
        
        # Call to BroydenTridiagonal(...): (line 494)
        # Processing the call keyword arguments (line 494)
        kwargs_207986 = {}
        # Getting the type of 'BroydenTridiagonal' (line 494)
        BroydenTridiagonal_207985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 494)
        BroydenTridiagonal_call_result_207987 = invoke(stypy.reporting.localization.Localization(__file__, 494, 12), BroydenTridiagonal_207985, *[], **kwargs_207986)
        
        # Assigning a type to the variable 'p' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'p', BroydenTridiagonal_call_result_207987)
        
        
        # Obtaining an instance of the builtin type 'list' (line 495)
        list_207988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 495)
        # Adding element type (line 495)
        str_207989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 20), 'str', '2-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 19), list_207988, str_207989)
        # Adding element type (line 495)
        str_207990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 31), 'str', '3-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 19), list_207988, str_207990)
        # Adding element type (line 495)
        str_207991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 42), 'str', 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 19), list_207988, str_207991)
        
        # Testing the type of a for loop iterable (line 495)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 495, 8), list_207988)
        # Getting the type of the for loop variable (line 495)
        for_loop_var_207992 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 495, 8), list_207988)
        # Assigning a type to the variable 'jac' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'jac', for_loop_var_207992)
        # SSA begins for a for statement (line 495)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 496):
        
        # Call to least_squares(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'p' (line 496)
        p_207994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 38), 'p', False)
        # Obtaining the member 'fun' of a type (line 496)
        fun_207995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 38), p_207994, 'fun')
        # Getting the type of 'p' (line 496)
        p_207996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 45), 'p', False)
        # Obtaining the member 'x0' of a type (line 496)
        x0_207997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 45), p_207996, 'x0')
        # Getting the type of 'jac' (line 496)
        jac_207998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 51), 'jac', False)
        # Processing the call keyword arguments (line 496)
        # Getting the type of 'self' (line 496)
        self_207999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 63), 'self', False)
        # Obtaining the member 'method' of a type (line 496)
        method_208000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 63), self_207999, 'method')
        keyword_208001 = method_208000
        kwargs_208002 = {'method': keyword_208001}
        # Getting the type of 'least_squares' (line 496)
        least_squares_207993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 24), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 496)
        least_squares_call_result_208003 = invoke(stypy.reporting.localization.Localization(__file__, 496, 24), least_squares_207993, *[fun_207995, x0_207997, jac_207998], **kwargs_208002)
        
        # Assigning a type to the variable 'res_dense' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'res_dense', least_squares_call_result_208003)
        
        # Assigning a Call to a Name (line 497):
        
        # Call to least_squares(...): (line 497)
        # Processing the call arguments (line 497)
        # Getting the type of 'p' (line 498)
        p_208005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 'p', False)
        # Obtaining the member 'fun' of a type (line 498)
        fun_208006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 16), p_208005, 'fun')
        # Getting the type of 'p' (line 498)
        p_208007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 23), 'p', False)
        # Obtaining the member 'x0' of a type (line 498)
        x0_208008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 23), p_208007, 'x0')
        # Getting the type of 'jac' (line 498)
        jac_208009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 29), 'jac', False)
        # Processing the call keyword arguments (line 497)
        # Getting the type of 'self' (line 498)
        self_208010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 40), 'self', False)
        # Obtaining the member 'method' of a type (line 498)
        method_208011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 40), self_208010, 'method')
        keyword_208012 = method_208011
        # Getting the type of 'p' (line 499)
        p_208013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 29), 'p', False)
        # Obtaining the member 'sparsity' of a type (line 499)
        sparsity_208014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 29), p_208013, 'sparsity')
        keyword_208015 = sparsity_208014
        kwargs_208016 = {'jac_sparsity': keyword_208015, 'method': keyword_208012}
        # Getting the type of 'least_squares' (line 497)
        least_squares_208004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 25), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 497)
        least_squares_call_result_208017 = invoke(stypy.reporting.localization.Localization(__file__, 497, 25), least_squares_208004, *[fun_208006, x0_208008, jac_208009], **kwargs_208016)
        
        # Assigning a type to the variable 'res_sparse' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'res_sparse', least_squares_call_result_208017)
        
        # Call to assert_equal(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of 'res_dense' (line 500)
        res_dense_208019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 25), 'res_dense', False)
        # Obtaining the member 'nfev' of a type (line 500)
        nfev_208020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 25), res_dense_208019, 'nfev')
        # Getting the type of 'res_sparse' (line 500)
        res_sparse_208021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 41), 'res_sparse', False)
        # Obtaining the member 'nfev' of a type (line 500)
        nfev_208022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 41), res_sparse_208021, 'nfev')
        # Processing the call keyword arguments (line 500)
        kwargs_208023 = {}
        # Getting the type of 'assert_equal' (line 500)
        assert_equal_208018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 500)
        assert_equal_call_result_208024 = invoke(stypy.reporting.localization.Localization(__file__, 500, 12), assert_equal_208018, *[nfev_208020, nfev_208022], **kwargs_208023)
        
        
        # Call to assert_allclose(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of 'res_dense' (line 501)
        res_dense_208026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 28), 'res_dense', False)
        # Obtaining the member 'x' of a type (line 501)
        x_208027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 28), res_dense_208026, 'x')
        # Getting the type of 'res_sparse' (line 501)
        res_sparse_208028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 41), 'res_sparse', False)
        # Obtaining the member 'x' of a type (line 501)
        x_208029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 41), res_sparse_208028, 'x')
        # Processing the call keyword arguments (line 501)
        float_208030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 60), 'float')
        keyword_208031 = float_208030
        kwargs_208032 = {'atol': keyword_208031}
        # Getting the type of 'assert_allclose' (line 501)
        assert_allclose_208025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 501)
        assert_allclose_call_result_208033 = invoke(stypy.reporting.localization.Localization(__file__, 501, 12), assert_allclose_208025, *[x_208027, x_208029], **kwargs_208032)
        
        
        # Call to assert_allclose(...): (line 502)
        # Processing the call arguments (line 502)
        # Getting the type of 'res_dense' (line 502)
        res_dense_208035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 28), 'res_dense', False)
        # Obtaining the member 'cost' of a type (line 502)
        cost_208036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 28), res_dense_208035, 'cost')
        int_208037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 44), 'int')
        # Processing the call keyword arguments (line 502)
        float_208038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 52), 'float')
        keyword_208039 = float_208038
        kwargs_208040 = {'atol': keyword_208039}
        # Getting the type of 'assert_allclose' (line 502)
        assert_allclose_208034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 502)
        assert_allclose_call_result_208041 = invoke(stypy.reporting.localization.Localization(__file__, 502, 12), assert_allclose_208034, *[cost_208036, int_208037], **kwargs_208040)
        
        
        # Call to assert_allclose(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'res_sparse' (line 503)
        res_sparse_208043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 28), 'res_sparse', False)
        # Obtaining the member 'cost' of a type (line 503)
        cost_208044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 28), res_sparse_208043, 'cost')
        int_208045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 45), 'int')
        # Processing the call keyword arguments (line 503)
        float_208046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 53), 'float')
        keyword_208047 = float_208046
        kwargs_208048 = {'atol': keyword_208047}
        # Getting the type of 'assert_allclose' (line 503)
        assert_allclose_208042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 503)
        assert_allclose_call_result_208049 = invoke(stypy.reporting.localization.Localization(__file__, 503, 12), assert_allclose_208042, *[cost_208044, int_208045], **kwargs_208048)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_numerical_jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_numerical_jac' in the type store
        # Getting the type of 'stypy_return_type' (line 493)
        stypy_return_type_208050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208050)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_numerical_jac'
        return stypy_return_type_208050


    @norecursion
    def test_with_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_with_bounds'
        module_type_store = module_type_store.open_function_context('test_with_bounds', 505, 4, False)
        # Assigning a type to the variable 'self' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SparseMixin.test_with_bounds.__dict__.__setitem__('stypy_localization', localization)
        SparseMixin.test_with_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SparseMixin.test_with_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        SparseMixin.test_with_bounds.__dict__.__setitem__('stypy_function_name', 'SparseMixin.test_with_bounds')
        SparseMixin.test_with_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        SparseMixin.test_with_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        SparseMixin.test_with_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SparseMixin.test_with_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        SparseMixin.test_with_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        SparseMixin.test_with_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SparseMixin.test_with_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.test_with_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_with_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_with_bounds(...)' code ##################

        
        # Assigning a Call to a Name (line 506):
        
        # Call to BroydenTridiagonal(...): (line 506)
        # Processing the call keyword arguments (line 506)
        kwargs_208052 = {}
        # Getting the type of 'BroydenTridiagonal' (line 506)
        BroydenTridiagonal_208051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 12), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 506)
        BroydenTridiagonal_call_result_208053 = invoke(stypy.reporting.localization.Localization(__file__, 506, 12), BroydenTridiagonal_208051, *[], **kwargs_208052)
        
        # Assigning a type to the variable 'p' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'p', BroydenTridiagonal_call_result_208053)
        
        
        # Call to product(...): (line 507)
        # Processing the call arguments (line 507)
        
        # Obtaining an instance of the builtin type 'list' (line 508)
        list_208055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 508)
        # Adding element type (line 508)
        # Getting the type of 'p' (line 508)
        p_208056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 17), 'p', False)
        # Obtaining the member 'jac' of a type (line 508)
        jac_208057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 17), p_208056, 'jac')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 16), list_208055, jac_208057)
        # Adding element type (line 508)
        str_208058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 24), 'str', '2-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 16), list_208055, str_208058)
        # Adding element type (line 508)
        str_208059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 35), 'str', '3-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 16), list_208055, str_208059)
        # Adding element type (line 508)
        str_208060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 46), 'str', 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 16), list_208055, str_208060)
        
        
        # Obtaining an instance of the builtin type 'list' (line 508)
        list_208061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 508)
        # Adding element type (line 508)
        # Getting the type of 'None' (line 508)
        None_208062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 54), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 53), list_208061, None_208062)
        # Adding element type (line 508)
        # Getting the type of 'p' (line 508)
        p_208063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 60), 'p', False)
        # Obtaining the member 'sparsity' of a type (line 508)
        sparsity_208064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 60), p_208063, 'sparsity')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 53), list_208061, sparsity_208064)
        
        # Processing the call keyword arguments (line 507)
        kwargs_208065 = {}
        # Getting the type of 'product' (line 507)
        product_208054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 33), 'product', False)
        # Calling product(args, kwargs) (line 507)
        product_call_result_208066 = invoke(stypy.reporting.localization.Localization(__file__, 507, 33), product_208054, *[list_208055, list_208061], **kwargs_208065)
        
        # Testing the type of a for loop iterable (line 507)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 507, 8), product_call_result_208066)
        # Getting the type of the for loop variable (line 507)
        for_loop_var_208067 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 507, 8), product_call_result_208066)
        # Assigning a type to the variable 'jac' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'jac', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 8), for_loop_var_208067))
        # Assigning a type to the variable 'jac_sparsity' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'jac_sparsity', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 8), for_loop_var_208067))
        # SSA begins for a for statement (line 507)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 509):
        
        # Call to least_squares(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'p' (line 510)
        p_208069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'p', False)
        # Obtaining the member 'fun' of a type (line 510)
        fun_208070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 16), p_208069, 'fun')
        # Getting the type of 'p' (line 510)
        p_208071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 23), 'p', False)
        # Obtaining the member 'x0' of a type (line 510)
        x0_208072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 23), p_208071, 'x0')
        # Getting the type of 'jac' (line 510)
        jac_208073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 29), 'jac', False)
        # Processing the call keyword arguments (line 509)
        
        # Obtaining an instance of the builtin type 'tuple' (line 510)
        tuple_208074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 510)
        # Adding element type (line 510)
        # Getting the type of 'p' (line 510)
        p_208075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 42), 'p', False)
        # Obtaining the member 'lb' of a type (line 510)
        lb_208076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 42), p_208075, 'lb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 42), tuple_208074, lb_208076)
        # Adding element type (line 510)
        # Getting the type of 'np' (line 510)
        np_208077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 48), 'np', False)
        # Obtaining the member 'inf' of a type (line 510)
        inf_208078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 48), np_208077, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 42), tuple_208074, inf_208078)
        
        keyword_208079 = tuple_208074
        # Getting the type of 'self' (line 511)
        self_208080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 23), 'self', False)
        # Obtaining the member 'method' of a type (line 511)
        method_208081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 23), self_208080, 'method')
        keyword_208082 = method_208081
        # Getting the type of 'jac_sparsity' (line 511)
        jac_sparsity_208083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 48), 'jac_sparsity', False)
        keyword_208084 = jac_sparsity_208083
        kwargs_208085 = {'jac_sparsity': keyword_208084, 'bounds': keyword_208079, 'method': keyword_208082}
        # Getting the type of 'least_squares' (line 509)
        least_squares_208068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 20), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 509)
        least_squares_call_result_208086 = invoke(stypy.reporting.localization.Localization(__file__, 509, 20), least_squares_208068, *[fun_208070, x0_208072, jac_208073], **kwargs_208085)
        
        # Assigning a type to the variable 'res_1' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'res_1', least_squares_call_result_208086)
        
        # Assigning a Call to a Name (line 512):
        
        # Call to least_squares(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'p' (line 513)
        p_208088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 16), 'p', False)
        # Obtaining the member 'fun' of a type (line 513)
        fun_208089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 16), p_208088, 'fun')
        # Getting the type of 'p' (line 513)
        p_208090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 23), 'p', False)
        # Obtaining the member 'x0' of a type (line 513)
        x0_208091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 23), p_208090, 'x0')
        # Getting the type of 'jac' (line 513)
        jac_208092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 29), 'jac', False)
        # Processing the call keyword arguments (line 512)
        
        # Obtaining an instance of the builtin type 'tuple' (line 513)
        tuple_208093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 513)
        # Adding element type (line 513)
        
        # Getting the type of 'np' (line 513)
        np_208094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 43), 'np', False)
        # Obtaining the member 'inf' of a type (line 513)
        inf_208095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 43), np_208094, 'inf')
        # Applying the 'usub' unary operator (line 513)
        result___neg___208096 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 42), 'usub', inf_208095)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 42), tuple_208093, result___neg___208096)
        # Adding element type (line 513)
        # Getting the type of 'p' (line 513)
        p_208097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 51), 'p', False)
        # Obtaining the member 'ub' of a type (line 513)
        ub_208098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 51), p_208097, 'ub')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 42), tuple_208093, ub_208098)
        
        keyword_208099 = tuple_208093
        # Getting the type of 'self' (line 514)
        self_208100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 23), 'self', False)
        # Obtaining the member 'method' of a type (line 514)
        method_208101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 23), self_208100, 'method')
        keyword_208102 = method_208101
        # Getting the type of 'jac_sparsity' (line 514)
        jac_sparsity_208103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 49), 'jac_sparsity', False)
        keyword_208104 = jac_sparsity_208103
        kwargs_208105 = {'jac_sparsity': keyword_208104, 'bounds': keyword_208099, 'method': keyword_208102}
        # Getting the type of 'least_squares' (line 512)
        least_squares_208087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 20), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 512)
        least_squares_call_result_208106 = invoke(stypy.reporting.localization.Localization(__file__, 512, 20), least_squares_208087, *[fun_208089, x0_208091, jac_208092], **kwargs_208105)
        
        # Assigning a type to the variable 'res_2' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'res_2', least_squares_call_result_208106)
        
        # Assigning a Call to a Name (line 515):
        
        # Call to least_squares(...): (line 515)
        # Processing the call arguments (line 515)
        # Getting the type of 'p' (line 516)
        p_208108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 16), 'p', False)
        # Obtaining the member 'fun' of a type (line 516)
        fun_208109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 16), p_208108, 'fun')
        # Getting the type of 'p' (line 516)
        p_208110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 23), 'p', False)
        # Obtaining the member 'x0' of a type (line 516)
        x0_208111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 23), p_208110, 'x0')
        # Getting the type of 'jac' (line 516)
        jac_208112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 29), 'jac', False)
        # Processing the call keyword arguments (line 515)
        
        # Obtaining an instance of the builtin type 'tuple' (line 516)
        tuple_208113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 516)
        # Adding element type (line 516)
        # Getting the type of 'p' (line 516)
        p_208114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 42), 'p', False)
        # Obtaining the member 'lb' of a type (line 516)
        lb_208115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 42), p_208114, 'lb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 42), tuple_208113, lb_208115)
        # Adding element type (line 516)
        # Getting the type of 'p' (line 516)
        p_208116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 48), 'p', False)
        # Obtaining the member 'ub' of a type (line 516)
        ub_208117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 48), p_208116, 'ub')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 42), tuple_208113, ub_208117)
        
        keyword_208118 = tuple_208113
        # Getting the type of 'self' (line 517)
        self_208119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 23), 'self', False)
        # Obtaining the member 'method' of a type (line 517)
        method_208120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 23), self_208119, 'method')
        keyword_208121 = method_208120
        # Getting the type of 'jac_sparsity' (line 517)
        jac_sparsity_208122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 49), 'jac_sparsity', False)
        keyword_208123 = jac_sparsity_208122
        kwargs_208124 = {'jac_sparsity': keyword_208123, 'bounds': keyword_208118, 'method': keyword_208121}
        # Getting the type of 'least_squares' (line 515)
        least_squares_208107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 20), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 515)
        least_squares_call_result_208125 = invoke(stypy.reporting.localization.Localization(__file__, 515, 20), least_squares_208107, *[fun_208109, x0_208111, jac_208112], **kwargs_208124)
        
        # Assigning a type to the variable 'res_3' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'res_3', least_squares_call_result_208125)
        
        # Call to assert_allclose(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'res_1' (line 518)
        res_1_208127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 28), 'res_1', False)
        # Obtaining the member 'optimality' of a type (line 518)
        optimality_208128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 28), res_1_208127, 'optimality')
        int_208129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 46), 'int')
        # Processing the call keyword arguments (line 518)
        float_208130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 54), 'float')
        keyword_208131 = float_208130
        kwargs_208132 = {'atol': keyword_208131}
        # Getting the type of 'assert_allclose' (line 518)
        assert_allclose_208126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 518)
        assert_allclose_call_result_208133 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), assert_allclose_208126, *[optimality_208128, int_208129], **kwargs_208132)
        
        
        # Call to assert_allclose(...): (line 519)
        # Processing the call arguments (line 519)
        # Getting the type of 'res_2' (line 519)
        res_2_208135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 28), 'res_2', False)
        # Obtaining the member 'optimality' of a type (line 519)
        optimality_208136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 28), res_2_208135, 'optimality')
        int_208137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 46), 'int')
        # Processing the call keyword arguments (line 519)
        float_208138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 54), 'float')
        keyword_208139 = float_208138
        kwargs_208140 = {'atol': keyword_208139}
        # Getting the type of 'assert_allclose' (line 519)
        assert_allclose_208134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 519)
        assert_allclose_call_result_208141 = invoke(stypy.reporting.localization.Localization(__file__, 519, 12), assert_allclose_208134, *[optimality_208136, int_208137], **kwargs_208140)
        
        
        # Call to assert_allclose(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 'res_3' (line 520)
        res_3_208143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 28), 'res_3', False)
        # Obtaining the member 'optimality' of a type (line 520)
        optimality_208144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 28), res_3_208143, 'optimality')
        int_208145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 46), 'int')
        # Processing the call keyword arguments (line 520)
        float_208146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 54), 'float')
        keyword_208147 = float_208146
        kwargs_208148 = {'atol': keyword_208147}
        # Getting the type of 'assert_allclose' (line 520)
        assert_allclose_208142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 520)
        assert_allclose_call_result_208149 = invoke(stypy.reporting.localization.Localization(__file__, 520, 12), assert_allclose_208142, *[optimality_208144, int_208145], **kwargs_208148)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_with_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_with_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 505)
        stypy_return_type_208150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208150)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_with_bounds'
        return stypy_return_type_208150


    @norecursion
    def test_wrong_jac_sparsity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_wrong_jac_sparsity'
        module_type_store = module_type_store.open_function_context('test_wrong_jac_sparsity', 522, 4, False)
        # Assigning a type to the variable 'self' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SparseMixin.test_wrong_jac_sparsity.__dict__.__setitem__('stypy_localization', localization)
        SparseMixin.test_wrong_jac_sparsity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SparseMixin.test_wrong_jac_sparsity.__dict__.__setitem__('stypy_type_store', module_type_store)
        SparseMixin.test_wrong_jac_sparsity.__dict__.__setitem__('stypy_function_name', 'SparseMixin.test_wrong_jac_sparsity')
        SparseMixin.test_wrong_jac_sparsity.__dict__.__setitem__('stypy_param_names_list', [])
        SparseMixin.test_wrong_jac_sparsity.__dict__.__setitem__('stypy_varargs_param_name', None)
        SparseMixin.test_wrong_jac_sparsity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SparseMixin.test_wrong_jac_sparsity.__dict__.__setitem__('stypy_call_defaults', defaults)
        SparseMixin.test_wrong_jac_sparsity.__dict__.__setitem__('stypy_call_varargs', varargs)
        SparseMixin.test_wrong_jac_sparsity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SparseMixin.test_wrong_jac_sparsity.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.test_wrong_jac_sparsity', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_wrong_jac_sparsity', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_wrong_jac_sparsity(...)' code ##################

        
        # Assigning a Call to a Name (line 523):
        
        # Call to BroydenTridiagonal(...): (line 523)
        # Processing the call keyword arguments (line 523)
        kwargs_208152 = {}
        # Getting the type of 'BroydenTridiagonal' (line 523)
        BroydenTridiagonal_208151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 523)
        BroydenTridiagonal_call_result_208153 = invoke(stypy.reporting.localization.Localization(__file__, 523, 12), BroydenTridiagonal_208151, *[], **kwargs_208152)
        
        # Assigning a type to the variable 'p' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'p', BroydenTridiagonal_call_result_208153)
        
        # Assigning a Subscript to a Name (line 524):
        
        # Obtaining the type of the subscript
        int_208154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 31), 'int')
        slice_208155 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 524, 19), None, int_208154, None)
        # Getting the type of 'p' (line 524)
        p_208156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 19), 'p')
        # Obtaining the member 'sparsity' of a type (line 524)
        sparsity_208157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 19), p_208156, 'sparsity')
        # Obtaining the member '__getitem__' of a type (line 524)
        getitem___208158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 19), sparsity_208157, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 524)
        subscript_call_result_208159 = invoke(stypy.reporting.localization.Localization(__file__, 524, 19), getitem___208158, slice_208155)
        
        # Assigning a type to the variable 'sparsity' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'sparsity', subscript_call_result_208159)
        
        # Call to assert_raises(...): (line 525)
        # Processing the call arguments (line 525)
        # Getting the type of 'ValueError' (line 525)
        ValueError_208161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 525)
        least_squares_208162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 34), 'least_squares', False)
        # Getting the type of 'p' (line 525)
        p_208163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 49), 'p', False)
        # Obtaining the member 'fun' of a type (line 525)
        fun_208164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 49), p_208163, 'fun')
        # Getting the type of 'p' (line 525)
        p_208165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 56), 'p', False)
        # Obtaining the member 'x0' of a type (line 525)
        x0_208166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 56), p_208165, 'x0')
        # Processing the call keyword arguments (line 525)
        # Getting the type of 'sparsity' (line 526)
        sparsity_208167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 35), 'sparsity', False)
        keyword_208168 = sparsity_208167
        # Getting the type of 'self' (line 526)
        self_208169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 52), 'self', False)
        # Obtaining the member 'method' of a type (line 526)
        method_208170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 52), self_208169, 'method')
        keyword_208171 = method_208170
        kwargs_208172 = {'jac_sparsity': keyword_208168, 'method': keyword_208171}
        # Getting the type of 'assert_raises' (line 525)
        assert_raises_208160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 525)
        assert_raises_call_result_208173 = invoke(stypy.reporting.localization.Localization(__file__, 525, 8), assert_raises_208160, *[ValueError_208161, least_squares_208162, fun_208164, x0_208166], **kwargs_208172)
        
        
        # ################# End of 'test_wrong_jac_sparsity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_wrong_jac_sparsity' in the type store
        # Getting the type of 'stypy_return_type' (line 522)
        stypy_return_type_208174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208174)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_wrong_jac_sparsity'
        return stypy_return_type_208174


    @norecursion
    def test_linear_operator(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_linear_operator'
        module_type_store = module_type_store.open_function_context('test_linear_operator', 528, 4, False)
        # Assigning a type to the variable 'self' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SparseMixin.test_linear_operator.__dict__.__setitem__('stypy_localization', localization)
        SparseMixin.test_linear_operator.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SparseMixin.test_linear_operator.__dict__.__setitem__('stypy_type_store', module_type_store)
        SparseMixin.test_linear_operator.__dict__.__setitem__('stypy_function_name', 'SparseMixin.test_linear_operator')
        SparseMixin.test_linear_operator.__dict__.__setitem__('stypy_param_names_list', [])
        SparseMixin.test_linear_operator.__dict__.__setitem__('stypy_varargs_param_name', None)
        SparseMixin.test_linear_operator.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SparseMixin.test_linear_operator.__dict__.__setitem__('stypy_call_defaults', defaults)
        SparseMixin.test_linear_operator.__dict__.__setitem__('stypy_call_varargs', varargs)
        SparseMixin.test_linear_operator.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SparseMixin.test_linear_operator.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.test_linear_operator', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_linear_operator', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_linear_operator(...)' code ##################

        
        # Assigning a Call to a Name (line 529):
        
        # Call to BroydenTridiagonal(...): (line 529)
        # Processing the call keyword arguments (line 529)
        str_208176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 36), 'str', 'operator')
        keyword_208177 = str_208176
        kwargs_208178 = {'mode': keyword_208177}
        # Getting the type of 'BroydenTridiagonal' (line 529)
        BroydenTridiagonal_208175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 529)
        BroydenTridiagonal_call_result_208179 = invoke(stypy.reporting.localization.Localization(__file__, 529, 12), BroydenTridiagonal_208175, *[], **kwargs_208178)
        
        # Assigning a type to the variable 'p' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'p', BroydenTridiagonal_call_result_208179)
        
        # Assigning a Call to a Name (line 530):
        
        # Call to least_squares(...): (line 530)
        # Processing the call arguments (line 530)
        # Getting the type of 'p' (line 530)
        p_208181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 28), 'p', False)
        # Obtaining the member 'fun' of a type (line 530)
        fun_208182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 28), p_208181, 'fun')
        # Getting the type of 'p' (line 530)
        p_208183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 35), 'p', False)
        # Obtaining the member 'x0' of a type (line 530)
        x0_208184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 35), p_208183, 'x0')
        # Getting the type of 'p' (line 530)
        p_208185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 41), 'p', False)
        # Obtaining the member 'jac' of a type (line 530)
        jac_208186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 41), p_208185, 'jac')
        # Processing the call keyword arguments (line 530)
        # Getting the type of 'self' (line 530)
        self_208187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 55), 'self', False)
        # Obtaining the member 'method' of a type (line 530)
        method_208188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 55), self_208187, 'method')
        keyword_208189 = method_208188
        kwargs_208190 = {'method': keyword_208189}
        # Getting the type of 'least_squares' (line 530)
        least_squares_208180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 530)
        least_squares_call_result_208191 = invoke(stypy.reporting.localization.Localization(__file__, 530, 14), least_squares_208180, *[fun_208182, x0_208184, jac_208186], **kwargs_208190)
        
        # Assigning a type to the variable 'res' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'res', least_squares_call_result_208191)
        
        # Call to assert_allclose(...): (line 531)
        # Processing the call arguments (line 531)
        # Getting the type of 'res' (line 531)
        res_208193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 24), 'res', False)
        # Obtaining the member 'cost' of a type (line 531)
        cost_208194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 24), res_208193, 'cost')
        float_208195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 34), 'float')
        # Processing the call keyword arguments (line 531)
        float_208196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 44), 'float')
        keyword_208197 = float_208196
        kwargs_208198 = {'atol': keyword_208197}
        # Getting the type of 'assert_allclose' (line 531)
        assert_allclose_208192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 531)
        assert_allclose_call_result_208199 = invoke(stypy.reporting.localization.Localization(__file__, 531, 8), assert_allclose_208192, *[cost_208194, float_208195], **kwargs_208198)
        
        
        # Call to assert_raises(...): (line 532)
        # Processing the call arguments (line 532)
        # Getting the type of 'ValueError' (line 532)
        ValueError_208201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 532)
        least_squares_208202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 34), 'least_squares', False)
        # Getting the type of 'p' (line 532)
        p_208203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 49), 'p', False)
        # Obtaining the member 'fun' of a type (line 532)
        fun_208204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 49), p_208203, 'fun')
        # Getting the type of 'p' (line 532)
        p_208205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 56), 'p', False)
        # Obtaining the member 'x0' of a type (line 532)
        x0_208206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 56), p_208205, 'x0')
        # Getting the type of 'p' (line 532)
        p_208207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 62), 'p', False)
        # Obtaining the member 'jac' of a type (line 532)
        jac_208208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 62), p_208207, 'jac')
        # Processing the call keyword arguments (line 532)
        # Getting the type of 'self' (line 533)
        self_208209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 29), 'self', False)
        # Obtaining the member 'method' of a type (line 533)
        method_208210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 29), self_208209, 'method')
        keyword_208211 = method_208210
        str_208212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 52), 'str', 'exact')
        keyword_208213 = str_208212
        kwargs_208214 = {'tr_solver': keyword_208213, 'method': keyword_208211}
        # Getting the type of 'assert_raises' (line 532)
        assert_raises_208200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 532)
        assert_raises_call_result_208215 = invoke(stypy.reporting.localization.Localization(__file__, 532, 8), assert_raises_208200, *[ValueError_208201, least_squares_208202, fun_208204, x0_208206, jac_208208], **kwargs_208214)
        
        
        # ################# End of 'test_linear_operator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_linear_operator' in the type store
        # Getting the type of 'stypy_return_type' (line 528)
        stypy_return_type_208216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208216)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_linear_operator'
        return stypy_return_type_208216


    @norecursion
    def test_x_scale_jac_scale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_scale_jac_scale'
        module_type_store = module_type_store.open_function_context('test_x_scale_jac_scale', 535, 4, False)
        # Assigning a type to the variable 'self' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SparseMixin.test_x_scale_jac_scale.__dict__.__setitem__('stypy_localization', localization)
        SparseMixin.test_x_scale_jac_scale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SparseMixin.test_x_scale_jac_scale.__dict__.__setitem__('stypy_type_store', module_type_store)
        SparseMixin.test_x_scale_jac_scale.__dict__.__setitem__('stypy_function_name', 'SparseMixin.test_x_scale_jac_scale')
        SparseMixin.test_x_scale_jac_scale.__dict__.__setitem__('stypy_param_names_list', [])
        SparseMixin.test_x_scale_jac_scale.__dict__.__setitem__('stypy_varargs_param_name', None)
        SparseMixin.test_x_scale_jac_scale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SparseMixin.test_x_scale_jac_scale.__dict__.__setitem__('stypy_call_defaults', defaults)
        SparseMixin.test_x_scale_jac_scale.__dict__.__setitem__('stypy_call_varargs', varargs)
        SparseMixin.test_x_scale_jac_scale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SparseMixin.test_x_scale_jac_scale.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.test_x_scale_jac_scale', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_scale_jac_scale', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_scale_jac_scale(...)' code ##################

        
        # Assigning a Call to a Name (line 536):
        
        # Call to BroydenTridiagonal(...): (line 536)
        # Processing the call keyword arguments (line 536)
        kwargs_208218 = {}
        # Getting the type of 'BroydenTridiagonal' (line 536)
        BroydenTridiagonal_208217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 12), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 536)
        BroydenTridiagonal_call_result_208219 = invoke(stypy.reporting.localization.Localization(__file__, 536, 12), BroydenTridiagonal_208217, *[], **kwargs_208218)
        
        # Assigning a type to the variable 'p' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'p', BroydenTridiagonal_call_result_208219)
        
        # Assigning a Call to a Name (line 537):
        
        # Call to least_squares(...): (line 537)
        # Processing the call arguments (line 537)
        # Getting the type of 'p' (line 537)
        p_208221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 28), 'p', False)
        # Obtaining the member 'fun' of a type (line 537)
        fun_208222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 28), p_208221, 'fun')
        # Getting the type of 'p' (line 537)
        p_208223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 35), 'p', False)
        # Obtaining the member 'x0' of a type (line 537)
        x0_208224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 35), p_208223, 'x0')
        # Getting the type of 'p' (line 537)
        p_208225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 41), 'p', False)
        # Obtaining the member 'jac' of a type (line 537)
        jac_208226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 41), p_208225, 'jac')
        # Processing the call keyword arguments (line 537)
        # Getting the type of 'self' (line 537)
        self_208227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 55), 'self', False)
        # Obtaining the member 'method' of a type (line 537)
        method_208228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 55), self_208227, 'method')
        keyword_208229 = method_208228
        str_208230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 36), 'str', 'jac')
        keyword_208231 = str_208230
        kwargs_208232 = {'x_scale': keyword_208231, 'method': keyword_208229}
        # Getting the type of 'least_squares' (line 537)
        least_squares_208220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 537)
        least_squares_call_result_208233 = invoke(stypy.reporting.localization.Localization(__file__, 537, 14), least_squares_208220, *[fun_208222, x0_208224, jac_208226], **kwargs_208232)
        
        # Assigning a type to the variable 'res' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'res', least_squares_call_result_208233)
        
        # Call to assert_allclose(...): (line 539)
        # Processing the call arguments (line 539)
        # Getting the type of 'res' (line 539)
        res_208235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 24), 'res', False)
        # Obtaining the member 'cost' of a type (line 539)
        cost_208236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 24), res_208235, 'cost')
        float_208237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 34), 'float')
        # Processing the call keyword arguments (line 539)
        float_208238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 44), 'float')
        keyword_208239 = float_208238
        kwargs_208240 = {'atol': keyword_208239}
        # Getting the type of 'assert_allclose' (line 539)
        assert_allclose_208234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 539)
        assert_allclose_call_result_208241 = invoke(stypy.reporting.localization.Localization(__file__, 539, 8), assert_allclose_208234, *[cost_208236, float_208237], **kwargs_208240)
        
        
        # Assigning a Call to a Name (line 541):
        
        # Call to BroydenTridiagonal(...): (line 541)
        # Processing the call keyword arguments (line 541)
        str_208243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 36), 'str', 'operator')
        keyword_208244 = str_208243
        kwargs_208245 = {'mode': keyword_208244}
        # Getting the type of 'BroydenTridiagonal' (line 541)
        BroydenTridiagonal_208242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 541)
        BroydenTridiagonal_call_result_208246 = invoke(stypy.reporting.localization.Localization(__file__, 541, 12), BroydenTridiagonal_208242, *[], **kwargs_208245)
        
        # Assigning a type to the variable 'p' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'p', BroydenTridiagonal_call_result_208246)
        
        # Call to assert_raises(...): (line 542)
        # Processing the call arguments (line 542)
        # Getting the type of 'ValueError' (line 542)
        ValueError_208248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 542)
        least_squares_208249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 34), 'least_squares', False)
        # Getting the type of 'p' (line 542)
        p_208250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 49), 'p', False)
        # Obtaining the member 'fun' of a type (line 542)
        fun_208251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 49), p_208250, 'fun')
        # Getting the type of 'p' (line 542)
        p_208252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 56), 'p', False)
        # Obtaining the member 'x0' of a type (line 542)
        x0_208253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 56), p_208252, 'x0')
        # Getting the type of 'p' (line 542)
        p_208254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 62), 'p', False)
        # Obtaining the member 'jac' of a type (line 542)
        jac_208255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 62), p_208254, 'jac')
        # Processing the call keyword arguments (line 542)
        # Getting the type of 'self' (line 543)
        self_208256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 29), 'self', False)
        # Obtaining the member 'method' of a type (line 543)
        method_208257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 29), self_208256, 'method')
        keyword_208258 = method_208257
        str_208259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 50), 'str', 'jac')
        keyword_208260 = str_208259
        kwargs_208261 = {'x_scale': keyword_208260, 'method': keyword_208258}
        # Getting the type of 'assert_raises' (line 542)
        assert_raises_208247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 542)
        assert_raises_call_result_208262 = invoke(stypy.reporting.localization.Localization(__file__, 542, 8), assert_raises_208247, *[ValueError_208248, least_squares_208249, fun_208251, x0_208253, jac_208255], **kwargs_208261)
        
        
        # ################# End of 'test_x_scale_jac_scale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_scale_jac_scale' in the type store
        # Getting the type of 'stypy_return_type' (line 535)
        stypy_return_type_208263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208263)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_scale_jac_scale'
        return stypy_return_type_208263


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 445, 0, False)
        # Assigning a type to the variable 'self' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseMixin.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SparseMixin' (line 445)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 0), 'SparseMixin', SparseMixin)
# Declaration of the 'LossFunctionMixin' class

class LossFunctionMixin(object, ):

    @norecursion
    def test_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_options'
        module_type_store = module_type_store.open_function_context('test_options', 547, 4, False)
        # Assigning a type to the variable 'self' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LossFunctionMixin.test_options.__dict__.__setitem__('stypy_localization', localization)
        LossFunctionMixin.test_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LossFunctionMixin.test_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        LossFunctionMixin.test_options.__dict__.__setitem__('stypy_function_name', 'LossFunctionMixin.test_options')
        LossFunctionMixin.test_options.__dict__.__setitem__('stypy_param_names_list', [])
        LossFunctionMixin.test_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        LossFunctionMixin.test_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LossFunctionMixin.test_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        LossFunctionMixin.test_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        LossFunctionMixin.test_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LossFunctionMixin.test_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LossFunctionMixin.test_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_options(...)' code ##################

        
        # Getting the type of 'LOSSES' (line 548)
        LOSSES_208264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 20), 'LOSSES')
        # Testing the type of a for loop iterable (line 548)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 548, 8), LOSSES_208264)
        # Getting the type of the for loop variable (line 548)
        for_loop_var_208265 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 548, 8), LOSSES_208264)
        # Assigning a type to the variable 'loss' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'loss', for_loop_var_208265)
        # SSA begins for a for statement (line 548)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 549):
        
        # Call to least_squares(...): (line 549)
        # Processing the call arguments (line 549)
        # Getting the type of 'fun_trivial' (line 549)
        fun_trivial_208267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 32), 'fun_trivial', False)
        float_208268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 45), 'float')
        # Processing the call keyword arguments (line 549)
        # Getting the type of 'loss' (line 549)
        loss_208269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 55), 'loss', False)
        keyword_208270 = loss_208269
        # Getting the type of 'self' (line 550)
        self_208271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 39), 'self', False)
        # Obtaining the member 'method' of a type (line 550)
        method_208272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 39), self_208271, 'method')
        keyword_208273 = method_208272
        kwargs_208274 = {'loss': keyword_208270, 'method': keyword_208273}
        # Getting the type of 'least_squares' (line 549)
        least_squares_208266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 18), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 549)
        least_squares_call_result_208275 = invoke(stypy.reporting.localization.Localization(__file__, 549, 18), least_squares_208266, *[fun_trivial_208267, float_208268], **kwargs_208274)
        
        # Assigning a type to the variable 'res' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'res', least_squares_call_result_208275)
        
        # Call to assert_allclose(...): (line 551)
        # Processing the call arguments (line 551)
        # Getting the type of 'res' (line 551)
        res_208277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 551)
        x_208278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 28), res_208277, 'x')
        int_208279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 35), 'int')
        # Processing the call keyword arguments (line 551)
        float_208280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 43), 'float')
        keyword_208281 = float_208280
        kwargs_208282 = {'atol': keyword_208281}
        # Getting the type of 'assert_allclose' (line 551)
        assert_allclose_208276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 551)
        assert_allclose_call_result_208283 = invoke(stypy.reporting.localization.Localization(__file__, 551, 12), assert_allclose_208276, *[x_208278, int_208279], **kwargs_208282)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_raises(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'ValueError' (line 553)
        ValueError_208285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 553)
        least_squares_208286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 553)
        fun_trivial_208287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 49), 'fun_trivial', False)
        float_208288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 62), 'float')
        # Processing the call keyword arguments (line 553)
        str_208289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 27), 'str', 'hinge')
        keyword_208290 = str_208289
        # Getting the type of 'self' (line 554)
        self_208291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 43), 'self', False)
        # Obtaining the member 'method' of a type (line 554)
        method_208292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 43), self_208291, 'method')
        keyword_208293 = method_208292
        kwargs_208294 = {'loss': keyword_208290, 'method': keyword_208293}
        # Getting the type of 'assert_raises' (line 553)
        assert_raises_208284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 553)
        assert_raises_call_result_208295 = invoke(stypy.reporting.localization.Localization(__file__, 553, 8), assert_raises_208284, *[ValueError_208285, least_squares_208286, fun_trivial_208287, float_208288], **kwargs_208294)
        
        
        # ################# End of 'test_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_options' in the type store
        # Getting the type of 'stypy_return_type' (line 547)
        stypy_return_type_208296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208296)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_options'
        return stypy_return_type_208296


    @norecursion
    def test_fun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_fun'
        module_type_store = module_type_store.open_function_context('test_fun', 556, 4, False)
        # Assigning a type to the variable 'self' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LossFunctionMixin.test_fun.__dict__.__setitem__('stypy_localization', localization)
        LossFunctionMixin.test_fun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LossFunctionMixin.test_fun.__dict__.__setitem__('stypy_type_store', module_type_store)
        LossFunctionMixin.test_fun.__dict__.__setitem__('stypy_function_name', 'LossFunctionMixin.test_fun')
        LossFunctionMixin.test_fun.__dict__.__setitem__('stypy_param_names_list', [])
        LossFunctionMixin.test_fun.__dict__.__setitem__('stypy_varargs_param_name', None)
        LossFunctionMixin.test_fun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LossFunctionMixin.test_fun.__dict__.__setitem__('stypy_call_defaults', defaults)
        LossFunctionMixin.test_fun.__dict__.__setitem__('stypy_call_varargs', varargs)
        LossFunctionMixin.test_fun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LossFunctionMixin.test_fun.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LossFunctionMixin.test_fun', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_fun', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_fun(...)' code ##################

        
        # Getting the type of 'LOSSES' (line 559)
        LOSSES_208297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 20), 'LOSSES')
        # Testing the type of a for loop iterable (line 559)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 559, 8), LOSSES_208297)
        # Getting the type of the for loop variable (line 559)
        for_loop_var_208298 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 559, 8), LOSSES_208297)
        # Assigning a type to the variable 'loss' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'loss', for_loop_var_208298)
        # SSA begins for a for statement (line 559)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 560):
        
        # Call to least_squares(...): (line 560)
        # Processing the call arguments (line 560)
        # Getting the type of 'fun_trivial' (line 560)
        fun_trivial_208300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 32), 'fun_trivial', False)
        float_208301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 45), 'float')
        # Processing the call keyword arguments (line 560)
        # Getting the type of 'loss' (line 560)
        loss_208302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 55), 'loss', False)
        keyword_208303 = loss_208302
        # Getting the type of 'self' (line 561)
        self_208304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 39), 'self', False)
        # Obtaining the member 'method' of a type (line 561)
        method_208305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 39), self_208304, 'method')
        keyword_208306 = method_208305
        kwargs_208307 = {'loss': keyword_208303, 'method': keyword_208306}
        # Getting the type of 'least_squares' (line 560)
        least_squares_208299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 18), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 560)
        least_squares_call_result_208308 = invoke(stypy.reporting.localization.Localization(__file__, 560, 18), least_squares_208299, *[fun_trivial_208300, float_208301], **kwargs_208307)
        
        # Assigning a type to the variable 'res' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 12), 'res', least_squares_call_result_208308)
        
        # Call to assert_equal(...): (line 562)
        # Processing the call arguments (line 562)
        # Getting the type of 'res' (line 562)
        res_208310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 25), 'res', False)
        # Obtaining the member 'fun' of a type (line 562)
        fun_208311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 25), res_208310, 'fun')
        
        # Call to fun_trivial(...): (line 562)
        # Processing the call arguments (line 562)
        # Getting the type of 'res' (line 562)
        res_208313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 46), 'res', False)
        # Obtaining the member 'x' of a type (line 562)
        x_208314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 46), res_208313, 'x')
        # Processing the call keyword arguments (line 562)
        kwargs_208315 = {}
        # Getting the type of 'fun_trivial' (line 562)
        fun_trivial_208312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 34), 'fun_trivial', False)
        # Calling fun_trivial(args, kwargs) (line 562)
        fun_trivial_call_result_208316 = invoke(stypy.reporting.localization.Localization(__file__, 562, 34), fun_trivial_208312, *[x_208314], **kwargs_208315)
        
        # Processing the call keyword arguments (line 562)
        kwargs_208317 = {}
        # Getting the type of 'assert_equal' (line 562)
        assert_equal_208309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 562)
        assert_equal_call_result_208318 = invoke(stypy.reporting.localization.Localization(__file__, 562, 12), assert_equal_208309, *[fun_208311, fun_trivial_call_result_208316], **kwargs_208317)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_fun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_fun' in the type store
        # Getting the type of 'stypy_return_type' (line 556)
        stypy_return_type_208319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208319)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_fun'
        return stypy_return_type_208319


    @norecursion
    def test_grad(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_grad'
        module_type_store = module_type_store.open_function_context('test_grad', 564, 4, False)
        # Assigning a type to the variable 'self' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LossFunctionMixin.test_grad.__dict__.__setitem__('stypy_localization', localization)
        LossFunctionMixin.test_grad.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LossFunctionMixin.test_grad.__dict__.__setitem__('stypy_type_store', module_type_store)
        LossFunctionMixin.test_grad.__dict__.__setitem__('stypy_function_name', 'LossFunctionMixin.test_grad')
        LossFunctionMixin.test_grad.__dict__.__setitem__('stypy_param_names_list', [])
        LossFunctionMixin.test_grad.__dict__.__setitem__('stypy_varargs_param_name', None)
        LossFunctionMixin.test_grad.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LossFunctionMixin.test_grad.__dict__.__setitem__('stypy_call_defaults', defaults)
        LossFunctionMixin.test_grad.__dict__.__setitem__('stypy_call_varargs', varargs)
        LossFunctionMixin.test_grad.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LossFunctionMixin.test_grad.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LossFunctionMixin.test_grad', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_grad', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_grad(...)' code ##################

        
        # Assigning a Call to a Name (line 567):
        
        # Call to array(...): (line 567)
        # Processing the call arguments (line 567)
        
        # Obtaining an instance of the builtin type 'list' (line 567)
        list_208322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 567)
        # Adding element type (line 567)
        float_208323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 21), list_208322, float_208323)
        
        # Processing the call keyword arguments (line 567)
        kwargs_208324 = {}
        # Getting the type of 'np' (line 567)
        np_208320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 567)
        array_208321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 12), np_208320, 'array')
        # Calling array(args, kwargs) (line 567)
        array_call_result_208325 = invoke(stypy.reporting.localization.Localization(__file__, 567, 12), array_208321, *[list_208322], **kwargs_208324)
        
        # Assigning a type to the variable 'x' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'x', array_call_result_208325)
        
        # Assigning a Call to a Name (line 569):
        
        # Call to least_squares(...): (line 569)
        # Processing the call arguments (line 569)
        # Getting the type of 'fun_trivial' (line 569)
        fun_trivial_208327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 569)
        x_208328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 569)
        jac_trivial_208329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 569)
        str_208330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 62), 'str', 'linear')
        keyword_208331 = str_208330
        int_208332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 37), 'int')
        keyword_208333 = int_208332
        # Getting the type of 'self' (line 570)
        self_208334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 570)
        method_208335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 47), self_208334, 'method')
        keyword_208336 = method_208335
        kwargs_208337 = {'loss': keyword_208331, 'max_nfev': keyword_208333, 'method': keyword_208336}
        # Getting the type of 'least_squares' (line 569)
        least_squares_208326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 569)
        least_squares_call_result_208338 = invoke(stypy.reporting.localization.Localization(__file__, 569, 14), least_squares_208326, *[fun_trivial_208327, x_208328, jac_trivial_208329], **kwargs_208337)
        
        # Assigning a type to the variable 'res' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'res', least_squares_call_result_208338)
        
        # Call to assert_equal(...): (line 571)
        # Processing the call arguments (line 571)
        # Getting the type of 'res' (line 571)
        res_208340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 21), 'res', False)
        # Obtaining the member 'grad' of a type (line 571)
        grad_208341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 21), res_208340, 'grad')
        int_208342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 31), 'int')
        # Getting the type of 'x' (line 571)
        x_208343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 35), 'x', False)
        # Applying the binary operator '*' (line 571)
        result_mul_208344 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 31), '*', int_208342, x_208343)
        
        # Getting the type of 'x' (line 571)
        x_208345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 40), 'x', False)
        int_208346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 43), 'int')
        # Applying the binary operator '**' (line 571)
        result_pow_208347 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 40), '**', x_208345, int_208346)
        
        int_208348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 47), 'int')
        # Applying the binary operator '+' (line 571)
        result_add_208349 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 40), '+', result_pow_208347, int_208348)
        
        # Applying the binary operator '*' (line 571)
        result_mul_208350 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 37), '*', result_mul_208344, result_add_208349)
        
        # Processing the call keyword arguments (line 571)
        kwargs_208351 = {}
        # Getting the type of 'assert_equal' (line 571)
        assert_equal_208339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 571)
        assert_equal_call_result_208352 = invoke(stypy.reporting.localization.Localization(__file__, 571, 8), assert_equal_208339, *[grad_208341, result_mul_208350], **kwargs_208351)
        
        
        # Assigning a Call to a Name (line 573):
        
        # Call to least_squares(...): (line 573)
        # Processing the call arguments (line 573)
        # Getting the type of 'fun_trivial' (line 573)
        fun_trivial_208354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 573)
        x_208355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 573)
        jac_trivial_208356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 573)
        str_208357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 62), 'str', 'huber')
        keyword_208358 = str_208357
        int_208359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 37), 'int')
        keyword_208360 = int_208359
        # Getting the type of 'self' (line 574)
        self_208361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 574)
        method_208362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 47), self_208361, 'method')
        keyword_208363 = method_208362
        kwargs_208364 = {'loss': keyword_208358, 'max_nfev': keyword_208360, 'method': keyword_208363}
        # Getting the type of 'least_squares' (line 573)
        least_squares_208353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 573)
        least_squares_call_result_208365 = invoke(stypy.reporting.localization.Localization(__file__, 573, 14), least_squares_208353, *[fun_trivial_208354, x_208355, jac_trivial_208356], **kwargs_208364)
        
        # Assigning a type to the variable 'res' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'res', least_squares_call_result_208365)
        
        # Call to assert_equal(...): (line 575)
        # Processing the call arguments (line 575)
        # Getting the type of 'res' (line 575)
        res_208367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 21), 'res', False)
        # Obtaining the member 'grad' of a type (line 575)
        grad_208368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 21), res_208367, 'grad')
        int_208369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 31), 'int')
        # Getting the type of 'x' (line 575)
        x_208370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 35), 'x', False)
        # Applying the binary operator '*' (line 575)
        result_mul_208371 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 31), '*', int_208369, x_208370)
        
        # Processing the call keyword arguments (line 575)
        kwargs_208372 = {}
        # Getting the type of 'assert_equal' (line 575)
        assert_equal_208366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 575)
        assert_equal_call_result_208373 = invoke(stypy.reporting.localization.Localization(__file__, 575, 8), assert_equal_208366, *[grad_208368, result_mul_208371], **kwargs_208372)
        
        
        # Assigning a Call to a Name (line 577):
        
        # Call to least_squares(...): (line 577)
        # Processing the call arguments (line 577)
        # Getting the type of 'fun_trivial' (line 577)
        fun_trivial_208375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 577)
        x_208376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 577)
        jac_trivial_208377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 577)
        str_208378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 62), 'str', 'soft_l1')
        keyword_208379 = str_208378
        int_208380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 37), 'int')
        keyword_208381 = int_208380
        # Getting the type of 'self' (line 578)
        self_208382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 578)
        method_208383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 47), self_208382, 'method')
        keyword_208384 = method_208383
        kwargs_208385 = {'loss': keyword_208379, 'max_nfev': keyword_208381, 'method': keyword_208384}
        # Getting the type of 'least_squares' (line 577)
        least_squares_208374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 577)
        least_squares_call_result_208386 = invoke(stypy.reporting.localization.Localization(__file__, 577, 14), least_squares_208374, *[fun_trivial_208375, x_208376, jac_trivial_208377], **kwargs_208385)
        
        # Assigning a type to the variable 'res' (line 577)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'res', least_squares_call_result_208386)
        
        # Call to assert_allclose(...): (line 579)
        # Processing the call arguments (line 579)
        # Getting the type of 'res' (line 579)
        res_208388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 24), 'res', False)
        # Obtaining the member 'grad' of a type (line 579)
        grad_208389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 24), res_208388, 'grad')
        int_208390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 24), 'int')
        # Getting the type of 'x' (line 580)
        x_208391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 28), 'x', False)
        # Applying the binary operator '*' (line 580)
        result_mul_208392 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 24), '*', int_208390, x_208391)
        
        # Getting the type of 'x' (line 580)
        x_208393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 33), 'x', False)
        int_208394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 36), 'int')
        # Applying the binary operator '**' (line 580)
        result_pow_208395 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 33), '**', x_208393, int_208394)
        
        int_208396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 40), 'int')
        # Applying the binary operator '+' (line 580)
        result_add_208397 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 33), '+', result_pow_208395, int_208396)
        
        # Applying the binary operator '*' (line 580)
        result_mul_208398 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 30), '*', result_mul_208392, result_add_208397)
        
        int_208399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 46), 'int')
        # Getting the type of 'x' (line 580)
        x_208400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 51), 'x', False)
        int_208401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 54), 'int')
        # Applying the binary operator '**' (line 580)
        result_pow_208402 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 51), '**', x_208400, int_208401)
        
        int_208403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 58), 'int')
        # Applying the binary operator '+' (line 580)
        result_add_208404 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 51), '+', result_pow_208402, int_208403)
        
        int_208405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 62), 'int')
        # Applying the binary operator '**' (line 580)
        result_pow_208406 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 50), '**', result_add_208404, int_208405)
        
        # Applying the binary operator '+' (line 580)
        result_add_208407 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 46), '+', int_208399, result_pow_208406)
        
        float_208408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 66), 'float')
        # Applying the binary operator '**' (line 580)
        result_pow_208409 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 45), '**', result_add_208407, float_208408)
        
        # Applying the binary operator 'div' (line 580)
        result_div_208410 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 43), 'div', result_mul_208398, result_pow_208409)
        
        # Processing the call keyword arguments (line 579)
        kwargs_208411 = {}
        # Getting the type of 'assert_allclose' (line 579)
        assert_allclose_208387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 579)
        assert_allclose_call_result_208412 = invoke(stypy.reporting.localization.Localization(__file__, 579, 8), assert_allclose_208387, *[grad_208389, result_div_208410], **kwargs_208411)
        
        
        # Assigning a Call to a Name (line 582):
        
        # Call to least_squares(...): (line 582)
        # Processing the call arguments (line 582)
        # Getting the type of 'fun_trivial' (line 582)
        fun_trivial_208414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 582)
        x_208415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 582)
        jac_trivial_208416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 582)
        str_208417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 62), 'str', 'cauchy')
        keyword_208418 = str_208417
        int_208419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 37), 'int')
        keyword_208420 = int_208419
        # Getting the type of 'self' (line 583)
        self_208421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 583)
        method_208422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 47), self_208421, 'method')
        keyword_208423 = method_208422
        kwargs_208424 = {'loss': keyword_208418, 'max_nfev': keyword_208420, 'method': keyword_208423}
        # Getting the type of 'least_squares' (line 582)
        least_squares_208413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 582)
        least_squares_call_result_208425 = invoke(stypy.reporting.localization.Localization(__file__, 582, 14), least_squares_208413, *[fun_trivial_208414, x_208415, jac_trivial_208416], **kwargs_208424)
        
        # Assigning a type to the variable 'res' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'res', least_squares_call_result_208425)
        
        # Call to assert_allclose(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 'res' (line 584)
        res_208427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 24), 'res', False)
        # Obtaining the member 'grad' of a type (line 584)
        grad_208428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 24), res_208427, 'grad')
        int_208429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 34), 'int')
        # Getting the type of 'x' (line 584)
        x_208430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 38), 'x', False)
        # Applying the binary operator '*' (line 584)
        result_mul_208431 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 34), '*', int_208429, x_208430)
        
        # Getting the type of 'x' (line 584)
        x_208432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 43), 'x', False)
        int_208433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 46), 'int')
        # Applying the binary operator '**' (line 584)
        result_pow_208434 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 43), '**', x_208432, int_208433)
        
        int_208435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 50), 'int')
        # Applying the binary operator '+' (line 584)
        result_add_208436 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 43), '+', result_pow_208434, int_208435)
        
        # Applying the binary operator '*' (line 584)
        result_mul_208437 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 40), '*', result_mul_208431, result_add_208436)
        
        int_208438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 56), 'int')
        # Getting the type of 'x' (line 584)
        x_208439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 61), 'x', False)
        int_208440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 64), 'int')
        # Applying the binary operator '**' (line 584)
        result_pow_208441 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 61), '**', x_208439, int_208440)
        
        int_208442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 68), 'int')
        # Applying the binary operator '+' (line 584)
        result_add_208443 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 61), '+', result_pow_208441, int_208442)
        
        int_208444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 72), 'int')
        # Applying the binary operator '**' (line 584)
        result_pow_208445 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 60), '**', result_add_208443, int_208444)
        
        # Applying the binary operator '+' (line 584)
        result_add_208446 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 56), '+', int_208438, result_pow_208445)
        
        # Applying the binary operator 'div' (line 584)
        result_div_208447 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 53), 'div', result_mul_208437, result_add_208446)
        
        # Processing the call keyword arguments (line 584)
        kwargs_208448 = {}
        # Getting the type of 'assert_allclose' (line 584)
        assert_allclose_208426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 584)
        assert_allclose_call_result_208449 = invoke(stypy.reporting.localization.Localization(__file__, 584, 8), assert_allclose_208426, *[grad_208428, result_div_208447], **kwargs_208448)
        
        
        # Assigning a Call to a Name (line 586):
        
        # Call to least_squares(...): (line 586)
        # Processing the call arguments (line 586)
        # Getting the type of 'fun_trivial' (line 586)
        fun_trivial_208451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 586)
        x_208452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 586)
        jac_trivial_208453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 586)
        str_208454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 62), 'str', 'arctan')
        keyword_208455 = str_208454
        int_208456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 37), 'int')
        keyword_208457 = int_208456
        # Getting the type of 'self' (line 587)
        self_208458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 587)
        method_208459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 47), self_208458, 'method')
        keyword_208460 = method_208459
        kwargs_208461 = {'loss': keyword_208455, 'max_nfev': keyword_208457, 'method': keyword_208460}
        # Getting the type of 'least_squares' (line 586)
        least_squares_208450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 586)
        least_squares_call_result_208462 = invoke(stypy.reporting.localization.Localization(__file__, 586, 14), least_squares_208450, *[fun_trivial_208451, x_208452, jac_trivial_208453], **kwargs_208461)
        
        # Assigning a type to the variable 'res' (line 586)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'res', least_squares_call_result_208462)
        
        # Call to assert_allclose(...): (line 588)
        # Processing the call arguments (line 588)
        # Getting the type of 'res' (line 588)
        res_208464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 24), 'res', False)
        # Obtaining the member 'grad' of a type (line 588)
        grad_208465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 24), res_208464, 'grad')
        int_208466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 34), 'int')
        # Getting the type of 'x' (line 588)
        x_208467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 38), 'x', False)
        # Applying the binary operator '*' (line 588)
        result_mul_208468 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 34), '*', int_208466, x_208467)
        
        # Getting the type of 'x' (line 588)
        x_208469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 43), 'x', False)
        int_208470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 46), 'int')
        # Applying the binary operator '**' (line 588)
        result_pow_208471 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 43), '**', x_208469, int_208470)
        
        int_208472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 50), 'int')
        # Applying the binary operator '+' (line 588)
        result_add_208473 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 43), '+', result_pow_208471, int_208472)
        
        # Applying the binary operator '*' (line 588)
        result_mul_208474 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 40), '*', result_mul_208468, result_add_208473)
        
        int_208475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 56), 'int')
        # Getting the type of 'x' (line 588)
        x_208476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 61), 'x', False)
        int_208477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 64), 'int')
        # Applying the binary operator '**' (line 588)
        result_pow_208478 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 61), '**', x_208476, int_208477)
        
        int_208479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 68), 'int')
        # Applying the binary operator '+' (line 588)
        result_add_208480 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 61), '+', result_pow_208478, int_208479)
        
        int_208481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 72), 'int')
        # Applying the binary operator '**' (line 588)
        result_pow_208482 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 60), '**', result_add_208480, int_208481)
        
        # Applying the binary operator '+' (line 588)
        result_add_208483 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 56), '+', int_208475, result_pow_208482)
        
        # Applying the binary operator 'div' (line 588)
        result_div_208484 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 53), 'div', result_mul_208474, result_add_208483)
        
        # Processing the call keyword arguments (line 588)
        kwargs_208485 = {}
        # Getting the type of 'assert_allclose' (line 588)
        assert_allclose_208463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 588)
        assert_allclose_call_result_208486 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), assert_allclose_208463, *[grad_208465, result_div_208484], **kwargs_208485)
        
        
        # Assigning a Call to a Name (line 590):
        
        # Call to least_squares(...): (line 590)
        # Processing the call arguments (line 590)
        # Getting the type of 'fun_trivial' (line 590)
        fun_trivial_208488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 590)
        x_208489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 590)
        jac_trivial_208490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 590)
        # Getting the type of 'cubic_soft_l1' (line 590)
        cubic_soft_l1_208491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 62), 'cubic_soft_l1', False)
        keyword_208492 = cubic_soft_l1_208491
        int_208493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 37), 'int')
        keyword_208494 = int_208493
        # Getting the type of 'self' (line 591)
        self_208495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 591)
        method_208496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 47), self_208495, 'method')
        keyword_208497 = method_208496
        kwargs_208498 = {'loss': keyword_208492, 'max_nfev': keyword_208494, 'method': keyword_208497}
        # Getting the type of 'least_squares' (line 590)
        least_squares_208487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 590)
        least_squares_call_result_208499 = invoke(stypy.reporting.localization.Localization(__file__, 590, 14), least_squares_208487, *[fun_trivial_208488, x_208489, jac_trivial_208490], **kwargs_208498)
        
        # Assigning a type to the variable 'res' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'res', least_squares_call_result_208499)
        
        # Call to assert_allclose(...): (line 592)
        # Processing the call arguments (line 592)
        # Getting the type of 'res' (line 592)
        res_208501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 24), 'res', False)
        # Obtaining the member 'grad' of a type (line 592)
        grad_208502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 24), res_208501, 'grad')
        int_208503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 24), 'int')
        # Getting the type of 'x' (line 593)
        x_208504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 28), 'x', False)
        # Applying the binary operator '*' (line 593)
        result_mul_208505 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 24), '*', int_208503, x_208504)
        
        # Getting the type of 'x' (line 593)
        x_208506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 33), 'x', False)
        int_208507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 36), 'int')
        # Applying the binary operator '**' (line 593)
        result_pow_208508 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 33), '**', x_208506, int_208507)
        
        int_208509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 40), 'int')
        # Applying the binary operator '+' (line 593)
        result_add_208510 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 33), '+', result_pow_208508, int_208509)
        
        # Applying the binary operator '*' (line 593)
        result_mul_208511 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 30), '*', result_mul_208505, result_add_208510)
        
        int_208512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 46), 'int')
        # Getting the type of 'x' (line 593)
        x_208513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 51), 'x', False)
        int_208514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 54), 'int')
        # Applying the binary operator '**' (line 593)
        result_pow_208515 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 51), '**', x_208513, int_208514)
        
        int_208516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 58), 'int')
        # Applying the binary operator '+' (line 593)
        result_add_208517 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 51), '+', result_pow_208515, int_208516)
        
        int_208518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 62), 'int')
        # Applying the binary operator '**' (line 593)
        result_pow_208519 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 50), '**', result_add_208517, int_208518)
        
        # Applying the binary operator '+' (line 593)
        result_add_208520 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 46), '+', int_208512, result_pow_208519)
        
        int_208521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 67), 'int')
        int_208522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 69), 'int')
        # Applying the binary operator 'div' (line 593)
        result_div_208523 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 67), 'div', int_208521, int_208522)
        
        # Applying the binary operator '**' (line 593)
        result_pow_208524 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 45), '**', result_add_208520, result_div_208523)
        
        # Applying the binary operator 'div' (line 593)
        result_div_208525 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 43), 'div', result_mul_208511, result_pow_208524)
        
        # Processing the call keyword arguments (line 592)
        kwargs_208526 = {}
        # Getting the type of 'assert_allclose' (line 592)
        assert_allclose_208500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 592)
        assert_allclose_call_result_208527 = invoke(stypy.reporting.localization.Localization(__file__, 592, 8), assert_allclose_208500, *[grad_208502, result_div_208525], **kwargs_208526)
        
        
        # ################# End of 'test_grad(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_grad' in the type store
        # Getting the type of 'stypy_return_type' (line 564)
        stypy_return_type_208528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208528)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_grad'
        return stypy_return_type_208528


    @norecursion
    def test_jac(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_jac'
        module_type_store = module_type_store.open_function_context('test_jac', 595, 4, False)
        # Assigning a type to the variable 'self' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LossFunctionMixin.test_jac.__dict__.__setitem__('stypy_localization', localization)
        LossFunctionMixin.test_jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LossFunctionMixin.test_jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        LossFunctionMixin.test_jac.__dict__.__setitem__('stypy_function_name', 'LossFunctionMixin.test_jac')
        LossFunctionMixin.test_jac.__dict__.__setitem__('stypy_param_names_list', [])
        LossFunctionMixin.test_jac.__dict__.__setitem__('stypy_varargs_param_name', None)
        LossFunctionMixin.test_jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LossFunctionMixin.test_jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        LossFunctionMixin.test_jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        LossFunctionMixin.test_jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LossFunctionMixin.test_jac.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LossFunctionMixin.test_jac', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_jac', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_jac(...)' code ##################

        
        # Assigning a Num to a Name (line 604):
        float_208529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 12), 'float')
        # Assigning a type to the variable 'x' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'x', float_208529)
        
        # Assigning a BinOp to a Name (line 605):
        # Getting the type of 'x' (line 605)
        x_208530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'x')
        int_208531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 15), 'int')
        # Applying the binary operator '**' (line 605)
        result_pow_208532 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 12), '**', x_208530, int_208531)
        
        int_208533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 19), 'int')
        # Applying the binary operator '+' (line 605)
        result_add_208534 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 12), '+', result_pow_208532, int_208533)
        
        # Assigning a type to the variable 'f' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'f', result_add_208534)
        
        # Assigning a Call to a Name (line 607):
        
        # Call to least_squares(...): (line 607)
        # Processing the call arguments (line 607)
        # Getting the type of 'fun_trivial' (line 607)
        fun_trivial_208536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 607)
        x_208537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 607)
        jac_trivial_208538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 607)
        str_208539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 62), 'str', 'linear')
        keyword_208540 = str_208539
        int_208541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 37), 'int')
        keyword_208542 = int_208541
        # Getting the type of 'self' (line 608)
        self_208543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 608)
        method_208544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 47), self_208543, 'method')
        keyword_208545 = method_208544
        kwargs_208546 = {'loss': keyword_208540, 'max_nfev': keyword_208542, 'method': keyword_208545}
        # Getting the type of 'least_squares' (line 607)
        least_squares_208535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 607)
        least_squares_call_result_208547 = invoke(stypy.reporting.localization.Localization(__file__, 607, 14), least_squares_208535, *[fun_trivial_208536, x_208537, jac_trivial_208538], **kwargs_208546)
        
        # Assigning a type to the variable 'res' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'res', least_squares_call_result_208547)
        
        # Call to assert_equal(...): (line 609)
        # Processing the call arguments (line 609)
        # Getting the type of 'res' (line 609)
        res_208549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 21), 'res', False)
        # Obtaining the member 'jac' of a type (line 609)
        jac_208550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 21), res_208549, 'jac')
        int_208551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 30), 'int')
        # Getting the type of 'x' (line 609)
        x_208552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 34), 'x', False)
        # Applying the binary operator '*' (line 609)
        result_mul_208553 = python_operator(stypy.reporting.localization.Localization(__file__, 609, 30), '*', int_208551, x_208552)
        
        # Processing the call keyword arguments (line 609)
        kwargs_208554 = {}
        # Getting the type of 'assert_equal' (line 609)
        assert_equal_208548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 609)
        assert_equal_call_result_208555 = invoke(stypy.reporting.localization.Localization(__file__, 609, 8), assert_equal_208548, *[jac_208550, result_mul_208553], **kwargs_208554)
        
        
        # Assigning a Call to a Name (line 613):
        
        # Call to least_squares(...): (line 613)
        # Processing the call arguments (line 613)
        # Getting the type of 'fun_trivial' (line 613)
        fun_trivial_208557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 613)
        x_208558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 613)
        jac_trivial_208559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 613)
        str_208560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 62), 'str', 'huber')
        keyword_208561 = str_208560
        int_208562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 37), 'int')
        keyword_208563 = int_208562
        # Getting the type of 'self' (line 614)
        self_208564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 614)
        method_208565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 47), self_208564, 'method')
        keyword_208566 = method_208565
        kwargs_208567 = {'loss': keyword_208561, 'max_nfev': keyword_208563, 'method': keyword_208566}
        # Getting the type of 'least_squares' (line 613)
        least_squares_208556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 613)
        least_squares_call_result_208568 = invoke(stypy.reporting.localization.Localization(__file__, 613, 14), least_squares_208556, *[fun_trivial_208557, x_208558, jac_trivial_208559], **kwargs_208567)
        
        # Assigning a type to the variable 'res' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'res', least_squares_call_result_208568)
        
        # Call to assert_equal(...): (line 615)
        # Processing the call arguments (line 615)
        # Getting the type of 'res' (line 615)
        res_208570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 21), 'res', False)
        # Obtaining the member 'jac' of a type (line 615)
        jac_208571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 21), res_208570, 'jac')
        int_208572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 30), 'int')
        # Getting the type of 'x' (line 615)
        x_208573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 34), 'x', False)
        # Applying the binary operator '*' (line 615)
        result_mul_208574 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 30), '*', int_208572, x_208573)
        
        # Getting the type of 'EPS' (line 615)
        EPS_208575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 38), 'EPS', False)
        float_208576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 43), 'float')
        # Applying the binary operator '**' (line 615)
        result_pow_208577 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 38), '**', EPS_208575, float_208576)
        
        # Applying the binary operator '*' (line 615)
        result_mul_208578 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 36), '*', result_mul_208574, result_pow_208577)
        
        # Processing the call keyword arguments (line 615)
        kwargs_208579 = {}
        # Getting the type of 'assert_equal' (line 615)
        assert_equal_208569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 615)
        assert_equal_call_result_208580 = invoke(stypy.reporting.localization.Localization(__file__, 615, 8), assert_equal_208569, *[jac_208571, result_mul_208578], **kwargs_208579)
        
        
        # Assigning a Call to a Name (line 619):
        
        # Call to least_squares(...): (line 619)
        # Processing the call arguments (line 619)
        # Getting the type of 'fun_trivial' (line 619)
        fun_trivial_208582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 619)
        x_208583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 619)
        jac_trivial_208584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 619)
        str_208585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 62), 'str', 'huber')
        keyword_208586 = str_208585
        int_208587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 36), 'int')
        keyword_208588 = int_208587
        int_208589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 49), 'int')
        keyword_208590 = int_208589
        kwargs_208591 = {'loss': keyword_208586, 'f_scale': keyword_208588, 'max_nfev': keyword_208590}
        # Getting the type of 'least_squares' (line 619)
        least_squares_208581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 619)
        least_squares_call_result_208592 = invoke(stypy.reporting.localization.Localization(__file__, 619, 14), least_squares_208581, *[fun_trivial_208582, x_208583, jac_trivial_208584], **kwargs_208591)
        
        # Assigning a type to the variable 'res' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'res', least_squares_call_result_208592)
        
        # Call to assert_equal(...): (line 621)
        # Processing the call arguments (line 621)
        # Getting the type of 'res' (line 621)
        res_208594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 21), 'res', False)
        # Obtaining the member 'jac' of a type (line 621)
        jac_208595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 21), res_208594, 'jac')
        int_208596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 30), 'int')
        # Getting the type of 'x' (line 621)
        x_208597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 34), 'x', False)
        # Applying the binary operator '*' (line 621)
        result_mul_208598 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 30), '*', int_208596, x_208597)
        
        # Processing the call keyword arguments (line 621)
        kwargs_208599 = {}
        # Getting the type of 'assert_equal' (line 621)
        assert_equal_208593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 621)
        assert_equal_call_result_208600 = invoke(stypy.reporting.localization.Localization(__file__, 621, 8), assert_equal_208593, *[jac_208595, result_mul_208598], **kwargs_208599)
        
        
        # Assigning a Call to a Name (line 624):
        
        # Call to least_squares(...): (line 624)
        # Processing the call arguments (line 624)
        # Getting the type of 'fun_trivial' (line 624)
        fun_trivial_208602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 624)
        x_208603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 624)
        jac_trivial_208604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 624)
        str_208605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 62), 'str', 'soft_l1')
        keyword_208606 = str_208605
        int_208607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 37), 'int')
        keyword_208608 = int_208607
        # Getting the type of 'self' (line 625)
        self_208609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 625)
        method_208610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 47), self_208609, 'method')
        keyword_208611 = method_208610
        kwargs_208612 = {'loss': keyword_208606, 'max_nfev': keyword_208608, 'method': keyword_208611}
        # Getting the type of 'least_squares' (line 624)
        least_squares_208601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 624)
        least_squares_call_result_208613 = invoke(stypy.reporting.localization.Localization(__file__, 624, 14), least_squares_208601, *[fun_trivial_208602, x_208603, jac_trivial_208604], **kwargs_208612)
        
        # Assigning a type to the variable 'res' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'res', least_squares_call_result_208613)
        
        # Call to assert_allclose(...): (line 626)
        # Processing the call arguments (line 626)
        # Getting the type of 'res' (line 626)
        res_208615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 24), 'res', False)
        # Obtaining the member 'jac' of a type (line 626)
        jac_208616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 24), res_208615, 'jac')
        int_208617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 33), 'int')
        # Getting the type of 'x' (line 626)
        x_208618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 37), 'x', False)
        # Applying the binary operator '*' (line 626)
        result_mul_208619 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 33), '*', int_208617, x_208618)
        
        int_208620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 42), 'int')
        # Getting the type of 'f' (line 626)
        f_208621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 46), 'f', False)
        int_208622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 49), 'int')
        # Applying the binary operator '**' (line 626)
        result_pow_208623 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 46), '**', f_208621, int_208622)
        
        # Applying the binary operator '+' (line 626)
        result_add_208624 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 42), '+', int_208620, result_pow_208623)
        
        float_208625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 53), 'float')
        # Applying the binary operator '**' (line 626)
        result_pow_208626 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 41), '**', result_add_208624, float_208625)
        
        # Applying the binary operator '*' (line 626)
        result_mul_208627 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 39), '*', result_mul_208619, result_pow_208626)
        
        # Processing the call keyword arguments (line 626)
        kwargs_208628 = {}
        # Getting the type of 'assert_allclose' (line 626)
        assert_allclose_208614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 626)
        assert_allclose_call_result_208629 = invoke(stypy.reporting.localization.Localization(__file__, 626, 8), assert_allclose_208614, *[jac_208616, result_mul_208627], **kwargs_208628)
        
        
        # Assigning a Call to a Name (line 630):
        
        # Call to least_squares(...): (line 630)
        # Processing the call arguments (line 630)
        # Getting the type of 'fun_trivial' (line 630)
        fun_trivial_208631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 630)
        x_208632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 630)
        jac_trivial_208633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 630)
        str_208634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 62), 'str', 'cauchy')
        keyword_208635 = str_208634
        int_208636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 37), 'int')
        keyword_208637 = int_208636
        # Getting the type of 'self' (line 631)
        self_208638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 631)
        method_208639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 47), self_208638, 'method')
        keyword_208640 = method_208639
        kwargs_208641 = {'loss': keyword_208635, 'max_nfev': keyword_208637, 'method': keyword_208640}
        # Getting the type of 'least_squares' (line 630)
        least_squares_208630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 630)
        least_squares_call_result_208642 = invoke(stypy.reporting.localization.Localization(__file__, 630, 14), least_squares_208630, *[fun_trivial_208631, x_208632, jac_trivial_208633], **kwargs_208641)
        
        # Assigning a type to the variable 'res' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 8), 'res', least_squares_call_result_208642)
        
        # Call to assert_allclose(...): (line 632)
        # Processing the call arguments (line 632)
        # Getting the type of 'res' (line 632)
        res_208644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 24), 'res', False)
        # Obtaining the member 'jac' of a type (line 632)
        jac_208645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 24), res_208644, 'jac')
        int_208646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 33), 'int')
        # Getting the type of 'x' (line 632)
        x_208647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 37), 'x', False)
        # Applying the binary operator '*' (line 632)
        result_mul_208648 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 33), '*', int_208646, x_208647)
        
        # Getting the type of 'EPS' (line 632)
        EPS_208649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 41), 'EPS', False)
        float_208650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 46), 'float')
        # Applying the binary operator '**' (line 632)
        result_pow_208651 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 41), '**', EPS_208649, float_208650)
        
        # Applying the binary operator '*' (line 632)
        result_mul_208652 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 39), '*', result_mul_208648, result_pow_208651)
        
        # Processing the call keyword arguments (line 632)
        kwargs_208653 = {}
        # Getting the type of 'assert_allclose' (line 632)
        assert_allclose_208643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 632)
        assert_allclose_call_result_208654 = invoke(stypy.reporting.localization.Localization(__file__, 632, 8), assert_allclose_208643, *[jac_208645, result_mul_208652], **kwargs_208653)
        
        
        # Assigning a Call to a Name (line 635):
        
        # Call to least_squares(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'fun_trivial' (line 635)
        fun_trivial_208656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 635)
        x_208657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 635)
        jac_trivial_208658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 635)
        str_208659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 62), 'str', 'cauchy')
        keyword_208660 = str_208659
        int_208661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 36), 'int')
        keyword_208662 = int_208661
        int_208663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 49), 'int')
        keyword_208664 = int_208663
        # Getting the type of 'self' (line 636)
        self_208665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 59), 'self', False)
        # Obtaining the member 'method' of a type (line 636)
        method_208666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 59), self_208665, 'method')
        keyword_208667 = method_208666
        kwargs_208668 = {'loss': keyword_208660, 'f_scale': keyword_208662, 'max_nfev': keyword_208664, 'method': keyword_208667}
        # Getting the type of 'least_squares' (line 635)
        least_squares_208655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 635)
        least_squares_call_result_208669 = invoke(stypy.reporting.localization.Localization(__file__, 635, 14), least_squares_208655, *[fun_trivial_208656, x_208657, jac_trivial_208658], **kwargs_208668)
        
        # Assigning a type to the variable 'res' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 8), 'res', least_squares_call_result_208669)
        
        # Assigning a BinOp to a Name (line 637):
        # Getting the type of 'f' (line 637)
        f_208670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 13), 'f')
        int_208671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 17), 'int')
        # Applying the binary operator 'div' (line 637)
        result_div_208672 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 13), 'div', f_208670, int_208671)
        
        # Assigning a type to the variable 'fs' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'fs', result_div_208672)
        
        # Call to assert_allclose(...): (line 638)
        # Processing the call arguments (line 638)
        # Getting the type of 'res' (line 638)
        res_208674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 24), 'res', False)
        # Obtaining the member 'jac' of a type (line 638)
        jac_208675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 24), res_208674, 'jac')
        int_208676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 33), 'int')
        # Getting the type of 'x' (line 638)
        x_208677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 37), 'x', False)
        # Applying the binary operator '*' (line 638)
        result_mul_208678 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 33), '*', int_208676, x_208677)
        
        int_208679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 42), 'int')
        # Getting the type of 'fs' (line 638)
        fs_208680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 46), 'fs', False)
        int_208681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 50), 'int')
        # Applying the binary operator '**' (line 638)
        result_pow_208682 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 46), '**', fs_208680, int_208681)
        
        # Applying the binary operator '-' (line 638)
        result_sub_208683 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 42), '-', int_208679, result_pow_208682)
        
        float_208684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 54), 'float')
        # Applying the binary operator '**' (line 638)
        result_pow_208685 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 41), '**', result_sub_208683, float_208684)
        
        # Applying the binary operator '*' (line 638)
        result_mul_208686 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 39), '*', result_mul_208678, result_pow_208685)
        
        int_208687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 61), 'int')
        # Getting the type of 'fs' (line 638)
        fs_208688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 65), 'fs', False)
        int_208689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 69), 'int')
        # Applying the binary operator '**' (line 638)
        result_pow_208690 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 65), '**', fs_208688, int_208689)
        
        # Applying the binary operator '+' (line 638)
        result_add_208691 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 61), '+', int_208687, result_pow_208690)
        
        # Applying the binary operator 'div' (line 638)
        result_div_208692 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 58), 'div', result_mul_208686, result_add_208691)
        
        # Processing the call keyword arguments (line 638)
        kwargs_208693 = {}
        # Getting the type of 'assert_allclose' (line 638)
        assert_allclose_208673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 638)
        assert_allclose_call_result_208694 = invoke(stypy.reporting.localization.Localization(__file__, 638, 8), assert_allclose_208673, *[jac_208675, result_div_208692], **kwargs_208693)
        
        
        # Assigning a Call to a Name (line 641):
        
        # Call to least_squares(...): (line 641)
        # Processing the call arguments (line 641)
        # Getting the type of 'fun_trivial' (line 641)
        fun_trivial_208696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 641)
        x_208697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 641)
        jac_trivial_208698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 641)
        str_208699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 62), 'str', 'arctan')
        keyword_208700 = str_208699
        int_208701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 37), 'int')
        keyword_208702 = int_208701
        # Getting the type of 'self' (line 642)
        self_208703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 642)
        method_208704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 47), self_208703, 'method')
        keyword_208705 = method_208704
        kwargs_208706 = {'loss': keyword_208700, 'max_nfev': keyword_208702, 'method': keyword_208705}
        # Getting the type of 'least_squares' (line 641)
        least_squares_208695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 641)
        least_squares_call_result_208707 = invoke(stypy.reporting.localization.Localization(__file__, 641, 14), least_squares_208695, *[fun_trivial_208696, x_208697, jac_trivial_208698], **kwargs_208706)
        
        # Assigning a type to the variable 'res' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'res', least_squares_call_result_208707)
        
        # Call to assert_allclose(...): (line 643)
        # Processing the call arguments (line 643)
        # Getting the type of 'res' (line 643)
        res_208709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 24), 'res', False)
        # Obtaining the member 'jac' of a type (line 643)
        jac_208710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 24), res_208709, 'jac')
        int_208711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 33), 'int')
        # Getting the type of 'x' (line 643)
        x_208712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 37), 'x', False)
        # Applying the binary operator '*' (line 643)
        result_mul_208713 = python_operator(stypy.reporting.localization.Localization(__file__, 643, 33), '*', int_208711, x_208712)
        
        # Getting the type of 'EPS' (line 643)
        EPS_208714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 41), 'EPS', False)
        float_208715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 46), 'float')
        # Applying the binary operator '**' (line 643)
        result_pow_208716 = python_operator(stypy.reporting.localization.Localization(__file__, 643, 41), '**', EPS_208714, float_208715)
        
        # Applying the binary operator '*' (line 643)
        result_mul_208717 = python_operator(stypy.reporting.localization.Localization(__file__, 643, 39), '*', result_mul_208713, result_pow_208716)
        
        # Processing the call keyword arguments (line 643)
        kwargs_208718 = {}
        # Getting the type of 'assert_allclose' (line 643)
        assert_allclose_208708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 643)
        assert_allclose_call_result_208719 = invoke(stypy.reporting.localization.Localization(__file__, 643, 8), assert_allclose_208708, *[jac_208710, result_mul_208717], **kwargs_208718)
        
        
        # Assigning a Call to a Name (line 646):
        
        # Call to least_squares(...): (line 646)
        # Processing the call arguments (line 646)
        # Getting the type of 'fun_trivial' (line 646)
        fun_trivial_208721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 646)
        x_208722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 646)
        jac_trivial_208723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 646)
        str_208724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 62), 'str', 'arctan')
        keyword_208725 = str_208724
        float_208726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 36), 'float')
        keyword_208727 = float_208726
        int_208728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 51), 'int')
        keyword_208729 = int_208728
        # Getting the type of 'self' (line 647)
        self_208730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 61), 'self', False)
        # Obtaining the member 'method' of a type (line 647)
        method_208731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 61), self_208730, 'method')
        keyword_208732 = method_208731
        kwargs_208733 = {'loss': keyword_208725, 'f_scale': keyword_208727, 'max_nfev': keyword_208729, 'method': keyword_208732}
        # Getting the type of 'least_squares' (line 646)
        least_squares_208720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 646)
        least_squares_call_result_208734 = invoke(stypy.reporting.localization.Localization(__file__, 646, 14), least_squares_208720, *[fun_trivial_208721, x_208722, jac_trivial_208723], **kwargs_208733)
        
        # Assigning a type to the variable 'res' (line 646)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'res', least_squares_call_result_208734)
        
        # Assigning a BinOp to a Name (line 648):
        # Getting the type of 'f' (line 648)
        f_208735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 13), 'f')
        int_208736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 17), 'int')
        # Applying the binary operator 'div' (line 648)
        result_div_208737 = python_operator(stypy.reporting.localization.Localization(__file__, 648, 13), 'div', f_208735, int_208736)
        
        # Assigning a type to the variable 'fs' (line 648)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 8), 'fs', result_div_208737)
        
        # Call to assert_allclose(...): (line 649)
        # Processing the call arguments (line 649)
        # Getting the type of 'res' (line 649)
        res_208739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 24), 'res', False)
        # Obtaining the member 'jac' of a type (line 649)
        jac_208740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 24), res_208739, 'jac')
        int_208741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 33), 'int')
        # Getting the type of 'x' (line 649)
        x_208742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 37), 'x', False)
        # Applying the binary operator '*' (line 649)
        result_mul_208743 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 33), '*', int_208741, x_208742)
        
        int_208744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 42), 'int')
        int_208745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 46), 'int')
        # Getting the type of 'fs' (line 649)
        fs_208746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 50), 'fs', False)
        int_208747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 54), 'int')
        # Applying the binary operator '**' (line 649)
        result_pow_208748 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 50), '**', fs_208746, int_208747)
        
        # Applying the binary operator '*' (line 649)
        result_mul_208749 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 46), '*', int_208745, result_pow_208748)
        
        # Applying the binary operator '-' (line 649)
        result_sub_208750 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 42), '-', int_208744, result_mul_208749)
        
        float_208751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 58), 'float')
        # Applying the binary operator '**' (line 649)
        result_pow_208752 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 41), '**', result_sub_208750, float_208751)
        
        # Applying the binary operator '*' (line 649)
        result_mul_208753 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 39), '*', result_mul_208743, result_pow_208752)
        
        int_208754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 65), 'int')
        # Getting the type of 'fs' (line 649)
        fs_208755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 69), 'fs', False)
        int_208756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 73), 'int')
        # Applying the binary operator '**' (line 649)
        result_pow_208757 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 69), '**', fs_208755, int_208756)
        
        # Applying the binary operator '+' (line 649)
        result_add_208758 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 65), '+', int_208754, result_pow_208757)
        
        # Applying the binary operator 'div' (line 649)
        result_div_208759 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 62), 'div', result_mul_208753, result_add_208758)
        
        # Processing the call keyword arguments (line 649)
        kwargs_208760 = {}
        # Getting the type of 'assert_allclose' (line 649)
        assert_allclose_208738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 649)
        assert_allclose_call_result_208761 = invoke(stypy.reporting.localization.Localization(__file__, 649, 8), assert_allclose_208738, *[jac_208740, result_div_208759], **kwargs_208760)
        
        
        # Assigning a Call to a Name (line 652):
        
        # Call to least_squares(...): (line 652)
        # Processing the call arguments (line 652)
        # Getting the type of 'fun_trivial' (line 652)
        fun_trivial_208763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 652)
        x_208764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 652)
        jac_trivial_208765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 652)
        # Getting the type of 'cubic_soft_l1' (line 652)
        cubic_soft_l1_208766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 62), 'cubic_soft_l1', False)
        keyword_208767 = cubic_soft_l1_208766
        int_208768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 37), 'int')
        keyword_208769 = int_208768
        kwargs_208770 = {'loss': keyword_208767, 'max_nfev': keyword_208769}
        # Getting the type of 'least_squares' (line 652)
        least_squares_208762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 652)
        least_squares_call_result_208771 = invoke(stypy.reporting.localization.Localization(__file__, 652, 14), least_squares_208762, *[fun_trivial_208763, x_208764, jac_trivial_208765], **kwargs_208770)
        
        # Assigning a type to the variable 'res' (line 652)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'res', least_squares_call_result_208771)
        
        # Call to assert_allclose(...): (line 654)
        # Processing the call arguments (line 654)
        # Getting the type of 'res' (line 654)
        res_208773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 24), 'res', False)
        # Obtaining the member 'jac' of a type (line 654)
        jac_208774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 24), res_208773, 'jac')
        int_208775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 33), 'int')
        # Getting the type of 'x' (line 654)
        x_208776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 37), 'x', False)
        # Applying the binary operator '*' (line 654)
        result_mul_208777 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 33), '*', int_208775, x_208776)
        
        # Getting the type of 'EPS' (line 654)
        EPS_208778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 41), 'EPS', False)
        float_208779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 46), 'float')
        # Applying the binary operator '**' (line 654)
        result_pow_208780 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 41), '**', EPS_208778, float_208779)
        
        # Applying the binary operator '*' (line 654)
        result_mul_208781 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 39), '*', result_mul_208777, result_pow_208780)
        
        # Processing the call keyword arguments (line 654)
        kwargs_208782 = {}
        # Getting the type of 'assert_allclose' (line 654)
        assert_allclose_208772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 654)
        assert_allclose_call_result_208783 = invoke(stypy.reporting.localization.Localization(__file__, 654, 8), assert_allclose_208772, *[jac_208774, result_mul_208781], **kwargs_208782)
        
        
        # Assigning a Call to a Name (line 657):
        
        # Call to least_squares(...): (line 657)
        # Processing the call arguments (line 657)
        # Getting the type of 'fun_trivial' (line 657)
        fun_trivial_208785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 28), 'fun_trivial', False)
        # Getting the type of 'x' (line 657)
        x_208786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 41), 'x', False)
        # Getting the type of 'jac_trivial' (line 657)
        jac_trivial_208787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 44), 'jac_trivial', False)
        # Processing the call keyword arguments (line 657)
        # Getting the type of 'cubic_soft_l1' (line 658)
        cubic_soft_l1_208788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 33), 'cubic_soft_l1', False)
        keyword_208789 = cubic_soft_l1_208788
        int_208790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 56), 'int')
        keyword_208791 = int_208790
        int_208792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 68), 'int')
        keyword_208793 = int_208792
        kwargs_208794 = {'loss': keyword_208789, 'f_scale': keyword_208791, 'max_nfev': keyword_208793}
        # Getting the type of 'least_squares' (line 657)
        least_squares_208784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 657)
        least_squares_call_result_208795 = invoke(stypy.reporting.localization.Localization(__file__, 657, 14), least_squares_208784, *[fun_trivial_208785, x_208786, jac_trivial_208787], **kwargs_208794)
        
        # Assigning a type to the variable 'res' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'res', least_squares_call_result_208795)
        
        # Assigning a BinOp to a Name (line 659):
        # Getting the type of 'f' (line 659)
        f_208796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 13), 'f')
        int_208797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 17), 'int')
        # Applying the binary operator 'div' (line 659)
        result_div_208798 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 13), 'div', f_208796, int_208797)
        
        # Assigning a type to the variable 'fs' (line 659)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 8), 'fs', result_div_208798)
        
        # Call to assert_allclose(...): (line 660)
        # Processing the call arguments (line 660)
        # Getting the type of 'res' (line 660)
        res_208800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 24), 'res', False)
        # Obtaining the member 'jac' of a type (line 660)
        jac_208801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 24), res_208800, 'jac')
        int_208802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 24), 'int')
        # Getting the type of 'x' (line 661)
        x_208803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 28), 'x', False)
        # Applying the binary operator '*' (line 661)
        result_mul_208804 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 24), '*', int_208802, x_208803)
        
        int_208805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 33), 'int')
        # Getting the type of 'fs' (line 661)
        fs_208806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 37), 'fs', False)
        int_208807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 41), 'int')
        # Applying the binary operator '**' (line 661)
        result_pow_208808 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 37), '**', fs_208806, int_208807)
        
        int_208809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 45), 'int')
        # Applying the binary operator 'div' (line 661)
        result_div_208810 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 37), 'div', result_pow_208808, int_208809)
        
        # Applying the binary operator '-' (line 661)
        result_sub_208811 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 33), '-', int_208805, result_div_208810)
        
        float_208812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 49), 'float')
        # Applying the binary operator '**' (line 661)
        result_pow_208813 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 32), '**', result_sub_208811, float_208812)
        
        # Applying the binary operator '*' (line 661)
        result_mul_208814 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 30), '*', result_mul_208804, result_pow_208813)
        
        int_208815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 56), 'int')
        # Getting the type of 'fs' (line 661)
        fs_208816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 60), 'fs', False)
        int_208817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 64), 'int')
        # Applying the binary operator '**' (line 661)
        result_pow_208818 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 60), '**', fs_208816, int_208817)
        
        # Applying the binary operator '+' (line 661)
        result_add_208819 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 56), '+', int_208815, result_pow_208818)
        
        int_208820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 69), 'int')
        int_208821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 72), 'int')
        # Applying the binary operator 'div' (line 661)
        result_div_208822 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 69), 'div', int_208820, int_208821)
        
        # Applying the binary operator '**' (line 661)
        result_pow_208823 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 55), '**', result_add_208819, result_div_208822)
        
        # Applying the binary operator '*' (line 661)
        result_mul_208824 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 53), '*', result_mul_208814, result_pow_208823)
        
        # Processing the call keyword arguments (line 660)
        kwargs_208825 = {}
        # Getting the type of 'assert_allclose' (line 660)
        assert_allclose_208799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 660)
        assert_allclose_call_result_208826 = invoke(stypy.reporting.localization.Localization(__file__, 660, 8), assert_allclose_208799, *[jac_208801, result_mul_208824], **kwargs_208825)
        
        
        # ################# End of 'test_jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_jac' in the type store
        # Getting the type of 'stypy_return_type' (line 595)
        stypy_return_type_208827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208827)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_jac'
        return stypy_return_type_208827


    @norecursion
    def test_robustness(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_robustness'
        module_type_store = module_type_store.open_function_context('test_robustness', 663, 4, False)
        # Assigning a type to the variable 'self' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LossFunctionMixin.test_robustness.__dict__.__setitem__('stypy_localization', localization)
        LossFunctionMixin.test_robustness.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LossFunctionMixin.test_robustness.__dict__.__setitem__('stypy_type_store', module_type_store)
        LossFunctionMixin.test_robustness.__dict__.__setitem__('stypy_function_name', 'LossFunctionMixin.test_robustness')
        LossFunctionMixin.test_robustness.__dict__.__setitem__('stypy_param_names_list', [])
        LossFunctionMixin.test_robustness.__dict__.__setitem__('stypy_varargs_param_name', None)
        LossFunctionMixin.test_robustness.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LossFunctionMixin.test_robustness.__dict__.__setitem__('stypy_call_defaults', defaults)
        LossFunctionMixin.test_robustness.__dict__.__setitem__('stypy_call_varargs', varargs)
        LossFunctionMixin.test_robustness.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LossFunctionMixin.test_robustness.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LossFunctionMixin.test_robustness', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_robustness', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_robustness(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 664)
        list_208828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 664)
        # Adding element type (line 664)
        float_208829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 21), list_208828, float_208829)
        # Adding element type (line 664)
        float_208830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 21), list_208828, float_208830)
        
        # Testing the type of a for loop iterable (line 664)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 664, 8), list_208828)
        # Getting the type of the for loop variable (line 664)
        for_loop_var_208831 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 664, 8), list_208828)
        # Assigning a type to the variable 'noise' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 8), 'noise', for_loop_var_208831)
        # SSA begins for a for statement (line 664)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 665):
        
        # Call to ExponentialFittingProblem(...): (line 665)
        # Processing the call arguments (line 665)
        int_208833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 42), 'int')
        float_208834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 45), 'float')
        # Getting the type of 'noise' (line 665)
        noise_208835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 50), 'noise', False)
        # Processing the call keyword arguments (line 665)
        int_208836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 69), 'int')
        keyword_208837 = int_208836
        kwargs_208838 = {'random_seed': keyword_208837}
        # Getting the type of 'ExponentialFittingProblem' (line 665)
        ExponentialFittingProblem_208832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 16), 'ExponentialFittingProblem', False)
        # Calling ExponentialFittingProblem(args, kwargs) (line 665)
        ExponentialFittingProblem_call_result_208839 = invoke(stypy.reporting.localization.Localization(__file__, 665, 16), ExponentialFittingProblem_208832, *[int_208833, float_208834, noise_208835], **kwargs_208838)
        
        # Assigning a type to the variable 'p' (line 665)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'p', ExponentialFittingProblem_call_result_208839)
        
        
        # Obtaining an instance of the builtin type 'list' (line 667)
        list_208840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 667)
        # Adding element type (line 667)
        str_208841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 24), 'str', '2-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 23), list_208840, str_208841)
        # Adding element type (line 667)
        str_208842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 35), 'str', '3-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 23), list_208840, str_208842)
        # Adding element type (line 667)
        str_208843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 46), 'str', 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 23), list_208840, str_208843)
        # Adding element type (line 667)
        # Getting the type of 'p' (line 667)
        p_208844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 52), 'p')
        # Obtaining the member 'jac' of a type (line 667)
        jac_208845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 52), p_208844, 'jac')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 23), list_208840, jac_208845)
        
        # Testing the type of a for loop iterable (line 667)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 667, 12), list_208840)
        # Getting the type of the for loop variable (line 667)
        for_loop_var_208846 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 667, 12), list_208840)
        # Assigning a type to the variable 'jac' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'jac', for_loop_var_208846)
        # SSA begins for a for statement (line 667)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 668):
        
        # Call to least_squares(...): (line 668)
        # Processing the call arguments (line 668)
        # Getting the type of 'p' (line 668)
        p_208848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 40), 'p', False)
        # Obtaining the member 'fun' of a type (line 668)
        fun_208849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 40), p_208848, 'fun')
        # Getting the type of 'p' (line 668)
        p_208850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 47), 'p', False)
        # Obtaining the member 'p0' of a type (line 668)
        p0_208851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 47), p_208850, 'p0')
        # Processing the call keyword arguments (line 668)
        # Getting the type of 'jac' (line 668)
        jac_208852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 57), 'jac', False)
        keyword_208853 = jac_208852
        # Getting the type of 'self' (line 669)
        self_208854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 47), 'self', False)
        # Obtaining the member 'method' of a type (line 669)
        method_208855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 47), self_208854, 'method')
        keyword_208856 = method_208855
        kwargs_208857 = {'jac': keyword_208853, 'method': keyword_208856}
        # Getting the type of 'least_squares' (line 668)
        least_squares_208847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 26), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 668)
        least_squares_call_result_208858 = invoke(stypy.reporting.localization.Localization(__file__, 668, 26), least_squares_208847, *[fun_208849, p0_208851], **kwargs_208857)
        
        # Assigning a type to the variable 'res_lsq' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 16), 'res_lsq', least_squares_call_result_208858)
        
        # Call to assert_allclose(...): (line 670)
        # Processing the call arguments (line 670)
        # Getting the type of 'res_lsq' (line 670)
        res_lsq_208860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 32), 'res_lsq', False)
        # Obtaining the member 'optimality' of a type (line 670)
        optimality_208861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 32), res_lsq_208860, 'optimality')
        int_208862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 52), 'int')
        # Processing the call keyword arguments (line 670)
        float_208863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 60), 'float')
        keyword_208864 = float_208863
        kwargs_208865 = {'atol': keyword_208864}
        # Getting the type of 'assert_allclose' (line 670)
        assert_allclose_208859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 670)
        assert_allclose_call_result_208866 = invoke(stypy.reporting.localization.Localization(__file__, 670, 16), assert_allclose_208859, *[optimality_208861, int_208862], **kwargs_208865)
        
        
        # Getting the type of 'LOSSES' (line 671)
        LOSSES_208867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 28), 'LOSSES')
        # Testing the type of a for loop iterable (line 671)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 671, 16), LOSSES_208867)
        # Getting the type of the for loop variable (line 671)
        for_loop_var_208868 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 671, 16), LOSSES_208867)
        # Assigning a type to the variable 'loss' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'loss', for_loop_var_208868)
        # SSA begins for a for statement (line 671)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'loss' (line 672)
        loss_208869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 23), 'loss')
        str_208870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 31), 'str', 'linear')
        # Applying the binary operator '==' (line 672)
        result_eq_208871 = python_operator(stypy.reporting.localization.Localization(__file__, 672, 23), '==', loss_208869, str_208870)
        
        # Testing the type of an if condition (line 672)
        if_condition_208872 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 672, 20), result_eq_208871)
        # Assigning a type to the variable 'if_condition_208872' (line 672)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 20), 'if_condition_208872', if_condition_208872)
        # SSA begins for if statement (line 672)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 672)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 674):
        
        # Call to least_squares(...): (line 674)
        # Processing the call arguments (line 674)
        # Getting the type of 'p' (line 675)
        p_208874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 24), 'p', False)
        # Obtaining the member 'fun' of a type (line 675)
        fun_208875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 24), p_208874, 'fun')
        # Getting the type of 'p' (line 675)
        p_208876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 31), 'p', False)
        # Obtaining the member 'p0' of a type (line 675)
        p0_208877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 31), p_208876, 'p0')
        # Processing the call keyword arguments (line 674)
        # Getting the type of 'jac' (line 675)
        jac_208878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 41), 'jac', False)
        keyword_208879 = jac_208878
        # Getting the type of 'loss' (line 675)
        loss_208880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 51), 'loss', False)
        keyword_208881 = loss_208880
        # Getting the type of 'noise' (line 675)
        noise_208882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 65), 'noise', False)
        keyword_208883 = noise_208882
        # Getting the type of 'self' (line 676)
        self_208884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 31), 'self', False)
        # Obtaining the member 'method' of a type (line 676)
        method_208885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 31), self_208884, 'method')
        keyword_208886 = method_208885
        kwargs_208887 = {'loss': keyword_208881, 'f_scale': keyword_208883, 'jac': keyword_208879, 'method': keyword_208886}
        # Getting the type of 'least_squares' (line 674)
        least_squares_208873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 33), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 674)
        least_squares_call_result_208888 = invoke(stypy.reporting.localization.Localization(__file__, 674, 33), least_squares_208873, *[fun_208875, p0_208877], **kwargs_208887)
        
        # Assigning a type to the variable 'res_robust' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 20), 'res_robust', least_squares_call_result_208888)
        
        # Call to assert_allclose(...): (line 677)
        # Processing the call arguments (line 677)
        # Getting the type of 'res_robust' (line 677)
        res_robust_208890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 36), 'res_robust', False)
        # Obtaining the member 'optimality' of a type (line 677)
        optimality_208891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 36), res_robust_208890, 'optimality')
        int_208892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 59), 'int')
        # Processing the call keyword arguments (line 677)
        float_208893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 67), 'float')
        keyword_208894 = float_208893
        kwargs_208895 = {'atol': keyword_208894}
        # Getting the type of 'assert_allclose' (line 677)
        assert_allclose_208889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 20), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 677)
        assert_allclose_call_result_208896 = invoke(stypy.reporting.localization.Localization(__file__, 677, 20), assert_allclose_208889, *[optimality_208891, int_208892], **kwargs_208895)
        
        
        # Call to assert_(...): (line 678)
        # Processing the call arguments (line 678)
        
        
        # Call to norm(...): (line 678)
        # Processing the call arguments (line 678)
        # Getting the type of 'res_robust' (line 678)
        res_robust_208899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 33), 'res_robust', False)
        # Obtaining the member 'x' of a type (line 678)
        x_208900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 33), res_robust_208899, 'x')
        # Getting the type of 'p' (line 678)
        p_208901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 48), 'p', False)
        # Obtaining the member 'p_opt' of a type (line 678)
        p_opt_208902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 48), p_208901, 'p_opt')
        # Applying the binary operator '-' (line 678)
        result_sub_208903 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 33), '-', x_208900, p_opt_208902)
        
        # Processing the call keyword arguments (line 678)
        kwargs_208904 = {}
        # Getting the type of 'norm' (line 678)
        norm_208898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 28), 'norm', False)
        # Calling norm(args, kwargs) (line 678)
        norm_call_result_208905 = invoke(stypy.reporting.localization.Localization(__file__, 678, 28), norm_208898, *[result_sub_208903], **kwargs_208904)
        
        
        # Call to norm(...): (line 679)
        # Processing the call arguments (line 679)
        # Getting the type of 'res_lsq' (line 679)
        res_lsq_208907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 33), 'res_lsq', False)
        # Obtaining the member 'x' of a type (line 679)
        x_208908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 33), res_lsq_208907, 'x')
        # Getting the type of 'p' (line 679)
        p_208909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 45), 'p', False)
        # Obtaining the member 'p_opt' of a type (line 679)
        p_opt_208910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 45), p_208909, 'p_opt')
        # Applying the binary operator '-' (line 679)
        result_sub_208911 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 33), '-', x_208908, p_opt_208910)
        
        # Processing the call keyword arguments (line 679)
        kwargs_208912 = {}
        # Getting the type of 'norm' (line 679)
        norm_208906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 28), 'norm', False)
        # Calling norm(args, kwargs) (line 679)
        norm_call_result_208913 = invoke(stypy.reporting.localization.Localization(__file__, 679, 28), norm_208906, *[result_sub_208911], **kwargs_208912)
        
        # Applying the binary operator '<' (line 678)
        result_lt_208914 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 28), '<', norm_call_result_208905, norm_call_result_208913)
        
        # Processing the call keyword arguments (line 678)
        kwargs_208915 = {}
        # Getting the type of 'assert_' (line 678)
        assert__208897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 20), 'assert_', False)
        # Calling assert_(args, kwargs) (line 678)
        assert__call_result_208916 = invoke(stypy.reporting.localization.Localization(__file__, 678, 20), assert__208897, *[result_lt_208914], **kwargs_208915)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_robustness(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_robustness' in the type store
        # Getting the type of 'stypy_return_type' (line 663)
        stypy_return_type_208917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208917)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_robustness'
        return stypy_return_type_208917


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 546, 0, False)
        # Assigning a type to the variable 'self' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LossFunctionMixin.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LossFunctionMixin' (line 546)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 0), 'LossFunctionMixin', LossFunctionMixin)
# Declaration of the 'TestDogbox' class
# Getting the type of 'BaseMixin' (line 682)
BaseMixin_208918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 17), 'BaseMixin')
# Getting the type of 'BoundsMixin' (line 682)
BoundsMixin_208919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 28), 'BoundsMixin')
# Getting the type of 'SparseMixin' (line 682)
SparseMixin_208920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 41), 'SparseMixin')
# Getting the type of 'LossFunctionMixin' (line 682)
LossFunctionMixin_208921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 54), 'LossFunctionMixin')

class TestDogbox(BaseMixin_208918, BoundsMixin_208919, SparseMixin_208920, LossFunctionMixin_208921, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 682, 0, False)
        # Assigning a type to the variable 'self' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDogbox.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDogbox' (line 682)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 0), 'TestDogbox', TestDogbox)

# Assigning a Str to a Name (line 683):
str_208922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 13), 'str', 'dogbox')
# Getting the type of 'TestDogbox'
TestDogbox_208923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDogbox')
# Setting the type of the member 'method' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDogbox_208923, 'method', str_208922)
# Declaration of the 'TestTRF' class
# Getting the type of 'BaseMixin' (line 686)
BaseMixin_208924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 14), 'BaseMixin')
# Getting the type of 'BoundsMixin' (line 686)
BoundsMixin_208925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 25), 'BoundsMixin')
# Getting the type of 'SparseMixin' (line 686)
SparseMixin_208926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 38), 'SparseMixin')
# Getting the type of 'LossFunctionMixin' (line 686)
LossFunctionMixin_208927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 51), 'LossFunctionMixin')

class TestTRF(BaseMixin_208924, BoundsMixin_208925, SparseMixin_208926, LossFunctionMixin_208927, ):

    @norecursion
    def test_lsmr_regularization(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_lsmr_regularization'
        module_type_store = module_type_store.open_function_context('test_lsmr_regularization', 689, 4, False)
        # Assigning a type to the variable 'self' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTRF.test_lsmr_regularization.__dict__.__setitem__('stypy_localization', localization)
        TestTRF.test_lsmr_regularization.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTRF.test_lsmr_regularization.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTRF.test_lsmr_regularization.__dict__.__setitem__('stypy_function_name', 'TestTRF.test_lsmr_regularization')
        TestTRF.test_lsmr_regularization.__dict__.__setitem__('stypy_param_names_list', [])
        TestTRF.test_lsmr_regularization.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTRF.test_lsmr_regularization.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTRF.test_lsmr_regularization.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTRF.test_lsmr_regularization.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTRF.test_lsmr_regularization.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTRF.test_lsmr_regularization.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTRF.test_lsmr_regularization', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_lsmr_regularization', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_lsmr_regularization(...)' code ##################

        
        # Assigning a Call to a Name (line 690):
        
        # Call to BroydenTridiagonal(...): (line 690)
        # Processing the call keyword arguments (line 690)
        kwargs_208929 = {}
        # Getting the type of 'BroydenTridiagonal' (line 690)
        BroydenTridiagonal_208928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 12), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 690)
        BroydenTridiagonal_call_result_208930 = invoke(stypy.reporting.localization.Localization(__file__, 690, 12), BroydenTridiagonal_208928, *[], **kwargs_208929)
        
        # Assigning a type to the variable 'p' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'p', BroydenTridiagonal_call_result_208930)
        
        
        # Obtaining an instance of the builtin type 'list' (line 691)
        list_208931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 691)
        # Adding element type (line 691)
        # Getting the type of 'True' (line 691)
        True_208932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 27), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 691, 26), list_208931, True_208932)
        # Adding element type (line 691)
        # Getting the type of 'False' (line 691)
        False_208933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 33), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 691, 26), list_208931, False_208933)
        
        # Testing the type of a for loop iterable (line 691)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 691, 8), list_208931)
        # Getting the type of the for loop variable (line 691)
        for_loop_var_208934 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 691, 8), list_208931)
        # Assigning a type to the variable 'regularize' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 8), 'regularize', for_loop_var_208934)
        # SSA begins for a for statement (line 691)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 692):
        
        # Call to least_squares(...): (line 692)
        # Processing the call arguments (line 692)
        # Getting the type of 'p' (line 692)
        p_208936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 32), 'p', False)
        # Obtaining the member 'fun' of a type (line 692)
        fun_208937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 32), p_208936, 'fun')
        # Getting the type of 'p' (line 692)
        p_208938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 39), 'p', False)
        # Obtaining the member 'x0' of a type (line 692)
        x0_208939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 39), p_208938, 'x0')
        # Getting the type of 'p' (line 692)
        p_208940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 45), 'p', False)
        # Obtaining the member 'jac' of a type (line 692)
        jac_208941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 45), p_208940, 'jac')
        # Processing the call keyword arguments (line 692)
        str_208942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 59), 'str', 'trf')
        keyword_208943 = str_208942
        
        # Obtaining an instance of the builtin type 'dict' (line 693)
        dict_208944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 43), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 693)
        # Adding element type (key, value) (line 693)
        str_208945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 44), 'str', 'regularize')
        # Getting the type of 'regularize' (line 693)
        regularize_208946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 58), 'regularize', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 693, 43), dict_208944, (str_208945, regularize_208946))
        
        keyword_208947 = dict_208944
        kwargs_208948 = {'tr_options': keyword_208947, 'method': keyword_208943}
        # Getting the type of 'least_squares' (line 692)
        least_squares_208935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 18), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 692)
        least_squares_call_result_208949 = invoke(stypy.reporting.localization.Localization(__file__, 692, 18), least_squares_208935, *[fun_208937, x0_208939, jac_208941], **kwargs_208948)
        
        # Assigning a type to the variable 'res' (line 692)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 12), 'res', least_squares_call_result_208949)
        
        # Call to assert_allclose(...): (line 694)
        # Processing the call arguments (line 694)
        # Getting the type of 'res' (line 694)
        res_208951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 28), 'res', False)
        # Obtaining the member 'cost' of a type (line 694)
        cost_208952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 28), res_208951, 'cost')
        int_208953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 38), 'int')
        # Processing the call keyword arguments (line 694)
        float_208954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 46), 'float')
        keyword_208955 = float_208954
        kwargs_208956 = {'atol': keyword_208955}
        # Getting the type of 'assert_allclose' (line 694)
        assert_allclose_208950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 694)
        assert_allclose_call_result_208957 = invoke(stypy.reporting.localization.Localization(__file__, 694, 12), assert_allclose_208950, *[cost_208952, int_208953], **kwargs_208956)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_lsmr_regularization(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_lsmr_regularization' in the type store
        # Getting the type of 'stypy_return_type' (line 689)
        stypy_return_type_208958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208958)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_lsmr_regularization'
        return stypy_return_type_208958


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 686, 0, False)
        # Assigning a type to the variable 'self' (line 687)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTRF.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTRF' (line 686)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 0), 'TestTRF', TestTRF)

# Assigning a Str to a Name (line 687):
str_208959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 13), 'str', 'trf')
# Getting the type of 'TestTRF'
TestTRF_208960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestTRF')
# Setting the type of the member 'method' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestTRF_208960, 'method', str_208959)
# Declaration of the 'TestLM' class
# Getting the type of 'BaseMixin' (line 697)
BaseMixin_208961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 13), 'BaseMixin')

class TestLM(BaseMixin_208961, ):

    @norecursion
    def test_bounds_not_supported(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bounds_not_supported'
        module_type_store = module_type_store.open_function_context('test_bounds_not_supported', 700, 4, False)
        # Assigning a type to the variable 'self' (line 701)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLM.test_bounds_not_supported.__dict__.__setitem__('stypy_localization', localization)
        TestLM.test_bounds_not_supported.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLM.test_bounds_not_supported.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLM.test_bounds_not_supported.__dict__.__setitem__('stypy_function_name', 'TestLM.test_bounds_not_supported')
        TestLM.test_bounds_not_supported.__dict__.__setitem__('stypy_param_names_list', [])
        TestLM.test_bounds_not_supported.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLM.test_bounds_not_supported.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLM.test_bounds_not_supported.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLM.test_bounds_not_supported.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLM.test_bounds_not_supported.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLM.test_bounds_not_supported.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLM.test_bounds_not_supported', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bounds_not_supported', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bounds_not_supported(...)' code ##################

        
        # Call to assert_raises(...): (line 701)
        # Processing the call arguments (line 701)
        # Getting the type of 'ValueError' (line 701)
        ValueError_208963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 701)
        least_squares_208964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 701)
        fun_trivial_208965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 49), 'fun_trivial', False)
        float_208966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 22), 'float')
        # Processing the call keyword arguments (line 701)
        
        # Obtaining an instance of the builtin type 'tuple' (line 702)
        tuple_208967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 702)
        # Adding element type (line 702)
        float_208968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 35), tuple_208967, float_208968)
        # Adding element type (line 702)
        float_208969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 35), tuple_208967, float_208969)
        
        keyword_208970 = tuple_208967
        str_208971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 54), 'str', 'lm')
        keyword_208972 = str_208971
        kwargs_208973 = {'bounds': keyword_208970, 'method': keyword_208972}
        # Getting the type of 'assert_raises' (line 701)
        assert_raises_208962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 701)
        assert_raises_call_result_208974 = invoke(stypy.reporting.localization.Localization(__file__, 701, 8), assert_raises_208962, *[ValueError_208963, least_squares_208964, fun_trivial_208965, float_208966], **kwargs_208973)
        
        
        # ################# End of 'test_bounds_not_supported(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bounds_not_supported' in the type store
        # Getting the type of 'stypy_return_type' (line 700)
        stypy_return_type_208975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208975)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bounds_not_supported'
        return stypy_return_type_208975


    @norecursion
    def test_m_less_n_not_supported(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_m_less_n_not_supported'
        module_type_store = module_type_store.open_function_context('test_m_less_n_not_supported', 704, 4, False)
        # Assigning a type to the variable 'self' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLM.test_m_less_n_not_supported.__dict__.__setitem__('stypy_localization', localization)
        TestLM.test_m_less_n_not_supported.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLM.test_m_less_n_not_supported.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLM.test_m_less_n_not_supported.__dict__.__setitem__('stypy_function_name', 'TestLM.test_m_less_n_not_supported')
        TestLM.test_m_less_n_not_supported.__dict__.__setitem__('stypy_param_names_list', [])
        TestLM.test_m_less_n_not_supported.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLM.test_m_less_n_not_supported.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLM.test_m_less_n_not_supported.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLM.test_m_less_n_not_supported.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLM.test_m_less_n_not_supported.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLM.test_m_less_n_not_supported.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLM.test_m_less_n_not_supported', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_m_less_n_not_supported', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_m_less_n_not_supported(...)' code ##################

        
        # Assigning a List to a Name (line 705):
        
        # Obtaining an instance of the builtin type 'list' (line 705)
        list_208976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 705)
        # Adding element type (line 705)
        int_208977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 13), list_208976, int_208977)
        # Adding element type (line 705)
        int_208978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 13), list_208976, int_208978)
        
        # Assigning a type to the variable 'x0' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'x0', list_208976)
        
        # Call to assert_raises(...): (line 706)
        # Processing the call arguments (line 706)
        # Getting the type of 'ValueError' (line 706)
        ValueError_208980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 706)
        least_squares_208981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 34), 'least_squares', False)
        # Getting the type of 'fun_rosenbrock_cropped' (line 706)
        fun_rosenbrock_cropped_208982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 49), 'fun_rosenbrock_cropped', False)
        # Getting the type of 'x0' (line 706)
        x0_208983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 73), 'x0', False)
        # Processing the call keyword arguments (line 706)
        str_208984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 29), 'str', 'lm')
        keyword_208985 = str_208984
        kwargs_208986 = {'method': keyword_208985}
        # Getting the type of 'assert_raises' (line 706)
        assert_raises_208979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 706)
        assert_raises_call_result_208987 = invoke(stypy.reporting.localization.Localization(__file__, 706, 8), assert_raises_208979, *[ValueError_208980, least_squares_208981, fun_rosenbrock_cropped_208982, x0_208983], **kwargs_208986)
        
        
        # ################# End of 'test_m_less_n_not_supported(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_m_less_n_not_supported' in the type store
        # Getting the type of 'stypy_return_type' (line 704)
        stypy_return_type_208988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208988)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_m_less_n_not_supported'
        return stypy_return_type_208988


    @norecursion
    def test_sparse_not_supported(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sparse_not_supported'
        module_type_store = module_type_store.open_function_context('test_sparse_not_supported', 709, 4, False)
        # Assigning a type to the variable 'self' (line 710)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLM.test_sparse_not_supported.__dict__.__setitem__('stypy_localization', localization)
        TestLM.test_sparse_not_supported.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLM.test_sparse_not_supported.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLM.test_sparse_not_supported.__dict__.__setitem__('stypy_function_name', 'TestLM.test_sparse_not_supported')
        TestLM.test_sparse_not_supported.__dict__.__setitem__('stypy_param_names_list', [])
        TestLM.test_sparse_not_supported.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLM.test_sparse_not_supported.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLM.test_sparse_not_supported.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLM.test_sparse_not_supported.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLM.test_sparse_not_supported.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLM.test_sparse_not_supported.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLM.test_sparse_not_supported', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sparse_not_supported', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sparse_not_supported(...)' code ##################

        
        # Assigning a Call to a Name (line 710):
        
        # Call to BroydenTridiagonal(...): (line 710)
        # Processing the call keyword arguments (line 710)
        kwargs_208990 = {}
        # Getting the type of 'BroydenTridiagonal' (line 710)
        BroydenTridiagonal_208989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 12), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 710)
        BroydenTridiagonal_call_result_208991 = invoke(stypy.reporting.localization.Localization(__file__, 710, 12), BroydenTridiagonal_208989, *[], **kwargs_208990)
        
        # Assigning a type to the variable 'p' (line 710)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 8), 'p', BroydenTridiagonal_call_result_208991)
        
        # Call to assert_raises(...): (line 711)
        # Processing the call arguments (line 711)
        # Getting the type of 'ValueError' (line 711)
        ValueError_208993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 711)
        least_squares_208994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 34), 'least_squares', False)
        # Getting the type of 'p' (line 711)
        p_208995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 49), 'p', False)
        # Obtaining the member 'fun' of a type (line 711)
        fun_208996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 49), p_208995, 'fun')
        # Getting the type of 'p' (line 711)
        p_208997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 56), 'p', False)
        # Obtaining the member 'x0' of a type (line 711)
        x0_208998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 56), p_208997, 'x0')
        # Getting the type of 'p' (line 711)
        p_208999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 62), 'p', False)
        # Obtaining the member 'jac' of a type (line 711)
        jac_209000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 62), p_208999, 'jac')
        # Processing the call keyword arguments (line 711)
        str_209001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 29), 'str', 'lm')
        keyword_209002 = str_209001
        kwargs_209003 = {'method': keyword_209002}
        # Getting the type of 'assert_raises' (line 711)
        assert_raises_208992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 711)
        assert_raises_call_result_209004 = invoke(stypy.reporting.localization.Localization(__file__, 711, 8), assert_raises_208992, *[ValueError_208993, least_squares_208994, fun_208996, x0_208998, jac_209000], **kwargs_209003)
        
        
        # ################# End of 'test_sparse_not_supported(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sparse_not_supported' in the type store
        # Getting the type of 'stypy_return_type' (line 709)
        stypy_return_type_209005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209005)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sparse_not_supported'
        return stypy_return_type_209005


    @norecursion
    def test_jac_sparsity_not_supported(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_jac_sparsity_not_supported'
        module_type_store = module_type_store.open_function_context('test_jac_sparsity_not_supported', 714, 4, False)
        # Assigning a type to the variable 'self' (line 715)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLM.test_jac_sparsity_not_supported.__dict__.__setitem__('stypy_localization', localization)
        TestLM.test_jac_sparsity_not_supported.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLM.test_jac_sparsity_not_supported.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLM.test_jac_sparsity_not_supported.__dict__.__setitem__('stypy_function_name', 'TestLM.test_jac_sparsity_not_supported')
        TestLM.test_jac_sparsity_not_supported.__dict__.__setitem__('stypy_param_names_list', [])
        TestLM.test_jac_sparsity_not_supported.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLM.test_jac_sparsity_not_supported.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLM.test_jac_sparsity_not_supported.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLM.test_jac_sparsity_not_supported.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLM.test_jac_sparsity_not_supported.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLM.test_jac_sparsity_not_supported.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLM.test_jac_sparsity_not_supported', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_jac_sparsity_not_supported', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_jac_sparsity_not_supported(...)' code ##################

        
        # Call to assert_raises(...): (line 715)
        # Processing the call arguments (line 715)
        # Getting the type of 'ValueError' (line 715)
        ValueError_209007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 715)
        least_squares_209008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 715)
        fun_trivial_209009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 49), 'fun_trivial', False)
        float_209010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 62), 'float')
        # Processing the call keyword arguments (line 715)
        
        # Obtaining an instance of the builtin type 'list' (line 716)
        list_209011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 716)
        # Adding element type (line 716)
        int_209012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 716, 35), list_209011, int_209012)
        
        keyword_209013 = list_209011
        str_209014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 47), 'str', 'lm')
        keyword_209015 = str_209014
        kwargs_209016 = {'jac_sparsity': keyword_209013, 'method': keyword_209015}
        # Getting the type of 'assert_raises' (line 715)
        assert_raises_209006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 715)
        assert_raises_call_result_209017 = invoke(stypy.reporting.localization.Localization(__file__, 715, 8), assert_raises_209006, *[ValueError_209007, least_squares_209008, fun_trivial_209009, float_209010], **kwargs_209016)
        
        
        # ################# End of 'test_jac_sparsity_not_supported(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_jac_sparsity_not_supported' in the type store
        # Getting the type of 'stypy_return_type' (line 714)
        stypy_return_type_209018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209018)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_jac_sparsity_not_supported'
        return stypy_return_type_209018


    @norecursion
    def test_LinearOperator_not_supported(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_LinearOperator_not_supported'
        module_type_store = module_type_store.open_function_context('test_LinearOperator_not_supported', 718, 4, False)
        # Assigning a type to the variable 'self' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLM.test_LinearOperator_not_supported.__dict__.__setitem__('stypy_localization', localization)
        TestLM.test_LinearOperator_not_supported.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLM.test_LinearOperator_not_supported.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLM.test_LinearOperator_not_supported.__dict__.__setitem__('stypy_function_name', 'TestLM.test_LinearOperator_not_supported')
        TestLM.test_LinearOperator_not_supported.__dict__.__setitem__('stypy_param_names_list', [])
        TestLM.test_LinearOperator_not_supported.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLM.test_LinearOperator_not_supported.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLM.test_LinearOperator_not_supported.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLM.test_LinearOperator_not_supported.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLM.test_LinearOperator_not_supported.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLM.test_LinearOperator_not_supported.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLM.test_LinearOperator_not_supported', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_LinearOperator_not_supported', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_LinearOperator_not_supported(...)' code ##################

        
        # Assigning a Call to a Name (line 719):
        
        # Call to BroydenTridiagonal(...): (line 719)
        # Processing the call keyword arguments (line 719)
        str_209020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 36), 'str', 'operator')
        keyword_209021 = str_209020
        kwargs_209022 = {'mode': keyword_209021}
        # Getting the type of 'BroydenTridiagonal' (line 719)
        BroydenTridiagonal_209019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'BroydenTridiagonal', False)
        # Calling BroydenTridiagonal(args, kwargs) (line 719)
        BroydenTridiagonal_call_result_209023 = invoke(stypy.reporting.localization.Localization(__file__, 719, 12), BroydenTridiagonal_209019, *[], **kwargs_209022)
        
        # Assigning a type to the variable 'p' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'p', BroydenTridiagonal_call_result_209023)
        
        # Call to assert_raises(...): (line 720)
        # Processing the call arguments (line 720)
        # Getting the type of 'ValueError' (line 720)
        ValueError_209025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 720)
        least_squares_209026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 34), 'least_squares', False)
        # Getting the type of 'p' (line 720)
        p_209027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 49), 'p', False)
        # Obtaining the member 'fun' of a type (line 720)
        fun_209028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 49), p_209027, 'fun')
        # Getting the type of 'p' (line 720)
        p_209029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 56), 'p', False)
        # Obtaining the member 'x0' of a type (line 720)
        x0_209030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 56), p_209029, 'x0')
        # Getting the type of 'p' (line 720)
        p_209031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 62), 'p', False)
        # Obtaining the member 'jac' of a type (line 720)
        jac_209032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 62), p_209031, 'jac')
        # Processing the call keyword arguments (line 720)
        str_209033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 29), 'str', 'lm')
        keyword_209034 = str_209033
        kwargs_209035 = {'method': keyword_209034}
        # Getting the type of 'assert_raises' (line 720)
        assert_raises_209024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 720)
        assert_raises_call_result_209036 = invoke(stypy.reporting.localization.Localization(__file__, 720, 8), assert_raises_209024, *[ValueError_209025, least_squares_209026, fun_209028, x0_209030, jac_209032], **kwargs_209035)
        
        
        # ################# End of 'test_LinearOperator_not_supported(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_LinearOperator_not_supported' in the type store
        # Getting the type of 'stypy_return_type' (line 718)
        stypy_return_type_209037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209037)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_LinearOperator_not_supported'
        return stypy_return_type_209037


    @norecursion
    def test_loss(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_loss'
        module_type_store = module_type_store.open_function_context('test_loss', 723, 4, False)
        # Assigning a type to the variable 'self' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLM.test_loss.__dict__.__setitem__('stypy_localization', localization)
        TestLM.test_loss.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLM.test_loss.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLM.test_loss.__dict__.__setitem__('stypy_function_name', 'TestLM.test_loss')
        TestLM.test_loss.__dict__.__setitem__('stypy_param_names_list', [])
        TestLM.test_loss.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLM.test_loss.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLM.test_loss.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLM.test_loss.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLM.test_loss.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLM.test_loss.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLM.test_loss', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_loss', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_loss(...)' code ##################

        
        # Assigning a Call to a Name (line 724):
        
        # Call to least_squares(...): (line 724)
        # Processing the call arguments (line 724)
        # Getting the type of 'fun_trivial' (line 724)
        fun_trivial_209039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 28), 'fun_trivial', False)
        float_209040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 41), 'float')
        # Processing the call keyword arguments (line 724)
        str_209041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 51), 'str', 'linear')
        keyword_209042 = str_209041
        str_209043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 68), 'str', 'lm')
        keyword_209044 = str_209043
        kwargs_209045 = {'loss': keyword_209042, 'method': keyword_209044}
        # Getting the type of 'least_squares' (line 724)
        least_squares_209038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 14), 'least_squares', False)
        # Calling least_squares(args, kwargs) (line 724)
        least_squares_call_result_209046 = invoke(stypy.reporting.localization.Localization(__file__, 724, 14), least_squares_209038, *[fun_trivial_209039, float_209040], **kwargs_209045)
        
        # Assigning a type to the variable 'res' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'res', least_squares_call_result_209046)
        
        # Call to assert_allclose(...): (line 725)
        # Processing the call arguments (line 725)
        # Getting the type of 'res' (line 725)
        res_209048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 725)
        x_209049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 24), res_209048, 'x')
        float_209050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 31), 'float')
        # Processing the call keyword arguments (line 725)
        float_209051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 41), 'float')
        keyword_209052 = float_209051
        kwargs_209053 = {'atol': keyword_209052}
        # Getting the type of 'assert_allclose' (line 725)
        assert_allclose_209047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 725)
        assert_allclose_call_result_209054 = invoke(stypy.reporting.localization.Localization(__file__, 725, 8), assert_allclose_209047, *[x_209049, float_209050], **kwargs_209053)
        
        
        # Call to assert_raises(...): (line 727)
        # Processing the call arguments (line 727)
        # Getting the type of 'ValueError' (line 727)
        ValueError_209056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 22), 'ValueError', False)
        # Getting the type of 'least_squares' (line 727)
        least_squares_209057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 34), 'least_squares', False)
        # Getting the type of 'fun_trivial' (line 727)
        fun_trivial_209058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 49), 'fun_trivial', False)
        float_209059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 62), 'float')
        # Processing the call keyword arguments (line 727)
        str_209060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 29), 'str', 'lm')
        keyword_209061 = str_209060
        str_209062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 40), 'str', 'huber')
        keyword_209063 = str_209062
        kwargs_209064 = {'loss': keyword_209063, 'method': keyword_209061}
        # Getting the type of 'assert_raises' (line 727)
        assert_raises_209055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 727)
        assert_raises_call_result_209065 = invoke(stypy.reporting.localization.Localization(__file__, 727, 8), assert_raises_209055, *[ValueError_209056, least_squares_209057, fun_trivial_209058, float_209059], **kwargs_209064)
        
        
        # ################# End of 'test_loss(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_loss' in the type store
        # Getting the type of 'stypy_return_type' (line 723)
        stypy_return_type_209066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209066)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_loss'
        return stypy_return_type_209066


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 697, 0, False)
        # Assigning a type to the variable 'self' (line 698)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLM.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLM' (line 697)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 0), 'TestLM', TestLM)

# Assigning a Str to a Name (line 698):
str_209067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 13), 'str', 'lm')
# Getting the type of 'TestLM'
TestLM_209068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestLM')
# Setting the type of the member 'method' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestLM_209068, 'method', str_209067)

@norecursion
def test_basic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_basic'
    module_type_store = module_type_store.open_function_context('test_basic', 731, 0, False)
    
    # Passed parameters checking function
    test_basic.stypy_localization = localization
    test_basic.stypy_type_of_self = None
    test_basic.stypy_type_store = module_type_store
    test_basic.stypy_function_name = 'test_basic'
    test_basic.stypy_param_names_list = []
    test_basic.stypy_varargs_param_name = None
    test_basic.stypy_kwargs_param_name = None
    test_basic.stypy_call_defaults = defaults
    test_basic.stypy_call_varargs = varargs
    test_basic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_basic', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_basic', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_basic(...)' code ##################

    
    # Assigning a Call to a Name (line 733):
    
    # Call to least_squares(...): (line 733)
    # Processing the call arguments (line 733)
    # Getting the type of 'fun_trivial' (line 733)
    fun_trivial_209070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 24), 'fun_trivial', False)
    float_209071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 37), 'float')
    # Processing the call keyword arguments (line 733)
    kwargs_209072 = {}
    # Getting the type of 'least_squares' (line 733)
    least_squares_209069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 10), 'least_squares', False)
    # Calling least_squares(args, kwargs) (line 733)
    least_squares_call_result_209073 = invoke(stypy.reporting.localization.Localization(__file__, 733, 10), least_squares_209069, *[fun_trivial_209070, float_209071], **kwargs_209072)
    
    # Assigning a type to the variable 'res' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), 'res', least_squares_call_result_209073)
    
    # Call to assert_allclose(...): (line 734)
    # Processing the call arguments (line 734)
    # Getting the type of 'res' (line 734)
    res_209075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 20), 'res', False)
    # Obtaining the member 'x' of a type (line 734)
    x_209076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 20), res_209075, 'x')
    int_209077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 27), 'int')
    # Processing the call keyword arguments (line 734)
    float_209078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 35), 'float')
    keyword_209079 = float_209078
    kwargs_209080 = {'atol': keyword_209079}
    # Getting the type of 'assert_allclose' (line 734)
    assert_allclose_209074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 734)
    assert_allclose_call_result_209081 = invoke(stypy.reporting.localization.Localization(__file__, 734, 4), assert_allclose_209074, *[x_209076, int_209077], **kwargs_209080)
    
    
    # ################# End of 'test_basic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_basic' in the type store
    # Getting the type of 'stypy_return_type' (line 731)
    stypy_return_type_209082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_209082)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_basic'
    return stypy_return_type_209082

# Assigning a type to the variable 'test_basic' (line 731)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 0), 'test_basic', test_basic)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
