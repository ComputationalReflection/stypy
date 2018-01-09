
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import sys
4: 
5: try:
6:     from StringIO import StringIO
7: except ImportError:
8:     from io import StringIO
9: 
10: import numpy as np
11: from numpy.testing import (assert_, assert_array_equal, assert_allclose,
12:                            assert_equal)
13: from pytest import raises as assert_raises
14: 
15: from scipy.sparse import coo_matrix
16: from scipy.special import erf
17: from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
18:                                   estimate_bc_jac, compute_jac_indices,
19:                                   construct_global_jac, solve_bvp)
20: 
21: 
22: def exp_fun(x, y):
23:     return np.vstack((y[1], y[0]))
24: 
25: 
26: def exp_fun_jac(x, y):
27:     df_dy = np.empty((2, 2, x.shape[0]))
28:     df_dy[0, 0] = 0
29:     df_dy[0, 1] = 1
30:     df_dy[1, 0] = 1
31:     df_dy[1, 1] = 0
32:     return df_dy
33: 
34: 
35: def exp_bc(ya, yb):
36:     return np.hstack((ya[0] - 1, yb[0]))
37: 
38: 
39: def exp_bc_complex(ya, yb):
40:     return np.hstack((ya[0] - 1 - 1j, yb[0]))
41: 
42: 
43: def exp_bc_jac(ya, yb):
44:     dbc_dya = np.array([
45:         [1, 0],
46:         [0, 0]
47:     ])
48:     dbc_dyb = np.array([
49:         [0, 0],
50:         [1, 0]
51:     ])
52:     return dbc_dya, dbc_dyb
53: 
54: 
55: def exp_sol(x):
56:     return (np.exp(-x) - np.exp(x - 2)) / (1 - np.exp(-2))
57: 
58: 
59: def sl_fun(x, y, p):
60:     return np.vstack((y[1], -p[0]**2 * y[0]))
61: 
62: 
63: def sl_fun_jac(x, y, p):
64:     n, m = y.shape
65:     df_dy = np.empty((n, 2, m))
66:     df_dy[0, 0] = 0
67:     df_dy[0, 1] = 1
68:     df_dy[1, 0] = -p[0]**2
69:     df_dy[1, 1] = 0
70: 
71:     df_dp = np.empty((n, 1, m))
72:     df_dp[0, 0] = 0
73:     df_dp[1, 0] = -2 * p[0] * y[0]
74: 
75:     return df_dy, df_dp
76: 
77: 
78: def sl_bc(ya, yb, p):
79:     return np.hstack((ya[0], yb[0], ya[1] - p[0]))
80: 
81: 
82: def sl_bc_jac(ya, yb, p):
83:     dbc_dya = np.zeros((3, 2))
84:     dbc_dya[0, 0] = 1
85:     dbc_dya[2, 1] = 1
86: 
87:     dbc_dyb = np.zeros((3, 2))
88:     dbc_dyb[1, 0] = 1
89: 
90:     dbc_dp = np.zeros((3, 1))
91:     dbc_dp[2, 0] = -1
92: 
93:     return dbc_dya, dbc_dyb, dbc_dp
94: 
95: 
96: def sl_sol(x, p):
97:     return np.sin(p[0] * x)
98: 
99: 
100: def emden_fun(x, y):
101:     return np.vstack((y[1], -y[0]**5))
102: 
103: 
104: def emden_fun_jac(x, y):
105:     df_dy = np.empty((2, 2, x.shape[0]))
106:     df_dy[0, 0] = 0
107:     df_dy[0, 1] = 1
108:     df_dy[1, 0] = -5 * y[0]**4
109:     df_dy[1, 1] = 0
110:     return df_dy
111: 
112: 
113: def emden_bc(ya, yb):
114:     return np.array([ya[1], yb[0] - (3/4)**0.5])
115: 
116: 
117: def emden_bc_jac(ya, yb):
118:     dbc_dya = np.array([
119:         [0, 1],
120:         [0, 0]
121:     ])
122:     dbc_dyb = np.array([
123:         [0, 0],
124:         [1, 0]
125:     ])
126:     return dbc_dya, dbc_dyb
127: 
128: 
129: def emden_sol(x):
130:     return (1 + x**2/3)**-0.5
131: 
132: 
133: def undefined_fun(x, y):
134:     return np.zeros_like(y)
135: 
136: 
137: def undefined_bc(ya, yb):
138:     return np.array([ya[0], yb[0] - 1])
139: 
140: 
141: def big_fun(x, y):
142:     f = np.zeros_like(y)
143:     f[::2] = y[1::2]
144:     return f
145: 
146: 
147: def big_bc(ya, yb):
148:     return np.hstack((ya[::2], yb[::2] - 1))
149: 
150: 
151: def big_sol(x, n):
152:     y = np.ones((2 * n, x.size))
153:     y[::2] = x
154:     return x
155: 
156: 
157: def shock_fun(x, y):
158:     eps = 1e-3
159:     return np.vstack((
160:         y[1],
161:         -(x * y[1] + eps * np.pi**2 * np.cos(np.pi * x) +
162:           np.pi * x * np.sin(np.pi * x)) / eps
163:     ))
164: 
165: 
166: def shock_bc(ya, yb):
167:     return np.array([ya[0] + 2, yb[0]])
168: 
169: 
170: def shock_sol(x):
171:     eps = 1e-3
172:     k = np.sqrt(2 * eps)
173:     return np.cos(np.pi * x) + erf(x / k) / erf(1 / k)
174: 
175: 
176: def test_modify_mesh():
177:     x = np.array([0, 1, 3, 9], dtype=float)
178:     x_new = modify_mesh(x, np.array([0]), np.array([2]))
179:     assert_array_equal(x_new, np.array([0, 0.5, 1, 3, 5, 7, 9]))
180: 
181:     x = np.array([-6, -3, 0, 3, 6], dtype=float)
182:     x_new = modify_mesh(x, np.array([1], dtype=int), np.array([0, 2, 3]))
183:     assert_array_equal(x_new, [-6, -5, -4, -3, -1.5, 0, 1, 2, 3, 4, 5, 6])
184: 
185: 
186: def test_compute_fun_jac():
187:     x = np.linspace(0, 1, 5)
188:     y = np.empty((2, x.shape[0]))
189:     y[0] = 0.01
190:     y[1] = 0.02
191:     p = np.array([])
192:     df_dy, df_dp = estimate_fun_jac(lambda x, y, p: exp_fun(x, y), x, y, p)
193:     df_dy_an = exp_fun_jac(x, y)
194:     assert_allclose(df_dy, df_dy_an)
195:     assert_(df_dp is None)
196: 
197:     x = np.linspace(0, np.pi, 5)
198:     y = np.empty((2, x.shape[0]))
199:     y[0] = np.sin(x)
200:     y[1] = np.cos(x)
201:     p = np.array([1.0])
202:     df_dy, df_dp = estimate_fun_jac(sl_fun, x, y, p)
203:     df_dy_an, df_dp_an = sl_fun_jac(x, y, p)
204:     assert_allclose(df_dy, df_dy_an)
205:     assert_allclose(df_dp, df_dp_an)
206: 
207:     x = np.linspace(0, 1, 10)
208:     y = np.empty((2, x.shape[0]))
209:     y[0] = (3/4)**0.5
210:     y[1] = 1e-4
211:     p = np.array([])
212:     df_dy, df_dp = estimate_fun_jac(lambda x, y, p: emden_fun(x, y), x, y, p)
213:     df_dy_an = emden_fun_jac(x, y)
214:     assert_allclose(df_dy, df_dy_an)
215:     assert_(df_dp is None)
216: 
217: 
218: def test_compute_bc_jac():
219:     ya = np.array([-1.0, 2])
220:     yb = np.array([0.5, 3])
221:     p = np.array([])
222:     dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(
223:         lambda ya, yb, p: exp_bc(ya, yb), ya, yb, p)
224:     dbc_dya_an, dbc_dyb_an = exp_bc_jac(ya, yb)
225:     assert_allclose(dbc_dya, dbc_dya_an)
226:     assert_allclose(dbc_dyb, dbc_dyb_an)
227:     assert_(dbc_dp is None)
228: 
229:     ya = np.array([0.0, 1])
230:     yb = np.array([0.0, -1])
231:     p = np.array([0.5])
232:     dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(sl_bc, ya, yb, p)
233:     dbc_dya_an, dbc_dyb_an, dbc_dp_an = sl_bc_jac(ya, yb, p)
234:     assert_allclose(dbc_dya, dbc_dya_an)
235:     assert_allclose(dbc_dyb, dbc_dyb_an)
236:     assert_allclose(dbc_dp, dbc_dp_an)
237: 
238:     ya = np.array([0.5, 100])
239:     yb = np.array([-1000, 10.5])
240:     p = np.array([])
241:     dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(
242:         lambda ya, yb, p: emden_bc(ya, yb), ya, yb, p)
243:     dbc_dya_an, dbc_dyb_an = emden_bc_jac(ya, yb)
244:     assert_allclose(dbc_dya, dbc_dya_an)
245:     assert_allclose(dbc_dyb, dbc_dyb_an)
246:     assert_(dbc_dp is None)
247: 
248: 
249: def test_compute_jac_indices():
250:     n = 2
251:     m = 4
252:     k = 2
253:     i, j = compute_jac_indices(n, m, k)
254:     s = coo_matrix((np.ones_like(i), (i, j))).toarray()
255:     s_true = np.array([
256:         [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
257:         [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
258:         [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
259:         [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
260:         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
261:         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
262:         [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
263:         [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
264:         [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
265:         [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
266:     ])
267:     assert_array_equal(s, s_true)
268: 
269: 
270: def test_compute_global_jac():
271:     n = 2
272:     m = 5
273:     k = 1
274:     i_jac, j_jac = compute_jac_indices(2, 5, 1)
275:     x = np.linspace(0, 1, 5)
276:     h = np.diff(x)
277:     y = np.vstack((np.sin(np.pi * x), np.pi * np.cos(np.pi * x)))
278:     p = np.array([3.0])
279: 
280:     f = sl_fun(x, y, p)
281: 
282:     x_middle = x[:-1] + 0.5 * h
283:     y_middle = 0.5 * (y[:, :-1] + y[:, 1:]) - h/8 * (f[:, 1:] - f[:, :-1])
284: 
285:     df_dy, df_dp = sl_fun_jac(x, y, p)
286:     df_dy_middle, df_dp_middle = sl_fun_jac(x_middle, y_middle, p)
287:     dbc_dya, dbc_dyb, dbc_dp = sl_bc_jac(y[:, 0], y[:, -1], p)
288: 
289:     J = construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle,
290:                              df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp)
291:     J = J.toarray()
292: 
293:     def J_block(h, p):
294:         return np.array([
295:             [h**2*p**2/12 - 1, -0.5*h, -h**2*p**2/12 + 1, -0.5*h],
296:             [0.5*h*p**2, h**2*p**2/12 - 1, 0.5*h*p**2, 1 - h**2*p**2/12]
297:         ])
298: 
299:     J_true = np.zeros((m * n + k, m * n + k))
300:     for i in range(m - 1):
301:         J_true[i * n: (i + 1) * n, i * n: (i + 2) * n] = J_block(h[i], p)
302: 
303:     J_true[:(m - 1) * n:2, -1] = p * h**2/6 * (y[0, :-1] - y[0, 1:])
304:     J_true[1:(m - 1) * n:2, -1] = p * (h * (y[0, :-1] + y[0, 1:]) +
305:                                        h**2/6 * (y[1, :-1] - y[1, 1:]))
306: 
307:     J_true[8, 0] = 1
308:     J_true[9, 8] = 1
309:     J_true[10, 1] = 1
310:     J_true[10, 10] = -1
311: 
312:     assert_allclose(J, J_true, rtol=1e-10)
313: 
314:     df_dy, df_dp = estimate_fun_jac(sl_fun, x, y, p)
315:     df_dy_middle, df_dp_middle = estimate_fun_jac(sl_fun, x_middle, y_middle, p)
316:     dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(sl_bc, y[:, 0], y[:, -1], p)
317:     J = construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle,
318:                              df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp)
319:     J = J.toarray()
320:     assert_allclose(J, J_true, rtol=1e-8, atol=1e-9)
321: 
322: 
323: def test_parameter_validation():
324:     x = [0, 1, 0.5]
325:     y = np.zeros((2, 3))
326:     assert_raises(ValueError, solve_bvp, exp_fun, exp_bc, x, y)
327: 
328:     x = np.linspace(0, 1, 5)
329:     y = np.zeros((2, 4))
330:     assert_raises(ValueError, solve_bvp, exp_fun, exp_bc, x, y)
331: 
332:     fun = lambda x, y, p: exp_fun(x, y)
333:     bc = lambda ya, yb, p: exp_bc(ya, yb)
334: 
335:     y = np.zeros((2, x.shape[0]))
336:     assert_raises(ValueError, solve_bvp, fun, bc, x, y, p=[1])
337: 
338:     def wrong_shape_fun(x, y):
339:         return np.zeros(3)
340: 
341:     assert_raises(ValueError, solve_bvp, wrong_shape_fun, bc, x, y)
342: 
343:     S = np.array([[0, 0]])
344:     assert_raises(ValueError, solve_bvp, exp_fun, exp_bc, x, y, S=S)
345: 
346: 
347: def test_no_params():
348:     x = np.linspace(0, 1, 5)
349:     x_test = np.linspace(0, 1, 100)
350:     y = np.zeros((2, x.shape[0]))
351:     for fun_jac in [None, exp_fun_jac]:
352:         for bc_jac in [None, exp_bc_jac]:
353:             sol = solve_bvp(exp_fun, exp_bc, x, y, fun_jac=fun_jac,
354:                             bc_jac=bc_jac)
355: 
356:             assert_equal(sol.status, 0)
357:             assert_(sol.success)
358: 
359:             assert_equal(sol.x.size, 5)
360: 
361:             sol_test = sol.sol(x_test)
362: 
363:             assert_allclose(sol_test[0], exp_sol(x_test), atol=1e-5)
364: 
365:             f_test = exp_fun(x_test, sol_test)
366:             r = sol.sol(x_test, 1) - f_test
367:             rel_res = r / (1 + np.abs(f_test))
368:             norm_res = np.sum(rel_res**2, axis=0)**0.5
369:             assert_(np.all(norm_res < 1e-3))
370: 
371:             assert_(np.all(sol.rms_residuals < 1e-3))
372:             assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
373:             assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)
374: 
375: 
376: def test_with_params():
377:     x = np.linspace(0, np.pi, 5)
378:     x_test = np.linspace(0, np.pi, 100)
379:     y = np.ones((2, x.shape[0]))
380: 
381:     for fun_jac in [None, sl_fun_jac]:
382:         for bc_jac in [None, sl_bc_jac]:
383:             sol = solve_bvp(sl_fun, sl_bc, x, y, p=[0.5], fun_jac=fun_jac,
384:                             bc_jac=bc_jac)
385: 
386:             assert_equal(sol.status, 0)
387:             assert_(sol.success)
388: 
389:             assert_(sol.x.size < 10)
390: 
391:             assert_allclose(sol.p, [1], rtol=1e-4)
392: 
393:             sol_test = sol.sol(x_test)
394: 
395:             assert_allclose(sol_test[0], sl_sol(x_test, [1]),
396:                             rtol=1e-4, atol=1e-4)
397: 
398:             f_test = sl_fun(x_test, sol_test, [1])
399:             r = sol.sol(x_test, 1) - f_test
400:             rel_res = r / (1 + np.abs(f_test))
401:             norm_res = np.sum(rel_res ** 2, axis=0) ** 0.5
402:             assert_(np.all(norm_res < 1e-3))
403: 
404:             assert_(np.all(sol.rms_residuals < 1e-3))
405:             assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
406:             assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)
407: 
408: 
409: def test_singular_term():
410:     x = np.linspace(0, 1, 10)
411:     x_test = np.linspace(0.05, 1, 100)
412:     y = np.empty((2, 10))
413:     y[0] = (3/4)**0.5
414:     y[1] = 1e-4
415:     S = np.array([[0, 0], [0, -2]])
416: 
417:     for fun_jac in [None, emden_fun_jac]:
418:         for bc_jac in [None, emden_bc_jac]:
419:             sol = solve_bvp(emden_fun, emden_bc, x, y, S=S, fun_jac=fun_jac,
420:                             bc_jac=bc_jac)
421: 
422:             assert_equal(sol.status, 0)
423:             assert_(sol.success)
424: 
425:             assert_equal(sol.x.size, 10)
426: 
427:             sol_test = sol.sol(x_test)
428:             assert_allclose(sol_test[0], emden_sol(x_test), atol=1e-5)
429: 
430:             f_test = emden_fun(x_test, sol_test) + S.dot(sol_test) / x_test
431:             r = sol.sol(x_test, 1) - f_test
432:             rel_res = r / (1 + np.abs(f_test))
433:             norm_res = np.sum(rel_res ** 2, axis=0) ** 0.5
434: 
435:             assert_(np.all(norm_res < 1e-3))
436:             assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
437:             assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)
438: 
439: 
440: def test_complex():
441:     # The test is essentially the same as test_no_params, but boundary
442:     # conditions are turned into complex.
443:     x = np.linspace(0, 1, 5)
444:     x_test = np.linspace(0, 1, 100)
445:     y = np.zeros((2, x.shape[0]), dtype=complex)
446:     for fun_jac in [None, exp_fun_jac]:
447:         for bc_jac in [None, exp_bc_jac]:
448:             sol = solve_bvp(exp_fun, exp_bc_complex, x, y, fun_jac=fun_jac,
449:                             bc_jac=bc_jac)
450: 
451:             assert_equal(sol.status, 0)
452:             assert_(sol.success)
453: 
454:             sol_test = sol.sol(x_test)
455: 
456:             assert_allclose(sol_test[0].real, exp_sol(x_test), atol=1e-5)
457:             assert_allclose(sol_test[0].imag, exp_sol(x_test), atol=1e-5)
458: 
459:             f_test = exp_fun(x_test, sol_test)
460:             r = sol.sol(x_test, 1) - f_test
461:             rel_res = r / (1 + np.abs(f_test))
462:             norm_res = np.sum(np.real(rel_res * np.conj(rel_res)),
463:                               axis=0) ** 0.5
464:             assert_(np.all(norm_res < 1e-3))
465: 
466:             assert_(np.all(sol.rms_residuals < 1e-3))
467:             assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
468:             assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)
469: 
470: 
471: def test_failures():
472:     x = np.linspace(0, 1, 2)
473:     y = np.zeros((2, x.size))
474:     res = solve_bvp(exp_fun, exp_bc, x, y, tol=1e-5, max_nodes=5)
475:     assert_equal(res.status, 1)
476:     assert_(not res.success)
477: 
478:     x = np.linspace(0, 1, 5)
479:     y = np.zeros((2, x.size))
480:     res = solve_bvp(undefined_fun, undefined_bc, x, y)
481:     assert_equal(res.status, 2)
482:     assert_(not res.success)
483: 
484: 
485: def test_big_problem():
486:     n = 30
487:     x = np.linspace(0, 1, 5)
488:     y = np.zeros((2 * n, x.size))
489:     sol = solve_bvp(big_fun, big_bc, x, y)
490: 
491:     assert_equal(sol.status, 0)
492:     assert_(sol.success)
493: 
494:     sol_test = sol.sol(x)
495: 
496:     assert_allclose(sol_test[0], big_sol(x, n))
497: 
498:     f_test = big_fun(x, sol_test)
499:     r = sol.sol(x, 1) - f_test
500:     rel_res = r / (1 + np.abs(f_test))
501:     norm_res = np.sum(np.real(rel_res * np.conj(rel_res)), axis=0) ** 0.5
502:     assert_(np.all(norm_res < 1e-3))
503: 
504:     assert_(np.all(sol.rms_residuals < 1e-3))
505:     assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
506:     assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)
507: 
508: 
509: def test_shock_layer():
510:     x = np.linspace(-1, 1, 5)
511:     x_test = np.linspace(-1, 1, 100)
512:     y = np.zeros((2, x.size))
513:     sol = solve_bvp(shock_fun, shock_bc, x, y)
514: 
515:     assert_equal(sol.status, 0)
516:     assert_(sol.success)
517: 
518:     assert_(sol.x.size < 110)
519: 
520:     sol_test = sol.sol(x_test)
521:     assert_allclose(sol_test[0], shock_sol(x_test), rtol=1e-5, atol=1e-5)
522: 
523:     f_test = shock_fun(x_test, sol_test)
524:     r = sol.sol(x_test, 1) - f_test
525:     rel_res = r / (1 + np.abs(f_test))
526:     norm_res = np.sum(rel_res ** 2, axis=0) ** 0.5
527: 
528:     assert_(np.all(norm_res < 1e-3))
529:     assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
530:     assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)
531: 
532: 
533: def test_verbose():
534:     # Smoke test that checks the printing does something and does not crash
535:     x = np.linspace(0, 1, 5)
536:     y = np.zeros((2, x.shape[0]))
537:     for verbose in [0, 1, 2]:
538:         old_stdout = sys.stdout
539:         sys.stdout = StringIO()
540:         try:
541:             sol = solve_bvp(exp_fun, exp_bc, x, y, verbose=verbose)
542:             text = sys.stdout.getvalue()
543:         finally:
544:             sys.stdout = old_stdout
545: 
546:         assert_(sol.success)
547:         if verbose == 0:
548:             assert_(not text, text)
549:         if verbose >= 1:
550:             assert_("Solved in" in text, text)
551:         if verbose >= 2:
552:             assert_("Max residual" in text, text)
553: 
554: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)



# SSA begins for try-except statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))

# 'from StringIO import StringIO' statement (line 6)
try:
    from StringIO import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'StringIO', None, module_type_store, ['StringIO'], [StringIO])

# SSA branch for the except part of a try statement (line 5)
# SSA branch for the except 'ImportError' branch of a try statement (line 5)
module_type_store.open_ssa_branch('except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))

# 'from io import StringIO' statement (line 8)
try:
    from io import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'io', None, module_type_store, ['StringIO'], [StringIO])

# SSA join for try-except statement (line 5)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_38840 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_38840) is not StypyTypeError):

    if (import_38840 != 'pyd_module'):
        __import__(import_38840)
        sys_modules_38841 = sys.modules[import_38840]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_38841.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_38840)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.testing import assert_, assert_array_equal, assert_allclose, assert_equal' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_38842 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.testing')

if (type(import_38842) is not StypyTypeError):

    if (import_38842 != 'pyd_module'):
        __import__(import_38842)
        sys_modules_38843 = sys.modules[import_38842]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.testing', sys_modules_38843.module_type_store, module_type_store, ['assert_', 'assert_array_equal', 'assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_38843, sys_modules_38843.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_array_equal, assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_array_equal', 'assert_allclose', 'assert_equal'], [assert_, assert_array_equal, assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.testing', import_38842)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from pytest import assert_raises' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_38844 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'pytest')

if (type(import_38844) is not StypyTypeError):

    if (import_38844 != 'pyd_module'):
        __import__(import_38844)
        sys_modules_38845 = sys.modules[import_38844]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'pytest', sys_modules_38845.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_38845, sys_modules_38845.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'pytest', import_38844)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.sparse import coo_matrix' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_38846 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse')

if (type(import_38846) is not StypyTypeError):

    if (import_38846 != 'pyd_module'):
        __import__(import_38846)
        sys_modules_38847 = sys.modules[import_38846]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse', sys_modules_38847.module_type_store, module_type_store, ['coo_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_38847, sys_modules_38847.module_type_store, module_type_store)
    else:
        from scipy.sparse import coo_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse', None, module_type_store, ['coo_matrix'], [coo_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse', import_38846)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.special import erf' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_38848 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.special')

if (type(import_38848) is not StypyTypeError):

    if (import_38848 != 'pyd_module'):
        __import__(import_38848)
        sys_modules_38849 = sys.modules[import_38848]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.special', sys_modules_38849.module_type_store, module_type_store, ['erf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_38849, sys_modules_38849.module_type_store, module_type_store)
    else:
        from scipy.special import erf

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.special', None, module_type_store, ['erf'], [erf])

else:
    # Assigning a type to the variable 'scipy.special' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.special', import_38848)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.integrate._bvp import modify_mesh, estimate_fun_jac, estimate_bc_jac, compute_jac_indices, construct_global_jac, solve_bvp' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_38850 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.integrate._bvp')

if (type(import_38850) is not StypyTypeError):

    if (import_38850 != 'pyd_module'):
        __import__(import_38850)
        sys_modules_38851 = sys.modules[import_38850]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.integrate._bvp', sys_modules_38851.module_type_store, module_type_store, ['modify_mesh', 'estimate_fun_jac', 'estimate_bc_jac', 'compute_jac_indices', 'construct_global_jac', 'solve_bvp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_38851, sys_modules_38851.module_type_store, module_type_store)
    else:
        from scipy.integrate._bvp import modify_mesh, estimate_fun_jac, estimate_bc_jac, compute_jac_indices, construct_global_jac, solve_bvp

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.integrate._bvp', None, module_type_store, ['modify_mesh', 'estimate_fun_jac', 'estimate_bc_jac', 'compute_jac_indices', 'construct_global_jac', 'solve_bvp'], [modify_mesh, estimate_fun_jac, estimate_bc_jac, compute_jac_indices, construct_global_jac, solve_bvp])

else:
    # Assigning a type to the variable 'scipy.integrate._bvp' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.integrate._bvp', import_38850)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')


@norecursion
def exp_fun(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'exp_fun'
    module_type_store = module_type_store.open_function_context('exp_fun', 22, 0, False)
    
    # Passed parameters checking function
    exp_fun.stypy_localization = localization
    exp_fun.stypy_type_of_self = None
    exp_fun.stypy_type_store = module_type_store
    exp_fun.stypy_function_name = 'exp_fun'
    exp_fun.stypy_param_names_list = ['x', 'y']
    exp_fun.stypy_varargs_param_name = None
    exp_fun.stypy_kwargs_param_name = None
    exp_fun.stypy_call_defaults = defaults
    exp_fun.stypy_call_varargs = varargs
    exp_fun.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exp_fun', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exp_fun', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exp_fun(...)' code ##################

    
    # Call to vstack(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Obtaining an instance of the builtin type 'tuple' (line 23)
    tuple_38854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 23)
    # Adding element type (line 23)
    
    # Obtaining the type of the subscript
    int_38855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'int')
    # Getting the type of 'y' (line 23)
    y_38856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 22), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 23)
    getitem___38857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 22), y_38856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 23)
    subscript_call_result_38858 = invoke(stypy.reporting.localization.Localization(__file__, 23, 22), getitem___38857, int_38855)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 22), tuple_38854, subscript_call_result_38858)
    # Adding element type (line 23)
    
    # Obtaining the type of the subscript
    int_38859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'int')
    # Getting the type of 'y' (line 23)
    y_38860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 28), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 23)
    getitem___38861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 28), y_38860, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 23)
    subscript_call_result_38862 = invoke(stypy.reporting.localization.Localization(__file__, 23, 28), getitem___38861, int_38859)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 22), tuple_38854, subscript_call_result_38862)
    
    # Processing the call keyword arguments (line 23)
    kwargs_38863 = {}
    # Getting the type of 'np' (line 23)
    np_38852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'np', False)
    # Obtaining the member 'vstack' of a type (line 23)
    vstack_38853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 11), np_38852, 'vstack')
    # Calling vstack(args, kwargs) (line 23)
    vstack_call_result_38864 = invoke(stypy.reporting.localization.Localization(__file__, 23, 11), vstack_38853, *[tuple_38854], **kwargs_38863)
    
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type', vstack_call_result_38864)
    
    # ################# End of 'exp_fun(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exp_fun' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_38865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38865)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exp_fun'
    return stypy_return_type_38865

# Assigning a type to the variable 'exp_fun' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'exp_fun', exp_fun)

@norecursion
def exp_fun_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'exp_fun_jac'
    module_type_store = module_type_store.open_function_context('exp_fun_jac', 26, 0, False)
    
    # Passed parameters checking function
    exp_fun_jac.stypy_localization = localization
    exp_fun_jac.stypy_type_of_self = None
    exp_fun_jac.stypy_type_store = module_type_store
    exp_fun_jac.stypy_function_name = 'exp_fun_jac'
    exp_fun_jac.stypy_param_names_list = ['x', 'y']
    exp_fun_jac.stypy_varargs_param_name = None
    exp_fun_jac.stypy_kwargs_param_name = None
    exp_fun_jac.stypy_call_defaults = defaults
    exp_fun_jac.stypy_call_varargs = varargs
    exp_fun_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exp_fun_jac', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exp_fun_jac', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exp_fun_jac(...)' code ##################

    
    # Assigning a Call to a Name (line 27):
    
    # Assigning a Call to a Name (line 27):
    
    # Call to empty(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_38868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    int_38869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), tuple_38868, int_38869)
    # Adding element type (line 27)
    int_38870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), tuple_38868, int_38870)
    # Adding element type (line 27)
    
    # Obtaining the type of the subscript
    int_38871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 36), 'int')
    # Getting the type of 'x' (line 27)
    x_38872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 28), 'x', False)
    # Obtaining the member 'shape' of a type (line 27)
    shape_38873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 28), x_38872, 'shape')
    # Obtaining the member '__getitem__' of a type (line 27)
    getitem___38874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 28), shape_38873, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 27)
    subscript_call_result_38875 = invoke(stypy.reporting.localization.Localization(__file__, 27, 28), getitem___38874, int_38871)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), tuple_38868, subscript_call_result_38875)
    
    # Processing the call keyword arguments (line 27)
    kwargs_38876 = {}
    # Getting the type of 'np' (line 27)
    np_38866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'np', False)
    # Obtaining the member 'empty' of a type (line 27)
    empty_38867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), np_38866, 'empty')
    # Calling empty(args, kwargs) (line 27)
    empty_call_result_38877 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), empty_38867, *[tuple_38868], **kwargs_38876)
    
    # Assigning a type to the variable 'df_dy' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'df_dy', empty_call_result_38877)
    
    # Assigning a Num to a Subscript (line 28):
    
    # Assigning a Num to a Subscript (line 28):
    int_38878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 18), 'int')
    # Getting the type of 'df_dy' (line 28)
    df_dy_38879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'df_dy')
    
    # Obtaining an instance of the builtin type 'tuple' (line 28)
    tuple_38880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 28)
    # Adding element type (line 28)
    int_38881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), tuple_38880, int_38881)
    # Adding element type (line 28)
    int_38882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), tuple_38880, int_38882)
    
    # Storing an element on a container (line 28)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 4), df_dy_38879, (tuple_38880, int_38878))
    
    # Assigning a Num to a Subscript (line 29):
    
    # Assigning a Num to a Subscript (line 29):
    int_38883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 18), 'int')
    # Getting the type of 'df_dy' (line 29)
    df_dy_38884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'df_dy')
    
    # Obtaining an instance of the builtin type 'tuple' (line 29)
    tuple_38885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 29)
    # Adding element type (line 29)
    int_38886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), tuple_38885, int_38886)
    # Adding element type (line 29)
    int_38887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), tuple_38885, int_38887)
    
    # Storing an element on a container (line 29)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 4), df_dy_38884, (tuple_38885, int_38883))
    
    # Assigning a Num to a Subscript (line 30):
    
    # Assigning a Num to a Subscript (line 30):
    int_38888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 18), 'int')
    # Getting the type of 'df_dy' (line 30)
    df_dy_38889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'df_dy')
    
    # Obtaining an instance of the builtin type 'tuple' (line 30)
    tuple_38890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 30)
    # Adding element type (line 30)
    int_38891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 10), tuple_38890, int_38891)
    # Adding element type (line 30)
    int_38892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 10), tuple_38890, int_38892)
    
    # Storing an element on a container (line 30)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), df_dy_38889, (tuple_38890, int_38888))
    
    # Assigning a Num to a Subscript (line 31):
    
    # Assigning a Num to a Subscript (line 31):
    int_38893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'int')
    # Getting the type of 'df_dy' (line 31)
    df_dy_38894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'df_dy')
    
    # Obtaining an instance of the builtin type 'tuple' (line 31)
    tuple_38895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 31)
    # Adding element type (line 31)
    int_38896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 10), tuple_38895, int_38896)
    # Adding element type (line 31)
    int_38897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 10), tuple_38895, int_38897)
    
    # Storing an element on a container (line 31)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 4), df_dy_38894, (tuple_38895, int_38893))
    # Getting the type of 'df_dy' (line 32)
    df_dy_38898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'df_dy')
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type', df_dy_38898)
    
    # ################# End of 'exp_fun_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exp_fun_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_38899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38899)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exp_fun_jac'
    return stypy_return_type_38899

# Assigning a type to the variable 'exp_fun_jac' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'exp_fun_jac', exp_fun_jac)

@norecursion
def exp_bc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'exp_bc'
    module_type_store = module_type_store.open_function_context('exp_bc', 35, 0, False)
    
    # Passed parameters checking function
    exp_bc.stypy_localization = localization
    exp_bc.stypy_type_of_self = None
    exp_bc.stypy_type_store = module_type_store
    exp_bc.stypy_function_name = 'exp_bc'
    exp_bc.stypy_param_names_list = ['ya', 'yb']
    exp_bc.stypy_varargs_param_name = None
    exp_bc.stypy_kwargs_param_name = None
    exp_bc.stypy_call_defaults = defaults
    exp_bc.stypy_call_varargs = varargs
    exp_bc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exp_bc', ['ya', 'yb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exp_bc', localization, ['ya', 'yb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exp_bc(...)' code ##################

    
    # Call to hstack(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_38902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    
    # Obtaining the type of the subscript
    int_38903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'int')
    # Getting the type of 'ya' (line 36)
    ya_38904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'ya', False)
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___38905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 22), ya_38904, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_38906 = invoke(stypy.reporting.localization.Localization(__file__, 36, 22), getitem___38905, int_38903)
    
    int_38907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 30), 'int')
    # Applying the binary operator '-' (line 36)
    result_sub_38908 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 22), '-', subscript_call_result_38906, int_38907)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 22), tuple_38902, result_sub_38908)
    # Adding element type (line 36)
    
    # Obtaining the type of the subscript
    int_38909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 36), 'int')
    # Getting the type of 'yb' (line 36)
    yb_38910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 33), 'yb', False)
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___38911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 33), yb_38910, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_38912 = invoke(stypy.reporting.localization.Localization(__file__, 36, 33), getitem___38911, int_38909)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 22), tuple_38902, subscript_call_result_38912)
    
    # Processing the call keyword arguments (line 36)
    kwargs_38913 = {}
    # Getting the type of 'np' (line 36)
    np_38900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'np', False)
    # Obtaining the member 'hstack' of a type (line 36)
    hstack_38901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 11), np_38900, 'hstack')
    # Calling hstack(args, kwargs) (line 36)
    hstack_call_result_38914 = invoke(stypy.reporting.localization.Localization(__file__, 36, 11), hstack_38901, *[tuple_38902], **kwargs_38913)
    
    # Assigning a type to the variable 'stypy_return_type' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type', hstack_call_result_38914)
    
    # ################# End of 'exp_bc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exp_bc' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_38915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38915)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exp_bc'
    return stypy_return_type_38915

# Assigning a type to the variable 'exp_bc' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'exp_bc', exp_bc)

@norecursion
def exp_bc_complex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'exp_bc_complex'
    module_type_store = module_type_store.open_function_context('exp_bc_complex', 39, 0, False)
    
    # Passed parameters checking function
    exp_bc_complex.stypy_localization = localization
    exp_bc_complex.stypy_type_of_self = None
    exp_bc_complex.stypy_type_store = module_type_store
    exp_bc_complex.stypy_function_name = 'exp_bc_complex'
    exp_bc_complex.stypy_param_names_list = ['ya', 'yb']
    exp_bc_complex.stypy_varargs_param_name = None
    exp_bc_complex.stypy_kwargs_param_name = None
    exp_bc_complex.stypy_call_defaults = defaults
    exp_bc_complex.stypy_call_varargs = varargs
    exp_bc_complex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exp_bc_complex', ['ya', 'yb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exp_bc_complex', localization, ['ya', 'yb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exp_bc_complex(...)' code ##################

    
    # Call to hstack(...): (line 40)
    # Processing the call arguments (line 40)
    
    # Obtaining an instance of the builtin type 'tuple' (line 40)
    tuple_38918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 40)
    # Adding element type (line 40)
    
    # Obtaining the type of the subscript
    int_38919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'int')
    # Getting the type of 'ya' (line 40)
    ya_38920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'ya', False)
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___38921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 22), ya_38920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_38922 = invoke(stypy.reporting.localization.Localization(__file__, 40, 22), getitem___38921, int_38919)
    
    int_38923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 30), 'int')
    # Applying the binary operator '-' (line 40)
    result_sub_38924 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 22), '-', subscript_call_result_38922, int_38923)
    
    complex_38925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 34), 'complex')
    # Applying the binary operator '-' (line 40)
    result_sub_38926 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 32), '-', result_sub_38924, complex_38925)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 22), tuple_38918, result_sub_38926)
    # Adding element type (line 40)
    
    # Obtaining the type of the subscript
    int_38927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 41), 'int')
    # Getting the type of 'yb' (line 40)
    yb_38928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 38), 'yb', False)
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___38929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 38), yb_38928, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_38930 = invoke(stypy.reporting.localization.Localization(__file__, 40, 38), getitem___38929, int_38927)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 22), tuple_38918, subscript_call_result_38930)
    
    # Processing the call keyword arguments (line 40)
    kwargs_38931 = {}
    # Getting the type of 'np' (line 40)
    np_38916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'np', False)
    # Obtaining the member 'hstack' of a type (line 40)
    hstack_38917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 11), np_38916, 'hstack')
    # Calling hstack(args, kwargs) (line 40)
    hstack_call_result_38932 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), hstack_38917, *[tuple_38918], **kwargs_38931)
    
    # Assigning a type to the variable 'stypy_return_type' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type', hstack_call_result_38932)
    
    # ################# End of 'exp_bc_complex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exp_bc_complex' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_38933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38933)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exp_bc_complex'
    return stypy_return_type_38933

# Assigning a type to the variable 'exp_bc_complex' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'exp_bc_complex', exp_bc_complex)

@norecursion
def exp_bc_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'exp_bc_jac'
    module_type_store = module_type_store.open_function_context('exp_bc_jac', 43, 0, False)
    
    # Passed parameters checking function
    exp_bc_jac.stypy_localization = localization
    exp_bc_jac.stypy_type_of_self = None
    exp_bc_jac.stypy_type_store = module_type_store
    exp_bc_jac.stypy_function_name = 'exp_bc_jac'
    exp_bc_jac.stypy_param_names_list = ['ya', 'yb']
    exp_bc_jac.stypy_varargs_param_name = None
    exp_bc_jac.stypy_kwargs_param_name = None
    exp_bc_jac.stypy_call_defaults = defaults
    exp_bc_jac.stypy_call_varargs = varargs
    exp_bc_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exp_bc_jac', ['ya', 'yb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exp_bc_jac', localization, ['ya', 'yb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exp_bc_jac(...)' code ##################

    
    # Assigning a Call to a Name (line 44):
    
    # Assigning a Call to a Name (line 44):
    
    # Call to array(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_38936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    
    # Obtaining an instance of the builtin type 'list' (line 45)
    list_38937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 45)
    # Adding element type (line 45)
    int_38938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 8), list_38937, int_38938)
    # Adding element type (line 45)
    int_38939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 8), list_38937, int_38939)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), list_38936, list_38937)
    # Adding element type (line 44)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_38940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    int_38941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 8), list_38940, int_38941)
    # Adding element type (line 46)
    int_38942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 8), list_38940, int_38942)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), list_38936, list_38940)
    
    # Processing the call keyword arguments (line 44)
    kwargs_38943 = {}
    # Getting the type of 'np' (line 44)
    np_38934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 44)
    array_38935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 14), np_38934, 'array')
    # Calling array(args, kwargs) (line 44)
    array_call_result_38944 = invoke(stypy.reporting.localization.Localization(__file__, 44, 14), array_38935, *[list_38936], **kwargs_38943)
    
    # Assigning a type to the variable 'dbc_dya' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'dbc_dya', array_call_result_38944)
    
    # Assigning a Call to a Name (line 48):
    
    # Assigning a Call to a Name (line 48):
    
    # Call to array(...): (line 48)
    # Processing the call arguments (line 48)
    
    # Obtaining an instance of the builtin type 'list' (line 48)
    list_38947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 48)
    # Adding element type (line 48)
    
    # Obtaining an instance of the builtin type 'list' (line 49)
    list_38948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 49)
    # Adding element type (line 49)
    int_38949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 8), list_38948, int_38949)
    # Adding element type (line 49)
    int_38950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 8), list_38948, int_38950)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 23), list_38947, list_38948)
    # Adding element type (line 48)
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_38951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    # Adding element type (line 50)
    int_38952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 8), list_38951, int_38952)
    # Adding element type (line 50)
    int_38953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 8), list_38951, int_38953)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 23), list_38947, list_38951)
    
    # Processing the call keyword arguments (line 48)
    kwargs_38954 = {}
    # Getting the type of 'np' (line 48)
    np_38945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 48)
    array_38946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 14), np_38945, 'array')
    # Calling array(args, kwargs) (line 48)
    array_call_result_38955 = invoke(stypy.reporting.localization.Localization(__file__, 48, 14), array_38946, *[list_38947], **kwargs_38954)
    
    # Assigning a type to the variable 'dbc_dyb' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'dbc_dyb', array_call_result_38955)
    
    # Obtaining an instance of the builtin type 'tuple' (line 52)
    tuple_38956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 52)
    # Adding element type (line 52)
    # Getting the type of 'dbc_dya' (line 52)
    dbc_dya_38957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'dbc_dya')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 11), tuple_38956, dbc_dya_38957)
    # Adding element type (line 52)
    # Getting the type of 'dbc_dyb' (line 52)
    dbc_dyb_38958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'dbc_dyb')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 11), tuple_38956, dbc_dyb_38958)
    
    # Assigning a type to the variable 'stypy_return_type' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type', tuple_38956)
    
    # ################# End of 'exp_bc_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exp_bc_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_38959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38959)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exp_bc_jac'
    return stypy_return_type_38959

# Assigning a type to the variable 'exp_bc_jac' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'exp_bc_jac', exp_bc_jac)

@norecursion
def exp_sol(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'exp_sol'
    module_type_store = module_type_store.open_function_context('exp_sol', 55, 0, False)
    
    # Passed parameters checking function
    exp_sol.stypy_localization = localization
    exp_sol.stypy_type_of_self = None
    exp_sol.stypy_type_store = module_type_store
    exp_sol.stypy_function_name = 'exp_sol'
    exp_sol.stypy_param_names_list = ['x']
    exp_sol.stypy_varargs_param_name = None
    exp_sol.stypy_kwargs_param_name = None
    exp_sol.stypy_call_defaults = defaults
    exp_sol.stypy_call_varargs = varargs
    exp_sol.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exp_sol', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exp_sol', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exp_sol(...)' code ##################

    
    # Call to exp(...): (line 56)
    # Processing the call arguments (line 56)
    
    # Getting the type of 'x' (line 56)
    x_38962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'x', False)
    # Applying the 'usub' unary operator (line 56)
    result___neg___38963 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 19), 'usub', x_38962)
    
    # Processing the call keyword arguments (line 56)
    kwargs_38964 = {}
    # Getting the type of 'np' (line 56)
    np_38960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'np', False)
    # Obtaining the member 'exp' of a type (line 56)
    exp_38961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), np_38960, 'exp')
    # Calling exp(args, kwargs) (line 56)
    exp_call_result_38965 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), exp_38961, *[result___neg___38963], **kwargs_38964)
    
    
    # Call to exp(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'x' (line 56)
    x_38968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 32), 'x', False)
    int_38969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 36), 'int')
    # Applying the binary operator '-' (line 56)
    result_sub_38970 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 32), '-', x_38968, int_38969)
    
    # Processing the call keyword arguments (line 56)
    kwargs_38971 = {}
    # Getting the type of 'np' (line 56)
    np_38966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'np', False)
    # Obtaining the member 'exp' of a type (line 56)
    exp_38967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 25), np_38966, 'exp')
    # Calling exp(args, kwargs) (line 56)
    exp_call_result_38972 = invoke(stypy.reporting.localization.Localization(__file__, 56, 25), exp_38967, *[result_sub_38970], **kwargs_38971)
    
    # Applying the binary operator '-' (line 56)
    result_sub_38973 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 12), '-', exp_call_result_38965, exp_call_result_38972)
    
    int_38974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 43), 'int')
    
    # Call to exp(...): (line 56)
    # Processing the call arguments (line 56)
    int_38977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 54), 'int')
    # Processing the call keyword arguments (line 56)
    kwargs_38978 = {}
    # Getting the type of 'np' (line 56)
    np_38975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 47), 'np', False)
    # Obtaining the member 'exp' of a type (line 56)
    exp_38976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 47), np_38975, 'exp')
    # Calling exp(args, kwargs) (line 56)
    exp_call_result_38979 = invoke(stypy.reporting.localization.Localization(__file__, 56, 47), exp_38976, *[int_38977], **kwargs_38978)
    
    # Applying the binary operator '-' (line 56)
    result_sub_38980 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 43), '-', int_38974, exp_call_result_38979)
    
    # Applying the binary operator 'div' (line 56)
    result_div_38981 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 11), 'div', result_sub_38973, result_sub_38980)
    
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type', result_div_38981)
    
    # ################# End of 'exp_sol(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exp_sol' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_38982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38982)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exp_sol'
    return stypy_return_type_38982

# Assigning a type to the variable 'exp_sol' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'exp_sol', exp_sol)

@norecursion
def sl_fun(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sl_fun'
    module_type_store = module_type_store.open_function_context('sl_fun', 59, 0, False)
    
    # Passed parameters checking function
    sl_fun.stypy_localization = localization
    sl_fun.stypy_type_of_self = None
    sl_fun.stypy_type_store = module_type_store
    sl_fun.stypy_function_name = 'sl_fun'
    sl_fun.stypy_param_names_list = ['x', 'y', 'p']
    sl_fun.stypy_varargs_param_name = None
    sl_fun.stypy_kwargs_param_name = None
    sl_fun.stypy_call_defaults = defaults
    sl_fun.stypy_call_varargs = varargs
    sl_fun.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sl_fun', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sl_fun', localization, ['x', 'y', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sl_fun(...)' code ##################

    
    # Call to vstack(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Obtaining an instance of the builtin type 'tuple' (line 60)
    tuple_38985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 60)
    # Adding element type (line 60)
    
    # Obtaining the type of the subscript
    int_38986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 24), 'int')
    # Getting the type of 'y' (line 60)
    y_38987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___38988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 22), y_38987, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_38989 = invoke(stypy.reporting.localization.Localization(__file__, 60, 22), getitem___38988, int_38986)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 22), tuple_38985, subscript_call_result_38989)
    # Adding element type (line 60)
    
    
    # Obtaining the type of the subscript
    int_38990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 31), 'int')
    # Getting the type of 'p' (line 60)
    p_38991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___38992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 29), p_38991, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_38993 = invoke(stypy.reporting.localization.Localization(__file__, 60, 29), getitem___38992, int_38990)
    
    int_38994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 35), 'int')
    # Applying the binary operator '**' (line 60)
    result_pow_38995 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 29), '**', subscript_call_result_38993, int_38994)
    
    # Applying the 'usub' unary operator (line 60)
    result___neg___38996 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 28), 'usub', result_pow_38995)
    
    
    # Obtaining the type of the subscript
    int_38997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 41), 'int')
    # Getting the type of 'y' (line 60)
    y_38998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 39), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___38999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 39), y_38998, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_39000 = invoke(stypy.reporting.localization.Localization(__file__, 60, 39), getitem___38999, int_38997)
    
    # Applying the binary operator '*' (line 60)
    result_mul_39001 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 28), '*', result___neg___38996, subscript_call_result_39000)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 22), tuple_38985, result_mul_39001)
    
    # Processing the call keyword arguments (line 60)
    kwargs_39002 = {}
    # Getting the type of 'np' (line 60)
    np_38983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'np', False)
    # Obtaining the member 'vstack' of a type (line 60)
    vstack_38984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 11), np_38983, 'vstack')
    # Calling vstack(args, kwargs) (line 60)
    vstack_call_result_39003 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), vstack_38984, *[tuple_38985], **kwargs_39002)
    
    # Assigning a type to the variable 'stypy_return_type' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type', vstack_call_result_39003)
    
    # ################# End of 'sl_fun(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sl_fun' in the type store
    # Getting the type of 'stypy_return_type' (line 59)
    stypy_return_type_39004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39004)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sl_fun'
    return stypy_return_type_39004

# Assigning a type to the variable 'sl_fun' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'sl_fun', sl_fun)

@norecursion
def sl_fun_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sl_fun_jac'
    module_type_store = module_type_store.open_function_context('sl_fun_jac', 63, 0, False)
    
    # Passed parameters checking function
    sl_fun_jac.stypy_localization = localization
    sl_fun_jac.stypy_type_of_self = None
    sl_fun_jac.stypy_type_store = module_type_store
    sl_fun_jac.stypy_function_name = 'sl_fun_jac'
    sl_fun_jac.stypy_param_names_list = ['x', 'y', 'p']
    sl_fun_jac.stypy_varargs_param_name = None
    sl_fun_jac.stypy_kwargs_param_name = None
    sl_fun_jac.stypy_call_defaults = defaults
    sl_fun_jac.stypy_call_varargs = varargs
    sl_fun_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sl_fun_jac', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sl_fun_jac', localization, ['x', 'y', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sl_fun_jac(...)' code ##################

    
    # Assigning a Attribute to a Tuple (line 64):
    
    # Assigning a Subscript to a Name (line 64):
    
    # Obtaining the type of the subscript
    int_39005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'int')
    # Getting the type of 'y' (line 64)
    y_39006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'y')
    # Obtaining the member 'shape' of a type (line 64)
    shape_39007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), y_39006, 'shape')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___39008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 4), shape_39007, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_39009 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), getitem___39008, int_39005)
    
    # Assigning a type to the variable 'tuple_var_assignment_38796' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_38796', subscript_call_result_39009)
    
    # Assigning a Subscript to a Name (line 64):
    
    # Obtaining the type of the subscript
    int_39010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'int')
    # Getting the type of 'y' (line 64)
    y_39011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'y')
    # Obtaining the member 'shape' of a type (line 64)
    shape_39012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), y_39011, 'shape')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___39013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 4), shape_39012, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_39014 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), getitem___39013, int_39010)
    
    # Assigning a type to the variable 'tuple_var_assignment_38797' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_38797', subscript_call_result_39014)
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'tuple_var_assignment_38796' (line 64)
    tuple_var_assignment_38796_39015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_38796')
    # Assigning a type to the variable 'n' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'n', tuple_var_assignment_38796_39015)
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'tuple_var_assignment_38797' (line 64)
    tuple_var_assignment_38797_39016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_38797')
    # Assigning a type to the variable 'm' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 7), 'm', tuple_var_assignment_38797_39016)
    
    # Assigning a Call to a Name (line 65):
    
    # Assigning a Call to a Name (line 65):
    
    # Call to empty(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Obtaining an instance of the builtin type 'tuple' (line 65)
    tuple_39019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 65)
    # Adding element type (line 65)
    # Getting the type of 'n' (line 65)
    n_39020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 22), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), tuple_39019, n_39020)
    # Adding element type (line 65)
    int_39021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), tuple_39019, int_39021)
    # Adding element type (line 65)
    # Getting the type of 'm' (line 65)
    m_39022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), tuple_39019, m_39022)
    
    # Processing the call keyword arguments (line 65)
    kwargs_39023 = {}
    # Getting the type of 'np' (line 65)
    np_39017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'np', False)
    # Obtaining the member 'empty' of a type (line 65)
    empty_39018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), np_39017, 'empty')
    # Calling empty(args, kwargs) (line 65)
    empty_call_result_39024 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), empty_39018, *[tuple_39019], **kwargs_39023)
    
    # Assigning a type to the variable 'df_dy' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'df_dy', empty_call_result_39024)
    
    # Assigning a Num to a Subscript (line 66):
    
    # Assigning a Num to a Subscript (line 66):
    int_39025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 18), 'int')
    # Getting the type of 'df_dy' (line 66)
    df_dy_39026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'df_dy')
    
    # Obtaining an instance of the builtin type 'tuple' (line 66)
    tuple_39027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 66)
    # Adding element type (line 66)
    int_39028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 10), tuple_39027, int_39028)
    # Adding element type (line 66)
    int_39029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 10), tuple_39027, int_39029)
    
    # Storing an element on a container (line 66)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 4), df_dy_39026, (tuple_39027, int_39025))
    
    # Assigning a Num to a Subscript (line 67):
    
    # Assigning a Num to a Subscript (line 67):
    int_39030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 18), 'int')
    # Getting the type of 'df_dy' (line 67)
    df_dy_39031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'df_dy')
    
    # Obtaining an instance of the builtin type 'tuple' (line 67)
    tuple_39032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 67)
    # Adding element type (line 67)
    int_39033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), tuple_39032, int_39033)
    # Adding element type (line 67)
    int_39034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), tuple_39032, int_39034)
    
    # Storing an element on a container (line 67)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 4), df_dy_39031, (tuple_39032, int_39030))
    
    # Assigning a UnaryOp to a Subscript (line 68):
    
    # Assigning a UnaryOp to a Subscript (line 68):
    
    
    # Obtaining the type of the subscript
    int_39035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'int')
    # Getting the type of 'p' (line 68)
    p_39036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'p')
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___39037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), p_39036, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_39038 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), getitem___39037, int_39035)
    
    int_39039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'int')
    # Applying the binary operator '**' (line 68)
    result_pow_39040 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 19), '**', subscript_call_result_39038, int_39039)
    
    # Applying the 'usub' unary operator (line 68)
    result___neg___39041 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 18), 'usub', result_pow_39040)
    
    # Getting the type of 'df_dy' (line 68)
    df_dy_39042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'df_dy')
    
    # Obtaining an instance of the builtin type 'tuple' (line 68)
    tuple_39043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 68)
    # Adding element type (line 68)
    int_39044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 10), tuple_39043, int_39044)
    # Adding element type (line 68)
    int_39045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 10), tuple_39043, int_39045)
    
    # Storing an element on a container (line 68)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 4), df_dy_39042, (tuple_39043, result___neg___39041))
    
    # Assigning a Num to a Subscript (line 69):
    
    # Assigning a Num to a Subscript (line 69):
    int_39046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 18), 'int')
    # Getting the type of 'df_dy' (line 69)
    df_dy_39047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'df_dy')
    
    # Obtaining an instance of the builtin type 'tuple' (line 69)
    tuple_39048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 69)
    # Adding element type (line 69)
    int_39049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), tuple_39048, int_39049)
    # Adding element type (line 69)
    int_39050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), tuple_39048, int_39050)
    
    # Storing an element on a container (line 69)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 4), df_dy_39047, (tuple_39048, int_39046))
    
    # Assigning a Call to a Name (line 71):
    
    # Assigning a Call to a Name (line 71):
    
    # Call to empty(...): (line 71)
    # Processing the call arguments (line 71)
    
    # Obtaining an instance of the builtin type 'tuple' (line 71)
    tuple_39053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 71)
    # Adding element type (line 71)
    # Getting the type of 'n' (line 71)
    n_39054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), tuple_39053, n_39054)
    # Adding element type (line 71)
    int_39055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), tuple_39053, int_39055)
    # Adding element type (line 71)
    # Getting the type of 'm' (line 71)
    m_39056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 28), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 22), tuple_39053, m_39056)
    
    # Processing the call keyword arguments (line 71)
    kwargs_39057 = {}
    # Getting the type of 'np' (line 71)
    np_39051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'np', False)
    # Obtaining the member 'empty' of a type (line 71)
    empty_39052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), np_39051, 'empty')
    # Calling empty(args, kwargs) (line 71)
    empty_call_result_39058 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), empty_39052, *[tuple_39053], **kwargs_39057)
    
    # Assigning a type to the variable 'df_dp' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'df_dp', empty_call_result_39058)
    
    # Assigning a Num to a Subscript (line 72):
    
    # Assigning a Num to a Subscript (line 72):
    int_39059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 18), 'int')
    # Getting the type of 'df_dp' (line 72)
    df_dp_39060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'df_dp')
    
    # Obtaining an instance of the builtin type 'tuple' (line 72)
    tuple_39061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 72)
    # Adding element type (line 72)
    int_39062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 10), tuple_39061, int_39062)
    # Adding element type (line 72)
    int_39063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 10), tuple_39061, int_39063)
    
    # Storing an element on a container (line 72)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 4), df_dp_39060, (tuple_39061, int_39059))
    
    # Assigning a BinOp to a Subscript (line 73):
    
    # Assigning a BinOp to a Subscript (line 73):
    int_39064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 18), 'int')
    
    # Obtaining the type of the subscript
    int_39065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 25), 'int')
    # Getting the type of 'p' (line 73)
    p_39066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'p')
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___39067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), p_39066, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
    subscript_call_result_39068 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), getitem___39067, int_39065)
    
    # Applying the binary operator '*' (line 73)
    result_mul_39069 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 18), '*', int_39064, subscript_call_result_39068)
    
    
    # Obtaining the type of the subscript
    int_39070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 32), 'int')
    # Getting the type of 'y' (line 73)
    y_39071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'y')
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___39072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 30), y_39071, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
    subscript_call_result_39073 = invoke(stypy.reporting.localization.Localization(__file__, 73, 30), getitem___39072, int_39070)
    
    # Applying the binary operator '*' (line 73)
    result_mul_39074 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 28), '*', result_mul_39069, subscript_call_result_39073)
    
    # Getting the type of 'df_dp' (line 73)
    df_dp_39075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'df_dp')
    
    # Obtaining an instance of the builtin type 'tuple' (line 73)
    tuple_39076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 73)
    # Adding element type (line 73)
    int_39077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 10), tuple_39076, int_39077)
    # Adding element type (line 73)
    int_39078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 10), tuple_39076, int_39078)
    
    # Storing an element on a container (line 73)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 4), df_dp_39075, (tuple_39076, result_mul_39074))
    
    # Obtaining an instance of the builtin type 'tuple' (line 75)
    tuple_39079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 75)
    # Adding element type (line 75)
    # Getting the type of 'df_dy' (line 75)
    df_dy_39080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'df_dy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 11), tuple_39079, df_dy_39080)
    # Adding element type (line 75)
    # Getting the type of 'df_dp' (line 75)
    df_dp_39081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'df_dp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 11), tuple_39079, df_dp_39081)
    
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type', tuple_39079)
    
    # ################# End of 'sl_fun_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sl_fun_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 63)
    stypy_return_type_39082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39082)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sl_fun_jac'
    return stypy_return_type_39082

# Assigning a type to the variable 'sl_fun_jac' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'sl_fun_jac', sl_fun_jac)

@norecursion
def sl_bc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sl_bc'
    module_type_store = module_type_store.open_function_context('sl_bc', 78, 0, False)
    
    # Passed parameters checking function
    sl_bc.stypy_localization = localization
    sl_bc.stypy_type_of_self = None
    sl_bc.stypy_type_store = module_type_store
    sl_bc.stypy_function_name = 'sl_bc'
    sl_bc.stypy_param_names_list = ['ya', 'yb', 'p']
    sl_bc.stypy_varargs_param_name = None
    sl_bc.stypy_kwargs_param_name = None
    sl_bc.stypy_call_defaults = defaults
    sl_bc.stypy_call_varargs = varargs
    sl_bc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sl_bc', ['ya', 'yb', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sl_bc', localization, ['ya', 'yb', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sl_bc(...)' code ##################

    
    # Call to hstack(...): (line 79)
    # Processing the call arguments (line 79)
    
    # Obtaining an instance of the builtin type 'tuple' (line 79)
    tuple_39085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 79)
    # Adding element type (line 79)
    
    # Obtaining the type of the subscript
    int_39086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 25), 'int')
    # Getting the type of 'ya' (line 79)
    ya_39087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 22), 'ya', False)
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___39088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 22), ya_39087, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_39089 = invoke(stypy.reporting.localization.Localization(__file__, 79, 22), getitem___39088, int_39086)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 22), tuple_39085, subscript_call_result_39089)
    # Adding element type (line 79)
    
    # Obtaining the type of the subscript
    int_39090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 32), 'int')
    # Getting the type of 'yb' (line 79)
    yb_39091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'yb', False)
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___39092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 29), yb_39091, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_39093 = invoke(stypy.reporting.localization.Localization(__file__, 79, 29), getitem___39092, int_39090)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 22), tuple_39085, subscript_call_result_39093)
    # Adding element type (line 79)
    
    # Obtaining the type of the subscript
    int_39094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 39), 'int')
    # Getting the type of 'ya' (line 79)
    ya_39095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 36), 'ya', False)
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___39096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 36), ya_39095, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_39097 = invoke(stypy.reporting.localization.Localization(__file__, 79, 36), getitem___39096, int_39094)
    
    
    # Obtaining the type of the subscript
    int_39098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 46), 'int')
    # Getting the type of 'p' (line 79)
    p_39099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 44), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___39100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 44), p_39099, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_39101 = invoke(stypy.reporting.localization.Localization(__file__, 79, 44), getitem___39100, int_39098)
    
    # Applying the binary operator '-' (line 79)
    result_sub_39102 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 36), '-', subscript_call_result_39097, subscript_call_result_39101)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 22), tuple_39085, result_sub_39102)
    
    # Processing the call keyword arguments (line 79)
    kwargs_39103 = {}
    # Getting the type of 'np' (line 79)
    np_39083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'np', False)
    # Obtaining the member 'hstack' of a type (line 79)
    hstack_39084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), np_39083, 'hstack')
    # Calling hstack(args, kwargs) (line 79)
    hstack_call_result_39104 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), hstack_39084, *[tuple_39085], **kwargs_39103)
    
    # Assigning a type to the variable 'stypy_return_type' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type', hstack_call_result_39104)
    
    # ################# End of 'sl_bc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sl_bc' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_39105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39105)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sl_bc'
    return stypy_return_type_39105

# Assigning a type to the variable 'sl_bc' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'sl_bc', sl_bc)

@norecursion
def sl_bc_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sl_bc_jac'
    module_type_store = module_type_store.open_function_context('sl_bc_jac', 82, 0, False)
    
    # Passed parameters checking function
    sl_bc_jac.stypy_localization = localization
    sl_bc_jac.stypy_type_of_self = None
    sl_bc_jac.stypy_type_store = module_type_store
    sl_bc_jac.stypy_function_name = 'sl_bc_jac'
    sl_bc_jac.stypy_param_names_list = ['ya', 'yb', 'p']
    sl_bc_jac.stypy_varargs_param_name = None
    sl_bc_jac.stypy_kwargs_param_name = None
    sl_bc_jac.stypy_call_defaults = defaults
    sl_bc_jac.stypy_call_varargs = varargs
    sl_bc_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sl_bc_jac', ['ya', 'yb', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sl_bc_jac', localization, ['ya', 'yb', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sl_bc_jac(...)' code ##################

    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to zeros(...): (line 83)
    # Processing the call arguments (line 83)
    
    # Obtaining an instance of the builtin type 'tuple' (line 83)
    tuple_39108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 83)
    # Adding element type (line 83)
    int_39109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 24), tuple_39108, int_39109)
    # Adding element type (line 83)
    int_39110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 24), tuple_39108, int_39110)
    
    # Processing the call keyword arguments (line 83)
    kwargs_39111 = {}
    # Getting the type of 'np' (line 83)
    np_39106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 14), 'np', False)
    # Obtaining the member 'zeros' of a type (line 83)
    zeros_39107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 14), np_39106, 'zeros')
    # Calling zeros(args, kwargs) (line 83)
    zeros_call_result_39112 = invoke(stypy.reporting.localization.Localization(__file__, 83, 14), zeros_39107, *[tuple_39108], **kwargs_39111)
    
    # Assigning a type to the variable 'dbc_dya' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'dbc_dya', zeros_call_result_39112)
    
    # Assigning a Num to a Subscript (line 84):
    
    # Assigning a Num to a Subscript (line 84):
    int_39113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 20), 'int')
    # Getting the type of 'dbc_dya' (line 84)
    dbc_dya_39114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'dbc_dya')
    
    # Obtaining an instance of the builtin type 'tuple' (line 84)
    tuple_39115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 84)
    # Adding element type (line 84)
    int_39116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 12), tuple_39115, int_39116)
    # Adding element type (line 84)
    int_39117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 12), tuple_39115, int_39117)
    
    # Storing an element on a container (line 84)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 4), dbc_dya_39114, (tuple_39115, int_39113))
    
    # Assigning a Num to a Subscript (line 85):
    
    # Assigning a Num to a Subscript (line 85):
    int_39118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 20), 'int')
    # Getting the type of 'dbc_dya' (line 85)
    dbc_dya_39119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'dbc_dya')
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_39120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    int_39121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 12), tuple_39120, int_39121)
    # Adding element type (line 85)
    int_39122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 12), tuple_39120, int_39122)
    
    # Storing an element on a container (line 85)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 4), dbc_dya_39119, (tuple_39120, int_39118))
    
    # Assigning a Call to a Name (line 87):
    
    # Assigning a Call to a Name (line 87):
    
    # Call to zeros(...): (line 87)
    # Processing the call arguments (line 87)
    
    # Obtaining an instance of the builtin type 'tuple' (line 87)
    tuple_39125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 87)
    # Adding element type (line 87)
    int_39126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 24), tuple_39125, int_39126)
    # Adding element type (line 87)
    int_39127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 24), tuple_39125, int_39127)
    
    # Processing the call keyword arguments (line 87)
    kwargs_39128 = {}
    # Getting the type of 'np' (line 87)
    np_39123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 14), 'np', False)
    # Obtaining the member 'zeros' of a type (line 87)
    zeros_39124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 14), np_39123, 'zeros')
    # Calling zeros(args, kwargs) (line 87)
    zeros_call_result_39129 = invoke(stypy.reporting.localization.Localization(__file__, 87, 14), zeros_39124, *[tuple_39125], **kwargs_39128)
    
    # Assigning a type to the variable 'dbc_dyb' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'dbc_dyb', zeros_call_result_39129)
    
    # Assigning a Num to a Subscript (line 88):
    
    # Assigning a Num to a Subscript (line 88):
    int_39130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'int')
    # Getting the type of 'dbc_dyb' (line 88)
    dbc_dyb_39131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'dbc_dyb')
    
    # Obtaining an instance of the builtin type 'tuple' (line 88)
    tuple_39132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 88)
    # Adding element type (line 88)
    int_39133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 12), tuple_39132, int_39133)
    # Adding element type (line 88)
    int_39134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 12), tuple_39132, int_39134)
    
    # Storing an element on a container (line 88)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 4), dbc_dyb_39131, (tuple_39132, int_39130))
    
    # Assigning a Call to a Name (line 90):
    
    # Assigning a Call to a Name (line 90):
    
    # Call to zeros(...): (line 90)
    # Processing the call arguments (line 90)
    
    # Obtaining an instance of the builtin type 'tuple' (line 90)
    tuple_39137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 90)
    # Adding element type (line 90)
    int_39138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 23), tuple_39137, int_39138)
    # Adding element type (line 90)
    int_39139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 23), tuple_39137, int_39139)
    
    # Processing the call keyword arguments (line 90)
    kwargs_39140 = {}
    # Getting the type of 'np' (line 90)
    np_39135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 13), 'np', False)
    # Obtaining the member 'zeros' of a type (line 90)
    zeros_39136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 13), np_39135, 'zeros')
    # Calling zeros(args, kwargs) (line 90)
    zeros_call_result_39141 = invoke(stypy.reporting.localization.Localization(__file__, 90, 13), zeros_39136, *[tuple_39137], **kwargs_39140)
    
    # Assigning a type to the variable 'dbc_dp' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'dbc_dp', zeros_call_result_39141)
    
    # Assigning a Num to a Subscript (line 91):
    
    # Assigning a Num to a Subscript (line 91):
    int_39142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'int')
    # Getting the type of 'dbc_dp' (line 91)
    dbc_dp_39143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'dbc_dp')
    
    # Obtaining an instance of the builtin type 'tuple' (line 91)
    tuple_39144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 91)
    # Adding element type (line 91)
    int_39145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 11), tuple_39144, int_39145)
    # Adding element type (line 91)
    int_39146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 11), tuple_39144, int_39146)
    
    # Storing an element on a container (line 91)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 4), dbc_dp_39143, (tuple_39144, int_39142))
    
    # Obtaining an instance of the builtin type 'tuple' (line 93)
    tuple_39147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 93)
    # Adding element type (line 93)
    # Getting the type of 'dbc_dya' (line 93)
    dbc_dya_39148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'dbc_dya')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 11), tuple_39147, dbc_dya_39148)
    # Adding element type (line 93)
    # Getting the type of 'dbc_dyb' (line 93)
    dbc_dyb_39149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'dbc_dyb')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 11), tuple_39147, dbc_dyb_39149)
    # Adding element type (line 93)
    # Getting the type of 'dbc_dp' (line 93)
    dbc_dp_39150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 29), 'dbc_dp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 11), tuple_39147, dbc_dp_39150)
    
    # Assigning a type to the variable 'stypy_return_type' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type', tuple_39147)
    
    # ################# End of 'sl_bc_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sl_bc_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_39151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39151)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sl_bc_jac'
    return stypy_return_type_39151

# Assigning a type to the variable 'sl_bc_jac' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'sl_bc_jac', sl_bc_jac)

@norecursion
def sl_sol(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sl_sol'
    module_type_store = module_type_store.open_function_context('sl_sol', 96, 0, False)
    
    # Passed parameters checking function
    sl_sol.stypy_localization = localization
    sl_sol.stypy_type_of_self = None
    sl_sol.stypy_type_store = module_type_store
    sl_sol.stypy_function_name = 'sl_sol'
    sl_sol.stypy_param_names_list = ['x', 'p']
    sl_sol.stypy_varargs_param_name = None
    sl_sol.stypy_kwargs_param_name = None
    sl_sol.stypy_call_defaults = defaults
    sl_sol.stypy_call_varargs = varargs
    sl_sol.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sl_sol', ['x', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sl_sol', localization, ['x', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sl_sol(...)' code ##################

    
    # Call to sin(...): (line 97)
    # Processing the call arguments (line 97)
    
    # Obtaining the type of the subscript
    int_39154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 20), 'int')
    # Getting the type of 'p' (line 97)
    p_39155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___39156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 18), p_39155, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_39157 = invoke(stypy.reporting.localization.Localization(__file__, 97, 18), getitem___39156, int_39154)
    
    # Getting the type of 'x' (line 97)
    x_39158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 25), 'x', False)
    # Applying the binary operator '*' (line 97)
    result_mul_39159 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 18), '*', subscript_call_result_39157, x_39158)
    
    # Processing the call keyword arguments (line 97)
    kwargs_39160 = {}
    # Getting the type of 'np' (line 97)
    np_39152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'np', False)
    # Obtaining the member 'sin' of a type (line 97)
    sin_39153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), np_39152, 'sin')
    # Calling sin(args, kwargs) (line 97)
    sin_call_result_39161 = invoke(stypy.reporting.localization.Localization(__file__, 97, 11), sin_39153, *[result_mul_39159], **kwargs_39160)
    
    # Assigning a type to the variable 'stypy_return_type' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type', sin_call_result_39161)
    
    # ################# End of 'sl_sol(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sl_sol' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_39162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39162)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sl_sol'
    return stypy_return_type_39162

# Assigning a type to the variable 'sl_sol' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'sl_sol', sl_sol)

@norecursion
def emden_fun(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'emden_fun'
    module_type_store = module_type_store.open_function_context('emden_fun', 100, 0, False)
    
    # Passed parameters checking function
    emden_fun.stypy_localization = localization
    emden_fun.stypy_type_of_self = None
    emden_fun.stypy_type_store = module_type_store
    emden_fun.stypy_function_name = 'emden_fun'
    emden_fun.stypy_param_names_list = ['x', 'y']
    emden_fun.stypy_varargs_param_name = None
    emden_fun.stypy_kwargs_param_name = None
    emden_fun.stypy_call_defaults = defaults
    emden_fun.stypy_call_varargs = varargs
    emden_fun.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'emden_fun', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'emden_fun', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'emden_fun(...)' code ##################

    
    # Call to vstack(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Obtaining an instance of the builtin type 'tuple' (line 101)
    tuple_39165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 101)
    # Adding element type (line 101)
    
    # Obtaining the type of the subscript
    int_39166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 24), 'int')
    # Getting the type of 'y' (line 101)
    y_39167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___39168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 22), y_39167, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_39169 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), getitem___39168, int_39166)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 22), tuple_39165, subscript_call_result_39169)
    # Adding element type (line 101)
    
    
    # Obtaining the type of the subscript
    int_39170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 31), 'int')
    # Getting the type of 'y' (line 101)
    y_39171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 29), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___39172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 29), y_39171, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_39173 = invoke(stypy.reporting.localization.Localization(__file__, 101, 29), getitem___39172, int_39170)
    
    int_39174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 35), 'int')
    # Applying the binary operator '**' (line 101)
    result_pow_39175 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 29), '**', subscript_call_result_39173, int_39174)
    
    # Applying the 'usub' unary operator (line 101)
    result___neg___39176 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 28), 'usub', result_pow_39175)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 22), tuple_39165, result___neg___39176)
    
    # Processing the call keyword arguments (line 101)
    kwargs_39177 = {}
    # Getting the type of 'np' (line 101)
    np_39163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'np', False)
    # Obtaining the member 'vstack' of a type (line 101)
    vstack_39164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 11), np_39163, 'vstack')
    # Calling vstack(args, kwargs) (line 101)
    vstack_call_result_39178 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), vstack_39164, *[tuple_39165], **kwargs_39177)
    
    # Assigning a type to the variable 'stypy_return_type' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type', vstack_call_result_39178)
    
    # ################# End of 'emden_fun(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'emden_fun' in the type store
    # Getting the type of 'stypy_return_type' (line 100)
    stypy_return_type_39179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39179)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'emden_fun'
    return stypy_return_type_39179

# Assigning a type to the variable 'emden_fun' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'emden_fun', emden_fun)

@norecursion
def emden_fun_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'emden_fun_jac'
    module_type_store = module_type_store.open_function_context('emden_fun_jac', 104, 0, False)
    
    # Passed parameters checking function
    emden_fun_jac.stypy_localization = localization
    emden_fun_jac.stypy_type_of_self = None
    emden_fun_jac.stypy_type_store = module_type_store
    emden_fun_jac.stypy_function_name = 'emden_fun_jac'
    emden_fun_jac.stypy_param_names_list = ['x', 'y']
    emden_fun_jac.stypy_varargs_param_name = None
    emden_fun_jac.stypy_kwargs_param_name = None
    emden_fun_jac.stypy_call_defaults = defaults
    emden_fun_jac.stypy_call_varargs = varargs
    emden_fun_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'emden_fun_jac', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'emden_fun_jac', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'emden_fun_jac(...)' code ##################

    
    # Assigning a Call to a Name (line 105):
    
    # Assigning a Call to a Name (line 105):
    
    # Call to empty(...): (line 105)
    # Processing the call arguments (line 105)
    
    # Obtaining an instance of the builtin type 'tuple' (line 105)
    tuple_39182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 105)
    # Adding element type (line 105)
    int_39183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 22), tuple_39182, int_39183)
    # Adding element type (line 105)
    int_39184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 22), tuple_39182, int_39184)
    # Adding element type (line 105)
    
    # Obtaining the type of the subscript
    int_39185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 36), 'int')
    # Getting the type of 'x' (line 105)
    x_39186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 28), 'x', False)
    # Obtaining the member 'shape' of a type (line 105)
    shape_39187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 28), x_39186, 'shape')
    # Obtaining the member '__getitem__' of a type (line 105)
    getitem___39188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 28), shape_39187, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 105)
    subscript_call_result_39189 = invoke(stypy.reporting.localization.Localization(__file__, 105, 28), getitem___39188, int_39185)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 22), tuple_39182, subscript_call_result_39189)
    
    # Processing the call keyword arguments (line 105)
    kwargs_39190 = {}
    # Getting the type of 'np' (line 105)
    np_39180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'np', False)
    # Obtaining the member 'empty' of a type (line 105)
    empty_39181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), np_39180, 'empty')
    # Calling empty(args, kwargs) (line 105)
    empty_call_result_39191 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), empty_39181, *[tuple_39182], **kwargs_39190)
    
    # Assigning a type to the variable 'df_dy' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'df_dy', empty_call_result_39191)
    
    # Assigning a Num to a Subscript (line 106):
    
    # Assigning a Num to a Subscript (line 106):
    int_39192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 18), 'int')
    # Getting the type of 'df_dy' (line 106)
    df_dy_39193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'df_dy')
    
    # Obtaining an instance of the builtin type 'tuple' (line 106)
    tuple_39194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 106)
    # Adding element type (line 106)
    int_39195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 10), tuple_39194, int_39195)
    # Adding element type (line 106)
    int_39196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 10), tuple_39194, int_39196)
    
    # Storing an element on a container (line 106)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 4), df_dy_39193, (tuple_39194, int_39192))
    
    # Assigning a Num to a Subscript (line 107):
    
    # Assigning a Num to a Subscript (line 107):
    int_39197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 18), 'int')
    # Getting the type of 'df_dy' (line 107)
    df_dy_39198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'df_dy')
    
    # Obtaining an instance of the builtin type 'tuple' (line 107)
    tuple_39199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 107)
    # Adding element type (line 107)
    int_39200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 10), tuple_39199, int_39200)
    # Adding element type (line 107)
    int_39201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 10), tuple_39199, int_39201)
    
    # Storing an element on a container (line 107)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 4), df_dy_39198, (tuple_39199, int_39197))
    
    # Assigning a BinOp to a Subscript (line 108):
    
    # Assigning a BinOp to a Subscript (line 108):
    int_39202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 18), 'int')
    
    # Obtaining the type of the subscript
    int_39203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 25), 'int')
    # Getting the type of 'y' (line 108)
    y_39204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'y')
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___39205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 23), y_39204, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_39206 = invoke(stypy.reporting.localization.Localization(__file__, 108, 23), getitem___39205, int_39203)
    
    int_39207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 29), 'int')
    # Applying the binary operator '**' (line 108)
    result_pow_39208 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 23), '**', subscript_call_result_39206, int_39207)
    
    # Applying the binary operator '*' (line 108)
    result_mul_39209 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 18), '*', int_39202, result_pow_39208)
    
    # Getting the type of 'df_dy' (line 108)
    df_dy_39210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'df_dy')
    
    # Obtaining an instance of the builtin type 'tuple' (line 108)
    tuple_39211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 108)
    # Adding element type (line 108)
    int_39212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 10), tuple_39211, int_39212)
    # Adding element type (line 108)
    int_39213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 10), tuple_39211, int_39213)
    
    # Storing an element on a container (line 108)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 4), df_dy_39210, (tuple_39211, result_mul_39209))
    
    # Assigning a Num to a Subscript (line 109):
    
    # Assigning a Num to a Subscript (line 109):
    int_39214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 18), 'int')
    # Getting the type of 'df_dy' (line 109)
    df_dy_39215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'df_dy')
    
    # Obtaining an instance of the builtin type 'tuple' (line 109)
    tuple_39216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 109)
    # Adding element type (line 109)
    int_39217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 10), tuple_39216, int_39217)
    # Adding element type (line 109)
    int_39218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 10), tuple_39216, int_39218)
    
    # Storing an element on a container (line 109)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 4), df_dy_39215, (tuple_39216, int_39214))
    # Getting the type of 'df_dy' (line 110)
    df_dy_39219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'df_dy')
    # Assigning a type to the variable 'stypy_return_type' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type', df_dy_39219)
    
    # ################# End of 'emden_fun_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'emden_fun_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_39220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39220)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'emden_fun_jac'
    return stypy_return_type_39220

# Assigning a type to the variable 'emden_fun_jac' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'emden_fun_jac', emden_fun_jac)

@norecursion
def emden_bc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'emden_bc'
    module_type_store = module_type_store.open_function_context('emden_bc', 113, 0, False)
    
    # Passed parameters checking function
    emden_bc.stypy_localization = localization
    emden_bc.stypy_type_of_self = None
    emden_bc.stypy_type_store = module_type_store
    emden_bc.stypy_function_name = 'emden_bc'
    emden_bc.stypy_param_names_list = ['ya', 'yb']
    emden_bc.stypy_varargs_param_name = None
    emden_bc.stypy_kwargs_param_name = None
    emden_bc.stypy_call_defaults = defaults
    emden_bc.stypy_call_varargs = varargs
    emden_bc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'emden_bc', ['ya', 'yb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'emden_bc', localization, ['ya', 'yb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'emden_bc(...)' code ##################

    
    # Call to array(...): (line 114)
    # Processing the call arguments (line 114)
    
    # Obtaining an instance of the builtin type 'list' (line 114)
    list_39223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 114)
    # Adding element type (line 114)
    
    # Obtaining the type of the subscript
    int_39224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 24), 'int')
    # Getting the type of 'ya' (line 114)
    ya_39225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'ya', False)
    # Obtaining the member '__getitem__' of a type (line 114)
    getitem___39226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 21), ya_39225, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 114)
    subscript_call_result_39227 = invoke(stypy.reporting.localization.Localization(__file__, 114, 21), getitem___39226, int_39224)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 20), list_39223, subscript_call_result_39227)
    # Adding element type (line 114)
    
    # Obtaining the type of the subscript
    int_39228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 31), 'int')
    # Getting the type of 'yb' (line 114)
    yb_39229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'yb', False)
    # Obtaining the member '__getitem__' of a type (line 114)
    getitem___39230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 28), yb_39229, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 114)
    subscript_call_result_39231 = invoke(stypy.reporting.localization.Localization(__file__, 114, 28), getitem___39230, int_39228)
    
    int_39232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 37), 'int')
    int_39233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 39), 'int')
    # Applying the binary operator 'div' (line 114)
    result_div_39234 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 37), 'div', int_39232, int_39233)
    
    float_39235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 43), 'float')
    # Applying the binary operator '**' (line 114)
    result_pow_39236 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 36), '**', result_div_39234, float_39235)
    
    # Applying the binary operator '-' (line 114)
    result_sub_39237 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 28), '-', subscript_call_result_39231, result_pow_39236)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 20), list_39223, result_sub_39237)
    
    # Processing the call keyword arguments (line 114)
    kwargs_39238 = {}
    # Getting the type of 'np' (line 114)
    np_39221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 114)
    array_39222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 11), np_39221, 'array')
    # Calling array(args, kwargs) (line 114)
    array_call_result_39239 = invoke(stypy.reporting.localization.Localization(__file__, 114, 11), array_39222, *[list_39223], **kwargs_39238)
    
    # Assigning a type to the variable 'stypy_return_type' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type', array_call_result_39239)
    
    # ################# End of 'emden_bc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'emden_bc' in the type store
    # Getting the type of 'stypy_return_type' (line 113)
    stypy_return_type_39240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39240)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'emden_bc'
    return stypy_return_type_39240

# Assigning a type to the variable 'emden_bc' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'emden_bc', emden_bc)

@norecursion
def emden_bc_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'emden_bc_jac'
    module_type_store = module_type_store.open_function_context('emden_bc_jac', 117, 0, False)
    
    # Passed parameters checking function
    emden_bc_jac.stypy_localization = localization
    emden_bc_jac.stypy_type_of_self = None
    emden_bc_jac.stypy_type_store = module_type_store
    emden_bc_jac.stypy_function_name = 'emden_bc_jac'
    emden_bc_jac.stypy_param_names_list = ['ya', 'yb']
    emden_bc_jac.stypy_varargs_param_name = None
    emden_bc_jac.stypy_kwargs_param_name = None
    emden_bc_jac.stypy_call_defaults = defaults
    emden_bc_jac.stypy_call_varargs = varargs
    emden_bc_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'emden_bc_jac', ['ya', 'yb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'emden_bc_jac', localization, ['ya', 'yb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'emden_bc_jac(...)' code ##################

    
    # Assigning a Call to a Name (line 118):
    
    # Assigning a Call to a Name (line 118):
    
    # Call to array(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Obtaining an instance of the builtin type 'list' (line 118)
    list_39243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 118)
    # Adding element type (line 118)
    
    # Obtaining an instance of the builtin type 'list' (line 119)
    list_39244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 119)
    # Adding element type (line 119)
    int_39245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 8), list_39244, int_39245)
    # Adding element type (line 119)
    int_39246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 8), list_39244, int_39246)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 23), list_39243, list_39244)
    # Adding element type (line 118)
    
    # Obtaining an instance of the builtin type 'list' (line 120)
    list_39247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 120)
    # Adding element type (line 120)
    int_39248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 8), list_39247, int_39248)
    # Adding element type (line 120)
    int_39249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 8), list_39247, int_39249)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 23), list_39243, list_39247)
    
    # Processing the call keyword arguments (line 118)
    kwargs_39250 = {}
    # Getting the type of 'np' (line 118)
    np_39241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 118)
    array_39242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 14), np_39241, 'array')
    # Calling array(args, kwargs) (line 118)
    array_call_result_39251 = invoke(stypy.reporting.localization.Localization(__file__, 118, 14), array_39242, *[list_39243], **kwargs_39250)
    
    # Assigning a type to the variable 'dbc_dya' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'dbc_dya', array_call_result_39251)
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to array(...): (line 122)
    # Processing the call arguments (line 122)
    
    # Obtaining an instance of the builtin type 'list' (line 122)
    list_39254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 122)
    # Adding element type (line 122)
    
    # Obtaining an instance of the builtin type 'list' (line 123)
    list_39255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 123)
    # Adding element type (line 123)
    int_39256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 8), list_39255, int_39256)
    # Adding element type (line 123)
    int_39257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 8), list_39255, int_39257)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 23), list_39254, list_39255)
    # Adding element type (line 122)
    
    # Obtaining an instance of the builtin type 'list' (line 124)
    list_39258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 124)
    # Adding element type (line 124)
    int_39259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), list_39258, int_39259)
    # Adding element type (line 124)
    int_39260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), list_39258, int_39260)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 23), list_39254, list_39258)
    
    # Processing the call keyword arguments (line 122)
    kwargs_39261 = {}
    # Getting the type of 'np' (line 122)
    np_39252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 122)
    array_39253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 14), np_39252, 'array')
    # Calling array(args, kwargs) (line 122)
    array_call_result_39262 = invoke(stypy.reporting.localization.Localization(__file__, 122, 14), array_39253, *[list_39254], **kwargs_39261)
    
    # Assigning a type to the variable 'dbc_dyb' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'dbc_dyb', array_call_result_39262)
    
    # Obtaining an instance of the builtin type 'tuple' (line 126)
    tuple_39263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 126)
    # Adding element type (line 126)
    # Getting the type of 'dbc_dya' (line 126)
    dbc_dya_39264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'dbc_dya')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 11), tuple_39263, dbc_dya_39264)
    # Adding element type (line 126)
    # Getting the type of 'dbc_dyb' (line 126)
    dbc_dyb_39265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'dbc_dyb')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 11), tuple_39263, dbc_dyb_39265)
    
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type', tuple_39263)
    
    # ################# End of 'emden_bc_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'emden_bc_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_39266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39266)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'emden_bc_jac'
    return stypy_return_type_39266

# Assigning a type to the variable 'emden_bc_jac' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'emden_bc_jac', emden_bc_jac)

@norecursion
def emden_sol(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'emden_sol'
    module_type_store = module_type_store.open_function_context('emden_sol', 129, 0, False)
    
    # Passed parameters checking function
    emden_sol.stypy_localization = localization
    emden_sol.stypy_type_of_self = None
    emden_sol.stypy_type_store = module_type_store
    emden_sol.stypy_function_name = 'emden_sol'
    emden_sol.stypy_param_names_list = ['x']
    emden_sol.stypy_varargs_param_name = None
    emden_sol.stypy_kwargs_param_name = None
    emden_sol.stypy_call_defaults = defaults
    emden_sol.stypy_call_varargs = varargs
    emden_sol.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'emden_sol', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'emden_sol', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'emden_sol(...)' code ##################

    int_39267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 12), 'int')
    # Getting the type of 'x' (line 130)
    x_39268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'x')
    int_39269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 19), 'int')
    # Applying the binary operator '**' (line 130)
    result_pow_39270 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 16), '**', x_39268, int_39269)
    
    int_39271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 21), 'int')
    # Applying the binary operator 'div' (line 130)
    result_div_39272 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 16), 'div', result_pow_39270, int_39271)
    
    # Applying the binary operator '+' (line 130)
    result_add_39273 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 12), '+', int_39267, result_div_39272)
    
    float_39274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 25), 'float')
    # Applying the binary operator '**' (line 130)
    result_pow_39275 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), '**', result_add_39273, float_39274)
    
    # Assigning a type to the variable 'stypy_return_type' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type', result_pow_39275)
    
    # ################# End of 'emden_sol(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'emden_sol' in the type store
    # Getting the type of 'stypy_return_type' (line 129)
    stypy_return_type_39276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39276)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'emden_sol'
    return stypy_return_type_39276

# Assigning a type to the variable 'emden_sol' (line 129)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'emden_sol', emden_sol)

@norecursion
def undefined_fun(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'undefined_fun'
    module_type_store = module_type_store.open_function_context('undefined_fun', 133, 0, False)
    
    # Passed parameters checking function
    undefined_fun.stypy_localization = localization
    undefined_fun.stypy_type_of_self = None
    undefined_fun.stypy_type_store = module_type_store
    undefined_fun.stypy_function_name = 'undefined_fun'
    undefined_fun.stypy_param_names_list = ['x', 'y']
    undefined_fun.stypy_varargs_param_name = None
    undefined_fun.stypy_kwargs_param_name = None
    undefined_fun.stypy_call_defaults = defaults
    undefined_fun.stypy_call_varargs = varargs
    undefined_fun.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'undefined_fun', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'undefined_fun', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'undefined_fun(...)' code ##################

    
    # Call to zeros_like(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'y' (line 134)
    y_39279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'y', False)
    # Processing the call keyword arguments (line 134)
    kwargs_39280 = {}
    # Getting the type of 'np' (line 134)
    np_39277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 134)
    zeros_like_39278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 11), np_39277, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 134)
    zeros_like_call_result_39281 = invoke(stypy.reporting.localization.Localization(__file__, 134, 11), zeros_like_39278, *[y_39279], **kwargs_39280)
    
    # Assigning a type to the variable 'stypy_return_type' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type', zeros_like_call_result_39281)
    
    # ################# End of 'undefined_fun(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'undefined_fun' in the type store
    # Getting the type of 'stypy_return_type' (line 133)
    stypy_return_type_39282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39282)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'undefined_fun'
    return stypy_return_type_39282

# Assigning a type to the variable 'undefined_fun' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'undefined_fun', undefined_fun)

@norecursion
def undefined_bc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'undefined_bc'
    module_type_store = module_type_store.open_function_context('undefined_bc', 137, 0, False)
    
    # Passed parameters checking function
    undefined_bc.stypy_localization = localization
    undefined_bc.stypy_type_of_self = None
    undefined_bc.stypy_type_store = module_type_store
    undefined_bc.stypy_function_name = 'undefined_bc'
    undefined_bc.stypy_param_names_list = ['ya', 'yb']
    undefined_bc.stypy_varargs_param_name = None
    undefined_bc.stypy_kwargs_param_name = None
    undefined_bc.stypy_call_defaults = defaults
    undefined_bc.stypy_call_varargs = varargs
    undefined_bc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'undefined_bc', ['ya', 'yb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'undefined_bc', localization, ['ya', 'yb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'undefined_bc(...)' code ##################

    
    # Call to array(...): (line 138)
    # Processing the call arguments (line 138)
    
    # Obtaining an instance of the builtin type 'list' (line 138)
    list_39285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 138)
    # Adding element type (line 138)
    
    # Obtaining the type of the subscript
    int_39286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 24), 'int')
    # Getting the type of 'ya' (line 138)
    ya_39287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'ya', False)
    # Obtaining the member '__getitem__' of a type (line 138)
    getitem___39288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 21), ya_39287, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 138)
    subscript_call_result_39289 = invoke(stypy.reporting.localization.Localization(__file__, 138, 21), getitem___39288, int_39286)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 20), list_39285, subscript_call_result_39289)
    # Adding element type (line 138)
    
    # Obtaining the type of the subscript
    int_39290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 31), 'int')
    # Getting the type of 'yb' (line 138)
    yb_39291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'yb', False)
    # Obtaining the member '__getitem__' of a type (line 138)
    getitem___39292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 28), yb_39291, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 138)
    subscript_call_result_39293 = invoke(stypy.reporting.localization.Localization(__file__, 138, 28), getitem___39292, int_39290)
    
    int_39294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 36), 'int')
    # Applying the binary operator '-' (line 138)
    result_sub_39295 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 28), '-', subscript_call_result_39293, int_39294)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 20), list_39285, result_sub_39295)
    
    # Processing the call keyword arguments (line 138)
    kwargs_39296 = {}
    # Getting the type of 'np' (line 138)
    np_39283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 138)
    array_39284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 11), np_39283, 'array')
    # Calling array(args, kwargs) (line 138)
    array_call_result_39297 = invoke(stypy.reporting.localization.Localization(__file__, 138, 11), array_39284, *[list_39285], **kwargs_39296)
    
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type', array_call_result_39297)
    
    # ################# End of 'undefined_bc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'undefined_bc' in the type store
    # Getting the type of 'stypy_return_type' (line 137)
    stypy_return_type_39298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39298)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'undefined_bc'
    return stypy_return_type_39298

# Assigning a type to the variable 'undefined_bc' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'undefined_bc', undefined_bc)

@norecursion
def big_fun(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'big_fun'
    module_type_store = module_type_store.open_function_context('big_fun', 141, 0, False)
    
    # Passed parameters checking function
    big_fun.stypy_localization = localization
    big_fun.stypy_type_of_self = None
    big_fun.stypy_type_store = module_type_store
    big_fun.stypy_function_name = 'big_fun'
    big_fun.stypy_param_names_list = ['x', 'y']
    big_fun.stypy_varargs_param_name = None
    big_fun.stypy_kwargs_param_name = None
    big_fun.stypy_call_defaults = defaults
    big_fun.stypy_call_varargs = varargs
    big_fun.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'big_fun', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'big_fun', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'big_fun(...)' code ##################

    
    # Assigning a Call to a Name (line 142):
    
    # Assigning a Call to a Name (line 142):
    
    # Call to zeros_like(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'y' (line 142)
    y_39301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 22), 'y', False)
    # Processing the call keyword arguments (line 142)
    kwargs_39302 = {}
    # Getting the type of 'np' (line 142)
    np_39299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 142)
    zeros_like_39300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), np_39299, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 142)
    zeros_like_call_result_39303 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), zeros_like_39300, *[y_39301], **kwargs_39302)
    
    # Assigning a type to the variable 'f' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'f', zeros_like_call_result_39303)
    
    # Assigning a Subscript to a Subscript (line 143):
    
    # Assigning a Subscript to a Subscript (line 143):
    
    # Obtaining the type of the subscript
    int_39304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 15), 'int')
    int_39305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 18), 'int')
    slice_39306 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 143, 13), int_39304, None, int_39305)
    # Getting the type of 'y' (line 143)
    y_39307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 13), 'y')
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___39308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 13), y_39307, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_39309 = invoke(stypy.reporting.localization.Localization(__file__, 143, 13), getitem___39308, slice_39306)
    
    # Getting the type of 'f' (line 143)
    f_39310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'f')
    int_39311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 8), 'int')
    slice_39312 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 143, 4), None, None, int_39311)
    # Storing an element on a container (line 143)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 4), f_39310, (slice_39312, subscript_call_result_39309))
    # Getting the type of 'f' (line 144)
    f_39313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'f')
    # Assigning a type to the variable 'stypy_return_type' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type', f_39313)
    
    # ################# End of 'big_fun(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'big_fun' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_39314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39314)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'big_fun'
    return stypy_return_type_39314

# Assigning a type to the variable 'big_fun' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'big_fun', big_fun)

@norecursion
def big_bc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'big_bc'
    module_type_store = module_type_store.open_function_context('big_bc', 147, 0, False)
    
    # Passed parameters checking function
    big_bc.stypy_localization = localization
    big_bc.stypy_type_of_self = None
    big_bc.stypy_type_store = module_type_store
    big_bc.stypy_function_name = 'big_bc'
    big_bc.stypy_param_names_list = ['ya', 'yb']
    big_bc.stypy_varargs_param_name = None
    big_bc.stypy_kwargs_param_name = None
    big_bc.stypy_call_defaults = defaults
    big_bc.stypy_call_varargs = varargs
    big_bc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'big_bc', ['ya', 'yb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'big_bc', localization, ['ya', 'yb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'big_bc(...)' code ##################

    
    # Call to hstack(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Obtaining an instance of the builtin type 'tuple' (line 148)
    tuple_39317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 148)
    # Adding element type (line 148)
    
    # Obtaining the type of the subscript
    int_39318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 27), 'int')
    slice_39319 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 148, 22), None, None, int_39318)
    # Getting the type of 'ya' (line 148)
    ya_39320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 22), 'ya', False)
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___39321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 22), ya_39320, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_39322 = invoke(stypy.reporting.localization.Localization(__file__, 148, 22), getitem___39321, slice_39319)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 22), tuple_39317, subscript_call_result_39322)
    # Adding element type (line 148)
    
    # Obtaining the type of the subscript
    int_39323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 36), 'int')
    slice_39324 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 148, 31), None, None, int_39323)
    # Getting the type of 'yb' (line 148)
    yb_39325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 31), 'yb', False)
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___39326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 31), yb_39325, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_39327 = invoke(stypy.reporting.localization.Localization(__file__, 148, 31), getitem___39326, slice_39324)
    
    int_39328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 41), 'int')
    # Applying the binary operator '-' (line 148)
    result_sub_39329 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 31), '-', subscript_call_result_39327, int_39328)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 22), tuple_39317, result_sub_39329)
    
    # Processing the call keyword arguments (line 148)
    kwargs_39330 = {}
    # Getting the type of 'np' (line 148)
    np_39315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'np', False)
    # Obtaining the member 'hstack' of a type (line 148)
    hstack_39316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 11), np_39315, 'hstack')
    # Calling hstack(args, kwargs) (line 148)
    hstack_call_result_39331 = invoke(stypy.reporting.localization.Localization(__file__, 148, 11), hstack_39316, *[tuple_39317], **kwargs_39330)
    
    # Assigning a type to the variable 'stypy_return_type' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type', hstack_call_result_39331)
    
    # ################# End of 'big_bc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'big_bc' in the type store
    # Getting the type of 'stypy_return_type' (line 147)
    stypy_return_type_39332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39332)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'big_bc'
    return stypy_return_type_39332

# Assigning a type to the variable 'big_bc' (line 147)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'big_bc', big_bc)

@norecursion
def big_sol(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'big_sol'
    module_type_store = module_type_store.open_function_context('big_sol', 151, 0, False)
    
    # Passed parameters checking function
    big_sol.stypy_localization = localization
    big_sol.stypy_type_of_self = None
    big_sol.stypy_type_store = module_type_store
    big_sol.stypy_function_name = 'big_sol'
    big_sol.stypy_param_names_list = ['x', 'n']
    big_sol.stypy_varargs_param_name = None
    big_sol.stypy_kwargs_param_name = None
    big_sol.stypy_call_defaults = defaults
    big_sol.stypy_call_varargs = varargs
    big_sol.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'big_sol', ['x', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'big_sol', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'big_sol(...)' code ##################

    
    # Assigning a Call to a Name (line 152):
    
    # Assigning a Call to a Name (line 152):
    
    # Call to ones(...): (line 152)
    # Processing the call arguments (line 152)
    
    # Obtaining an instance of the builtin type 'tuple' (line 152)
    tuple_39335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 152)
    # Adding element type (line 152)
    int_39336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 17), 'int')
    # Getting the type of 'n' (line 152)
    n_39337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 21), 'n', False)
    # Applying the binary operator '*' (line 152)
    result_mul_39338 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 17), '*', int_39336, n_39337)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 17), tuple_39335, result_mul_39338)
    # Adding element type (line 152)
    # Getting the type of 'x' (line 152)
    x_39339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'x', False)
    # Obtaining the member 'size' of a type (line 152)
    size_39340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 24), x_39339, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 17), tuple_39335, size_39340)
    
    # Processing the call keyword arguments (line 152)
    kwargs_39341 = {}
    # Getting the type of 'np' (line 152)
    np_39333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 152)
    ones_39334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), np_39333, 'ones')
    # Calling ones(args, kwargs) (line 152)
    ones_call_result_39342 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), ones_39334, *[tuple_39335], **kwargs_39341)
    
    # Assigning a type to the variable 'y' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'y', ones_call_result_39342)
    
    # Assigning a Name to a Subscript (line 153):
    
    # Assigning a Name to a Subscript (line 153):
    # Getting the type of 'x' (line 153)
    x_39343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 13), 'x')
    # Getting the type of 'y' (line 153)
    y_39344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'y')
    int_39345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 8), 'int')
    slice_39346 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 153, 4), None, None, int_39345)
    # Storing an element on a container (line 153)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 4), y_39344, (slice_39346, x_39343))
    # Getting the type of 'x' (line 154)
    x_39347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type', x_39347)
    
    # ################# End of 'big_sol(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'big_sol' in the type store
    # Getting the type of 'stypy_return_type' (line 151)
    stypy_return_type_39348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39348)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'big_sol'
    return stypy_return_type_39348

# Assigning a type to the variable 'big_sol' (line 151)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'big_sol', big_sol)

@norecursion
def shock_fun(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'shock_fun'
    module_type_store = module_type_store.open_function_context('shock_fun', 157, 0, False)
    
    # Passed parameters checking function
    shock_fun.stypy_localization = localization
    shock_fun.stypy_type_of_self = None
    shock_fun.stypy_type_store = module_type_store
    shock_fun.stypy_function_name = 'shock_fun'
    shock_fun.stypy_param_names_list = ['x', 'y']
    shock_fun.stypy_varargs_param_name = None
    shock_fun.stypy_kwargs_param_name = None
    shock_fun.stypy_call_defaults = defaults
    shock_fun.stypy_call_varargs = varargs
    shock_fun.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'shock_fun', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'shock_fun', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'shock_fun(...)' code ##################

    
    # Assigning a Num to a Name (line 158):
    
    # Assigning a Num to a Name (line 158):
    float_39349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 10), 'float')
    # Assigning a type to the variable 'eps' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'eps', float_39349)
    
    # Call to vstack(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Obtaining an instance of the builtin type 'tuple' (line 160)
    tuple_39352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 160)
    # Adding element type (line 160)
    
    # Obtaining the type of the subscript
    int_39353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 10), 'int')
    # Getting the type of 'y' (line 160)
    y_39354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___39355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), y_39354, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_39356 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), getitem___39355, int_39353)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 8), tuple_39352, subscript_call_result_39356)
    # Adding element type (line 160)
    
    # Getting the type of 'x' (line 161)
    x_39357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 10), 'x', False)
    
    # Obtaining the type of the subscript
    int_39358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'int')
    # Getting the type of 'y' (line 161)
    y_39359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 14), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 161)
    getitem___39360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 14), y_39359, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 161)
    subscript_call_result_39361 = invoke(stypy.reporting.localization.Localization(__file__, 161, 14), getitem___39360, int_39358)
    
    # Applying the binary operator '*' (line 161)
    result_mul_39362 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 10), '*', x_39357, subscript_call_result_39361)
    
    # Getting the type of 'eps' (line 161)
    eps_39363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 21), 'eps', False)
    # Getting the type of 'np' (line 161)
    np_39364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 27), 'np', False)
    # Obtaining the member 'pi' of a type (line 161)
    pi_39365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 27), np_39364, 'pi')
    int_39366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 34), 'int')
    # Applying the binary operator '**' (line 161)
    result_pow_39367 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 27), '**', pi_39365, int_39366)
    
    # Applying the binary operator '*' (line 161)
    result_mul_39368 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 21), '*', eps_39363, result_pow_39367)
    
    
    # Call to cos(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'np' (line 161)
    np_39371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 45), 'np', False)
    # Obtaining the member 'pi' of a type (line 161)
    pi_39372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 45), np_39371, 'pi')
    # Getting the type of 'x' (line 161)
    x_39373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 53), 'x', False)
    # Applying the binary operator '*' (line 161)
    result_mul_39374 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 45), '*', pi_39372, x_39373)
    
    # Processing the call keyword arguments (line 161)
    kwargs_39375 = {}
    # Getting the type of 'np' (line 161)
    np_39369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 38), 'np', False)
    # Obtaining the member 'cos' of a type (line 161)
    cos_39370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 38), np_39369, 'cos')
    # Calling cos(args, kwargs) (line 161)
    cos_call_result_39376 = invoke(stypy.reporting.localization.Localization(__file__, 161, 38), cos_39370, *[result_mul_39374], **kwargs_39375)
    
    # Applying the binary operator '*' (line 161)
    result_mul_39377 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 36), '*', result_mul_39368, cos_call_result_39376)
    
    # Applying the binary operator '+' (line 161)
    result_add_39378 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 10), '+', result_mul_39362, result_mul_39377)
    
    # Getting the type of 'np' (line 162)
    np_39379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 10), 'np', False)
    # Obtaining the member 'pi' of a type (line 162)
    pi_39380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 10), np_39379, 'pi')
    # Getting the type of 'x' (line 162)
    x_39381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 18), 'x', False)
    # Applying the binary operator '*' (line 162)
    result_mul_39382 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 10), '*', pi_39380, x_39381)
    
    
    # Call to sin(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'np' (line 162)
    np_39385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'np', False)
    # Obtaining the member 'pi' of a type (line 162)
    pi_39386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 29), np_39385, 'pi')
    # Getting the type of 'x' (line 162)
    x_39387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 37), 'x', False)
    # Applying the binary operator '*' (line 162)
    result_mul_39388 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 29), '*', pi_39386, x_39387)
    
    # Processing the call keyword arguments (line 162)
    kwargs_39389 = {}
    # Getting the type of 'np' (line 162)
    np_39383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 22), 'np', False)
    # Obtaining the member 'sin' of a type (line 162)
    sin_39384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 22), np_39383, 'sin')
    # Calling sin(args, kwargs) (line 162)
    sin_call_result_39390 = invoke(stypy.reporting.localization.Localization(__file__, 162, 22), sin_39384, *[result_mul_39388], **kwargs_39389)
    
    # Applying the binary operator '*' (line 162)
    result_mul_39391 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 20), '*', result_mul_39382, sin_call_result_39390)
    
    # Applying the binary operator '+' (line 161)
    result_add_39392 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 56), '+', result_add_39378, result_mul_39391)
    
    # Applying the 'usub' unary operator (line 161)
    result___neg___39393 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 8), 'usub', result_add_39392)
    
    # Getting the type of 'eps' (line 162)
    eps_39394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 43), 'eps', False)
    # Applying the binary operator 'div' (line 161)
    result_div_39395 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 8), 'div', result___neg___39393, eps_39394)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 8), tuple_39352, result_div_39395)
    
    # Processing the call keyword arguments (line 159)
    kwargs_39396 = {}
    # Getting the type of 'np' (line 159)
    np_39350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'np', False)
    # Obtaining the member 'vstack' of a type (line 159)
    vstack_39351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), np_39350, 'vstack')
    # Calling vstack(args, kwargs) (line 159)
    vstack_call_result_39397 = invoke(stypy.reporting.localization.Localization(__file__, 159, 11), vstack_39351, *[tuple_39352], **kwargs_39396)
    
    # Assigning a type to the variable 'stypy_return_type' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'stypy_return_type', vstack_call_result_39397)
    
    # ################# End of 'shock_fun(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'shock_fun' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_39398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39398)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'shock_fun'
    return stypy_return_type_39398

# Assigning a type to the variable 'shock_fun' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'shock_fun', shock_fun)

@norecursion
def shock_bc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'shock_bc'
    module_type_store = module_type_store.open_function_context('shock_bc', 166, 0, False)
    
    # Passed parameters checking function
    shock_bc.stypy_localization = localization
    shock_bc.stypy_type_of_self = None
    shock_bc.stypy_type_store = module_type_store
    shock_bc.stypy_function_name = 'shock_bc'
    shock_bc.stypy_param_names_list = ['ya', 'yb']
    shock_bc.stypy_varargs_param_name = None
    shock_bc.stypy_kwargs_param_name = None
    shock_bc.stypy_call_defaults = defaults
    shock_bc.stypy_call_varargs = varargs
    shock_bc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'shock_bc', ['ya', 'yb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'shock_bc', localization, ['ya', 'yb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'shock_bc(...)' code ##################

    
    # Call to array(...): (line 167)
    # Processing the call arguments (line 167)
    
    # Obtaining an instance of the builtin type 'list' (line 167)
    list_39401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 167)
    # Adding element type (line 167)
    
    # Obtaining the type of the subscript
    int_39402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 24), 'int')
    # Getting the type of 'ya' (line 167)
    ya_39403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'ya', False)
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___39404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 21), ya_39403, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_39405 = invoke(stypy.reporting.localization.Localization(__file__, 167, 21), getitem___39404, int_39402)
    
    int_39406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 29), 'int')
    # Applying the binary operator '+' (line 167)
    result_add_39407 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 21), '+', subscript_call_result_39405, int_39406)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 20), list_39401, result_add_39407)
    # Adding element type (line 167)
    
    # Obtaining the type of the subscript
    int_39408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 35), 'int')
    # Getting the type of 'yb' (line 167)
    yb_39409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 32), 'yb', False)
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___39410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 32), yb_39409, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_39411 = invoke(stypy.reporting.localization.Localization(__file__, 167, 32), getitem___39410, int_39408)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 20), list_39401, subscript_call_result_39411)
    
    # Processing the call keyword arguments (line 167)
    kwargs_39412 = {}
    # Getting the type of 'np' (line 167)
    np_39399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 167)
    array_39400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 11), np_39399, 'array')
    # Calling array(args, kwargs) (line 167)
    array_call_result_39413 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), array_39400, *[list_39401], **kwargs_39412)
    
    # Assigning a type to the variable 'stypy_return_type' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type', array_call_result_39413)
    
    # ################# End of 'shock_bc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'shock_bc' in the type store
    # Getting the type of 'stypy_return_type' (line 166)
    stypy_return_type_39414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39414)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'shock_bc'
    return stypy_return_type_39414

# Assigning a type to the variable 'shock_bc' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'shock_bc', shock_bc)

@norecursion
def shock_sol(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'shock_sol'
    module_type_store = module_type_store.open_function_context('shock_sol', 170, 0, False)
    
    # Passed parameters checking function
    shock_sol.stypy_localization = localization
    shock_sol.stypy_type_of_self = None
    shock_sol.stypy_type_store = module_type_store
    shock_sol.stypy_function_name = 'shock_sol'
    shock_sol.stypy_param_names_list = ['x']
    shock_sol.stypy_varargs_param_name = None
    shock_sol.stypy_kwargs_param_name = None
    shock_sol.stypy_call_defaults = defaults
    shock_sol.stypy_call_varargs = varargs
    shock_sol.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'shock_sol', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'shock_sol', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'shock_sol(...)' code ##################

    
    # Assigning a Num to a Name (line 171):
    
    # Assigning a Num to a Name (line 171):
    float_39415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 10), 'float')
    # Assigning a type to the variable 'eps' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'eps', float_39415)
    
    # Assigning a Call to a Name (line 172):
    
    # Assigning a Call to a Name (line 172):
    
    # Call to sqrt(...): (line 172)
    # Processing the call arguments (line 172)
    int_39418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 16), 'int')
    # Getting the type of 'eps' (line 172)
    eps_39419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'eps', False)
    # Applying the binary operator '*' (line 172)
    result_mul_39420 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 16), '*', int_39418, eps_39419)
    
    # Processing the call keyword arguments (line 172)
    kwargs_39421 = {}
    # Getting the type of 'np' (line 172)
    np_39416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 172)
    sqrt_39417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), np_39416, 'sqrt')
    # Calling sqrt(args, kwargs) (line 172)
    sqrt_call_result_39422 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), sqrt_39417, *[result_mul_39420], **kwargs_39421)
    
    # Assigning a type to the variable 'k' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'k', sqrt_call_result_39422)
    
    # Call to cos(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'np' (line 173)
    np_39425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 18), 'np', False)
    # Obtaining the member 'pi' of a type (line 173)
    pi_39426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 18), np_39425, 'pi')
    # Getting the type of 'x' (line 173)
    x_39427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 26), 'x', False)
    # Applying the binary operator '*' (line 173)
    result_mul_39428 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 18), '*', pi_39426, x_39427)
    
    # Processing the call keyword arguments (line 173)
    kwargs_39429 = {}
    # Getting the type of 'np' (line 173)
    np_39423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'np', False)
    # Obtaining the member 'cos' of a type (line 173)
    cos_39424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 11), np_39423, 'cos')
    # Calling cos(args, kwargs) (line 173)
    cos_call_result_39430 = invoke(stypy.reporting.localization.Localization(__file__, 173, 11), cos_39424, *[result_mul_39428], **kwargs_39429)
    
    
    # Call to erf(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'x' (line 173)
    x_39432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 35), 'x', False)
    # Getting the type of 'k' (line 173)
    k_39433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 39), 'k', False)
    # Applying the binary operator 'div' (line 173)
    result_div_39434 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 35), 'div', x_39432, k_39433)
    
    # Processing the call keyword arguments (line 173)
    kwargs_39435 = {}
    # Getting the type of 'erf' (line 173)
    erf_39431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 31), 'erf', False)
    # Calling erf(args, kwargs) (line 173)
    erf_call_result_39436 = invoke(stypy.reporting.localization.Localization(__file__, 173, 31), erf_39431, *[result_div_39434], **kwargs_39435)
    
    
    # Call to erf(...): (line 173)
    # Processing the call arguments (line 173)
    int_39438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 48), 'int')
    # Getting the type of 'k' (line 173)
    k_39439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 52), 'k', False)
    # Applying the binary operator 'div' (line 173)
    result_div_39440 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 48), 'div', int_39438, k_39439)
    
    # Processing the call keyword arguments (line 173)
    kwargs_39441 = {}
    # Getting the type of 'erf' (line 173)
    erf_39437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 44), 'erf', False)
    # Calling erf(args, kwargs) (line 173)
    erf_call_result_39442 = invoke(stypy.reporting.localization.Localization(__file__, 173, 44), erf_39437, *[result_div_39440], **kwargs_39441)
    
    # Applying the binary operator 'div' (line 173)
    result_div_39443 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 31), 'div', erf_call_result_39436, erf_call_result_39442)
    
    # Applying the binary operator '+' (line 173)
    result_add_39444 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 11), '+', cos_call_result_39430, result_div_39443)
    
    # Assigning a type to the variable 'stypy_return_type' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type', result_add_39444)
    
    # ################# End of 'shock_sol(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'shock_sol' in the type store
    # Getting the type of 'stypy_return_type' (line 170)
    stypy_return_type_39445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39445)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'shock_sol'
    return stypy_return_type_39445

# Assigning a type to the variable 'shock_sol' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'shock_sol', shock_sol)

@norecursion
def test_modify_mesh(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_modify_mesh'
    module_type_store = module_type_store.open_function_context('test_modify_mesh', 176, 0, False)
    
    # Passed parameters checking function
    test_modify_mesh.stypy_localization = localization
    test_modify_mesh.stypy_type_of_self = None
    test_modify_mesh.stypy_type_store = module_type_store
    test_modify_mesh.stypy_function_name = 'test_modify_mesh'
    test_modify_mesh.stypy_param_names_list = []
    test_modify_mesh.stypy_varargs_param_name = None
    test_modify_mesh.stypy_kwargs_param_name = None
    test_modify_mesh.stypy_call_defaults = defaults
    test_modify_mesh.stypy_call_varargs = varargs
    test_modify_mesh.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_modify_mesh', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_modify_mesh', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_modify_mesh(...)' code ##################

    
    # Assigning a Call to a Name (line 177):
    
    # Assigning a Call to a Name (line 177):
    
    # Call to array(...): (line 177)
    # Processing the call arguments (line 177)
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_39448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    # Adding element type (line 177)
    int_39449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 17), list_39448, int_39449)
    # Adding element type (line 177)
    int_39450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 17), list_39448, int_39450)
    # Adding element type (line 177)
    int_39451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 17), list_39448, int_39451)
    # Adding element type (line 177)
    int_39452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 17), list_39448, int_39452)
    
    # Processing the call keyword arguments (line 177)
    # Getting the type of 'float' (line 177)
    float_39453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 37), 'float', False)
    keyword_39454 = float_39453
    kwargs_39455 = {'dtype': keyword_39454}
    # Getting the type of 'np' (line 177)
    np_39446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 177)
    array_39447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), np_39446, 'array')
    # Calling array(args, kwargs) (line 177)
    array_call_result_39456 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), array_39447, *[list_39448], **kwargs_39455)
    
    # Assigning a type to the variable 'x' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'x', array_call_result_39456)
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to modify_mesh(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'x' (line 178)
    x_39458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'x', False)
    
    # Call to array(...): (line 178)
    # Processing the call arguments (line 178)
    
    # Obtaining an instance of the builtin type 'list' (line 178)
    list_39461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 178)
    # Adding element type (line 178)
    int_39462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 36), list_39461, int_39462)
    
    # Processing the call keyword arguments (line 178)
    kwargs_39463 = {}
    # Getting the type of 'np' (line 178)
    np_39459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'np', False)
    # Obtaining the member 'array' of a type (line 178)
    array_39460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 27), np_39459, 'array')
    # Calling array(args, kwargs) (line 178)
    array_call_result_39464 = invoke(stypy.reporting.localization.Localization(__file__, 178, 27), array_39460, *[list_39461], **kwargs_39463)
    
    
    # Call to array(...): (line 178)
    # Processing the call arguments (line 178)
    
    # Obtaining an instance of the builtin type 'list' (line 178)
    list_39467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 178)
    # Adding element type (line 178)
    int_39468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 51), list_39467, int_39468)
    
    # Processing the call keyword arguments (line 178)
    kwargs_39469 = {}
    # Getting the type of 'np' (line 178)
    np_39465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 42), 'np', False)
    # Obtaining the member 'array' of a type (line 178)
    array_39466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 42), np_39465, 'array')
    # Calling array(args, kwargs) (line 178)
    array_call_result_39470 = invoke(stypy.reporting.localization.Localization(__file__, 178, 42), array_39466, *[list_39467], **kwargs_39469)
    
    # Processing the call keyword arguments (line 178)
    kwargs_39471 = {}
    # Getting the type of 'modify_mesh' (line 178)
    modify_mesh_39457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'modify_mesh', False)
    # Calling modify_mesh(args, kwargs) (line 178)
    modify_mesh_call_result_39472 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), modify_mesh_39457, *[x_39458, array_call_result_39464, array_call_result_39470], **kwargs_39471)
    
    # Assigning a type to the variable 'x_new' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'x_new', modify_mesh_call_result_39472)
    
    # Call to assert_array_equal(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'x_new' (line 179)
    x_new_39474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 23), 'x_new', False)
    
    # Call to array(...): (line 179)
    # Processing the call arguments (line 179)
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_39477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    # Adding element type (line 179)
    int_39478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 39), list_39477, int_39478)
    # Adding element type (line 179)
    float_39479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 39), list_39477, float_39479)
    # Adding element type (line 179)
    int_39480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 39), list_39477, int_39480)
    # Adding element type (line 179)
    int_39481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 39), list_39477, int_39481)
    # Adding element type (line 179)
    int_39482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 39), list_39477, int_39482)
    # Adding element type (line 179)
    int_39483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 39), list_39477, int_39483)
    # Adding element type (line 179)
    int_39484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 39), list_39477, int_39484)
    
    # Processing the call keyword arguments (line 179)
    kwargs_39485 = {}
    # Getting the type of 'np' (line 179)
    np_39475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 30), 'np', False)
    # Obtaining the member 'array' of a type (line 179)
    array_39476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 30), np_39475, 'array')
    # Calling array(args, kwargs) (line 179)
    array_call_result_39486 = invoke(stypy.reporting.localization.Localization(__file__, 179, 30), array_39476, *[list_39477], **kwargs_39485)
    
    # Processing the call keyword arguments (line 179)
    kwargs_39487 = {}
    # Getting the type of 'assert_array_equal' (line 179)
    assert_array_equal_39473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 179)
    assert_array_equal_call_result_39488 = invoke(stypy.reporting.localization.Localization(__file__, 179, 4), assert_array_equal_39473, *[x_new_39474, array_call_result_39486], **kwargs_39487)
    
    
    # Assigning a Call to a Name (line 181):
    
    # Assigning a Call to a Name (line 181):
    
    # Call to array(...): (line 181)
    # Processing the call arguments (line 181)
    
    # Obtaining an instance of the builtin type 'list' (line 181)
    list_39491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 181)
    # Adding element type (line 181)
    int_39492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 17), list_39491, int_39492)
    # Adding element type (line 181)
    int_39493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 17), list_39491, int_39493)
    # Adding element type (line 181)
    int_39494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 17), list_39491, int_39494)
    # Adding element type (line 181)
    int_39495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 17), list_39491, int_39495)
    # Adding element type (line 181)
    int_39496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 17), list_39491, int_39496)
    
    # Processing the call keyword arguments (line 181)
    # Getting the type of 'float' (line 181)
    float_39497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 42), 'float', False)
    keyword_39498 = float_39497
    kwargs_39499 = {'dtype': keyword_39498}
    # Getting the type of 'np' (line 181)
    np_39489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 181)
    array_39490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), np_39489, 'array')
    # Calling array(args, kwargs) (line 181)
    array_call_result_39500 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), array_39490, *[list_39491], **kwargs_39499)
    
    # Assigning a type to the variable 'x' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'x', array_call_result_39500)
    
    # Assigning a Call to a Name (line 182):
    
    # Assigning a Call to a Name (line 182):
    
    # Call to modify_mesh(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'x' (line 182)
    x_39502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 24), 'x', False)
    
    # Call to array(...): (line 182)
    # Processing the call arguments (line 182)
    
    # Obtaining an instance of the builtin type 'list' (line 182)
    list_39505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 182)
    # Adding element type (line 182)
    int_39506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 36), list_39505, int_39506)
    
    # Processing the call keyword arguments (line 182)
    # Getting the type of 'int' (line 182)
    int_39507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 47), 'int', False)
    keyword_39508 = int_39507
    kwargs_39509 = {'dtype': keyword_39508}
    # Getting the type of 'np' (line 182)
    np_39503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 27), 'np', False)
    # Obtaining the member 'array' of a type (line 182)
    array_39504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 27), np_39503, 'array')
    # Calling array(args, kwargs) (line 182)
    array_call_result_39510 = invoke(stypy.reporting.localization.Localization(__file__, 182, 27), array_39504, *[list_39505], **kwargs_39509)
    
    
    # Call to array(...): (line 182)
    # Processing the call arguments (line 182)
    
    # Obtaining an instance of the builtin type 'list' (line 182)
    list_39513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 62), 'list')
    # Adding type elements to the builtin type 'list' instance (line 182)
    # Adding element type (line 182)
    int_39514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 63), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 62), list_39513, int_39514)
    # Adding element type (line 182)
    int_39515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 62), list_39513, int_39515)
    # Adding element type (line 182)
    int_39516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 69), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 62), list_39513, int_39516)
    
    # Processing the call keyword arguments (line 182)
    kwargs_39517 = {}
    # Getting the type of 'np' (line 182)
    np_39511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 53), 'np', False)
    # Obtaining the member 'array' of a type (line 182)
    array_39512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 53), np_39511, 'array')
    # Calling array(args, kwargs) (line 182)
    array_call_result_39518 = invoke(stypy.reporting.localization.Localization(__file__, 182, 53), array_39512, *[list_39513], **kwargs_39517)
    
    # Processing the call keyword arguments (line 182)
    kwargs_39519 = {}
    # Getting the type of 'modify_mesh' (line 182)
    modify_mesh_39501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'modify_mesh', False)
    # Calling modify_mesh(args, kwargs) (line 182)
    modify_mesh_call_result_39520 = invoke(stypy.reporting.localization.Localization(__file__, 182, 12), modify_mesh_39501, *[x_39502, array_call_result_39510, array_call_result_39518], **kwargs_39519)
    
    # Assigning a type to the variable 'x_new' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'x_new', modify_mesh_call_result_39520)
    
    # Call to assert_array_equal(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'x_new' (line 183)
    x_new_39522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 23), 'x_new', False)
    
    # Obtaining an instance of the builtin type 'list' (line 183)
    list_39523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 183)
    # Adding element type (line 183)
    int_39524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 30), list_39523, int_39524)
    # Adding element type (line 183)
    int_39525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 30), list_39523, int_39525)
    # Adding element type (line 183)
    int_39526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 30), list_39523, int_39526)
    # Adding element type (line 183)
    int_39527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 30), list_39523, int_39527)
    # Adding element type (line 183)
    float_39528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 47), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 30), list_39523, float_39528)
    # Adding element type (line 183)
    int_39529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 30), list_39523, int_39529)
    # Adding element type (line 183)
    int_39530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 56), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 30), list_39523, int_39530)
    # Adding element type (line 183)
    int_39531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 30), list_39523, int_39531)
    # Adding element type (line 183)
    int_39532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 30), list_39523, int_39532)
    # Adding element type (line 183)
    int_39533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 65), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 30), list_39523, int_39533)
    # Adding element type (line 183)
    int_39534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 68), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 30), list_39523, int_39534)
    # Adding element type (line 183)
    int_39535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 71), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 30), list_39523, int_39535)
    
    # Processing the call keyword arguments (line 183)
    kwargs_39536 = {}
    # Getting the type of 'assert_array_equal' (line 183)
    assert_array_equal_39521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 183)
    assert_array_equal_call_result_39537 = invoke(stypy.reporting.localization.Localization(__file__, 183, 4), assert_array_equal_39521, *[x_new_39522, list_39523], **kwargs_39536)
    
    
    # ################# End of 'test_modify_mesh(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_modify_mesh' in the type store
    # Getting the type of 'stypy_return_type' (line 176)
    stypy_return_type_39538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39538)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_modify_mesh'
    return stypy_return_type_39538

# Assigning a type to the variable 'test_modify_mesh' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'test_modify_mesh', test_modify_mesh)

@norecursion
def test_compute_fun_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_compute_fun_jac'
    module_type_store = module_type_store.open_function_context('test_compute_fun_jac', 186, 0, False)
    
    # Passed parameters checking function
    test_compute_fun_jac.stypy_localization = localization
    test_compute_fun_jac.stypy_type_of_self = None
    test_compute_fun_jac.stypy_type_store = module_type_store
    test_compute_fun_jac.stypy_function_name = 'test_compute_fun_jac'
    test_compute_fun_jac.stypy_param_names_list = []
    test_compute_fun_jac.stypy_varargs_param_name = None
    test_compute_fun_jac.stypy_kwargs_param_name = None
    test_compute_fun_jac.stypy_call_defaults = defaults
    test_compute_fun_jac.stypy_call_varargs = varargs
    test_compute_fun_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_compute_fun_jac', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_compute_fun_jac', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_compute_fun_jac(...)' code ##################

    
    # Assigning a Call to a Name (line 187):
    
    # Assigning a Call to a Name (line 187):
    
    # Call to linspace(...): (line 187)
    # Processing the call arguments (line 187)
    int_39541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 20), 'int')
    int_39542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 23), 'int')
    int_39543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 26), 'int')
    # Processing the call keyword arguments (line 187)
    kwargs_39544 = {}
    # Getting the type of 'np' (line 187)
    np_39539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 187)
    linspace_39540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), np_39539, 'linspace')
    # Calling linspace(args, kwargs) (line 187)
    linspace_call_result_39545 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), linspace_39540, *[int_39541, int_39542, int_39543], **kwargs_39544)
    
    # Assigning a type to the variable 'x' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'x', linspace_call_result_39545)
    
    # Assigning a Call to a Name (line 188):
    
    # Assigning a Call to a Name (line 188):
    
    # Call to empty(...): (line 188)
    # Processing the call arguments (line 188)
    
    # Obtaining an instance of the builtin type 'tuple' (line 188)
    tuple_39548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 188)
    # Adding element type (line 188)
    int_39549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 18), tuple_39548, int_39549)
    # Adding element type (line 188)
    
    # Obtaining the type of the subscript
    int_39550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 29), 'int')
    # Getting the type of 'x' (line 188)
    x_39551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 188)
    shape_39552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 21), x_39551, 'shape')
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___39553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 21), shape_39552, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_39554 = invoke(stypy.reporting.localization.Localization(__file__, 188, 21), getitem___39553, int_39550)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 18), tuple_39548, subscript_call_result_39554)
    
    # Processing the call keyword arguments (line 188)
    kwargs_39555 = {}
    # Getting the type of 'np' (line 188)
    np_39546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 188)
    empty_39547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), np_39546, 'empty')
    # Calling empty(args, kwargs) (line 188)
    empty_call_result_39556 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), empty_39547, *[tuple_39548], **kwargs_39555)
    
    # Assigning a type to the variable 'y' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'y', empty_call_result_39556)
    
    # Assigning a Num to a Subscript (line 189):
    
    # Assigning a Num to a Subscript (line 189):
    float_39557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 11), 'float')
    # Getting the type of 'y' (line 189)
    y_39558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'y')
    int_39559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 6), 'int')
    # Storing an element on a container (line 189)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 4), y_39558, (int_39559, float_39557))
    
    # Assigning a Num to a Subscript (line 190):
    
    # Assigning a Num to a Subscript (line 190):
    float_39560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 11), 'float')
    # Getting the type of 'y' (line 190)
    y_39561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'y')
    int_39562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 6), 'int')
    # Storing an element on a container (line 190)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 4), y_39561, (int_39562, float_39560))
    
    # Assigning a Call to a Name (line 191):
    
    # Assigning a Call to a Name (line 191):
    
    # Call to array(...): (line 191)
    # Processing the call arguments (line 191)
    
    # Obtaining an instance of the builtin type 'list' (line 191)
    list_39565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 191)
    
    # Processing the call keyword arguments (line 191)
    kwargs_39566 = {}
    # Getting the type of 'np' (line 191)
    np_39563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 191)
    array_39564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), np_39563, 'array')
    # Calling array(args, kwargs) (line 191)
    array_call_result_39567 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), array_39564, *[list_39565], **kwargs_39566)
    
    # Assigning a type to the variable 'p' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'p', array_call_result_39567)
    
    # Assigning a Call to a Tuple (line 192):
    
    # Assigning a Subscript to a Name (line 192):
    
    # Obtaining the type of the subscript
    int_39568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 4), 'int')
    
    # Call to estimate_fun_jac(...): (line 192)
    # Processing the call arguments (line 192)

    @norecursion
    def _stypy_temp_lambda_5(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_5'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_5', 192, 36, True)
        # Passed parameters checking function
        _stypy_temp_lambda_5.stypy_localization = localization
        _stypy_temp_lambda_5.stypy_type_of_self = None
        _stypy_temp_lambda_5.stypy_type_store = module_type_store
        _stypy_temp_lambda_5.stypy_function_name = '_stypy_temp_lambda_5'
        _stypy_temp_lambda_5.stypy_param_names_list = ['x', 'y', 'p']
        _stypy_temp_lambda_5.stypy_varargs_param_name = None
        _stypy_temp_lambda_5.stypy_kwargs_param_name = None
        _stypy_temp_lambda_5.stypy_call_defaults = defaults
        _stypy_temp_lambda_5.stypy_call_varargs = varargs
        _stypy_temp_lambda_5.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_5', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_5', ['x', 'y', 'p'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to exp_fun(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'x' (line 192)
        x_39571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 60), 'x', False)
        # Getting the type of 'y' (line 192)
        y_39572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 63), 'y', False)
        # Processing the call keyword arguments (line 192)
        kwargs_39573 = {}
        # Getting the type of 'exp_fun' (line 192)
        exp_fun_39570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 52), 'exp_fun', False)
        # Calling exp_fun(args, kwargs) (line 192)
        exp_fun_call_result_39574 = invoke(stypy.reporting.localization.Localization(__file__, 192, 52), exp_fun_39570, *[x_39571, y_39572], **kwargs_39573)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 36), 'stypy_return_type', exp_fun_call_result_39574)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_5' in the type store
        # Getting the type of 'stypy_return_type' (line 192)
        stypy_return_type_39575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 36), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39575)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_5'
        return stypy_return_type_39575

    # Assigning a type to the variable '_stypy_temp_lambda_5' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 36), '_stypy_temp_lambda_5', _stypy_temp_lambda_5)
    # Getting the type of '_stypy_temp_lambda_5' (line 192)
    _stypy_temp_lambda_5_39576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 36), '_stypy_temp_lambda_5')
    # Getting the type of 'x' (line 192)
    x_39577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 67), 'x', False)
    # Getting the type of 'y' (line 192)
    y_39578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 70), 'y', False)
    # Getting the type of 'p' (line 192)
    p_39579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 73), 'p', False)
    # Processing the call keyword arguments (line 192)
    kwargs_39580 = {}
    # Getting the type of 'estimate_fun_jac' (line 192)
    estimate_fun_jac_39569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 19), 'estimate_fun_jac', False)
    # Calling estimate_fun_jac(args, kwargs) (line 192)
    estimate_fun_jac_call_result_39581 = invoke(stypy.reporting.localization.Localization(__file__, 192, 19), estimate_fun_jac_39569, *[_stypy_temp_lambda_5_39576, x_39577, y_39578, p_39579], **kwargs_39580)
    
    # Obtaining the member '__getitem__' of a type (line 192)
    getitem___39582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 4), estimate_fun_jac_call_result_39581, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 192)
    subscript_call_result_39583 = invoke(stypy.reporting.localization.Localization(__file__, 192, 4), getitem___39582, int_39568)
    
    # Assigning a type to the variable 'tuple_var_assignment_38798' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'tuple_var_assignment_38798', subscript_call_result_39583)
    
    # Assigning a Subscript to a Name (line 192):
    
    # Obtaining the type of the subscript
    int_39584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 4), 'int')
    
    # Call to estimate_fun_jac(...): (line 192)
    # Processing the call arguments (line 192)

    @norecursion
    def _stypy_temp_lambda_6(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_6'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_6', 192, 36, True)
        # Passed parameters checking function
        _stypy_temp_lambda_6.stypy_localization = localization
        _stypy_temp_lambda_6.stypy_type_of_self = None
        _stypy_temp_lambda_6.stypy_type_store = module_type_store
        _stypy_temp_lambda_6.stypy_function_name = '_stypy_temp_lambda_6'
        _stypy_temp_lambda_6.stypy_param_names_list = ['x', 'y', 'p']
        _stypy_temp_lambda_6.stypy_varargs_param_name = None
        _stypy_temp_lambda_6.stypy_kwargs_param_name = None
        _stypy_temp_lambda_6.stypy_call_defaults = defaults
        _stypy_temp_lambda_6.stypy_call_varargs = varargs
        _stypy_temp_lambda_6.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_6', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_6', ['x', 'y', 'p'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to exp_fun(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'x' (line 192)
        x_39587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 60), 'x', False)
        # Getting the type of 'y' (line 192)
        y_39588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 63), 'y', False)
        # Processing the call keyword arguments (line 192)
        kwargs_39589 = {}
        # Getting the type of 'exp_fun' (line 192)
        exp_fun_39586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 52), 'exp_fun', False)
        # Calling exp_fun(args, kwargs) (line 192)
        exp_fun_call_result_39590 = invoke(stypy.reporting.localization.Localization(__file__, 192, 52), exp_fun_39586, *[x_39587, y_39588], **kwargs_39589)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 36), 'stypy_return_type', exp_fun_call_result_39590)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_6' in the type store
        # Getting the type of 'stypy_return_type' (line 192)
        stypy_return_type_39591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 36), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39591)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_6'
        return stypy_return_type_39591

    # Assigning a type to the variable '_stypy_temp_lambda_6' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 36), '_stypy_temp_lambda_6', _stypy_temp_lambda_6)
    # Getting the type of '_stypy_temp_lambda_6' (line 192)
    _stypy_temp_lambda_6_39592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 36), '_stypy_temp_lambda_6')
    # Getting the type of 'x' (line 192)
    x_39593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 67), 'x', False)
    # Getting the type of 'y' (line 192)
    y_39594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 70), 'y', False)
    # Getting the type of 'p' (line 192)
    p_39595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 73), 'p', False)
    # Processing the call keyword arguments (line 192)
    kwargs_39596 = {}
    # Getting the type of 'estimate_fun_jac' (line 192)
    estimate_fun_jac_39585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 19), 'estimate_fun_jac', False)
    # Calling estimate_fun_jac(args, kwargs) (line 192)
    estimate_fun_jac_call_result_39597 = invoke(stypy.reporting.localization.Localization(__file__, 192, 19), estimate_fun_jac_39585, *[_stypy_temp_lambda_6_39592, x_39593, y_39594, p_39595], **kwargs_39596)
    
    # Obtaining the member '__getitem__' of a type (line 192)
    getitem___39598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 4), estimate_fun_jac_call_result_39597, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 192)
    subscript_call_result_39599 = invoke(stypy.reporting.localization.Localization(__file__, 192, 4), getitem___39598, int_39584)
    
    # Assigning a type to the variable 'tuple_var_assignment_38799' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'tuple_var_assignment_38799', subscript_call_result_39599)
    
    # Assigning a Name to a Name (line 192):
    # Getting the type of 'tuple_var_assignment_38798' (line 192)
    tuple_var_assignment_38798_39600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'tuple_var_assignment_38798')
    # Assigning a type to the variable 'df_dy' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'df_dy', tuple_var_assignment_38798_39600)
    
    # Assigning a Name to a Name (line 192):
    # Getting the type of 'tuple_var_assignment_38799' (line 192)
    tuple_var_assignment_38799_39601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'tuple_var_assignment_38799')
    # Assigning a type to the variable 'df_dp' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), 'df_dp', tuple_var_assignment_38799_39601)
    
    # Assigning a Call to a Name (line 193):
    
    # Assigning a Call to a Name (line 193):
    
    # Call to exp_fun_jac(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'x' (line 193)
    x_39603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 27), 'x', False)
    # Getting the type of 'y' (line 193)
    y_39604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 30), 'y', False)
    # Processing the call keyword arguments (line 193)
    kwargs_39605 = {}
    # Getting the type of 'exp_fun_jac' (line 193)
    exp_fun_jac_39602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'exp_fun_jac', False)
    # Calling exp_fun_jac(args, kwargs) (line 193)
    exp_fun_jac_call_result_39606 = invoke(stypy.reporting.localization.Localization(__file__, 193, 15), exp_fun_jac_39602, *[x_39603, y_39604], **kwargs_39605)
    
    # Assigning a type to the variable 'df_dy_an' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'df_dy_an', exp_fun_jac_call_result_39606)
    
    # Call to assert_allclose(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'df_dy' (line 194)
    df_dy_39608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 20), 'df_dy', False)
    # Getting the type of 'df_dy_an' (line 194)
    df_dy_an_39609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 27), 'df_dy_an', False)
    # Processing the call keyword arguments (line 194)
    kwargs_39610 = {}
    # Getting the type of 'assert_allclose' (line 194)
    assert_allclose_39607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 194)
    assert_allclose_call_result_39611 = invoke(stypy.reporting.localization.Localization(__file__, 194, 4), assert_allclose_39607, *[df_dy_39608, df_dy_an_39609], **kwargs_39610)
    
    
    # Call to assert_(...): (line 195)
    # Processing the call arguments (line 195)
    
    # Getting the type of 'df_dp' (line 195)
    df_dp_39613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'df_dp', False)
    # Getting the type of 'None' (line 195)
    None_39614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 21), 'None', False)
    # Applying the binary operator 'is' (line 195)
    result_is__39615 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 12), 'is', df_dp_39613, None_39614)
    
    # Processing the call keyword arguments (line 195)
    kwargs_39616 = {}
    # Getting the type of 'assert_' (line 195)
    assert__39612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 195)
    assert__call_result_39617 = invoke(stypy.reporting.localization.Localization(__file__, 195, 4), assert__39612, *[result_is__39615], **kwargs_39616)
    
    
    # Assigning a Call to a Name (line 197):
    
    # Assigning a Call to a Name (line 197):
    
    # Call to linspace(...): (line 197)
    # Processing the call arguments (line 197)
    int_39620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 20), 'int')
    # Getting the type of 'np' (line 197)
    np_39621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 23), 'np', False)
    # Obtaining the member 'pi' of a type (line 197)
    pi_39622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 23), np_39621, 'pi')
    int_39623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 30), 'int')
    # Processing the call keyword arguments (line 197)
    kwargs_39624 = {}
    # Getting the type of 'np' (line 197)
    np_39618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 197)
    linspace_39619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), np_39618, 'linspace')
    # Calling linspace(args, kwargs) (line 197)
    linspace_call_result_39625 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), linspace_39619, *[int_39620, pi_39622, int_39623], **kwargs_39624)
    
    # Assigning a type to the variable 'x' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'x', linspace_call_result_39625)
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to empty(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Obtaining an instance of the builtin type 'tuple' (line 198)
    tuple_39628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 198)
    # Adding element type (line 198)
    int_39629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), tuple_39628, int_39629)
    # Adding element type (line 198)
    
    # Obtaining the type of the subscript
    int_39630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 29), 'int')
    # Getting the type of 'x' (line 198)
    x_39631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 198)
    shape_39632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 21), x_39631, 'shape')
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___39633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 21), shape_39632, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_39634 = invoke(stypy.reporting.localization.Localization(__file__, 198, 21), getitem___39633, int_39630)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), tuple_39628, subscript_call_result_39634)
    
    # Processing the call keyword arguments (line 198)
    kwargs_39635 = {}
    # Getting the type of 'np' (line 198)
    np_39626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 198)
    empty_39627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), np_39626, 'empty')
    # Calling empty(args, kwargs) (line 198)
    empty_call_result_39636 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), empty_39627, *[tuple_39628], **kwargs_39635)
    
    # Assigning a type to the variable 'y' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'y', empty_call_result_39636)
    
    # Assigning a Call to a Subscript (line 199):
    
    # Assigning a Call to a Subscript (line 199):
    
    # Call to sin(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'x' (line 199)
    x_39639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 18), 'x', False)
    # Processing the call keyword arguments (line 199)
    kwargs_39640 = {}
    # Getting the type of 'np' (line 199)
    np_39637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'np', False)
    # Obtaining the member 'sin' of a type (line 199)
    sin_39638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 11), np_39637, 'sin')
    # Calling sin(args, kwargs) (line 199)
    sin_call_result_39641 = invoke(stypy.reporting.localization.Localization(__file__, 199, 11), sin_39638, *[x_39639], **kwargs_39640)
    
    # Getting the type of 'y' (line 199)
    y_39642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'y')
    int_39643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 6), 'int')
    # Storing an element on a container (line 199)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 4), y_39642, (int_39643, sin_call_result_39641))
    
    # Assigning a Call to a Subscript (line 200):
    
    # Assigning a Call to a Subscript (line 200):
    
    # Call to cos(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'x' (line 200)
    x_39646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), 'x', False)
    # Processing the call keyword arguments (line 200)
    kwargs_39647 = {}
    # Getting the type of 'np' (line 200)
    np_39644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'np', False)
    # Obtaining the member 'cos' of a type (line 200)
    cos_39645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 11), np_39644, 'cos')
    # Calling cos(args, kwargs) (line 200)
    cos_call_result_39648 = invoke(stypy.reporting.localization.Localization(__file__, 200, 11), cos_39645, *[x_39646], **kwargs_39647)
    
    # Getting the type of 'y' (line 200)
    y_39649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'y')
    int_39650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 6), 'int')
    # Storing an element on a container (line 200)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 4), y_39649, (int_39650, cos_call_result_39648))
    
    # Assigning a Call to a Name (line 201):
    
    # Assigning a Call to a Name (line 201):
    
    # Call to array(...): (line 201)
    # Processing the call arguments (line 201)
    
    # Obtaining an instance of the builtin type 'list' (line 201)
    list_39653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 201)
    # Adding element type (line 201)
    float_39654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 17), list_39653, float_39654)
    
    # Processing the call keyword arguments (line 201)
    kwargs_39655 = {}
    # Getting the type of 'np' (line 201)
    np_39651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 201)
    array_39652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), np_39651, 'array')
    # Calling array(args, kwargs) (line 201)
    array_call_result_39656 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), array_39652, *[list_39653], **kwargs_39655)
    
    # Assigning a type to the variable 'p' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'p', array_call_result_39656)
    
    # Assigning a Call to a Tuple (line 202):
    
    # Assigning a Subscript to a Name (line 202):
    
    # Obtaining the type of the subscript
    int_39657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 4), 'int')
    
    # Call to estimate_fun_jac(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'sl_fun' (line 202)
    sl_fun_39659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 36), 'sl_fun', False)
    # Getting the type of 'x' (line 202)
    x_39660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 44), 'x', False)
    # Getting the type of 'y' (line 202)
    y_39661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 47), 'y', False)
    # Getting the type of 'p' (line 202)
    p_39662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 50), 'p', False)
    # Processing the call keyword arguments (line 202)
    kwargs_39663 = {}
    # Getting the type of 'estimate_fun_jac' (line 202)
    estimate_fun_jac_39658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'estimate_fun_jac', False)
    # Calling estimate_fun_jac(args, kwargs) (line 202)
    estimate_fun_jac_call_result_39664 = invoke(stypy.reporting.localization.Localization(__file__, 202, 19), estimate_fun_jac_39658, *[sl_fun_39659, x_39660, y_39661, p_39662], **kwargs_39663)
    
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___39665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 4), estimate_fun_jac_call_result_39664, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_39666 = invoke(stypy.reporting.localization.Localization(__file__, 202, 4), getitem___39665, int_39657)
    
    # Assigning a type to the variable 'tuple_var_assignment_38800' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'tuple_var_assignment_38800', subscript_call_result_39666)
    
    # Assigning a Subscript to a Name (line 202):
    
    # Obtaining the type of the subscript
    int_39667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 4), 'int')
    
    # Call to estimate_fun_jac(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'sl_fun' (line 202)
    sl_fun_39669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 36), 'sl_fun', False)
    # Getting the type of 'x' (line 202)
    x_39670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 44), 'x', False)
    # Getting the type of 'y' (line 202)
    y_39671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 47), 'y', False)
    # Getting the type of 'p' (line 202)
    p_39672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 50), 'p', False)
    # Processing the call keyword arguments (line 202)
    kwargs_39673 = {}
    # Getting the type of 'estimate_fun_jac' (line 202)
    estimate_fun_jac_39668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'estimate_fun_jac', False)
    # Calling estimate_fun_jac(args, kwargs) (line 202)
    estimate_fun_jac_call_result_39674 = invoke(stypy.reporting.localization.Localization(__file__, 202, 19), estimate_fun_jac_39668, *[sl_fun_39669, x_39670, y_39671, p_39672], **kwargs_39673)
    
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___39675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 4), estimate_fun_jac_call_result_39674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_39676 = invoke(stypy.reporting.localization.Localization(__file__, 202, 4), getitem___39675, int_39667)
    
    # Assigning a type to the variable 'tuple_var_assignment_38801' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'tuple_var_assignment_38801', subscript_call_result_39676)
    
    # Assigning a Name to a Name (line 202):
    # Getting the type of 'tuple_var_assignment_38800' (line 202)
    tuple_var_assignment_38800_39677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'tuple_var_assignment_38800')
    # Assigning a type to the variable 'df_dy' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'df_dy', tuple_var_assignment_38800_39677)
    
    # Assigning a Name to a Name (line 202):
    # Getting the type of 'tuple_var_assignment_38801' (line 202)
    tuple_var_assignment_38801_39678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'tuple_var_assignment_38801')
    # Assigning a type to the variable 'df_dp' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'df_dp', tuple_var_assignment_38801_39678)
    
    # Assigning a Call to a Tuple (line 203):
    
    # Assigning a Subscript to a Name (line 203):
    
    # Obtaining the type of the subscript
    int_39679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 4), 'int')
    
    # Call to sl_fun_jac(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'x' (line 203)
    x_39681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 36), 'x', False)
    # Getting the type of 'y' (line 203)
    y_39682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 39), 'y', False)
    # Getting the type of 'p' (line 203)
    p_39683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), 'p', False)
    # Processing the call keyword arguments (line 203)
    kwargs_39684 = {}
    # Getting the type of 'sl_fun_jac' (line 203)
    sl_fun_jac_39680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'sl_fun_jac', False)
    # Calling sl_fun_jac(args, kwargs) (line 203)
    sl_fun_jac_call_result_39685 = invoke(stypy.reporting.localization.Localization(__file__, 203, 25), sl_fun_jac_39680, *[x_39681, y_39682, p_39683], **kwargs_39684)
    
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___39686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 4), sl_fun_jac_call_result_39685, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_39687 = invoke(stypy.reporting.localization.Localization(__file__, 203, 4), getitem___39686, int_39679)
    
    # Assigning a type to the variable 'tuple_var_assignment_38802' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_38802', subscript_call_result_39687)
    
    # Assigning a Subscript to a Name (line 203):
    
    # Obtaining the type of the subscript
    int_39688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 4), 'int')
    
    # Call to sl_fun_jac(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'x' (line 203)
    x_39690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 36), 'x', False)
    # Getting the type of 'y' (line 203)
    y_39691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 39), 'y', False)
    # Getting the type of 'p' (line 203)
    p_39692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), 'p', False)
    # Processing the call keyword arguments (line 203)
    kwargs_39693 = {}
    # Getting the type of 'sl_fun_jac' (line 203)
    sl_fun_jac_39689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'sl_fun_jac', False)
    # Calling sl_fun_jac(args, kwargs) (line 203)
    sl_fun_jac_call_result_39694 = invoke(stypy.reporting.localization.Localization(__file__, 203, 25), sl_fun_jac_39689, *[x_39690, y_39691, p_39692], **kwargs_39693)
    
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___39695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 4), sl_fun_jac_call_result_39694, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_39696 = invoke(stypy.reporting.localization.Localization(__file__, 203, 4), getitem___39695, int_39688)
    
    # Assigning a type to the variable 'tuple_var_assignment_38803' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_38803', subscript_call_result_39696)
    
    # Assigning a Name to a Name (line 203):
    # Getting the type of 'tuple_var_assignment_38802' (line 203)
    tuple_var_assignment_38802_39697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_38802')
    # Assigning a type to the variable 'df_dy_an' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'df_dy_an', tuple_var_assignment_38802_39697)
    
    # Assigning a Name to a Name (line 203):
    # Getting the type of 'tuple_var_assignment_38803' (line 203)
    tuple_var_assignment_38803_39698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'tuple_var_assignment_38803')
    # Assigning a type to the variable 'df_dp_an' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 14), 'df_dp_an', tuple_var_assignment_38803_39698)
    
    # Call to assert_allclose(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'df_dy' (line 204)
    df_dy_39700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'df_dy', False)
    # Getting the type of 'df_dy_an' (line 204)
    df_dy_an_39701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 27), 'df_dy_an', False)
    # Processing the call keyword arguments (line 204)
    kwargs_39702 = {}
    # Getting the type of 'assert_allclose' (line 204)
    assert_allclose_39699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 204)
    assert_allclose_call_result_39703 = invoke(stypy.reporting.localization.Localization(__file__, 204, 4), assert_allclose_39699, *[df_dy_39700, df_dy_an_39701], **kwargs_39702)
    
    
    # Call to assert_allclose(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'df_dp' (line 205)
    df_dp_39705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'df_dp', False)
    # Getting the type of 'df_dp_an' (line 205)
    df_dp_an_39706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'df_dp_an', False)
    # Processing the call keyword arguments (line 205)
    kwargs_39707 = {}
    # Getting the type of 'assert_allclose' (line 205)
    assert_allclose_39704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 205)
    assert_allclose_call_result_39708 = invoke(stypy.reporting.localization.Localization(__file__, 205, 4), assert_allclose_39704, *[df_dp_39705, df_dp_an_39706], **kwargs_39707)
    
    
    # Assigning a Call to a Name (line 207):
    
    # Assigning a Call to a Name (line 207):
    
    # Call to linspace(...): (line 207)
    # Processing the call arguments (line 207)
    int_39711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 20), 'int')
    int_39712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 23), 'int')
    int_39713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 26), 'int')
    # Processing the call keyword arguments (line 207)
    kwargs_39714 = {}
    # Getting the type of 'np' (line 207)
    np_39709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 207)
    linspace_39710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), np_39709, 'linspace')
    # Calling linspace(args, kwargs) (line 207)
    linspace_call_result_39715 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), linspace_39710, *[int_39711, int_39712, int_39713], **kwargs_39714)
    
    # Assigning a type to the variable 'x' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'x', linspace_call_result_39715)
    
    # Assigning a Call to a Name (line 208):
    
    # Assigning a Call to a Name (line 208):
    
    # Call to empty(...): (line 208)
    # Processing the call arguments (line 208)
    
    # Obtaining an instance of the builtin type 'tuple' (line 208)
    tuple_39718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 208)
    # Adding element type (line 208)
    int_39719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 18), tuple_39718, int_39719)
    # Adding element type (line 208)
    
    # Obtaining the type of the subscript
    int_39720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 29), 'int')
    # Getting the type of 'x' (line 208)
    x_39721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 208)
    shape_39722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 21), x_39721, 'shape')
    # Obtaining the member '__getitem__' of a type (line 208)
    getitem___39723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 21), shape_39722, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 208)
    subscript_call_result_39724 = invoke(stypy.reporting.localization.Localization(__file__, 208, 21), getitem___39723, int_39720)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 18), tuple_39718, subscript_call_result_39724)
    
    # Processing the call keyword arguments (line 208)
    kwargs_39725 = {}
    # Getting the type of 'np' (line 208)
    np_39716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 208)
    empty_39717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), np_39716, 'empty')
    # Calling empty(args, kwargs) (line 208)
    empty_call_result_39726 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), empty_39717, *[tuple_39718], **kwargs_39725)
    
    # Assigning a type to the variable 'y' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'y', empty_call_result_39726)
    
    # Assigning a BinOp to a Subscript (line 209):
    
    # Assigning a BinOp to a Subscript (line 209):
    int_39727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 12), 'int')
    int_39728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 14), 'int')
    # Applying the binary operator 'div' (line 209)
    result_div_39729 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 12), 'div', int_39727, int_39728)
    
    float_39730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 18), 'float')
    # Applying the binary operator '**' (line 209)
    result_pow_39731 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 11), '**', result_div_39729, float_39730)
    
    # Getting the type of 'y' (line 209)
    y_39732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'y')
    int_39733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 6), 'int')
    # Storing an element on a container (line 209)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 4), y_39732, (int_39733, result_pow_39731))
    
    # Assigning a Num to a Subscript (line 210):
    
    # Assigning a Num to a Subscript (line 210):
    float_39734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 11), 'float')
    # Getting the type of 'y' (line 210)
    y_39735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'y')
    int_39736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 6), 'int')
    # Storing an element on a container (line 210)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 4), y_39735, (int_39736, float_39734))
    
    # Assigning a Call to a Name (line 211):
    
    # Assigning a Call to a Name (line 211):
    
    # Call to array(...): (line 211)
    # Processing the call arguments (line 211)
    
    # Obtaining an instance of the builtin type 'list' (line 211)
    list_39739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 211)
    
    # Processing the call keyword arguments (line 211)
    kwargs_39740 = {}
    # Getting the type of 'np' (line 211)
    np_39737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 211)
    array_39738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), np_39737, 'array')
    # Calling array(args, kwargs) (line 211)
    array_call_result_39741 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), array_39738, *[list_39739], **kwargs_39740)
    
    # Assigning a type to the variable 'p' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'p', array_call_result_39741)
    
    # Assigning a Call to a Tuple (line 212):
    
    # Assigning a Subscript to a Name (line 212):
    
    # Obtaining the type of the subscript
    int_39742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 4), 'int')
    
    # Call to estimate_fun_jac(...): (line 212)
    # Processing the call arguments (line 212)

    @norecursion
    def _stypy_temp_lambda_7(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_7'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_7', 212, 36, True)
        # Passed parameters checking function
        _stypy_temp_lambda_7.stypy_localization = localization
        _stypy_temp_lambda_7.stypy_type_of_self = None
        _stypy_temp_lambda_7.stypy_type_store = module_type_store
        _stypy_temp_lambda_7.stypy_function_name = '_stypy_temp_lambda_7'
        _stypy_temp_lambda_7.stypy_param_names_list = ['x', 'y', 'p']
        _stypy_temp_lambda_7.stypy_varargs_param_name = None
        _stypy_temp_lambda_7.stypy_kwargs_param_name = None
        _stypy_temp_lambda_7.stypy_call_defaults = defaults
        _stypy_temp_lambda_7.stypy_call_varargs = varargs
        _stypy_temp_lambda_7.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_7', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_7', ['x', 'y', 'p'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to emden_fun(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'x' (line 212)
        x_39745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 62), 'x', False)
        # Getting the type of 'y' (line 212)
        y_39746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 65), 'y', False)
        # Processing the call keyword arguments (line 212)
        kwargs_39747 = {}
        # Getting the type of 'emden_fun' (line 212)
        emden_fun_39744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 52), 'emden_fun', False)
        # Calling emden_fun(args, kwargs) (line 212)
        emden_fun_call_result_39748 = invoke(stypy.reporting.localization.Localization(__file__, 212, 52), emden_fun_39744, *[x_39745, y_39746], **kwargs_39747)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'stypy_return_type', emden_fun_call_result_39748)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_7' in the type store
        # Getting the type of 'stypy_return_type' (line 212)
        stypy_return_type_39749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39749)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_7'
        return stypy_return_type_39749

    # Assigning a type to the variable '_stypy_temp_lambda_7' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), '_stypy_temp_lambda_7', _stypy_temp_lambda_7)
    # Getting the type of '_stypy_temp_lambda_7' (line 212)
    _stypy_temp_lambda_7_39750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), '_stypy_temp_lambda_7')
    # Getting the type of 'x' (line 212)
    x_39751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 69), 'x', False)
    # Getting the type of 'y' (line 212)
    y_39752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 72), 'y', False)
    # Getting the type of 'p' (line 212)
    p_39753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 75), 'p', False)
    # Processing the call keyword arguments (line 212)
    kwargs_39754 = {}
    # Getting the type of 'estimate_fun_jac' (line 212)
    estimate_fun_jac_39743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 19), 'estimate_fun_jac', False)
    # Calling estimate_fun_jac(args, kwargs) (line 212)
    estimate_fun_jac_call_result_39755 = invoke(stypy.reporting.localization.Localization(__file__, 212, 19), estimate_fun_jac_39743, *[_stypy_temp_lambda_7_39750, x_39751, y_39752, p_39753], **kwargs_39754)
    
    # Obtaining the member '__getitem__' of a type (line 212)
    getitem___39756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 4), estimate_fun_jac_call_result_39755, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 212)
    subscript_call_result_39757 = invoke(stypy.reporting.localization.Localization(__file__, 212, 4), getitem___39756, int_39742)
    
    # Assigning a type to the variable 'tuple_var_assignment_38804' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'tuple_var_assignment_38804', subscript_call_result_39757)
    
    # Assigning a Subscript to a Name (line 212):
    
    # Obtaining the type of the subscript
    int_39758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 4), 'int')
    
    # Call to estimate_fun_jac(...): (line 212)
    # Processing the call arguments (line 212)

    @norecursion
    def _stypy_temp_lambda_8(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_8'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_8', 212, 36, True)
        # Passed parameters checking function
        _stypy_temp_lambda_8.stypy_localization = localization
        _stypy_temp_lambda_8.stypy_type_of_self = None
        _stypy_temp_lambda_8.stypy_type_store = module_type_store
        _stypy_temp_lambda_8.stypy_function_name = '_stypy_temp_lambda_8'
        _stypy_temp_lambda_8.stypy_param_names_list = ['x', 'y', 'p']
        _stypy_temp_lambda_8.stypy_varargs_param_name = None
        _stypy_temp_lambda_8.stypy_kwargs_param_name = None
        _stypy_temp_lambda_8.stypy_call_defaults = defaults
        _stypy_temp_lambda_8.stypy_call_varargs = varargs
        _stypy_temp_lambda_8.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_8', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_8', ['x', 'y', 'p'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to emden_fun(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'x' (line 212)
        x_39761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 62), 'x', False)
        # Getting the type of 'y' (line 212)
        y_39762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 65), 'y', False)
        # Processing the call keyword arguments (line 212)
        kwargs_39763 = {}
        # Getting the type of 'emden_fun' (line 212)
        emden_fun_39760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 52), 'emden_fun', False)
        # Calling emden_fun(args, kwargs) (line 212)
        emden_fun_call_result_39764 = invoke(stypy.reporting.localization.Localization(__file__, 212, 52), emden_fun_39760, *[x_39761, y_39762], **kwargs_39763)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'stypy_return_type', emden_fun_call_result_39764)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_8' in the type store
        # Getting the type of 'stypy_return_type' (line 212)
        stypy_return_type_39765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39765)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_8'
        return stypy_return_type_39765

    # Assigning a type to the variable '_stypy_temp_lambda_8' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), '_stypy_temp_lambda_8', _stypy_temp_lambda_8)
    # Getting the type of '_stypy_temp_lambda_8' (line 212)
    _stypy_temp_lambda_8_39766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), '_stypy_temp_lambda_8')
    # Getting the type of 'x' (line 212)
    x_39767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 69), 'x', False)
    # Getting the type of 'y' (line 212)
    y_39768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 72), 'y', False)
    # Getting the type of 'p' (line 212)
    p_39769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 75), 'p', False)
    # Processing the call keyword arguments (line 212)
    kwargs_39770 = {}
    # Getting the type of 'estimate_fun_jac' (line 212)
    estimate_fun_jac_39759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 19), 'estimate_fun_jac', False)
    # Calling estimate_fun_jac(args, kwargs) (line 212)
    estimate_fun_jac_call_result_39771 = invoke(stypy.reporting.localization.Localization(__file__, 212, 19), estimate_fun_jac_39759, *[_stypy_temp_lambda_8_39766, x_39767, y_39768, p_39769], **kwargs_39770)
    
    # Obtaining the member '__getitem__' of a type (line 212)
    getitem___39772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 4), estimate_fun_jac_call_result_39771, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 212)
    subscript_call_result_39773 = invoke(stypy.reporting.localization.Localization(__file__, 212, 4), getitem___39772, int_39758)
    
    # Assigning a type to the variable 'tuple_var_assignment_38805' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'tuple_var_assignment_38805', subscript_call_result_39773)
    
    # Assigning a Name to a Name (line 212):
    # Getting the type of 'tuple_var_assignment_38804' (line 212)
    tuple_var_assignment_38804_39774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'tuple_var_assignment_38804')
    # Assigning a type to the variable 'df_dy' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'df_dy', tuple_var_assignment_38804_39774)
    
    # Assigning a Name to a Name (line 212):
    # Getting the type of 'tuple_var_assignment_38805' (line 212)
    tuple_var_assignment_38805_39775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'tuple_var_assignment_38805')
    # Assigning a type to the variable 'df_dp' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 11), 'df_dp', tuple_var_assignment_38805_39775)
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to emden_fun_jac(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'x' (line 213)
    x_39777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 29), 'x', False)
    # Getting the type of 'y' (line 213)
    y_39778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 32), 'y', False)
    # Processing the call keyword arguments (line 213)
    kwargs_39779 = {}
    # Getting the type of 'emden_fun_jac' (line 213)
    emden_fun_jac_39776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'emden_fun_jac', False)
    # Calling emden_fun_jac(args, kwargs) (line 213)
    emden_fun_jac_call_result_39780 = invoke(stypy.reporting.localization.Localization(__file__, 213, 15), emden_fun_jac_39776, *[x_39777, y_39778], **kwargs_39779)
    
    # Assigning a type to the variable 'df_dy_an' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'df_dy_an', emden_fun_jac_call_result_39780)
    
    # Call to assert_allclose(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'df_dy' (line 214)
    df_dy_39782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'df_dy', False)
    # Getting the type of 'df_dy_an' (line 214)
    df_dy_an_39783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 27), 'df_dy_an', False)
    # Processing the call keyword arguments (line 214)
    kwargs_39784 = {}
    # Getting the type of 'assert_allclose' (line 214)
    assert_allclose_39781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 214)
    assert_allclose_call_result_39785 = invoke(stypy.reporting.localization.Localization(__file__, 214, 4), assert_allclose_39781, *[df_dy_39782, df_dy_an_39783], **kwargs_39784)
    
    
    # Call to assert_(...): (line 215)
    # Processing the call arguments (line 215)
    
    # Getting the type of 'df_dp' (line 215)
    df_dp_39787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'df_dp', False)
    # Getting the type of 'None' (line 215)
    None_39788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'None', False)
    # Applying the binary operator 'is' (line 215)
    result_is__39789 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 12), 'is', df_dp_39787, None_39788)
    
    # Processing the call keyword arguments (line 215)
    kwargs_39790 = {}
    # Getting the type of 'assert_' (line 215)
    assert__39786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 215)
    assert__call_result_39791 = invoke(stypy.reporting.localization.Localization(__file__, 215, 4), assert__39786, *[result_is__39789], **kwargs_39790)
    
    
    # ################# End of 'test_compute_fun_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_compute_fun_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 186)
    stypy_return_type_39792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39792)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_compute_fun_jac'
    return stypy_return_type_39792

# Assigning a type to the variable 'test_compute_fun_jac' (line 186)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'test_compute_fun_jac', test_compute_fun_jac)

@norecursion
def test_compute_bc_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_compute_bc_jac'
    module_type_store = module_type_store.open_function_context('test_compute_bc_jac', 218, 0, False)
    
    # Passed parameters checking function
    test_compute_bc_jac.stypy_localization = localization
    test_compute_bc_jac.stypy_type_of_self = None
    test_compute_bc_jac.stypy_type_store = module_type_store
    test_compute_bc_jac.stypy_function_name = 'test_compute_bc_jac'
    test_compute_bc_jac.stypy_param_names_list = []
    test_compute_bc_jac.stypy_varargs_param_name = None
    test_compute_bc_jac.stypy_kwargs_param_name = None
    test_compute_bc_jac.stypy_call_defaults = defaults
    test_compute_bc_jac.stypy_call_varargs = varargs
    test_compute_bc_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_compute_bc_jac', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_compute_bc_jac', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_compute_bc_jac(...)' code ##################

    
    # Assigning a Call to a Name (line 219):
    
    # Assigning a Call to a Name (line 219):
    
    # Call to array(...): (line 219)
    # Processing the call arguments (line 219)
    
    # Obtaining an instance of the builtin type 'list' (line 219)
    list_39795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 219)
    # Adding element type (line 219)
    float_39796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 18), list_39795, float_39796)
    # Adding element type (line 219)
    int_39797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 18), list_39795, int_39797)
    
    # Processing the call keyword arguments (line 219)
    kwargs_39798 = {}
    # Getting the type of 'np' (line 219)
    np_39793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 219)
    array_39794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 9), np_39793, 'array')
    # Calling array(args, kwargs) (line 219)
    array_call_result_39799 = invoke(stypy.reporting.localization.Localization(__file__, 219, 9), array_39794, *[list_39795], **kwargs_39798)
    
    # Assigning a type to the variable 'ya' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'ya', array_call_result_39799)
    
    # Assigning a Call to a Name (line 220):
    
    # Assigning a Call to a Name (line 220):
    
    # Call to array(...): (line 220)
    # Processing the call arguments (line 220)
    
    # Obtaining an instance of the builtin type 'list' (line 220)
    list_39802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 220)
    # Adding element type (line 220)
    float_39803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 18), list_39802, float_39803)
    # Adding element type (line 220)
    int_39804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 18), list_39802, int_39804)
    
    # Processing the call keyword arguments (line 220)
    kwargs_39805 = {}
    # Getting the type of 'np' (line 220)
    np_39800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 220)
    array_39801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 9), np_39800, 'array')
    # Calling array(args, kwargs) (line 220)
    array_call_result_39806 = invoke(stypy.reporting.localization.Localization(__file__, 220, 9), array_39801, *[list_39802], **kwargs_39805)
    
    # Assigning a type to the variable 'yb' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'yb', array_call_result_39806)
    
    # Assigning a Call to a Name (line 221):
    
    # Assigning a Call to a Name (line 221):
    
    # Call to array(...): (line 221)
    # Processing the call arguments (line 221)
    
    # Obtaining an instance of the builtin type 'list' (line 221)
    list_39809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 221)
    
    # Processing the call keyword arguments (line 221)
    kwargs_39810 = {}
    # Getting the type of 'np' (line 221)
    np_39807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 221)
    array_39808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), np_39807, 'array')
    # Calling array(args, kwargs) (line 221)
    array_call_result_39811 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), array_39808, *[list_39809], **kwargs_39810)
    
    # Assigning a type to the variable 'p' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'p', array_call_result_39811)
    
    # Assigning a Call to a Tuple (line 222):
    
    # Assigning a Subscript to a Name (line 222):
    
    # Obtaining the type of the subscript
    int_39812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 4), 'int')
    
    # Call to estimate_bc_jac(...): (line 222)
    # Processing the call arguments (line 222)

    @norecursion
    def _stypy_temp_lambda_9(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_9'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_9', 223, 8, True)
        # Passed parameters checking function
        _stypy_temp_lambda_9.stypy_localization = localization
        _stypy_temp_lambda_9.stypy_type_of_self = None
        _stypy_temp_lambda_9.stypy_type_store = module_type_store
        _stypy_temp_lambda_9.stypy_function_name = '_stypy_temp_lambda_9'
        _stypy_temp_lambda_9.stypy_param_names_list = ['ya', 'yb', 'p']
        _stypy_temp_lambda_9.stypy_varargs_param_name = None
        _stypy_temp_lambda_9.stypy_kwargs_param_name = None
        _stypy_temp_lambda_9.stypy_call_defaults = defaults
        _stypy_temp_lambda_9.stypy_call_varargs = varargs
        _stypy_temp_lambda_9.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_9', ['ya', 'yb', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_9', ['ya', 'yb', 'p'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to exp_bc(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'ya' (line 223)
        ya_39815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 33), 'ya', False)
        # Getting the type of 'yb' (line 223)
        yb_39816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 37), 'yb', False)
        # Processing the call keyword arguments (line 223)
        kwargs_39817 = {}
        # Getting the type of 'exp_bc' (line 223)
        exp_bc_39814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 26), 'exp_bc', False)
        # Calling exp_bc(args, kwargs) (line 223)
        exp_bc_call_result_39818 = invoke(stypy.reporting.localization.Localization(__file__, 223, 26), exp_bc_39814, *[ya_39815, yb_39816], **kwargs_39817)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type', exp_bc_call_result_39818)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_9' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_39819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39819)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_9'
        return stypy_return_type_39819

    # Assigning a type to the variable '_stypy_temp_lambda_9' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), '_stypy_temp_lambda_9', _stypy_temp_lambda_9)
    # Getting the type of '_stypy_temp_lambda_9' (line 223)
    _stypy_temp_lambda_9_39820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), '_stypy_temp_lambda_9')
    # Getting the type of 'ya' (line 223)
    ya_39821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), 'ya', False)
    # Getting the type of 'yb' (line 223)
    yb_39822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 46), 'yb', False)
    # Getting the type of 'p' (line 223)
    p_39823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 50), 'p', False)
    # Processing the call keyword arguments (line 222)
    kwargs_39824 = {}
    # Getting the type of 'estimate_bc_jac' (line 222)
    estimate_bc_jac_39813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 31), 'estimate_bc_jac', False)
    # Calling estimate_bc_jac(args, kwargs) (line 222)
    estimate_bc_jac_call_result_39825 = invoke(stypy.reporting.localization.Localization(__file__, 222, 31), estimate_bc_jac_39813, *[_stypy_temp_lambda_9_39820, ya_39821, yb_39822, p_39823], **kwargs_39824)
    
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___39826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 4), estimate_bc_jac_call_result_39825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_39827 = invoke(stypy.reporting.localization.Localization(__file__, 222, 4), getitem___39826, int_39812)
    
    # Assigning a type to the variable 'tuple_var_assignment_38806' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'tuple_var_assignment_38806', subscript_call_result_39827)
    
    # Assigning a Subscript to a Name (line 222):
    
    # Obtaining the type of the subscript
    int_39828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 4), 'int')
    
    # Call to estimate_bc_jac(...): (line 222)
    # Processing the call arguments (line 222)

    @norecursion
    def _stypy_temp_lambda_10(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_10'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_10', 223, 8, True)
        # Passed parameters checking function
        _stypy_temp_lambda_10.stypy_localization = localization
        _stypy_temp_lambda_10.stypy_type_of_self = None
        _stypy_temp_lambda_10.stypy_type_store = module_type_store
        _stypy_temp_lambda_10.stypy_function_name = '_stypy_temp_lambda_10'
        _stypy_temp_lambda_10.stypy_param_names_list = ['ya', 'yb', 'p']
        _stypy_temp_lambda_10.stypy_varargs_param_name = None
        _stypy_temp_lambda_10.stypy_kwargs_param_name = None
        _stypy_temp_lambda_10.stypy_call_defaults = defaults
        _stypy_temp_lambda_10.stypy_call_varargs = varargs
        _stypy_temp_lambda_10.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_10', ['ya', 'yb', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_10', ['ya', 'yb', 'p'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to exp_bc(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'ya' (line 223)
        ya_39831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 33), 'ya', False)
        # Getting the type of 'yb' (line 223)
        yb_39832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 37), 'yb', False)
        # Processing the call keyword arguments (line 223)
        kwargs_39833 = {}
        # Getting the type of 'exp_bc' (line 223)
        exp_bc_39830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 26), 'exp_bc', False)
        # Calling exp_bc(args, kwargs) (line 223)
        exp_bc_call_result_39834 = invoke(stypy.reporting.localization.Localization(__file__, 223, 26), exp_bc_39830, *[ya_39831, yb_39832], **kwargs_39833)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type', exp_bc_call_result_39834)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_10' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_39835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39835)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_10'
        return stypy_return_type_39835

    # Assigning a type to the variable '_stypy_temp_lambda_10' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), '_stypy_temp_lambda_10', _stypy_temp_lambda_10)
    # Getting the type of '_stypy_temp_lambda_10' (line 223)
    _stypy_temp_lambda_10_39836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), '_stypy_temp_lambda_10')
    # Getting the type of 'ya' (line 223)
    ya_39837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), 'ya', False)
    # Getting the type of 'yb' (line 223)
    yb_39838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 46), 'yb', False)
    # Getting the type of 'p' (line 223)
    p_39839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 50), 'p', False)
    # Processing the call keyword arguments (line 222)
    kwargs_39840 = {}
    # Getting the type of 'estimate_bc_jac' (line 222)
    estimate_bc_jac_39829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 31), 'estimate_bc_jac', False)
    # Calling estimate_bc_jac(args, kwargs) (line 222)
    estimate_bc_jac_call_result_39841 = invoke(stypy.reporting.localization.Localization(__file__, 222, 31), estimate_bc_jac_39829, *[_stypy_temp_lambda_10_39836, ya_39837, yb_39838, p_39839], **kwargs_39840)
    
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___39842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 4), estimate_bc_jac_call_result_39841, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_39843 = invoke(stypy.reporting.localization.Localization(__file__, 222, 4), getitem___39842, int_39828)
    
    # Assigning a type to the variable 'tuple_var_assignment_38807' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'tuple_var_assignment_38807', subscript_call_result_39843)
    
    # Assigning a Subscript to a Name (line 222):
    
    # Obtaining the type of the subscript
    int_39844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 4), 'int')
    
    # Call to estimate_bc_jac(...): (line 222)
    # Processing the call arguments (line 222)

    @norecursion
    def _stypy_temp_lambda_11(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_11'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_11', 223, 8, True)
        # Passed parameters checking function
        _stypy_temp_lambda_11.stypy_localization = localization
        _stypy_temp_lambda_11.stypy_type_of_self = None
        _stypy_temp_lambda_11.stypy_type_store = module_type_store
        _stypy_temp_lambda_11.stypy_function_name = '_stypy_temp_lambda_11'
        _stypy_temp_lambda_11.stypy_param_names_list = ['ya', 'yb', 'p']
        _stypy_temp_lambda_11.stypy_varargs_param_name = None
        _stypy_temp_lambda_11.stypy_kwargs_param_name = None
        _stypy_temp_lambda_11.stypy_call_defaults = defaults
        _stypy_temp_lambda_11.stypy_call_varargs = varargs
        _stypy_temp_lambda_11.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_11', ['ya', 'yb', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_11', ['ya', 'yb', 'p'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to exp_bc(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'ya' (line 223)
        ya_39847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 33), 'ya', False)
        # Getting the type of 'yb' (line 223)
        yb_39848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 37), 'yb', False)
        # Processing the call keyword arguments (line 223)
        kwargs_39849 = {}
        # Getting the type of 'exp_bc' (line 223)
        exp_bc_39846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 26), 'exp_bc', False)
        # Calling exp_bc(args, kwargs) (line 223)
        exp_bc_call_result_39850 = invoke(stypy.reporting.localization.Localization(__file__, 223, 26), exp_bc_39846, *[ya_39847, yb_39848], **kwargs_39849)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type', exp_bc_call_result_39850)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_11' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_39851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39851)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_11'
        return stypy_return_type_39851

    # Assigning a type to the variable '_stypy_temp_lambda_11' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), '_stypy_temp_lambda_11', _stypy_temp_lambda_11)
    # Getting the type of '_stypy_temp_lambda_11' (line 223)
    _stypy_temp_lambda_11_39852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), '_stypy_temp_lambda_11')
    # Getting the type of 'ya' (line 223)
    ya_39853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), 'ya', False)
    # Getting the type of 'yb' (line 223)
    yb_39854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 46), 'yb', False)
    # Getting the type of 'p' (line 223)
    p_39855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 50), 'p', False)
    # Processing the call keyword arguments (line 222)
    kwargs_39856 = {}
    # Getting the type of 'estimate_bc_jac' (line 222)
    estimate_bc_jac_39845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 31), 'estimate_bc_jac', False)
    # Calling estimate_bc_jac(args, kwargs) (line 222)
    estimate_bc_jac_call_result_39857 = invoke(stypy.reporting.localization.Localization(__file__, 222, 31), estimate_bc_jac_39845, *[_stypy_temp_lambda_11_39852, ya_39853, yb_39854, p_39855], **kwargs_39856)
    
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___39858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 4), estimate_bc_jac_call_result_39857, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_39859 = invoke(stypy.reporting.localization.Localization(__file__, 222, 4), getitem___39858, int_39844)
    
    # Assigning a type to the variable 'tuple_var_assignment_38808' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'tuple_var_assignment_38808', subscript_call_result_39859)
    
    # Assigning a Name to a Name (line 222):
    # Getting the type of 'tuple_var_assignment_38806' (line 222)
    tuple_var_assignment_38806_39860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'tuple_var_assignment_38806')
    # Assigning a type to the variable 'dbc_dya' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'dbc_dya', tuple_var_assignment_38806_39860)
    
    # Assigning a Name to a Name (line 222):
    # Getting the type of 'tuple_var_assignment_38807' (line 222)
    tuple_var_assignment_38807_39861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'tuple_var_assignment_38807')
    # Assigning a type to the variable 'dbc_dyb' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 13), 'dbc_dyb', tuple_var_assignment_38807_39861)
    
    # Assigning a Name to a Name (line 222):
    # Getting the type of 'tuple_var_assignment_38808' (line 222)
    tuple_var_assignment_38808_39862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'tuple_var_assignment_38808')
    # Assigning a type to the variable 'dbc_dp' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 22), 'dbc_dp', tuple_var_assignment_38808_39862)
    
    # Assigning a Call to a Tuple (line 224):
    
    # Assigning a Subscript to a Name (line 224):
    
    # Obtaining the type of the subscript
    int_39863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 4), 'int')
    
    # Call to exp_bc_jac(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'ya' (line 224)
    ya_39865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 40), 'ya', False)
    # Getting the type of 'yb' (line 224)
    yb_39866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 44), 'yb', False)
    # Processing the call keyword arguments (line 224)
    kwargs_39867 = {}
    # Getting the type of 'exp_bc_jac' (line 224)
    exp_bc_jac_39864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 29), 'exp_bc_jac', False)
    # Calling exp_bc_jac(args, kwargs) (line 224)
    exp_bc_jac_call_result_39868 = invoke(stypy.reporting.localization.Localization(__file__, 224, 29), exp_bc_jac_39864, *[ya_39865, yb_39866], **kwargs_39867)
    
    # Obtaining the member '__getitem__' of a type (line 224)
    getitem___39869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 4), exp_bc_jac_call_result_39868, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 224)
    subscript_call_result_39870 = invoke(stypy.reporting.localization.Localization(__file__, 224, 4), getitem___39869, int_39863)
    
    # Assigning a type to the variable 'tuple_var_assignment_38809' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'tuple_var_assignment_38809', subscript_call_result_39870)
    
    # Assigning a Subscript to a Name (line 224):
    
    # Obtaining the type of the subscript
    int_39871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 4), 'int')
    
    # Call to exp_bc_jac(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'ya' (line 224)
    ya_39873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 40), 'ya', False)
    # Getting the type of 'yb' (line 224)
    yb_39874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 44), 'yb', False)
    # Processing the call keyword arguments (line 224)
    kwargs_39875 = {}
    # Getting the type of 'exp_bc_jac' (line 224)
    exp_bc_jac_39872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 29), 'exp_bc_jac', False)
    # Calling exp_bc_jac(args, kwargs) (line 224)
    exp_bc_jac_call_result_39876 = invoke(stypy.reporting.localization.Localization(__file__, 224, 29), exp_bc_jac_39872, *[ya_39873, yb_39874], **kwargs_39875)
    
    # Obtaining the member '__getitem__' of a type (line 224)
    getitem___39877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 4), exp_bc_jac_call_result_39876, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 224)
    subscript_call_result_39878 = invoke(stypy.reporting.localization.Localization(__file__, 224, 4), getitem___39877, int_39871)
    
    # Assigning a type to the variable 'tuple_var_assignment_38810' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'tuple_var_assignment_38810', subscript_call_result_39878)
    
    # Assigning a Name to a Name (line 224):
    # Getting the type of 'tuple_var_assignment_38809' (line 224)
    tuple_var_assignment_38809_39879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'tuple_var_assignment_38809')
    # Assigning a type to the variable 'dbc_dya_an' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'dbc_dya_an', tuple_var_assignment_38809_39879)
    
    # Assigning a Name to a Name (line 224):
    # Getting the type of 'tuple_var_assignment_38810' (line 224)
    tuple_var_assignment_38810_39880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'tuple_var_assignment_38810')
    # Assigning a type to the variable 'dbc_dyb_an' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'dbc_dyb_an', tuple_var_assignment_38810_39880)
    
    # Call to assert_allclose(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'dbc_dya' (line 225)
    dbc_dya_39882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'dbc_dya', False)
    # Getting the type of 'dbc_dya_an' (line 225)
    dbc_dya_an_39883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 29), 'dbc_dya_an', False)
    # Processing the call keyword arguments (line 225)
    kwargs_39884 = {}
    # Getting the type of 'assert_allclose' (line 225)
    assert_allclose_39881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 225)
    assert_allclose_call_result_39885 = invoke(stypy.reporting.localization.Localization(__file__, 225, 4), assert_allclose_39881, *[dbc_dya_39882, dbc_dya_an_39883], **kwargs_39884)
    
    
    # Call to assert_allclose(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'dbc_dyb' (line 226)
    dbc_dyb_39887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'dbc_dyb', False)
    # Getting the type of 'dbc_dyb_an' (line 226)
    dbc_dyb_an_39888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 29), 'dbc_dyb_an', False)
    # Processing the call keyword arguments (line 226)
    kwargs_39889 = {}
    # Getting the type of 'assert_allclose' (line 226)
    assert_allclose_39886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 226)
    assert_allclose_call_result_39890 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), assert_allclose_39886, *[dbc_dyb_39887, dbc_dyb_an_39888], **kwargs_39889)
    
    
    # Call to assert_(...): (line 227)
    # Processing the call arguments (line 227)
    
    # Getting the type of 'dbc_dp' (line 227)
    dbc_dp_39892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'dbc_dp', False)
    # Getting the type of 'None' (line 227)
    None_39893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 22), 'None', False)
    # Applying the binary operator 'is' (line 227)
    result_is__39894 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 12), 'is', dbc_dp_39892, None_39893)
    
    # Processing the call keyword arguments (line 227)
    kwargs_39895 = {}
    # Getting the type of 'assert_' (line 227)
    assert__39891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 227)
    assert__call_result_39896 = invoke(stypy.reporting.localization.Localization(__file__, 227, 4), assert__39891, *[result_is__39894], **kwargs_39895)
    
    
    # Assigning a Call to a Name (line 229):
    
    # Assigning a Call to a Name (line 229):
    
    # Call to array(...): (line 229)
    # Processing the call arguments (line 229)
    
    # Obtaining an instance of the builtin type 'list' (line 229)
    list_39899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 229)
    # Adding element type (line 229)
    float_39900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 18), list_39899, float_39900)
    # Adding element type (line 229)
    int_39901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 18), list_39899, int_39901)
    
    # Processing the call keyword arguments (line 229)
    kwargs_39902 = {}
    # Getting the type of 'np' (line 229)
    np_39897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 229)
    array_39898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 9), np_39897, 'array')
    # Calling array(args, kwargs) (line 229)
    array_call_result_39903 = invoke(stypy.reporting.localization.Localization(__file__, 229, 9), array_39898, *[list_39899], **kwargs_39902)
    
    # Assigning a type to the variable 'ya' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'ya', array_call_result_39903)
    
    # Assigning a Call to a Name (line 230):
    
    # Assigning a Call to a Name (line 230):
    
    # Call to array(...): (line 230)
    # Processing the call arguments (line 230)
    
    # Obtaining an instance of the builtin type 'list' (line 230)
    list_39906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 230)
    # Adding element type (line 230)
    float_39907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 18), list_39906, float_39907)
    # Adding element type (line 230)
    int_39908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 18), list_39906, int_39908)
    
    # Processing the call keyword arguments (line 230)
    kwargs_39909 = {}
    # Getting the type of 'np' (line 230)
    np_39904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 230)
    array_39905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 9), np_39904, 'array')
    # Calling array(args, kwargs) (line 230)
    array_call_result_39910 = invoke(stypy.reporting.localization.Localization(__file__, 230, 9), array_39905, *[list_39906], **kwargs_39909)
    
    # Assigning a type to the variable 'yb' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'yb', array_call_result_39910)
    
    # Assigning a Call to a Name (line 231):
    
    # Assigning a Call to a Name (line 231):
    
    # Call to array(...): (line 231)
    # Processing the call arguments (line 231)
    
    # Obtaining an instance of the builtin type 'list' (line 231)
    list_39913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 231)
    # Adding element type (line 231)
    float_39914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 17), list_39913, float_39914)
    
    # Processing the call keyword arguments (line 231)
    kwargs_39915 = {}
    # Getting the type of 'np' (line 231)
    np_39911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 231)
    array_39912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), np_39911, 'array')
    # Calling array(args, kwargs) (line 231)
    array_call_result_39916 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), array_39912, *[list_39913], **kwargs_39915)
    
    # Assigning a type to the variable 'p' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'p', array_call_result_39916)
    
    # Assigning a Call to a Tuple (line 232):
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_39917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 4), 'int')
    
    # Call to estimate_bc_jac(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'sl_bc' (line 232)
    sl_bc_39919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 47), 'sl_bc', False)
    # Getting the type of 'ya' (line 232)
    ya_39920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 54), 'ya', False)
    # Getting the type of 'yb' (line 232)
    yb_39921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 58), 'yb', False)
    # Getting the type of 'p' (line 232)
    p_39922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 62), 'p', False)
    # Processing the call keyword arguments (line 232)
    kwargs_39923 = {}
    # Getting the type of 'estimate_bc_jac' (line 232)
    estimate_bc_jac_39918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 31), 'estimate_bc_jac', False)
    # Calling estimate_bc_jac(args, kwargs) (line 232)
    estimate_bc_jac_call_result_39924 = invoke(stypy.reporting.localization.Localization(__file__, 232, 31), estimate_bc_jac_39918, *[sl_bc_39919, ya_39920, yb_39921, p_39922], **kwargs_39923)
    
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___39925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 4), estimate_bc_jac_call_result_39924, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_39926 = invoke(stypy.reporting.localization.Localization(__file__, 232, 4), getitem___39925, int_39917)
    
    # Assigning a type to the variable 'tuple_var_assignment_38811' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_38811', subscript_call_result_39926)
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_39927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 4), 'int')
    
    # Call to estimate_bc_jac(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'sl_bc' (line 232)
    sl_bc_39929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 47), 'sl_bc', False)
    # Getting the type of 'ya' (line 232)
    ya_39930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 54), 'ya', False)
    # Getting the type of 'yb' (line 232)
    yb_39931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 58), 'yb', False)
    # Getting the type of 'p' (line 232)
    p_39932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 62), 'p', False)
    # Processing the call keyword arguments (line 232)
    kwargs_39933 = {}
    # Getting the type of 'estimate_bc_jac' (line 232)
    estimate_bc_jac_39928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 31), 'estimate_bc_jac', False)
    # Calling estimate_bc_jac(args, kwargs) (line 232)
    estimate_bc_jac_call_result_39934 = invoke(stypy.reporting.localization.Localization(__file__, 232, 31), estimate_bc_jac_39928, *[sl_bc_39929, ya_39930, yb_39931, p_39932], **kwargs_39933)
    
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___39935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 4), estimate_bc_jac_call_result_39934, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_39936 = invoke(stypy.reporting.localization.Localization(__file__, 232, 4), getitem___39935, int_39927)
    
    # Assigning a type to the variable 'tuple_var_assignment_38812' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_38812', subscript_call_result_39936)
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_39937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 4), 'int')
    
    # Call to estimate_bc_jac(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'sl_bc' (line 232)
    sl_bc_39939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 47), 'sl_bc', False)
    # Getting the type of 'ya' (line 232)
    ya_39940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 54), 'ya', False)
    # Getting the type of 'yb' (line 232)
    yb_39941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 58), 'yb', False)
    # Getting the type of 'p' (line 232)
    p_39942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 62), 'p', False)
    # Processing the call keyword arguments (line 232)
    kwargs_39943 = {}
    # Getting the type of 'estimate_bc_jac' (line 232)
    estimate_bc_jac_39938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 31), 'estimate_bc_jac', False)
    # Calling estimate_bc_jac(args, kwargs) (line 232)
    estimate_bc_jac_call_result_39944 = invoke(stypy.reporting.localization.Localization(__file__, 232, 31), estimate_bc_jac_39938, *[sl_bc_39939, ya_39940, yb_39941, p_39942], **kwargs_39943)
    
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___39945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 4), estimate_bc_jac_call_result_39944, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_39946 = invoke(stypy.reporting.localization.Localization(__file__, 232, 4), getitem___39945, int_39937)
    
    # Assigning a type to the variable 'tuple_var_assignment_38813' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_38813', subscript_call_result_39946)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_var_assignment_38811' (line 232)
    tuple_var_assignment_38811_39947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_38811')
    # Assigning a type to the variable 'dbc_dya' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'dbc_dya', tuple_var_assignment_38811_39947)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_var_assignment_38812' (line 232)
    tuple_var_assignment_38812_39948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_38812')
    # Assigning a type to the variable 'dbc_dyb' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 13), 'dbc_dyb', tuple_var_assignment_38812_39948)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_var_assignment_38813' (line 232)
    tuple_var_assignment_38813_39949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_38813')
    # Assigning a type to the variable 'dbc_dp' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 22), 'dbc_dp', tuple_var_assignment_38813_39949)
    
    # Assigning a Call to a Tuple (line 233):
    
    # Assigning a Subscript to a Name (line 233):
    
    # Obtaining the type of the subscript
    int_39950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 4), 'int')
    
    # Call to sl_bc_jac(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'ya' (line 233)
    ya_39952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 50), 'ya', False)
    # Getting the type of 'yb' (line 233)
    yb_39953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 54), 'yb', False)
    # Getting the type of 'p' (line 233)
    p_39954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 58), 'p', False)
    # Processing the call keyword arguments (line 233)
    kwargs_39955 = {}
    # Getting the type of 'sl_bc_jac' (line 233)
    sl_bc_jac_39951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 40), 'sl_bc_jac', False)
    # Calling sl_bc_jac(args, kwargs) (line 233)
    sl_bc_jac_call_result_39956 = invoke(stypy.reporting.localization.Localization(__file__, 233, 40), sl_bc_jac_39951, *[ya_39952, yb_39953, p_39954], **kwargs_39955)
    
    # Obtaining the member '__getitem__' of a type (line 233)
    getitem___39957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 4), sl_bc_jac_call_result_39956, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 233)
    subscript_call_result_39958 = invoke(stypy.reporting.localization.Localization(__file__, 233, 4), getitem___39957, int_39950)
    
    # Assigning a type to the variable 'tuple_var_assignment_38814' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'tuple_var_assignment_38814', subscript_call_result_39958)
    
    # Assigning a Subscript to a Name (line 233):
    
    # Obtaining the type of the subscript
    int_39959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 4), 'int')
    
    # Call to sl_bc_jac(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'ya' (line 233)
    ya_39961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 50), 'ya', False)
    # Getting the type of 'yb' (line 233)
    yb_39962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 54), 'yb', False)
    # Getting the type of 'p' (line 233)
    p_39963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 58), 'p', False)
    # Processing the call keyword arguments (line 233)
    kwargs_39964 = {}
    # Getting the type of 'sl_bc_jac' (line 233)
    sl_bc_jac_39960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 40), 'sl_bc_jac', False)
    # Calling sl_bc_jac(args, kwargs) (line 233)
    sl_bc_jac_call_result_39965 = invoke(stypy.reporting.localization.Localization(__file__, 233, 40), sl_bc_jac_39960, *[ya_39961, yb_39962, p_39963], **kwargs_39964)
    
    # Obtaining the member '__getitem__' of a type (line 233)
    getitem___39966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 4), sl_bc_jac_call_result_39965, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 233)
    subscript_call_result_39967 = invoke(stypy.reporting.localization.Localization(__file__, 233, 4), getitem___39966, int_39959)
    
    # Assigning a type to the variable 'tuple_var_assignment_38815' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'tuple_var_assignment_38815', subscript_call_result_39967)
    
    # Assigning a Subscript to a Name (line 233):
    
    # Obtaining the type of the subscript
    int_39968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 4), 'int')
    
    # Call to sl_bc_jac(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'ya' (line 233)
    ya_39970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 50), 'ya', False)
    # Getting the type of 'yb' (line 233)
    yb_39971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 54), 'yb', False)
    # Getting the type of 'p' (line 233)
    p_39972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 58), 'p', False)
    # Processing the call keyword arguments (line 233)
    kwargs_39973 = {}
    # Getting the type of 'sl_bc_jac' (line 233)
    sl_bc_jac_39969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 40), 'sl_bc_jac', False)
    # Calling sl_bc_jac(args, kwargs) (line 233)
    sl_bc_jac_call_result_39974 = invoke(stypy.reporting.localization.Localization(__file__, 233, 40), sl_bc_jac_39969, *[ya_39970, yb_39971, p_39972], **kwargs_39973)
    
    # Obtaining the member '__getitem__' of a type (line 233)
    getitem___39975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 4), sl_bc_jac_call_result_39974, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 233)
    subscript_call_result_39976 = invoke(stypy.reporting.localization.Localization(__file__, 233, 4), getitem___39975, int_39968)
    
    # Assigning a type to the variable 'tuple_var_assignment_38816' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'tuple_var_assignment_38816', subscript_call_result_39976)
    
    # Assigning a Name to a Name (line 233):
    # Getting the type of 'tuple_var_assignment_38814' (line 233)
    tuple_var_assignment_38814_39977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'tuple_var_assignment_38814')
    # Assigning a type to the variable 'dbc_dya_an' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'dbc_dya_an', tuple_var_assignment_38814_39977)
    
    # Assigning a Name to a Name (line 233):
    # Getting the type of 'tuple_var_assignment_38815' (line 233)
    tuple_var_assignment_38815_39978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'tuple_var_assignment_38815')
    # Assigning a type to the variable 'dbc_dyb_an' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'dbc_dyb_an', tuple_var_assignment_38815_39978)
    
    # Assigning a Name to a Name (line 233):
    # Getting the type of 'tuple_var_assignment_38816' (line 233)
    tuple_var_assignment_38816_39979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'tuple_var_assignment_38816')
    # Assigning a type to the variable 'dbc_dp_an' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 28), 'dbc_dp_an', tuple_var_assignment_38816_39979)
    
    # Call to assert_allclose(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'dbc_dya' (line 234)
    dbc_dya_39981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 20), 'dbc_dya', False)
    # Getting the type of 'dbc_dya_an' (line 234)
    dbc_dya_an_39982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 29), 'dbc_dya_an', False)
    # Processing the call keyword arguments (line 234)
    kwargs_39983 = {}
    # Getting the type of 'assert_allclose' (line 234)
    assert_allclose_39980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 234)
    assert_allclose_call_result_39984 = invoke(stypy.reporting.localization.Localization(__file__, 234, 4), assert_allclose_39980, *[dbc_dya_39981, dbc_dya_an_39982], **kwargs_39983)
    
    
    # Call to assert_allclose(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'dbc_dyb' (line 235)
    dbc_dyb_39986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'dbc_dyb', False)
    # Getting the type of 'dbc_dyb_an' (line 235)
    dbc_dyb_an_39987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 29), 'dbc_dyb_an', False)
    # Processing the call keyword arguments (line 235)
    kwargs_39988 = {}
    # Getting the type of 'assert_allclose' (line 235)
    assert_allclose_39985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 235)
    assert_allclose_call_result_39989 = invoke(stypy.reporting.localization.Localization(__file__, 235, 4), assert_allclose_39985, *[dbc_dyb_39986, dbc_dyb_an_39987], **kwargs_39988)
    
    
    # Call to assert_allclose(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'dbc_dp' (line 236)
    dbc_dp_39991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'dbc_dp', False)
    # Getting the type of 'dbc_dp_an' (line 236)
    dbc_dp_an_39992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 28), 'dbc_dp_an', False)
    # Processing the call keyword arguments (line 236)
    kwargs_39993 = {}
    # Getting the type of 'assert_allclose' (line 236)
    assert_allclose_39990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 236)
    assert_allclose_call_result_39994 = invoke(stypy.reporting.localization.Localization(__file__, 236, 4), assert_allclose_39990, *[dbc_dp_39991, dbc_dp_an_39992], **kwargs_39993)
    
    
    # Assigning a Call to a Name (line 238):
    
    # Assigning a Call to a Name (line 238):
    
    # Call to array(...): (line 238)
    # Processing the call arguments (line 238)
    
    # Obtaining an instance of the builtin type 'list' (line 238)
    list_39997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 238)
    # Adding element type (line 238)
    float_39998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 18), list_39997, float_39998)
    # Adding element type (line 238)
    int_39999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 18), list_39997, int_39999)
    
    # Processing the call keyword arguments (line 238)
    kwargs_40000 = {}
    # Getting the type of 'np' (line 238)
    np_39995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 238)
    array_39996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 9), np_39995, 'array')
    # Calling array(args, kwargs) (line 238)
    array_call_result_40001 = invoke(stypy.reporting.localization.Localization(__file__, 238, 9), array_39996, *[list_39997], **kwargs_40000)
    
    # Assigning a type to the variable 'ya' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'ya', array_call_result_40001)
    
    # Assigning a Call to a Name (line 239):
    
    # Assigning a Call to a Name (line 239):
    
    # Call to array(...): (line 239)
    # Processing the call arguments (line 239)
    
    # Obtaining an instance of the builtin type 'list' (line 239)
    list_40004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 239)
    # Adding element type (line 239)
    int_40005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 18), list_40004, int_40005)
    # Adding element type (line 239)
    float_40006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 18), list_40004, float_40006)
    
    # Processing the call keyword arguments (line 239)
    kwargs_40007 = {}
    # Getting the type of 'np' (line 239)
    np_40002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 239)
    array_40003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 9), np_40002, 'array')
    # Calling array(args, kwargs) (line 239)
    array_call_result_40008 = invoke(stypy.reporting.localization.Localization(__file__, 239, 9), array_40003, *[list_40004], **kwargs_40007)
    
    # Assigning a type to the variable 'yb' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'yb', array_call_result_40008)
    
    # Assigning a Call to a Name (line 240):
    
    # Assigning a Call to a Name (line 240):
    
    # Call to array(...): (line 240)
    # Processing the call arguments (line 240)
    
    # Obtaining an instance of the builtin type 'list' (line 240)
    list_40011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 240)
    
    # Processing the call keyword arguments (line 240)
    kwargs_40012 = {}
    # Getting the type of 'np' (line 240)
    np_40009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 240)
    array_40010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), np_40009, 'array')
    # Calling array(args, kwargs) (line 240)
    array_call_result_40013 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), array_40010, *[list_40011], **kwargs_40012)
    
    # Assigning a type to the variable 'p' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'p', array_call_result_40013)
    
    # Assigning a Call to a Tuple (line 241):
    
    # Assigning a Subscript to a Name (line 241):
    
    # Obtaining the type of the subscript
    int_40014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 4), 'int')
    
    # Call to estimate_bc_jac(...): (line 241)
    # Processing the call arguments (line 241)

    @norecursion
    def _stypy_temp_lambda_12(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_12'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_12', 242, 8, True)
        # Passed parameters checking function
        _stypy_temp_lambda_12.stypy_localization = localization
        _stypy_temp_lambda_12.stypy_type_of_self = None
        _stypy_temp_lambda_12.stypy_type_store = module_type_store
        _stypy_temp_lambda_12.stypy_function_name = '_stypy_temp_lambda_12'
        _stypy_temp_lambda_12.stypy_param_names_list = ['ya', 'yb', 'p']
        _stypy_temp_lambda_12.stypy_varargs_param_name = None
        _stypy_temp_lambda_12.stypy_kwargs_param_name = None
        _stypy_temp_lambda_12.stypy_call_defaults = defaults
        _stypy_temp_lambda_12.stypy_call_varargs = varargs
        _stypy_temp_lambda_12.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_12', ['ya', 'yb', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_12', ['ya', 'yb', 'p'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to emden_bc(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'ya' (line 242)
        ya_40017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 35), 'ya', False)
        # Getting the type of 'yb' (line 242)
        yb_40018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 39), 'yb', False)
        # Processing the call keyword arguments (line 242)
        kwargs_40019 = {}
        # Getting the type of 'emden_bc' (line 242)
        emden_bc_40016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 26), 'emden_bc', False)
        # Calling emden_bc(args, kwargs) (line 242)
        emden_bc_call_result_40020 = invoke(stypy.reporting.localization.Localization(__file__, 242, 26), emden_bc_40016, *[ya_40017, yb_40018], **kwargs_40019)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'stypy_return_type', emden_bc_call_result_40020)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_12' in the type store
        # Getting the type of 'stypy_return_type' (line 242)
        stypy_return_type_40021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40021)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_12'
        return stypy_return_type_40021

    # Assigning a type to the variable '_stypy_temp_lambda_12' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), '_stypy_temp_lambda_12', _stypy_temp_lambda_12)
    # Getting the type of '_stypy_temp_lambda_12' (line 242)
    _stypy_temp_lambda_12_40022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), '_stypy_temp_lambda_12')
    # Getting the type of 'ya' (line 242)
    ya_40023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 44), 'ya', False)
    # Getting the type of 'yb' (line 242)
    yb_40024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 48), 'yb', False)
    # Getting the type of 'p' (line 242)
    p_40025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 52), 'p', False)
    # Processing the call keyword arguments (line 241)
    kwargs_40026 = {}
    # Getting the type of 'estimate_bc_jac' (line 241)
    estimate_bc_jac_40015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 31), 'estimate_bc_jac', False)
    # Calling estimate_bc_jac(args, kwargs) (line 241)
    estimate_bc_jac_call_result_40027 = invoke(stypy.reporting.localization.Localization(__file__, 241, 31), estimate_bc_jac_40015, *[_stypy_temp_lambda_12_40022, ya_40023, yb_40024, p_40025], **kwargs_40026)
    
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___40028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 4), estimate_bc_jac_call_result_40027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_40029 = invoke(stypy.reporting.localization.Localization(__file__, 241, 4), getitem___40028, int_40014)
    
    # Assigning a type to the variable 'tuple_var_assignment_38817' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_38817', subscript_call_result_40029)
    
    # Assigning a Subscript to a Name (line 241):
    
    # Obtaining the type of the subscript
    int_40030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 4), 'int')
    
    # Call to estimate_bc_jac(...): (line 241)
    # Processing the call arguments (line 241)

    @norecursion
    def _stypy_temp_lambda_13(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_13'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_13', 242, 8, True)
        # Passed parameters checking function
        _stypy_temp_lambda_13.stypy_localization = localization
        _stypy_temp_lambda_13.stypy_type_of_self = None
        _stypy_temp_lambda_13.stypy_type_store = module_type_store
        _stypy_temp_lambda_13.stypy_function_name = '_stypy_temp_lambda_13'
        _stypy_temp_lambda_13.stypy_param_names_list = ['ya', 'yb', 'p']
        _stypy_temp_lambda_13.stypy_varargs_param_name = None
        _stypy_temp_lambda_13.stypy_kwargs_param_name = None
        _stypy_temp_lambda_13.stypy_call_defaults = defaults
        _stypy_temp_lambda_13.stypy_call_varargs = varargs
        _stypy_temp_lambda_13.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_13', ['ya', 'yb', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_13', ['ya', 'yb', 'p'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to emden_bc(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'ya' (line 242)
        ya_40033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 35), 'ya', False)
        # Getting the type of 'yb' (line 242)
        yb_40034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 39), 'yb', False)
        # Processing the call keyword arguments (line 242)
        kwargs_40035 = {}
        # Getting the type of 'emden_bc' (line 242)
        emden_bc_40032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 26), 'emden_bc', False)
        # Calling emden_bc(args, kwargs) (line 242)
        emden_bc_call_result_40036 = invoke(stypy.reporting.localization.Localization(__file__, 242, 26), emden_bc_40032, *[ya_40033, yb_40034], **kwargs_40035)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'stypy_return_type', emden_bc_call_result_40036)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_13' in the type store
        # Getting the type of 'stypy_return_type' (line 242)
        stypy_return_type_40037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40037)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_13'
        return stypy_return_type_40037

    # Assigning a type to the variable '_stypy_temp_lambda_13' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), '_stypy_temp_lambda_13', _stypy_temp_lambda_13)
    # Getting the type of '_stypy_temp_lambda_13' (line 242)
    _stypy_temp_lambda_13_40038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), '_stypy_temp_lambda_13')
    # Getting the type of 'ya' (line 242)
    ya_40039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 44), 'ya', False)
    # Getting the type of 'yb' (line 242)
    yb_40040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 48), 'yb', False)
    # Getting the type of 'p' (line 242)
    p_40041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 52), 'p', False)
    # Processing the call keyword arguments (line 241)
    kwargs_40042 = {}
    # Getting the type of 'estimate_bc_jac' (line 241)
    estimate_bc_jac_40031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 31), 'estimate_bc_jac', False)
    # Calling estimate_bc_jac(args, kwargs) (line 241)
    estimate_bc_jac_call_result_40043 = invoke(stypy.reporting.localization.Localization(__file__, 241, 31), estimate_bc_jac_40031, *[_stypy_temp_lambda_13_40038, ya_40039, yb_40040, p_40041], **kwargs_40042)
    
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___40044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 4), estimate_bc_jac_call_result_40043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_40045 = invoke(stypy.reporting.localization.Localization(__file__, 241, 4), getitem___40044, int_40030)
    
    # Assigning a type to the variable 'tuple_var_assignment_38818' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_38818', subscript_call_result_40045)
    
    # Assigning a Subscript to a Name (line 241):
    
    # Obtaining the type of the subscript
    int_40046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 4), 'int')
    
    # Call to estimate_bc_jac(...): (line 241)
    # Processing the call arguments (line 241)

    @norecursion
    def _stypy_temp_lambda_14(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_14'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_14', 242, 8, True)
        # Passed parameters checking function
        _stypy_temp_lambda_14.stypy_localization = localization
        _stypy_temp_lambda_14.stypy_type_of_self = None
        _stypy_temp_lambda_14.stypy_type_store = module_type_store
        _stypy_temp_lambda_14.stypy_function_name = '_stypy_temp_lambda_14'
        _stypy_temp_lambda_14.stypy_param_names_list = ['ya', 'yb', 'p']
        _stypy_temp_lambda_14.stypy_varargs_param_name = None
        _stypy_temp_lambda_14.stypy_kwargs_param_name = None
        _stypy_temp_lambda_14.stypy_call_defaults = defaults
        _stypy_temp_lambda_14.stypy_call_varargs = varargs
        _stypy_temp_lambda_14.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_14', ['ya', 'yb', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_14', ['ya', 'yb', 'p'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to emden_bc(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'ya' (line 242)
        ya_40049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 35), 'ya', False)
        # Getting the type of 'yb' (line 242)
        yb_40050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 39), 'yb', False)
        # Processing the call keyword arguments (line 242)
        kwargs_40051 = {}
        # Getting the type of 'emden_bc' (line 242)
        emden_bc_40048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 26), 'emden_bc', False)
        # Calling emden_bc(args, kwargs) (line 242)
        emden_bc_call_result_40052 = invoke(stypy.reporting.localization.Localization(__file__, 242, 26), emden_bc_40048, *[ya_40049, yb_40050], **kwargs_40051)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'stypy_return_type', emden_bc_call_result_40052)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_14' in the type store
        # Getting the type of 'stypy_return_type' (line 242)
        stypy_return_type_40053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40053)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_14'
        return stypy_return_type_40053

    # Assigning a type to the variable '_stypy_temp_lambda_14' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), '_stypy_temp_lambda_14', _stypy_temp_lambda_14)
    # Getting the type of '_stypy_temp_lambda_14' (line 242)
    _stypy_temp_lambda_14_40054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), '_stypy_temp_lambda_14')
    # Getting the type of 'ya' (line 242)
    ya_40055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 44), 'ya', False)
    # Getting the type of 'yb' (line 242)
    yb_40056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 48), 'yb', False)
    # Getting the type of 'p' (line 242)
    p_40057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 52), 'p', False)
    # Processing the call keyword arguments (line 241)
    kwargs_40058 = {}
    # Getting the type of 'estimate_bc_jac' (line 241)
    estimate_bc_jac_40047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 31), 'estimate_bc_jac', False)
    # Calling estimate_bc_jac(args, kwargs) (line 241)
    estimate_bc_jac_call_result_40059 = invoke(stypy.reporting.localization.Localization(__file__, 241, 31), estimate_bc_jac_40047, *[_stypy_temp_lambda_14_40054, ya_40055, yb_40056, p_40057], **kwargs_40058)
    
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___40060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 4), estimate_bc_jac_call_result_40059, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_40061 = invoke(stypy.reporting.localization.Localization(__file__, 241, 4), getitem___40060, int_40046)
    
    # Assigning a type to the variable 'tuple_var_assignment_38819' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_38819', subscript_call_result_40061)
    
    # Assigning a Name to a Name (line 241):
    # Getting the type of 'tuple_var_assignment_38817' (line 241)
    tuple_var_assignment_38817_40062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_38817')
    # Assigning a type to the variable 'dbc_dya' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'dbc_dya', tuple_var_assignment_38817_40062)
    
    # Assigning a Name to a Name (line 241):
    # Getting the type of 'tuple_var_assignment_38818' (line 241)
    tuple_var_assignment_38818_40063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_38818')
    # Assigning a type to the variable 'dbc_dyb' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 13), 'dbc_dyb', tuple_var_assignment_38818_40063)
    
    # Assigning a Name to a Name (line 241):
    # Getting the type of 'tuple_var_assignment_38819' (line 241)
    tuple_var_assignment_38819_40064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_38819')
    # Assigning a type to the variable 'dbc_dp' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 22), 'dbc_dp', tuple_var_assignment_38819_40064)
    
    # Assigning a Call to a Tuple (line 243):
    
    # Assigning a Subscript to a Name (line 243):
    
    # Obtaining the type of the subscript
    int_40065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 4), 'int')
    
    # Call to emden_bc_jac(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'ya' (line 243)
    ya_40067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 42), 'ya', False)
    # Getting the type of 'yb' (line 243)
    yb_40068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 46), 'yb', False)
    # Processing the call keyword arguments (line 243)
    kwargs_40069 = {}
    # Getting the type of 'emden_bc_jac' (line 243)
    emden_bc_jac_40066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 29), 'emden_bc_jac', False)
    # Calling emden_bc_jac(args, kwargs) (line 243)
    emden_bc_jac_call_result_40070 = invoke(stypy.reporting.localization.Localization(__file__, 243, 29), emden_bc_jac_40066, *[ya_40067, yb_40068], **kwargs_40069)
    
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___40071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 4), emden_bc_jac_call_result_40070, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_40072 = invoke(stypy.reporting.localization.Localization(__file__, 243, 4), getitem___40071, int_40065)
    
    # Assigning a type to the variable 'tuple_var_assignment_38820' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'tuple_var_assignment_38820', subscript_call_result_40072)
    
    # Assigning a Subscript to a Name (line 243):
    
    # Obtaining the type of the subscript
    int_40073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 4), 'int')
    
    # Call to emden_bc_jac(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'ya' (line 243)
    ya_40075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 42), 'ya', False)
    # Getting the type of 'yb' (line 243)
    yb_40076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 46), 'yb', False)
    # Processing the call keyword arguments (line 243)
    kwargs_40077 = {}
    # Getting the type of 'emden_bc_jac' (line 243)
    emden_bc_jac_40074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 29), 'emden_bc_jac', False)
    # Calling emden_bc_jac(args, kwargs) (line 243)
    emden_bc_jac_call_result_40078 = invoke(stypy.reporting.localization.Localization(__file__, 243, 29), emden_bc_jac_40074, *[ya_40075, yb_40076], **kwargs_40077)
    
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___40079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 4), emden_bc_jac_call_result_40078, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_40080 = invoke(stypy.reporting.localization.Localization(__file__, 243, 4), getitem___40079, int_40073)
    
    # Assigning a type to the variable 'tuple_var_assignment_38821' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'tuple_var_assignment_38821', subscript_call_result_40080)
    
    # Assigning a Name to a Name (line 243):
    # Getting the type of 'tuple_var_assignment_38820' (line 243)
    tuple_var_assignment_38820_40081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'tuple_var_assignment_38820')
    # Assigning a type to the variable 'dbc_dya_an' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'dbc_dya_an', tuple_var_assignment_38820_40081)
    
    # Assigning a Name to a Name (line 243):
    # Getting the type of 'tuple_var_assignment_38821' (line 243)
    tuple_var_assignment_38821_40082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'tuple_var_assignment_38821')
    # Assigning a type to the variable 'dbc_dyb_an' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'dbc_dyb_an', tuple_var_assignment_38821_40082)
    
    # Call to assert_allclose(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'dbc_dya' (line 244)
    dbc_dya_40084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'dbc_dya', False)
    # Getting the type of 'dbc_dya_an' (line 244)
    dbc_dya_an_40085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 29), 'dbc_dya_an', False)
    # Processing the call keyword arguments (line 244)
    kwargs_40086 = {}
    # Getting the type of 'assert_allclose' (line 244)
    assert_allclose_40083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 244)
    assert_allclose_call_result_40087 = invoke(stypy.reporting.localization.Localization(__file__, 244, 4), assert_allclose_40083, *[dbc_dya_40084, dbc_dya_an_40085], **kwargs_40086)
    
    
    # Call to assert_allclose(...): (line 245)
    # Processing the call arguments (line 245)
    # Getting the type of 'dbc_dyb' (line 245)
    dbc_dyb_40089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'dbc_dyb', False)
    # Getting the type of 'dbc_dyb_an' (line 245)
    dbc_dyb_an_40090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 29), 'dbc_dyb_an', False)
    # Processing the call keyword arguments (line 245)
    kwargs_40091 = {}
    # Getting the type of 'assert_allclose' (line 245)
    assert_allclose_40088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 245)
    assert_allclose_call_result_40092 = invoke(stypy.reporting.localization.Localization(__file__, 245, 4), assert_allclose_40088, *[dbc_dyb_40089, dbc_dyb_an_40090], **kwargs_40091)
    
    
    # Call to assert_(...): (line 246)
    # Processing the call arguments (line 246)
    
    # Getting the type of 'dbc_dp' (line 246)
    dbc_dp_40094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'dbc_dp', False)
    # Getting the type of 'None' (line 246)
    None_40095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 22), 'None', False)
    # Applying the binary operator 'is' (line 246)
    result_is__40096 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 12), 'is', dbc_dp_40094, None_40095)
    
    # Processing the call keyword arguments (line 246)
    kwargs_40097 = {}
    # Getting the type of 'assert_' (line 246)
    assert__40093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 246)
    assert__call_result_40098 = invoke(stypy.reporting.localization.Localization(__file__, 246, 4), assert__40093, *[result_is__40096], **kwargs_40097)
    
    
    # ################# End of 'test_compute_bc_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_compute_bc_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 218)
    stypy_return_type_40099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_40099)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_compute_bc_jac'
    return stypy_return_type_40099

# Assigning a type to the variable 'test_compute_bc_jac' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'test_compute_bc_jac', test_compute_bc_jac)

@norecursion
def test_compute_jac_indices(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_compute_jac_indices'
    module_type_store = module_type_store.open_function_context('test_compute_jac_indices', 249, 0, False)
    
    # Passed parameters checking function
    test_compute_jac_indices.stypy_localization = localization
    test_compute_jac_indices.stypy_type_of_self = None
    test_compute_jac_indices.stypy_type_store = module_type_store
    test_compute_jac_indices.stypy_function_name = 'test_compute_jac_indices'
    test_compute_jac_indices.stypy_param_names_list = []
    test_compute_jac_indices.stypy_varargs_param_name = None
    test_compute_jac_indices.stypy_kwargs_param_name = None
    test_compute_jac_indices.stypy_call_defaults = defaults
    test_compute_jac_indices.stypy_call_varargs = varargs
    test_compute_jac_indices.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_compute_jac_indices', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_compute_jac_indices', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_compute_jac_indices(...)' code ##################

    
    # Assigning a Num to a Name (line 250):
    
    # Assigning a Num to a Name (line 250):
    int_40100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 8), 'int')
    # Assigning a type to the variable 'n' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'n', int_40100)
    
    # Assigning a Num to a Name (line 251):
    
    # Assigning a Num to a Name (line 251):
    int_40101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 8), 'int')
    # Assigning a type to the variable 'm' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'm', int_40101)
    
    # Assigning a Num to a Name (line 252):
    
    # Assigning a Num to a Name (line 252):
    int_40102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 8), 'int')
    # Assigning a type to the variable 'k' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'k', int_40102)
    
    # Assigning a Call to a Tuple (line 253):
    
    # Assigning a Subscript to a Name (line 253):
    
    # Obtaining the type of the subscript
    int_40103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 4), 'int')
    
    # Call to compute_jac_indices(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 'n' (line 253)
    n_40105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 31), 'n', False)
    # Getting the type of 'm' (line 253)
    m_40106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 34), 'm', False)
    # Getting the type of 'k' (line 253)
    k_40107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 37), 'k', False)
    # Processing the call keyword arguments (line 253)
    kwargs_40108 = {}
    # Getting the type of 'compute_jac_indices' (line 253)
    compute_jac_indices_40104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'compute_jac_indices', False)
    # Calling compute_jac_indices(args, kwargs) (line 253)
    compute_jac_indices_call_result_40109 = invoke(stypy.reporting.localization.Localization(__file__, 253, 11), compute_jac_indices_40104, *[n_40105, m_40106, k_40107], **kwargs_40108)
    
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___40110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 4), compute_jac_indices_call_result_40109, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_40111 = invoke(stypy.reporting.localization.Localization(__file__, 253, 4), getitem___40110, int_40103)
    
    # Assigning a type to the variable 'tuple_var_assignment_38822' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_var_assignment_38822', subscript_call_result_40111)
    
    # Assigning a Subscript to a Name (line 253):
    
    # Obtaining the type of the subscript
    int_40112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 4), 'int')
    
    # Call to compute_jac_indices(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 'n' (line 253)
    n_40114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 31), 'n', False)
    # Getting the type of 'm' (line 253)
    m_40115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 34), 'm', False)
    # Getting the type of 'k' (line 253)
    k_40116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 37), 'k', False)
    # Processing the call keyword arguments (line 253)
    kwargs_40117 = {}
    # Getting the type of 'compute_jac_indices' (line 253)
    compute_jac_indices_40113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'compute_jac_indices', False)
    # Calling compute_jac_indices(args, kwargs) (line 253)
    compute_jac_indices_call_result_40118 = invoke(stypy.reporting.localization.Localization(__file__, 253, 11), compute_jac_indices_40113, *[n_40114, m_40115, k_40116], **kwargs_40117)
    
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___40119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 4), compute_jac_indices_call_result_40118, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_40120 = invoke(stypy.reporting.localization.Localization(__file__, 253, 4), getitem___40119, int_40112)
    
    # Assigning a type to the variable 'tuple_var_assignment_38823' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_var_assignment_38823', subscript_call_result_40120)
    
    # Assigning a Name to a Name (line 253):
    # Getting the type of 'tuple_var_assignment_38822' (line 253)
    tuple_var_assignment_38822_40121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_var_assignment_38822')
    # Assigning a type to the variable 'i' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'i', tuple_var_assignment_38822_40121)
    
    # Assigning a Name to a Name (line 253):
    # Getting the type of 'tuple_var_assignment_38823' (line 253)
    tuple_var_assignment_38823_40122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_var_assignment_38823')
    # Assigning a type to the variable 'j' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 7), 'j', tuple_var_assignment_38823_40122)
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to toarray(...): (line 254)
    # Processing the call keyword arguments (line 254)
    kwargs_40136 = {}
    
    # Call to coo_matrix(...): (line 254)
    # Processing the call arguments (line 254)
    
    # Obtaining an instance of the builtin type 'tuple' (line 254)
    tuple_40124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 254)
    # Adding element type (line 254)
    
    # Call to ones_like(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'i' (line 254)
    i_40127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 33), 'i', False)
    # Processing the call keyword arguments (line 254)
    kwargs_40128 = {}
    # Getting the type of 'np' (line 254)
    np_40125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 254)
    ones_like_40126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 20), np_40125, 'ones_like')
    # Calling ones_like(args, kwargs) (line 254)
    ones_like_call_result_40129 = invoke(stypy.reporting.localization.Localization(__file__, 254, 20), ones_like_40126, *[i_40127], **kwargs_40128)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 20), tuple_40124, ones_like_call_result_40129)
    # Adding element type (line 254)
    
    # Obtaining an instance of the builtin type 'tuple' (line 254)
    tuple_40130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 254)
    # Adding element type (line 254)
    # Getting the type of 'i' (line 254)
    i_40131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 38), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 38), tuple_40130, i_40131)
    # Adding element type (line 254)
    # Getting the type of 'j' (line 254)
    j_40132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 41), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 38), tuple_40130, j_40132)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 20), tuple_40124, tuple_40130)
    
    # Processing the call keyword arguments (line 254)
    kwargs_40133 = {}
    # Getting the type of 'coo_matrix' (line 254)
    coo_matrix_40123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 254)
    coo_matrix_call_result_40134 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), coo_matrix_40123, *[tuple_40124], **kwargs_40133)
    
    # Obtaining the member 'toarray' of a type (line 254)
    toarray_40135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), coo_matrix_call_result_40134, 'toarray')
    # Calling toarray(args, kwargs) (line 254)
    toarray_call_result_40137 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), toarray_40135, *[], **kwargs_40136)
    
    # Assigning a type to the variable 's' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 's', toarray_call_result_40137)
    
    # Assigning a Call to a Name (line 255):
    
    # Assigning a Call to a Name (line 255):
    
    # Call to array(...): (line 255)
    # Processing the call arguments (line 255)
    
    # Obtaining an instance of the builtin type 'list' (line 255)
    list_40140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 255)
    # Adding element type (line 255)
    
    # Obtaining an instance of the builtin type 'list' (line 256)
    list_40141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 256)
    # Adding element type (line 256)
    int_40142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 8), list_40141, int_40142)
    # Adding element type (line 256)
    int_40143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 8), list_40141, int_40143)
    # Adding element type (line 256)
    int_40144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 8), list_40141, int_40144)
    # Adding element type (line 256)
    int_40145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 8), list_40141, int_40145)
    # Adding element type (line 256)
    int_40146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 8), list_40141, int_40146)
    # Adding element type (line 256)
    int_40147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 8), list_40141, int_40147)
    # Adding element type (line 256)
    int_40148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 8), list_40141, int_40148)
    # Adding element type (line 256)
    int_40149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 8), list_40141, int_40149)
    # Adding element type (line 256)
    int_40150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 8), list_40141, int_40150)
    # Adding element type (line 256)
    int_40151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 8), list_40141, int_40151)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 22), list_40140, list_40141)
    # Adding element type (line 255)
    
    # Obtaining an instance of the builtin type 'list' (line 257)
    list_40152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 257)
    # Adding element type (line 257)
    int_40153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), list_40152, int_40153)
    # Adding element type (line 257)
    int_40154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), list_40152, int_40154)
    # Adding element type (line 257)
    int_40155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), list_40152, int_40155)
    # Adding element type (line 257)
    int_40156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), list_40152, int_40156)
    # Adding element type (line 257)
    int_40157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), list_40152, int_40157)
    # Adding element type (line 257)
    int_40158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), list_40152, int_40158)
    # Adding element type (line 257)
    int_40159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), list_40152, int_40159)
    # Adding element type (line 257)
    int_40160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), list_40152, int_40160)
    # Adding element type (line 257)
    int_40161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), list_40152, int_40161)
    # Adding element type (line 257)
    int_40162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), list_40152, int_40162)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 22), list_40140, list_40152)
    # Adding element type (line 255)
    
    # Obtaining an instance of the builtin type 'list' (line 258)
    list_40163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 258)
    # Adding element type (line 258)
    int_40164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 8), list_40163, int_40164)
    # Adding element type (line 258)
    int_40165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 8), list_40163, int_40165)
    # Adding element type (line 258)
    int_40166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 8), list_40163, int_40166)
    # Adding element type (line 258)
    int_40167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 8), list_40163, int_40167)
    # Adding element type (line 258)
    int_40168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 8), list_40163, int_40168)
    # Adding element type (line 258)
    int_40169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 8), list_40163, int_40169)
    # Adding element type (line 258)
    int_40170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 8), list_40163, int_40170)
    # Adding element type (line 258)
    int_40171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 8), list_40163, int_40171)
    # Adding element type (line 258)
    int_40172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 8), list_40163, int_40172)
    # Adding element type (line 258)
    int_40173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 8), list_40163, int_40173)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 22), list_40140, list_40163)
    # Adding element type (line 255)
    
    # Obtaining an instance of the builtin type 'list' (line 259)
    list_40174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 259)
    # Adding element type (line 259)
    int_40175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 8), list_40174, int_40175)
    # Adding element type (line 259)
    int_40176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 8), list_40174, int_40176)
    # Adding element type (line 259)
    int_40177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 8), list_40174, int_40177)
    # Adding element type (line 259)
    int_40178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 8), list_40174, int_40178)
    # Adding element type (line 259)
    int_40179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 8), list_40174, int_40179)
    # Adding element type (line 259)
    int_40180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 8), list_40174, int_40180)
    # Adding element type (line 259)
    int_40181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 8), list_40174, int_40181)
    # Adding element type (line 259)
    int_40182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 8), list_40174, int_40182)
    # Adding element type (line 259)
    int_40183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 8), list_40174, int_40183)
    # Adding element type (line 259)
    int_40184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 8), list_40174, int_40184)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 22), list_40140, list_40174)
    # Adding element type (line 255)
    
    # Obtaining an instance of the builtin type 'list' (line 260)
    list_40185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 260)
    # Adding element type (line 260)
    int_40186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 8), list_40185, int_40186)
    # Adding element type (line 260)
    int_40187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 8), list_40185, int_40187)
    # Adding element type (line 260)
    int_40188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 8), list_40185, int_40188)
    # Adding element type (line 260)
    int_40189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 8), list_40185, int_40189)
    # Adding element type (line 260)
    int_40190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 8), list_40185, int_40190)
    # Adding element type (line 260)
    int_40191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 8), list_40185, int_40191)
    # Adding element type (line 260)
    int_40192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 8), list_40185, int_40192)
    # Adding element type (line 260)
    int_40193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 8), list_40185, int_40193)
    # Adding element type (line 260)
    int_40194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 8), list_40185, int_40194)
    # Adding element type (line 260)
    int_40195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 8), list_40185, int_40195)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 22), list_40140, list_40185)
    # Adding element type (line 255)
    
    # Obtaining an instance of the builtin type 'list' (line 261)
    list_40196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 261)
    # Adding element type (line 261)
    int_40197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 8), list_40196, int_40197)
    # Adding element type (line 261)
    int_40198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 8), list_40196, int_40198)
    # Adding element type (line 261)
    int_40199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 8), list_40196, int_40199)
    # Adding element type (line 261)
    int_40200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 8), list_40196, int_40200)
    # Adding element type (line 261)
    int_40201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 8), list_40196, int_40201)
    # Adding element type (line 261)
    int_40202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 8), list_40196, int_40202)
    # Adding element type (line 261)
    int_40203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 8), list_40196, int_40203)
    # Adding element type (line 261)
    int_40204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 8), list_40196, int_40204)
    # Adding element type (line 261)
    int_40205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 8), list_40196, int_40205)
    # Adding element type (line 261)
    int_40206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 8), list_40196, int_40206)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 22), list_40140, list_40196)
    # Adding element type (line 255)
    
    # Obtaining an instance of the builtin type 'list' (line 262)
    list_40207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 262)
    # Adding element type (line 262)
    int_40208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 8), list_40207, int_40208)
    # Adding element type (line 262)
    int_40209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 8), list_40207, int_40209)
    # Adding element type (line 262)
    int_40210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 8), list_40207, int_40210)
    # Adding element type (line 262)
    int_40211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 8), list_40207, int_40211)
    # Adding element type (line 262)
    int_40212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 8), list_40207, int_40212)
    # Adding element type (line 262)
    int_40213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 8), list_40207, int_40213)
    # Adding element type (line 262)
    int_40214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 8), list_40207, int_40214)
    # Adding element type (line 262)
    int_40215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 8), list_40207, int_40215)
    # Adding element type (line 262)
    int_40216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 8), list_40207, int_40216)
    # Adding element type (line 262)
    int_40217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 8), list_40207, int_40217)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 22), list_40140, list_40207)
    # Adding element type (line 255)
    
    # Obtaining an instance of the builtin type 'list' (line 263)
    list_40218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 263)
    # Adding element type (line 263)
    int_40219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), list_40218, int_40219)
    # Adding element type (line 263)
    int_40220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), list_40218, int_40220)
    # Adding element type (line 263)
    int_40221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), list_40218, int_40221)
    # Adding element type (line 263)
    int_40222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), list_40218, int_40222)
    # Adding element type (line 263)
    int_40223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), list_40218, int_40223)
    # Adding element type (line 263)
    int_40224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), list_40218, int_40224)
    # Adding element type (line 263)
    int_40225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), list_40218, int_40225)
    # Adding element type (line 263)
    int_40226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), list_40218, int_40226)
    # Adding element type (line 263)
    int_40227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), list_40218, int_40227)
    # Adding element type (line 263)
    int_40228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), list_40218, int_40228)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 22), list_40140, list_40218)
    # Adding element type (line 255)
    
    # Obtaining an instance of the builtin type 'list' (line 264)
    list_40229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 264)
    # Adding element type (line 264)
    int_40230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 8), list_40229, int_40230)
    # Adding element type (line 264)
    int_40231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 8), list_40229, int_40231)
    # Adding element type (line 264)
    int_40232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 8), list_40229, int_40232)
    # Adding element type (line 264)
    int_40233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 8), list_40229, int_40233)
    # Adding element type (line 264)
    int_40234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 8), list_40229, int_40234)
    # Adding element type (line 264)
    int_40235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 8), list_40229, int_40235)
    # Adding element type (line 264)
    int_40236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 8), list_40229, int_40236)
    # Adding element type (line 264)
    int_40237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 8), list_40229, int_40237)
    # Adding element type (line 264)
    int_40238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 8), list_40229, int_40238)
    # Adding element type (line 264)
    int_40239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 8), list_40229, int_40239)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 22), list_40140, list_40229)
    # Adding element type (line 255)
    
    # Obtaining an instance of the builtin type 'list' (line 265)
    list_40240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 265)
    # Adding element type (line 265)
    int_40241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 8), list_40240, int_40241)
    # Adding element type (line 265)
    int_40242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 8), list_40240, int_40242)
    # Adding element type (line 265)
    int_40243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 8), list_40240, int_40243)
    # Adding element type (line 265)
    int_40244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 8), list_40240, int_40244)
    # Adding element type (line 265)
    int_40245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 8), list_40240, int_40245)
    # Adding element type (line 265)
    int_40246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 8), list_40240, int_40246)
    # Adding element type (line 265)
    int_40247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 8), list_40240, int_40247)
    # Adding element type (line 265)
    int_40248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 8), list_40240, int_40248)
    # Adding element type (line 265)
    int_40249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 8), list_40240, int_40249)
    # Adding element type (line 265)
    int_40250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 8), list_40240, int_40250)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 22), list_40140, list_40240)
    
    # Processing the call keyword arguments (line 255)
    kwargs_40251 = {}
    # Getting the type of 'np' (line 255)
    np_40138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 255)
    array_40139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 13), np_40138, 'array')
    # Calling array(args, kwargs) (line 255)
    array_call_result_40252 = invoke(stypy.reporting.localization.Localization(__file__, 255, 13), array_40139, *[list_40140], **kwargs_40251)
    
    # Assigning a type to the variable 's_true' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 's_true', array_call_result_40252)
    
    # Call to assert_array_equal(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 's' (line 267)
    s_40254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 's', False)
    # Getting the type of 's_true' (line 267)
    s_true_40255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 26), 's_true', False)
    # Processing the call keyword arguments (line 267)
    kwargs_40256 = {}
    # Getting the type of 'assert_array_equal' (line 267)
    assert_array_equal_40253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 267)
    assert_array_equal_call_result_40257 = invoke(stypy.reporting.localization.Localization(__file__, 267, 4), assert_array_equal_40253, *[s_40254, s_true_40255], **kwargs_40256)
    
    
    # ################# End of 'test_compute_jac_indices(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_compute_jac_indices' in the type store
    # Getting the type of 'stypy_return_type' (line 249)
    stypy_return_type_40258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_40258)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_compute_jac_indices'
    return stypy_return_type_40258

# Assigning a type to the variable 'test_compute_jac_indices' (line 249)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'test_compute_jac_indices', test_compute_jac_indices)

@norecursion
def test_compute_global_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_compute_global_jac'
    module_type_store = module_type_store.open_function_context('test_compute_global_jac', 270, 0, False)
    
    # Passed parameters checking function
    test_compute_global_jac.stypy_localization = localization
    test_compute_global_jac.stypy_type_of_self = None
    test_compute_global_jac.stypy_type_store = module_type_store
    test_compute_global_jac.stypy_function_name = 'test_compute_global_jac'
    test_compute_global_jac.stypy_param_names_list = []
    test_compute_global_jac.stypy_varargs_param_name = None
    test_compute_global_jac.stypy_kwargs_param_name = None
    test_compute_global_jac.stypy_call_defaults = defaults
    test_compute_global_jac.stypy_call_varargs = varargs
    test_compute_global_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_compute_global_jac', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_compute_global_jac', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_compute_global_jac(...)' code ##################

    
    # Assigning a Num to a Name (line 271):
    
    # Assigning a Num to a Name (line 271):
    int_40259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 8), 'int')
    # Assigning a type to the variable 'n' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'n', int_40259)
    
    # Assigning a Num to a Name (line 272):
    
    # Assigning a Num to a Name (line 272):
    int_40260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 8), 'int')
    # Assigning a type to the variable 'm' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'm', int_40260)
    
    # Assigning a Num to a Name (line 273):
    
    # Assigning a Num to a Name (line 273):
    int_40261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 8), 'int')
    # Assigning a type to the variable 'k' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'k', int_40261)
    
    # Assigning a Call to a Tuple (line 274):
    
    # Assigning a Subscript to a Name (line 274):
    
    # Obtaining the type of the subscript
    int_40262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 4), 'int')
    
    # Call to compute_jac_indices(...): (line 274)
    # Processing the call arguments (line 274)
    int_40264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 39), 'int')
    int_40265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 42), 'int')
    int_40266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 45), 'int')
    # Processing the call keyword arguments (line 274)
    kwargs_40267 = {}
    # Getting the type of 'compute_jac_indices' (line 274)
    compute_jac_indices_40263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 'compute_jac_indices', False)
    # Calling compute_jac_indices(args, kwargs) (line 274)
    compute_jac_indices_call_result_40268 = invoke(stypy.reporting.localization.Localization(__file__, 274, 19), compute_jac_indices_40263, *[int_40264, int_40265, int_40266], **kwargs_40267)
    
    # Obtaining the member '__getitem__' of a type (line 274)
    getitem___40269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 4), compute_jac_indices_call_result_40268, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 274)
    subscript_call_result_40270 = invoke(stypy.reporting.localization.Localization(__file__, 274, 4), getitem___40269, int_40262)
    
    # Assigning a type to the variable 'tuple_var_assignment_38824' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'tuple_var_assignment_38824', subscript_call_result_40270)
    
    # Assigning a Subscript to a Name (line 274):
    
    # Obtaining the type of the subscript
    int_40271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 4), 'int')
    
    # Call to compute_jac_indices(...): (line 274)
    # Processing the call arguments (line 274)
    int_40273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 39), 'int')
    int_40274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 42), 'int')
    int_40275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 45), 'int')
    # Processing the call keyword arguments (line 274)
    kwargs_40276 = {}
    # Getting the type of 'compute_jac_indices' (line 274)
    compute_jac_indices_40272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 'compute_jac_indices', False)
    # Calling compute_jac_indices(args, kwargs) (line 274)
    compute_jac_indices_call_result_40277 = invoke(stypy.reporting.localization.Localization(__file__, 274, 19), compute_jac_indices_40272, *[int_40273, int_40274, int_40275], **kwargs_40276)
    
    # Obtaining the member '__getitem__' of a type (line 274)
    getitem___40278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 4), compute_jac_indices_call_result_40277, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 274)
    subscript_call_result_40279 = invoke(stypy.reporting.localization.Localization(__file__, 274, 4), getitem___40278, int_40271)
    
    # Assigning a type to the variable 'tuple_var_assignment_38825' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'tuple_var_assignment_38825', subscript_call_result_40279)
    
    # Assigning a Name to a Name (line 274):
    # Getting the type of 'tuple_var_assignment_38824' (line 274)
    tuple_var_assignment_38824_40280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'tuple_var_assignment_38824')
    # Assigning a type to the variable 'i_jac' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'i_jac', tuple_var_assignment_38824_40280)
    
    # Assigning a Name to a Name (line 274):
    # Getting the type of 'tuple_var_assignment_38825' (line 274)
    tuple_var_assignment_38825_40281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'tuple_var_assignment_38825')
    # Assigning a type to the variable 'j_jac' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 11), 'j_jac', tuple_var_assignment_38825_40281)
    
    # Assigning a Call to a Name (line 275):
    
    # Assigning a Call to a Name (line 275):
    
    # Call to linspace(...): (line 275)
    # Processing the call arguments (line 275)
    int_40284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 20), 'int')
    int_40285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 23), 'int')
    int_40286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 26), 'int')
    # Processing the call keyword arguments (line 275)
    kwargs_40287 = {}
    # Getting the type of 'np' (line 275)
    np_40282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 275)
    linspace_40283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), np_40282, 'linspace')
    # Calling linspace(args, kwargs) (line 275)
    linspace_call_result_40288 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), linspace_40283, *[int_40284, int_40285, int_40286], **kwargs_40287)
    
    # Assigning a type to the variable 'x' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'x', linspace_call_result_40288)
    
    # Assigning a Call to a Name (line 276):
    
    # Assigning a Call to a Name (line 276):
    
    # Call to diff(...): (line 276)
    # Processing the call arguments (line 276)
    # Getting the type of 'x' (line 276)
    x_40291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'x', False)
    # Processing the call keyword arguments (line 276)
    kwargs_40292 = {}
    # Getting the type of 'np' (line 276)
    np_40289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'np', False)
    # Obtaining the member 'diff' of a type (line 276)
    diff_40290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), np_40289, 'diff')
    # Calling diff(args, kwargs) (line 276)
    diff_call_result_40293 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), diff_40290, *[x_40291], **kwargs_40292)
    
    # Assigning a type to the variable 'h' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'h', diff_call_result_40293)
    
    # Assigning a Call to a Name (line 277):
    
    # Assigning a Call to a Name (line 277):
    
    # Call to vstack(...): (line 277)
    # Processing the call arguments (line 277)
    
    # Obtaining an instance of the builtin type 'tuple' (line 277)
    tuple_40296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 277)
    # Adding element type (line 277)
    
    # Call to sin(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'np' (line 277)
    np_40299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 26), 'np', False)
    # Obtaining the member 'pi' of a type (line 277)
    pi_40300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 26), np_40299, 'pi')
    # Getting the type of 'x' (line 277)
    x_40301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 34), 'x', False)
    # Applying the binary operator '*' (line 277)
    result_mul_40302 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 26), '*', pi_40300, x_40301)
    
    # Processing the call keyword arguments (line 277)
    kwargs_40303 = {}
    # Getting the type of 'np' (line 277)
    np_40297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 19), 'np', False)
    # Obtaining the member 'sin' of a type (line 277)
    sin_40298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 19), np_40297, 'sin')
    # Calling sin(args, kwargs) (line 277)
    sin_call_result_40304 = invoke(stypy.reporting.localization.Localization(__file__, 277, 19), sin_40298, *[result_mul_40302], **kwargs_40303)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 19), tuple_40296, sin_call_result_40304)
    # Adding element type (line 277)
    # Getting the type of 'np' (line 277)
    np_40305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 38), 'np', False)
    # Obtaining the member 'pi' of a type (line 277)
    pi_40306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 38), np_40305, 'pi')
    
    # Call to cos(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'np' (line 277)
    np_40309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 53), 'np', False)
    # Obtaining the member 'pi' of a type (line 277)
    pi_40310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 53), np_40309, 'pi')
    # Getting the type of 'x' (line 277)
    x_40311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 61), 'x', False)
    # Applying the binary operator '*' (line 277)
    result_mul_40312 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 53), '*', pi_40310, x_40311)
    
    # Processing the call keyword arguments (line 277)
    kwargs_40313 = {}
    # Getting the type of 'np' (line 277)
    np_40307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 46), 'np', False)
    # Obtaining the member 'cos' of a type (line 277)
    cos_40308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 46), np_40307, 'cos')
    # Calling cos(args, kwargs) (line 277)
    cos_call_result_40314 = invoke(stypy.reporting.localization.Localization(__file__, 277, 46), cos_40308, *[result_mul_40312], **kwargs_40313)
    
    # Applying the binary operator '*' (line 277)
    result_mul_40315 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 38), '*', pi_40306, cos_call_result_40314)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 19), tuple_40296, result_mul_40315)
    
    # Processing the call keyword arguments (line 277)
    kwargs_40316 = {}
    # Getting the type of 'np' (line 277)
    np_40294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'np', False)
    # Obtaining the member 'vstack' of a type (line 277)
    vstack_40295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 8), np_40294, 'vstack')
    # Calling vstack(args, kwargs) (line 277)
    vstack_call_result_40317 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), vstack_40295, *[tuple_40296], **kwargs_40316)
    
    # Assigning a type to the variable 'y' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'y', vstack_call_result_40317)
    
    # Assigning a Call to a Name (line 278):
    
    # Assigning a Call to a Name (line 278):
    
    # Call to array(...): (line 278)
    # Processing the call arguments (line 278)
    
    # Obtaining an instance of the builtin type 'list' (line 278)
    list_40320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 278)
    # Adding element type (line 278)
    float_40321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 17), list_40320, float_40321)
    
    # Processing the call keyword arguments (line 278)
    kwargs_40322 = {}
    # Getting the type of 'np' (line 278)
    np_40318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 278)
    array_40319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), np_40318, 'array')
    # Calling array(args, kwargs) (line 278)
    array_call_result_40323 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), array_40319, *[list_40320], **kwargs_40322)
    
    # Assigning a type to the variable 'p' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'p', array_call_result_40323)
    
    # Assigning a Call to a Name (line 280):
    
    # Assigning a Call to a Name (line 280):
    
    # Call to sl_fun(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'x' (line 280)
    x_40325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'x', False)
    # Getting the type of 'y' (line 280)
    y_40326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 18), 'y', False)
    # Getting the type of 'p' (line 280)
    p_40327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 21), 'p', False)
    # Processing the call keyword arguments (line 280)
    kwargs_40328 = {}
    # Getting the type of 'sl_fun' (line 280)
    sl_fun_40324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'sl_fun', False)
    # Calling sl_fun(args, kwargs) (line 280)
    sl_fun_call_result_40329 = invoke(stypy.reporting.localization.Localization(__file__, 280, 8), sl_fun_40324, *[x_40325, y_40326, p_40327], **kwargs_40328)
    
    # Assigning a type to the variable 'f' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'f', sl_fun_call_result_40329)
    
    # Assigning a BinOp to a Name (line 282):
    
    # Assigning a BinOp to a Name (line 282):
    
    # Obtaining the type of the subscript
    int_40330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 18), 'int')
    slice_40331 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 282, 15), None, int_40330, None)
    # Getting the type of 'x' (line 282)
    x_40332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'x')
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___40333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 15), x_40332, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_40334 = invoke(stypy.reporting.localization.Localization(__file__, 282, 15), getitem___40333, slice_40331)
    
    float_40335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 24), 'float')
    # Getting the type of 'h' (line 282)
    h_40336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 30), 'h')
    # Applying the binary operator '*' (line 282)
    result_mul_40337 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 24), '*', float_40335, h_40336)
    
    # Applying the binary operator '+' (line 282)
    result_add_40338 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 15), '+', subscript_call_result_40334, result_mul_40337)
    
    # Assigning a type to the variable 'x_middle' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'x_middle', result_add_40338)
    
    # Assigning a BinOp to a Name (line 283):
    
    # Assigning a BinOp to a Name (line 283):
    float_40339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 15), 'float')
    
    # Obtaining the type of the subscript
    slice_40340 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 283, 22), None, None, None)
    int_40341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 28), 'int')
    slice_40342 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 283, 22), None, int_40341, None)
    # Getting the type of 'y' (line 283)
    y_40343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 22), 'y')
    # Obtaining the member '__getitem__' of a type (line 283)
    getitem___40344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 22), y_40343, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 283)
    subscript_call_result_40345 = invoke(stypy.reporting.localization.Localization(__file__, 283, 22), getitem___40344, (slice_40340, slice_40342))
    
    
    # Obtaining the type of the subscript
    slice_40346 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 283, 34), None, None, None)
    int_40347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 39), 'int')
    slice_40348 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 283, 34), int_40347, None, None)
    # Getting the type of 'y' (line 283)
    y_40349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 34), 'y')
    # Obtaining the member '__getitem__' of a type (line 283)
    getitem___40350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 34), y_40349, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 283)
    subscript_call_result_40351 = invoke(stypy.reporting.localization.Localization(__file__, 283, 34), getitem___40350, (slice_40346, slice_40348))
    
    # Applying the binary operator '+' (line 283)
    result_add_40352 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 22), '+', subscript_call_result_40345, subscript_call_result_40351)
    
    # Applying the binary operator '*' (line 283)
    result_mul_40353 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 15), '*', float_40339, result_add_40352)
    
    # Getting the type of 'h' (line 283)
    h_40354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 46), 'h')
    int_40355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 48), 'int')
    # Applying the binary operator 'div' (line 283)
    result_div_40356 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 46), 'div', h_40354, int_40355)
    
    
    # Obtaining the type of the subscript
    slice_40357 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 283, 53), None, None, None)
    int_40358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 58), 'int')
    slice_40359 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 283, 53), int_40358, None, None)
    # Getting the type of 'f' (line 283)
    f_40360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 53), 'f')
    # Obtaining the member '__getitem__' of a type (line 283)
    getitem___40361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 53), f_40360, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 283)
    subscript_call_result_40362 = invoke(stypy.reporting.localization.Localization(__file__, 283, 53), getitem___40361, (slice_40357, slice_40359))
    
    
    # Obtaining the type of the subscript
    slice_40363 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 283, 64), None, None, None)
    int_40364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 70), 'int')
    slice_40365 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 283, 64), None, int_40364, None)
    # Getting the type of 'f' (line 283)
    f_40366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 64), 'f')
    # Obtaining the member '__getitem__' of a type (line 283)
    getitem___40367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 64), f_40366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 283)
    subscript_call_result_40368 = invoke(stypy.reporting.localization.Localization(__file__, 283, 64), getitem___40367, (slice_40363, slice_40365))
    
    # Applying the binary operator '-' (line 283)
    result_sub_40369 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 53), '-', subscript_call_result_40362, subscript_call_result_40368)
    
    # Applying the binary operator '*' (line 283)
    result_mul_40370 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 50), '*', result_div_40356, result_sub_40369)
    
    # Applying the binary operator '-' (line 283)
    result_sub_40371 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 15), '-', result_mul_40353, result_mul_40370)
    
    # Assigning a type to the variable 'y_middle' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'y_middle', result_sub_40371)
    
    # Assigning a Call to a Tuple (line 285):
    
    # Assigning a Subscript to a Name (line 285):
    
    # Obtaining the type of the subscript
    int_40372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 4), 'int')
    
    # Call to sl_fun_jac(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'x' (line 285)
    x_40374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 30), 'x', False)
    # Getting the type of 'y' (line 285)
    y_40375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 33), 'y', False)
    # Getting the type of 'p' (line 285)
    p_40376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 36), 'p', False)
    # Processing the call keyword arguments (line 285)
    kwargs_40377 = {}
    # Getting the type of 'sl_fun_jac' (line 285)
    sl_fun_jac_40373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 19), 'sl_fun_jac', False)
    # Calling sl_fun_jac(args, kwargs) (line 285)
    sl_fun_jac_call_result_40378 = invoke(stypy.reporting.localization.Localization(__file__, 285, 19), sl_fun_jac_40373, *[x_40374, y_40375, p_40376], **kwargs_40377)
    
    # Obtaining the member '__getitem__' of a type (line 285)
    getitem___40379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 4), sl_fun_jac_call_result_40378, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 285)
    subscript_call_result_40380 = invoke(stypy.reporting.localization.Localization(__file__, 285, 4), getitem___40379, int_40372)
    
    # Assigning a type to the variable 'tuple_var_assignment_38826' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'tuple_var_assignment_38826', subscript_call_result_40380)
    
    # Assigning a Subscript to a Name (line 285):
    
    # Obtaining the type of the subscript
    int_40381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 4), 'int')
    
    # Call to sl_fun_jac(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'x' (line 285)
    x_40383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 30), 'x', False)
    # Getting the type of 'y' (line 285)
    y_40384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 33), 'y', False)
    # Getting the type of 'p' (line 285)
    p_40385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 36), 'p', False)
    # Processing the call keyword arguments (line 285)
    kwargs_40386 = {}
    # Getting the type of 'sl_fun_jac' (line 285)
    sl_fun_jac_40382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 19), 'sl_fun_jac', False)
    # Calling sl_fun_jac(args, kwargs) (line 285)
    sl_fun_jac_call_result_40387 = invoke(stypy.reporting.localization.Localization(__file__, 285, 19), sl_fun_jac_40382, *[x_40383, y_40384, p_40385], **kwargs_40386)
    
    # Obtaining the member '__getitem__' of a type (line 285)
    getitem___40388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 4), sl_fun_jac_call_result_40387, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 285)
    subscript_call_result_40389 = invoke(stypy.reporting.localization.Localization(__file__, 285, 4), getitem___40388, int_40381)
    
    # Assigning a type to the variable 'tuple_var_assignment_38827' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'tuple_var_assignment_38827', subscript_call_result_40389)
    
    # Assigning a Name to a Name (line 285):
    # Getting the type of 'tuple_var_assignment_38826' (line 285)
    tuple_var_assignment_38826_40390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'tuple_var_assignment_38826')
    # Assigning a type to the variable 'df_dy' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'df_dy', tuple_var_assignment_38826_40390)
    
    # Assigning a Name to a Name (line 285):
    # Getting the type of 'tuple_var_assignment_38827' (line 285)
    tuple_var_assignment_38827_40391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'tuple_var_assignment_38827')
    # Assigning a type to the variable 'df_dp' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'df_dp', tuple_var_assignment_38827_40391)
    
    # Assigning a Call to a Tuple (line 286):
    
    # Assigning a Subscript to a Name (line 286):
    
    # Obtaining the type of the subscript
    int_40392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 4), 'int')
    
    # Call to sl_fun_jac(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'x_middle' (line 286)
    x_middle_40394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 44), 'x_middle', False)
    # Getting the type of 'y_middle' (line 286)
    y_middle_40395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 54), 'y_middle', False)
    # Getting the type of 'p' (line 286)
    p_40396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 64), 'p', False)
    # Processing the call keyword arguments (line 286)
    kwargs_40397 = {}
    # Getting the type of 'sl_fun_jac' (line 286)
    sl_fun_jac_40393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 33), 'sl_fun_jac', False)
    # Calling sl_fun_jac(args, kwargs) (line 286)
    sl_fun_jac_call_result_40398 = invoke(stypy.reporting.localization.Localization(__file__, 286, 33), sl_fun_jac_40393, *[x_middle_40394, y_middle_40395, p_40396], **kwargs_40397)
    
    # Obtaining the member '__getitem__' of a type (line 286)
    getitem___40399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 4), sl_fun_jac_call_result_40398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 286)
    subscript_call_result_40400 = invoke(stypy.reporting.localization.Localization(__file__, 286, 4), getitem___40399, int_40392)
    
    # Assigning a type to the variable 'tuple_var_assignment_38828' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'tuple_var_assignment_38828', subscript_call_result_40400)
    
    # Assigning a Subscript to a Name (line 286):
    
    # Obtaining the type of the subscript
    int_40401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 4), 'int')
    
    # Call to sl_fun_jac(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'x_middle' (line 286)
    x_middle_40403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 44), 'x_middle', False)
    # Getting the type of 'y_middle' (line 286)
    y_middle_40404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 54), 'y_middle', False)
    # Getting the type of 'p' (line 286)
    p_40405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 64), 'p', False)
    # Processing the call keyword arguments (line 286)
    kwargs_40406 = {}
    # Getting the type of 'sl_fun_jac' (line 286)
    sl_fun_jac_40402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 33), 'sl_fun_jac', False)
    # Calling sl_fun_jac(args, kwargs) (line 286)
    sl_fun_jac_call_result_40407 = invoke(stypy.reporting.localization.Localization(__file__, 286, 33), sl_fun_jac_40402, *[x_middle_40403, y_middle_40404, p_40405], **kwargs_40406)
    
    # Obtaining the member '__getitem__' of a type (line 286)
    getitem___40408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 4), sl_fun_jac_call_result_40407, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 286)
    subscript_call_result_40409 = invoke(stypy.reporting.localization.Localization(__file__, 286, 4), getitem___40408, int_40401)
    
    # Assigning a type to the variable 'tuple_var_assignment_38829' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'tuple_var_assignment_38829', subscript_call_result_40409)
    
    # Assigning a Name to a Name (line 286):
    # Getting the type of 'tuple_var_assignment_38828' (line 286)
    tuple_var_assignment_38828_40410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'tuple_var_assignment_38828')
    # Assigning a type to the variable 'df_dy_middle' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'df_dy_middle', tuple_var_assignment_38828_40410)
    
    # Assigning a Name to a Name (line 286):
    # Getting the type of 'tuple_var_assignment_38829' (line 286)
    tuple_var_assignment_38829_40411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'tuple_var_assignment_38829')
    # Assigning a type to the variable 'df_dp_middle' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 18), 'df_dp_middle', tuple_var_assignment_38829_40411)
    
    # Assigning a Call to a Tuple (line 287):
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_40412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 4), 'int')
    
    # Call to sl_bc_jac(...): (line 287)
    # Processing the call arguments (line 287)
    
    # Obtaining the type of the subscript
    slice_40414 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 287, 41), None, None, None)
    int_40415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 46), 'int')
    # Getting the type of 'y' (line 287)
    y_40416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 41), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___40417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 41), y_40416, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_40418 = invoke(stypy.reporting.localization.Localization(__file__, 287, 41), getitem___40417, (slice_40414, int_40415))
    
    
    # Obtaining the type of the subscript
    slice_40419 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 287, 50), None, None, None)
    int_40420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 55), 'int')
    # Getting the type of 'y' (line 287)
    y_40421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 50), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___40422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 50), y_40421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_40423 = invoke(stypy.reporting.localization.Localization(__file__, 287, 50), getitem___40422, (slice_40419, int_40420))
    
    # Getting the type of 'p' (line 287)
    p_40424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 60), 'p', False)
    # Processing the call keyword arguments (line 287)
    kwargs_40425 = {}
    # Getting the type of 'sl_bc_jac' (line 287)
    sl_bc_jac_40413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'sl_bc_jac', False)
    # Calling sl_bc_jac(args, kwargs) (line 287)
    sl_bc_jac_call_result_40426 = invoke(stypy.reporting.localization.Localization(__file__, 287, 31), sl_bc_jac_40413, *[subscript_call_result_40418, subscript_call_result_40423, p_40424], **kwargs_40425)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___40427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 4), sl_bc_jac_call_result_40426, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_40428 = invoke(stypy.reporting.localization.Localization(__file__, 287, 4), getitem___40427, int_40412)
    
    # Assigning a type to the variable 'tuple_var_assignment_38830' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_38830', subscript_call_result_40428)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_40429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 4), 'int')
    
    # Call to sl_bc_jac(...): (line 287)
    # Processing the call arguments (line 287)
    
    # Obtaining the type of the subscript
    slice_40431 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 287, 41), None, None, None)
    int_40432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 46), 'int')
    # Getting the type of 'y' (line 287)
    y_40433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 41), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___40434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 41), y_40433, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_40435 = invoke(stypy.reporting.localization.Localization(__file__, 287, 41), getitem___40434, (slice_40431, int_40432))
    
    
    # Obtaining the type of the subscript
    slice_40436 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 287, 50), None, None, None)
    int_40437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 55), 'int')
    # Getting the type of 'y' (line 287)
    y_40438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 50), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___40439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 50), y_40438, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_40440 = invoke(stypy.reporting.localization.Localization(__file__, 287, 50), getitem___40439, (slice_40436, int_40437))
    
    # Getting the type of 'p' (line 287)
    p_40441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 60), 'p', False)
    # Processing the call keyword arguments (line 287)
    kwargs_40442 = {}
    # Getting the type of 'sl_bc_jac' (line 287)
    sl_bc_jac_40430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'sl_bc_jac', False)
    # Calling sl_bc_jac(args, kwargs) (line 287)
    sl_bc_jac_call_result_40443 = invoke(stypy.reporting.localization.Localization(__file__, 287, 31), sl_bc_jac_40430, *[subscript_call_result_40435, subscript_call_result_40440, p_40441], **kwargs_40442)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___40444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 4), sl_bc_jac_call_result_40443, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_40445 = invoke(stypy.reporting.localization.Localization(__file__, 287, 4), getitem___40444, int_40429)
    
    # Assigning a type to the variable 'tuple_var_assignment_38831' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_38831', subscript_call_result_40445)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_40446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 4), 'int')
    
    # Call to sl_bc_jac(...): (line 287)
    # Processing the call arguments (line 287)
    
    # Obtaining the type of the subscript
    slice_40448 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 287, 41), None, None, None)
    int_40449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 46), 'int')
    # Getting the type of 'y' (line 287)
    y_40450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 41), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___40451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 41), y_40450, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_40452 = invoke(stypy.reporting.localization.Localization(__file__, 287, 41), getitem___40451, (slice_40448, int_40449))
    
    
    # Obtaining the type of the subscript
    slice_40453 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 287, 50), None, None, None)
    int_40454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 55), 'int')
    # Getting the type of 'y' (line 287)
    y_40455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 50), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___40456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 50), y_40455, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_40457 = invoke(stypy.reporting.localization.Localization(__file__, 287, 50), getitem___40456, (slice_40453, int_40454))
    
    # Getting the type of 'p' (line 287)
    p_40458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 60), 'p', False)
    # Processing the call keyword arguments (line 287)
    kwargs_40459 = {}
    # Getting the type of 'sl_bc_jac' (line 287)
    sl_bc_jac_40447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'sl_bc_jac', False)
    # Calling sl_bc_jac(args, kwargs) (line 287)
    sl_bc_jac_call_result_40460 = invoke(stypy.reporting.localization.Localization(__file__, 287, 31), sl_bc_jac_40447, *[subscript_call_result_40452, subscript_call_result_40457, p_40458], **kwargs_40459)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___40461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 4), sl_bc_jac_call_result_40460, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_40462 = invoke(stypy.reporting.localization.Localization(__file__, 287, 4), getitem___40461, int_40446)
    
    # Assigning a type to the variable 'tuple_var_assignment_38832' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_38832', subscript_call_result_40462)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_38830' (line 287)
    tuple_var_assignment_38830_40463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_38830')
    # Assigning a type to the variable 'dbc_dya' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'dbc_dya', tuple_var_assignment_38830_40463)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_38831' (line 287)
    tuple_var_assignment_38831_40464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_38831')
    # Assigning a type to the variable 'dbc_dyb' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 13), 'dbc_dyb', tuple_var_assignment_38831_40464)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_38832' (line 287)
    tuple_var_assignment_38832_40465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_38832')
    # Assigning a type to the variable 'dbc_dp' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 22), 'dbc_dp', tuple_var_assignment_38832_40465)
    
    # Assigning a Call to a Name (line 289):
    
    # Assigning a Call to a Name (line 289):
    
    # Call to construct_global_jac(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'n' (line 289)
    n_40467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 29), 'n', False)
    # Getting the type of 'm' (line 289)
    m_40468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 32), 'm', False)
    # Getting the type of 'k' (line 289)
    k_40469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 35), 'k', False)
    # Getting the type of 'i_jac' (line 289)
    i_jac_40470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 38), 'i_jac', False)
    # Getting the type of 'j_jac' (line 289)
    j_jac_40471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 45), 'j_jac', False)
    # Getting the type of 'h' (line 289)
    h_40472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 52), 'h', False)
    # Getting the type of 'df_dy' (line 289)
    df_dy_40473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 55), 'df_dy', False)
    # Getting the type of 'df_dy_middle' (line 289)
    df_dy_middle_40474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 62), 'df_dy_middle', False)
    # Getting the type of 'df_dp' (line 290)
    df_dp_40475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 29), 'df_dp', False)
    # Getting the type of 'df_dp_middle' (line 290)
    df_dp_middle_40476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 36), 'df_dp_middle', False)
    # Getting the type of 'dbc_dya' (line 290)
    dbc_dya_40477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 50), 'dbc_dya', False)
    # Getting the type of 'dbc_dyb' (line 290)
    dbc_dyb_40478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 59), 'dbc_dyb', False)
    # Getting the type of 'dbc_dp' (line 290)
    dbc_dp_40479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 68), 'dbc_dp', False)
    # Processing the call keyword arguments (line 289)
    kwargs_40480 = {}
    # Getting the type of 'construct_global_jac' (line 289)
    construct_global_jac_40466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'construct_global_jac', False)
    # Calling construct_global_jac(args, kwargs) (line 289)
    construct_global_jac_call_result_40481 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), construct_global_jac_40466, *[n_40467, m_40468, k_40469, i_jac_40470, j_jac_40471, h_40472, df_dy_40473, df_dy_middle_40474, df_dp_40475, df_dp_middle_40476, dbc_dya_40477, dbc_dyb_40478, dbc_dp_40479], **kwargs_40480)
    
    # Assigning a type to the variable 'J' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'J', construct_global_jac_call_result_40481)
    
    # Assigning a Call to a Name (line 291):
    
    # Assigning a Call to a Name (line 291):
    
    # Call to toarray(...): (line 291)
    # Processing the call keyword arguments (line 291)
    kwargs_40484 = {}
    # Getting the type of 'J' (line 291)
    J_40482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'J', False)
    # Obtaining the member 'toarray' of a type (line 291)
    toarray_40483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), J_40482, 'toarray')
    # Calling toarray(args, kwargs) (line 291)
    toarray_call_result_40485 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), toarray_40483, *[], **kwargs_40484)
    
    # Assigning a type to the variable 'J' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'J', toarray_call_result_40485)

    @norecursion
    def J_block(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'J_block'
        module_type_store = module_type_store.open_function_context('J_block', 293, 4, False)
        
        # Passed parameters checking function
        J_block.stypy_localization = localization
        J_block.stypy_type_of_self = None
        J_block.stypy_type_store = module_type_store
        J_block.stypy_function_name = 'J_block'
        J_block.stypy_param_names_list = ['h', 'p']
        J_block.stypy_varargs_param_name = None
        J_block.stypy_kwargs_param_name = None
        J_block.stypy_call_defaults = defaults
        J_block.stypy_call_varargs = varargs
        J_block.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'J_block', ['h', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'J_block', localization, ['h', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'J_block(...)' code ##################

        
        # Call to array(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Obtaining an instance of the builtin type 'list' (line 294)
        list_40488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 294)
        # Adding element type (line 294)
        
        # Obtaining an instance of the builtin type 'list' (line 295)
        list_40489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 295)
        # Adding element type (line 295)
        # Getting the type of 'h' (line 295)
        h_40490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 13), 'h', False)
        int_40491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 16), 'int')
        # Applying the binary operator '**' (line 295)
        result_pow_40492 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 13), '**', h_40490, int_40491)
        
        # Getting the type of 'p' (line 295)
        p_40493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 18), 'p', False)
        int_40494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 21), 'int')
        # Applying the binary operator '**' (line 295)
        result_pow_40495 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 18), '**', p_40493, int_40494)
        
        # Applying the binary operator '*' (line 295)
        result_mul_40496 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 13), '*', result_pow_40492, result_pow_40495)
        
        int_40497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 23), 'int')
        # Applying the binary operator 'div' (line 295)
        result_div_40498 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 22), 'div', result_mul_40496, int_40497)
        
        int_40499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 28), 'int')
        # Applying the binary operator '-' (line 295)
        result_sub_40500 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 13), '-', result_div_40498, int_40499)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 12), list_40489, result_sub_40500)
        # Adding element type (line 295)
        float_40501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 31), 'float')
        # Getting the type of 'h' (line 295)
        h_40502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 36), 'h', False)
        # Applying the binary operator '*' (line 295)
        result_mul_40503 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 31), '*', float_40501, h_40502)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 12), list_40489, result_mul_40503)
        # Adding element type (line 295)
        
        # Getting the type of 'h' (line 295)
        h_40504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 40), 'h', False)
        int_40505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 43), 'int')
        # Applying the binary operator '**' (line 295)
        result_pow_40506 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 40), '**', h_40504, int_40505)
        
        # Applying the 'usub' unary operator (line 295)
        result___neg___40507 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 39), 'usub', result_pow_40506)
        
        # Getting the type of 'p' (line 295)
        p_40508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 45), 'p', False)
        int_40509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 48), 'int')
        # Applying the binary operator '**' (line 295)
        result_pow_40510 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 45), '**', p_40508, int_40509)
        
        # Applying the binary operator '*' (line 295)
        result_mul_40511 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 39), '*', result___neg___40507, result_pow_40510)
        
        int_40512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 50), 'int')
        # Applying the binary operator 'div' (line 295)
        result_div_40513 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 49), 'div', result_mul_40511, int_40512)
        
        int_40514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 55), 'int')
        # Applying the binary operator '+' (line 295)
        result_add_40515 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 39), '+', result_div_40513, int_40514)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 12), list_40489, result_add_40515)
        # Adding element type (line 295)
        float_40516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 58), 'float')
        # Getting the type of 'h' (line 295)
        h_40517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 63), 'h', False)
        # Applying the binary operator '*' (line 295)
        result_mul_40518 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 58), '*', float_40516, h_40517)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 12), list_40489, result_mul_40518)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 24), list_40488, list_40489)
        # Adding element type (line 294)
        
        # Obtaining an instance of the builtin type 'list' (line 296)
        list_40519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 296)
        # Adding element type (line 296)
        float_40520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 13), 'float')
        # Getting the type of 'h' (line 296)
        h_40521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 17), 'h', False)
        # Applying the binary operator '*' (line 296)
        result_mul_40522 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 13), '*', float_40520, h_40521)
        
        # Getting the type of 'p' (line 296)
        p_40523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 'p', False)
        int_40524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 22), 'int')
        # Applying the binary operator '**' (line 296)
        result_pow_40525 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 19), '**', p_40523, int_40524)
        
        # Applying the binary operator '*' (line 296)
        result_mul_40526 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 18), '*', result_mul_40522, result_pow_40525)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 12), list_40519, result_mul_40526)
        # Adding element type (line 296)
        # Getting the type of 'h' (line 296)
        h_40527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 25), 'h', False)
        int_40528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 28), 'int')
        # Applying the binary operator '**' (line 296)
        result_pow_40529 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 25), '**', h_40527, int_40528)
        
        # Getting the type of 'p' (line 296)
        p_40530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 30), 'p', False)
        int_40531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 33), 'int')
        # Applying the binary operator '**' (line 296)
        result_pow_40532 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 30), '**', p_40530, int_40531)
        
        # Applying the binary operator '*' (line 296)
        result_mul_40533 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 25), '*', result_pow_40529, result_pow_40532)
        
        int_40534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 35), 'int')
        # Applying the binary operator 'div' (line 296)
        result_div_40535 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 34), 'div', result_mul_40533, int_40534)
        
        int_40536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 40), 'int')
        # Applying the binary operator '-' (line 296)
        result_sub_40537 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 25), '-', result_div_40535, int_40536)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 12), list_40519, result_sub_40537)
        # Adding element type (line 296)
        float_40538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 43), 'float')
        # Getting the type of 'h' (line 296)
        h_40539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 47), 'h', False)
        # Applying the binary operator '*' (line 296)
        result_mul_40540 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 43), '*', float_40538, h_40539)
        
        # Getting the type of 'p' (line 296)
        p_40541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 49), 'p', False)
        int_40542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 52), 'int')
        # Applying the binary operator '**' (line 296)
        result_pow_40543 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 49), '**', p_40541, int_40542)
        
        # Applying the binary operator '*' (line 296)
        result_mul_40544 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 48), '*', result_mul_40540, result_pow_40543)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 12), list_40519, result_mul_40544)
        # Adding element type (line 296)
        int_40545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 55), 'int')
        # Getting the type of 'h' (line 296)
        h_40546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 59), 'h', False)
        int_40547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 62), 'int')
        # Applying the binary operator '**' (line 296)
        result_pow_40548 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 59), '**', h_40546, int_40547)
        
        # Getting the type of 'p' (line 296)
        p_40549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 64), 'p', False)
        int_40550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 67), 'int')
        # Applying the binary operator '**' (line 296)
        result_pow_40551 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 64), '**', p_40549, int_40550)
        
        # Applying the binary operator '*' (line 296)
        result_mul_40552 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 59), '*', result_pow_40548, result_pow_40551)
        
        int_40553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 69), 'int')
        # Applying the binary operator 'div' (line 296)
        result_div_40554 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 68), 'div', result_mul_40552, int_40553)
        
        # Applying the binary operator '-' (line 296)
        result_sub_40555 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 55), '-', int_40545, result_div_40554)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 12), list_40519, result_sub_40555)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 24), list_40488, list_40519)
        
        # Processing the call keyword arguments (line 294)
        kwargs_40556 = {}
        # Getting the type of 'np' (line 294)
        np_40486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 294)
        array_40487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 15), np_40486, 'array')
        # Calling array(args, kwargs) (line 294)
        array_call_result_40557 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), array_40487, *[list_40488], **kwargs_40556)
        
        # Assigning a type to the variable 'stypy_return_type' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'stypy_return_type', array_call_result_40557)
        
        # ################# End of 'J_block(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'J_block' in the type store
        # Getting the type of 'stypy_return_type' (line 293)
        stypy_return_type_40558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40558)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'J_block'
        return stypy_return_type_40558

    # Assigning a type to the variable 'J_block' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'J_block', J_block)
    
    # Assigning a Call to a Name (line 299):
    
    # Assigning a Call to a Name (line 299):
    
    # Call to zeros(...): (line 299)
    # Processing the call arguments (line 299)
    
    # Obtaining an instance of the builtin type 'tuple' (line 299)
    tuple_40561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 299)
    # Adding element type (line 299)
    # Getting the type of 'm' (line 299)
    m_40562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 23), 'm', False)
    # Getting the type of 'n' (line 299)
    n_40563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 27), 'n', False)
    # Applying the binary operator '*' (line 299)
    result_mul_40564 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 23), '*', m_40562, n_40563)
    
    # Getting the type of 'k' (line 299)
    k_40565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 31), 'k', False)
    # Applying the binary operator '+' (line 299)
    result_add_40566 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 23), '+', result_mul_40564, k_40565)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 23), tuple_40561, result_add_40566)
    # Adding element type (line 299)
    # Getting the type of 'm' (line 299)
    m_40567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 34), 'm', False)
    # Getting the type of 'n' (line 299)
    n_40568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 38), 'n', False)
    # Applying the binary operator '*' (line 299)
    result_mul_40569 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 34), '*', m_40567, n_40568)
    
    # Getting the type of 'k' (line 299)
    k_40570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 42), 'k', False)
    # Applying the binary operator '+' (line 299)
    result_add_40571 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 34), '+', result_mul_40569, k_40570)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 23), tuple_40561, result_add_40571)
    
    # Processing the call keyword arguments (line 299)
    kwargs_40572 = {}
    # Getting the type of 'np' (line 299)
    np_40559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 13), 'np', False)
    # Obtaining the member 'zeros' of a type (line 299)
    zeros_40560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 13), np_40559, 'zeros')
    # Calling zeros(args, kwargs) (line 299)
    zeros_call_result_40573 = invoke(stypy.reporting.localization.Localization(__file__, 299, 13), zeros_40560, *[tuple_40561], **kwargs_40572)
    
    # Assigning a type to the variable 'J_true' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'J_true', zeros_call_result_40573)
    
    
    # Call to range(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 'm' (line 300)
    m_40575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 19), 'm', False)
    int_40576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 23), 'int')
    # Applying the binary operator '-' (line 300)
    result_sub_40577 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 19), '-', m_40575, int_40576)
    
    # Processing the call keyword arguments (line 300)
    kwargs_40578 = {}
    # Getting the type of 'range' (line 300)
    range_40574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 13), 'range', False)
    # Calling range(args, kwargs) (line 300)
    range_call_result_40579 = invoke(stypy.reporting.localization.Localization(__file__, 300, 13), range_40574, *[result_sub_40577], **kwargs_40578)
    
    # Testing the type of a for loop iterable (line 300)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 300, 4), range_call_result_40579)
    # Getting the type of the for loop variable (line 300)
    for_loop_var_40580 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 300, 4), range_call_result_40579)
    # Assigning a type to the variable 'i' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'i', for_loop_var_40580)
    # SSA begins for a for statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 301):
    
    # Assigning a Call to a Subscript (line 301):
    
    # Call to J_block(...): (line 301)
    # Processing the call arguments (line 301)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 301)
    i_40582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 67), 'i', False)
    # Getting the type of 'h' (line 301)
    h_40583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 65), 'h', False)
    # Obtaining the member '__getitem__' of a type (line 301)
    getitem___40584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 65), h_40583, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 301)
    subscript_call_result_40585 = invoke(stypy.reporting.localization.Localization(__file__, 301, 65), getitem___40584, i_40582)
    
    # Getting the type of 'p' (line 301)
    p_40586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 71), 'p', False)
    # Processing the call keyword arguments (line 301)
    kwargs_40587 = {}
    # Getting the type of 'J_block' (line 301)
    J_block_40581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 57), 'J_block', False)
    # Calling J_block(args, kwargs) (line 301)
    J_block_call_result_40588 = invoke(stypy.reporting.localization.Localization(__file__, 301, 57), J_block_40581, *[subscript_call_result_40585, p_40586], **kwargs_40587)
    
    # Getting the type of 'J_true' (line 301)
    J_true_40589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'J_true')
    # Getting the type of 'i' (line 301)
    i_40590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 15), 'i')
    # Getting the type of 'n' (line 301)
    n_40591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 19), 'n')
    # Applying the binary operator '*' (line 301)
    result_mul_40592 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 15), '*', i_40590, n_40591)
    
    # Getting the type of 'i' (line 301)
    i_40593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 23), 'i')
    int_40594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 27), 'int')
    # Applying the binary operator '+' (line 301)
    result_add_40595 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 23), '+', i_40593, int_40594)
    
    # Getting the type of 'n' (line 301)
    n_40596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 32), 'n')
    # Applying the binary operator '*' (line 301)
    result_mul_40597 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 22), '*', result_add_40595, n_40596)
    
    slice_40598 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 301, 8), result_mul_40592, result_mul_40597, None)
    # Getting the type of 'i' (line 301)
    i_40599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 35), 'i')
    # Getting the type of 'n' (line 301)
    n_40600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 39), 'n')
    # Applying the binary operator '*' (line 301)
    result_mul_40601 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 35), '*', i_40599, n_40600)
    
    # Getting the type of 'i' (line 301)
    i_40602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 43), 'i')
    int_40603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 47), 'int')
    # Applying the binary operator '+' (line 301)
    result_add_40604 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 43), '+', i_40602, int_40603)
    
    # Getting the type of 'n' (line 301)
    n_40605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 52), 'n')
    # Applying the binary operator '*' (line 301)
    result_mul_40606 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 42), '*', result_add_40604, n_40605)
    
    slice_40607 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 301, 8), result_mul_40601, result_mul_40606, None)
    # Storing an element on a container (line 301)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 8), J_true_40589, ((slice_40598, slice_40607), J_block_call_result_40588))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 303):
    
    # Assigning a BinOp to a Subscript (line 303):
    # Getting the type of 'p' (line 303)
    p_40608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 33), 'p')
    # Getting the type of 'h' (line 303)
    h_40609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 37), 'h')
    int_40610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 40), 'int')
    # Applying the binary operator '**' (line 303)
    result_pow_40611 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 37), '**', h_40609, int_40610)
    
    # Applying the binary operator '*' (line 303)
    result_mul_40612 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 33), '*', p_40608, result_pow_40611)
    
    int_40613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 42), 'int')
    # Applying the binary operator 'div' (line 303)
    result_div_40614 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 41), 'div', result_mul_40612, int_40613)
    
    
    # Obtaining the type of the subscript
    int_40615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 49), 'int')
    int_40616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 53), 'int')
    slice_40617 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 303, 47), None, int_40616, None)
    # Getting the type of 'y' (line 303)
    y_40618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 47), 'y')
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___40619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 47), y_40618, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_40620 = invoke(stypy.reporting.localization.Localization(__file__, 303, 47), getitem___40619, (int_40615, slice_40617))
    
    
    # Obtaining the type of the subscript
    int_40621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 61), 'int')
    int_40622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 64), 'int')
    slice_40623 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 303, 59), int_40622, None, None)
    # Getting the type of 'y' (line 303)
    y_40624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 59), 'y')
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___40625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 59), y_40624, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_40626 = invoke(stypy.reporting.localization.Localization(__file__, 303, 59), getitem___40625, (int_40621, slice_40623))
    
    # Applying the binary operator '-' (line 303)
    result_sub_40627 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 47), '-', subscript_call_result_40620, subscript_call_result_40626)
    
    # Applying the binary operator '*' (line 303)
    result_mul_40628 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 44), '*', result_div_40614, result_sub_40627)
    
    # Getting the type of 'J_true' (line 303)
    J_true_40629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'J_true')
    # Getting the type of 'm' (line 303)
    m_40630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 13), 'm')
    int_40631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 17), 'int')
    # Applying the binary operator '-' (line 303)
    result_sub_40632 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 13), '-', m_40630, int_40631)
    
    # Getting the type of 'n' (line 303)
    n_40633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 22), 'n')
    # Applying the binary operator '*' (line 303)
    result_mul_40634 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 12), '*', result_sub_40632, n_40633)
    
    int_40635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 24), 'int')
    slice_40636 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 303, 4), None, result_mul_40634, int_40635)
    int_40637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 27), 'int')
    # Storing an element on a container (line 303)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 4), J_true_40629, ((slice_40636, int_40637), result_mul_40628))
    
    # Assigning a BinOp to a Subscript (line 304):
    
    # Assigning a BinOp to a Subscript (line 304):
    # Getting the type of 'p' (line 304)
    p_40638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 34), 'p')
    # Getting the type of 'h' (line 304)
    h_40639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 39), 'h')
    
    # Obtaining the type of the subscript
    int_40640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 46), 'int')
    int_40641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 50), 'int')
    slice_40642 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 304, 44), None, int_40641, None)
    # Getting the type of 'y' (line 304)
    y_40643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 44), 'y')
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___40644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 44), y_40643, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_40645 = invoke(stypy.reporting.localization.Localization(__file__, 304, 44), getitem___40644, (int_40640, slice_40642))
    
    
    # Obtaining the type of the subscript
    int_40646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 58), 'int')
    int_40647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 61), 'int')
    slice_40648 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 304, 56), int_40647, None, None)
    # Getting the type of 'y' (line 304)
    y_40649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 56), 'y')
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___40650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 56), y_40649, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_40651 = invoke(stypy.reporting.localization.Localization(__file__, 304, 56), getitem___40650, (int_40646, slice_40648))
    
    # Applying the binary operator '+' (line 304)
    result_add_40652 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 44), '+', subscript_call_result_40645, subscript_call_result_40651)
    
    # Applying the binary operator '*' (line 304)
    result_mul_40653 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 39), '*', h_40639, result_add_40652)
    
    # Getting the type of 'h' (line 305)
    h_40654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 39), 'h')
    int_40655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 42), 'int')
    # Applying the binary operator '**' (line 305)
    result_pow_40656 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 39), '**', h_40654, int_40655)
    
    int_40657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 44), 'int')
    # Applying the binary operator 'div' (line 305)
    result_div_40658 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 39), 'div', result_pow_40656, int_40657)
    
    
    # Obtaining the type of the subscript
    int_40659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 51), 'int')
    int_40660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 55), 'int')
    slice_40661 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 305, 49), None, int_40660, None)
    # Getting the type of 'y' (line 305)
    y_40662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 49), 'y')
    # Obtaining the member '__getitem__' of a type (line 305)
    getitem___40663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 49), y_40662, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 305)
    subscript_call_result_40664 = invoke(stypy.reporting.localization.Localization(__file__, 305, 49), getitem___40663, (int_40659, slice_40661))
    
    
    # Obtaining the type of the subscript
    int_40665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 63), 'int')
    int_40666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 66), 'int')
    slice_40667 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 305, 61), int_40666, None, None)
    # Getting the type of 'y' (line 305)
    y_40668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 61), 'y')
    # Obtaining the member '__getitem__' of a type (line 305)
    getitem___40669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 61), y_40668, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 305)
    subscript_call_result_40670 = invoke(stypy.reporting.localization.Localization(__file__, 305, 61), getitem___40669, (int_40665, slice_40667))
    
    # Applying the binary operator '-' (line 305)
    result_sub_40671 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 49), '-', subscript_call_result_40664, subscript_call_result_40670)
    
    # Applying the binary operator '*' (line 305)
    result_mul_40672 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 46), '*', result_div_40658, result_sub_40671)
    
    # Applying the binary operator '+' (line 304)
    result_add_40673 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 39), '+', result_mul_40653, result_mul_40672)
    
    # Applying the binary operator '*' (line 304)
    result_mul_40674 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 34), '*', p_40638, result_add_40673)
    
    # Getting the type of 'J_true' (line 304)
    J_true_40675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'J_true')
    int_40676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 11), 'int')
    # Getting the type of 'm' (line 304)
    m_40677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 14), 'm')
    int_40678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 18), 'int')
    # Applying the binary operator '-' (line 304)
    result_sub_40679 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 14), '-', m_40677, int_40678)
    
    # Getting the type of 'n' (line 304)
    n_40680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 23), 'n')
    # Applying the binary operator '*' (line 304)
    result_mul_40681 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 13), '*', result_sub_40679, n_40680)
    
    int_40682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 25), 'int')
    slice_40683 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 304, 4), int_40676, result_mul_40681, int_40682)
    int_40684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 28), 'int')
    # Storing an element on a container (line 304)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 4), J_true_40675, ((slice_40683, int_40684), result_mul_40674))
    
    # Assigning a Num to a Subscript (line 307):
    
    # Assigning a Num to a Subscript (line 307):
    int_40685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 19), 'int')
    # Getting the type of 'J_true' (line 307)
    J_true_40686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'J_true')
    
    # Obtaining an instance of the builtin type 'tuple' (line 307)
    tuple_40687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 307)
    # Adding element type (line 307)
    int_40688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 11), tuple_40687, int_40688)
    # Adding element type (line 307)
    int_40689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 11), tuple_40687, int_40689)
    
    # Storing an element on a container (line 307)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 4), J_true_40686, (tuple_40687, int_40685))
    
    # Assigning a Num to a Subscript (line 308):
    
    # Assigning a Num to a Subscript (line 308):
    int_40690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 19), 'int')
    # Getting the type of 'J_true' (line 308)
    J_true_40691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'J_true')
    
    # Obtaining an instance of the builtin type 'tuple' (line 308)
    tuple_40692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 308)
    # Adding element type (line 308)
    int_40693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 11), tuple_40692, int_40693)
    # Adding element type (line 308)
    int_40694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 11), tuple_40692, int_40694)
    
    # Storing an element on a container (line 308)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 4), J_true_40691, (tuple_40692, int_40690))
    
    # Assigning a Num to a Subscript (line 309):
    
    # Assigning a Num to a Subscript (line 309):
    int_40695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 20), 'int')
    # Getting the type of 'J_true' (line 309)
    J_true_40696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'J_true')
    
    # Obtaining an instance of the builtin type 'tuple' (line 309)
    tuple_40697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 309)
    # Adding element type (line 309)
    int_40698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 11), tuple_40697, int_40698)
    # Adding element type (line 309)
    int_40699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 11), tuple_40697, int_40699)
    
    # Storing an element on a container (line 309)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 4), J_true_40696, (tuple_40697, int_40695))
    
    # Assigning a Num to a Subscript (line 310):
    
    # Assigning a Num to a Subscript (line 310):
    int_40700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 21), 'int')
    # Getting the type of 'J_true' (line 310)
    J_true_40701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'J_true')
    
    # Obtaining an instance of the builtin type 'tuple' (line 310)
    tuple_40702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 310)
    # Adding element type (line 310)
    int_40703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 11), tuple_40702, int_40703)
    # Adding element type (line 310)
    int_40704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 11), tuple_40702, int_40704)
    
    # Storing an element on a container (line 310)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 4), J_true_40701, (tuple_40702, int_40700))
    
    # Call to assert_allclose(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'J' (line 312)
    J_40706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 20), 'J', False)
    # Getting the type of 'J_true' (line 312)
    J_true_40707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 23), 'J_true', False)
    # Processing the call keyword arguments (line 312)
    float_40708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 36), 'float')
    keyword_40709 = float_40708
    kwargs_40710 = {'rtol': keyword_40709}
    # Getting the type of 'assert_allclose' (line 312)
    assert_allclose_40705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 312)
    assert_allclose_call_result_40711 = invoke(stypy.reporting.localization.Localization(__file__, 312, 4), assert_allclose_40705, *[J_40706, J_true_40707], **kwargs_40710)
    
    
    # Assigning a Call to a Tuple (line 314):
    
    # Assigning a Subscript to a Name (line 314):
    
    # Obtaining the type of the subscript
    int_40712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 4), 'int')
    
    # Call to estimate_fun_jac(...): (line 314)
    # Processing the call arguments (line 314)
    # Getting the type of 'sl_fun' (line 314)
    sl_fun_40714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 36), 'sl_fun', False)
    # Getting the type of 'x' (line 314)
    x_40715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 44), 'x', False)
    # Getting the type of 'y' (line 314)
    y_40716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 47), 'y', False)
    # Getting the type of 'p' (line 314)
    p_40717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 50), 'p', False)
    # Processing the call keyword arguments (line 314)
    kwargs_40718 = {}
    # Getting the type of 'estimate_fun_jac' (line 314)
    estimate_fun_jac_40713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 19), 'estimate_fun_jac', False)
    # Calling estimate_fun_jac(args, kwargs) (line 314)
    estimate_fun_jac_call_result_40719 = invoke(stypy.reporting.localization.Localization(__file__, 314, 19), estimate_fun_jac_40713, *[sl_fun_40714, x_40715, y_40716, p_40717], **kwargs_40718)
    
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___40720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 4), estimate_fun_jac_call_result_40719, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_40721 = invoke(stypy.reporting.localization.Localization(__file__, 314, 4), getitem___40720, int_40712)
    
    # Assigning a type to the variable 'tuple_var_assignment_38833' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'tuple_var_assignment_38833', subscript_call_result_40721)
    
    # Assigning a Subscript to a Name (line 314):
    
    # Obtaining the type of the subscript
    int_40722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 4), 'int')
    
    # Call to estimate_fun_jac(...): (line 314)
    # Processing the call arguments (line 314)
    # Getting the type of 'sl_fun' (line 314)
    sl_fun_40724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 36), 'sl_fun', False)
    # Getting the type of 'x' (line 314)
    x_40725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 44), 'x', False)
    # Getting the type of 'y' (line 314)
    y_40726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 47), 'y', False)
    # Getting the type of 'p' (line 314)
    p_40727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 50), 'p', False)
    # Processing the call keyword arguments (line 314)
    kwargs_40728 = {}
    # Getting the type of 'estimate_fun_jac' (line 314)
    estimate_fun_jac_40723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 19), 'estimate_fun_jac', False)
    # Calling estimate_fun_jac(args, kwargs) (line 314)
    estimate_fun_jac_call_result_40729 = invoke(stypy.reporting.localization.Localization(__file__, 314, 19), estimate_fun_jac_40723, *[sl_fun_40724, x_40725, y_40726, p_40727], **kwargs_40728)
    
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___40730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 4), estimate_fun_jac_call_result_40729, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_40731 = invoke(stypy.reporting.localization.Localization(__file__, 314, 4), getitem___40730, int_40722)
    
    # Assigning a type to the variable 'tuple_var_assignment_38834' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'tuple_var_assignment_38834', subscript_call_result_40731)
    
    # Assigning a Name to a Name (line 314):
    # Getting the type of 'tuple_var_assignment_38833' (line 314)
    tuple_var_assignment_38833_40732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'tuple_var_assignment_38833')
    # Assigning a type to the variable 'df_dy' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'df_dy', tuple_var_assignment_38833_40732)
    
    # Assigning a Name to a Name (line 314):
    # Getting the type of 'tuple_var_assignment_38834' (line 314)
    tuple_var_assignment_38834_40733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'tuple_var_assignment_38834')
    # Assigning a type to the variable 'df_dp' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'df_dp', tuple_var_assignment_38834_40733)
    
    # Assigning a Call to a Tuple (line 315):
    
    # Assigning a Subscript to a Name (line 315):
    
    # Obtaining the type of the subscript
    int_40734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 4), 'int')
    
    # Call to estimate_fun_jac(...): (line 315)
    # Processing the call arguments (line 315)
    # Getting the type of 'sl_fun' (line 315)
    sl_fun_40736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 50), 'sl_fun', False)
    # Getting the type of 'x_middle' (line 315)
    x_middle_40737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 58), 'x_middle', False)
    # Getting the type of 'y_middle' (line 315)
    y_middle_40738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 68), 'y_middle', False)
    # Getting the type of 'p' (line 315)
    p_40739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 78), 'p', False)
    # Processing the call keyword arguments (line 315)
    kwargs_40740 = {}
    # Getting the type of 'estimate_fun_jac' (line 315)
    estimate_fun_jac_40735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 33), 'estimate_fun_jac', False)
    # Calling estimate_fun_jac(args, kwargs) (line 315)
    estimate_fun_jac_call_result_40741 = invoke(stypy.reporting.localization.Localization(__file__, 315, 33), estimate_fun_jac_40735, *[sl_fun_40736, x_middle_40737, y_middle_40738, p_40739], **kwargs_40740)
    
    # Obtaining the member '__getitem__' of a type (line 315)
    getitem___40742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 4), estimate_fun_jac_call_result_40741, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 315)
    subscript_call_result_40743 = invoke(stypy.reporting.localization.Localization(__file__, 315, 4), getitem___40742, int_40734)
    
    # Assigning a type to the variable 'tuple_var_assignment_38835' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'tuple_var_assignment_38835', subscript_call_result_40743)
    
    # Assigning a Subscript to a Name (line 315):
    
    # Obtaining the type of the subscript
    int_40744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 4), 'int')
    
    # Call to estimate_fun_jac(...): (line 315)
    # Processing the call arguments (line 315)
    # Getting the type of 'sl_fun' (line 315)
    sl_fun_40746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 50), 'sl_fun', False)
    # Getting the type of 'x_middle' (line 315)
    x_middle_40747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 58), 'x_middle', False)
    # Getting the type of 'y_middle' (line 315)
    y_middle_40748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 68), 'y_middle', False)
    # Getting the type of 'p' (line 315)
    p_40749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 78), 'p', False)
    # Processing the call keyword arguments (line 315)
    kwargs_40750 = {}
    # Getting the type of 'estimate_fun_jac' (line 315)
    estimate_fun_jac_40745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 33), 'estimate_fun_jac', False)
    # Calling estimate_fun_jac(args, kwargs) (line 315)
    estimate_fun_jac_call_result_40751 = invoke(stypy.reporting.localization.Localization(__file__, 315, 33), estimate_fun_jac_40745, *[sl_fun_40746, x_middle_40747, y_middle_40748, p_40749], **kwargs_40750)
    
    # Obtaining the member '__getitem__' of a type (line 315)
    getitem___40752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 4), estimate_fun_jac_call_result_40751, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 315)
    subscript_call_result_40753 = invoke(stypy.reporting.localization.Localization(__file__, 315, 4), getitem___40752, int_40744)
    
    # Assigning a type to the variable 'tuple_var_assignment_38836' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'tuple_var_assignment_38836', subscript_call_result_40753)
    
    # Assigning a Name to a Name (line 315):
    # Getting the type of 'tuple_var_assignment_38835' (line 315)
    tuple_var_assignment_38835_40754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'tuple_var_assignment_38835')
    # Assigning a type to the variable 'df_dy_middle' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'df_dy_middle', tuple_var_assignment_38835_40754)
    
    # Assigning a Name to a Name (line 315):
    # Getting the type of 'tuple_var_assignment_38836' (line 315)
    tuple_var_assignment_38836_40755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'tuple_var_assignment_38836')
    # Assigning a type to the variable 'df_dp_middle' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 18), 'df_dp_middle', tuple_var_assignment_38836_40755)
    
    # Assigning a Call to a Tuple (line 316):
    
    # Assigning a Subscript to a Name (line 316):
    
    # Obtaining the type of the subscript
    int_40756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 4), 'int')
    
    # Call to estimate_bc_jac(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'sl_bc' (line 316)
    sl_bc_40758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 47), 'sl_bc', False)
    
    # Obtaining the type of the subscript
    slice_40759 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 316, 54), None, None, None)
    int_40760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 59), 'int')
    # Getting the type of 'y' (line 316)
    y_40761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 54), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___40762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 54), y_40761, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_40763 = invoke(stypy.reporting.localization.Localization(__file__, 316, 54), getitem___40762, (slice_40759, int_40760))
    
    
    # Obtaining the type of the subscript
    slice_40764 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 316, 63), None, None, None)
    int_40765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 68), 'int')
    # Getting the type of 'y' (line 316)
    y_40766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 63), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___40767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 63), y_40766, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_40768 = invoke(stypy.reporting.localization.Localization(__file__, 316, 63), getitem___40767, (slice_40764, int_40765))
    
    # Getting the type of 'p' (line 316)
    p_40769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 73), 'p', False)
    # Processing the call keyword arguments (line 316)
    kwargs_40770 = {}
    # Getting the type of 'estimate_bc_jac' (line 316)
    estimate_bc_jac_40757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 31), 'estimate_bc_jac', False)
    # Calling estimate_bc_jac(args, kwargs) (line 316)
    estimate_bc_jac_call_result_40771 = invoke(stypy.reporting.localization.Localization(__file__, 316, 31), estimate_bc_jac_40757, *[sl_bc_40758, subscript_call_result_40763, subscript_call_result_40768, p_40769], **kwargs_40770)
    
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___40772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 4), estimate_bc_jac_call_result_40771, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_40773 = invoke(stypy.reporting.localization.Localization(__file__, 316, 4), getitem___40772, int_40756)
    
    # Assigning a type to the variable 'tuple_var_assignment_38837' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'tuple_var_assignment_38837', subscript_call_result_40773)
    
    # Assigning a Subscript to a Name (line 316):
    
    # Obtaining the type of the subscript
    int_40774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 4), 'int')
    
    # Call to estimate_bc_jac(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'sl_bc' (line 316)
    sl_bc_40776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 47), 'sl_bc', False)
    
    # Obtaining the type of the subscript
    slice_40777 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 316, 54), None, None, None)
    int_40778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 59), 'int')
    # Getting the type of 'y' (line 316)
    y_40779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 54), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___40780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 54), y_40779, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_40781 = invoke(stypy.reporting.localization.Localization(__file__, 316, 54), getitem___40780, (slice_40777, int_40778))
    
    
    # Obtaining the type of the subscript
    slice_40782 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 316, 63), None, None, None)
    int_40783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 68), 'int')
    # Getting the type of 'y' (line 316)
    y_40784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 63), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___40785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 63), y_40784, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_40786 = invoke(stypy.reporting.localization.Localization(__file__, 316, 63), getitem___40785, (slice_40782, int_40783))
    
    # Getting the type of 'p' (line 316)
    p_40787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 73), 'p', False)
    # Processing the call keyword arguments (line 316)
    kwargs_40788 = {}
    # Getting the type of 'estimate_bc_jac' (line 316)
    estimate_bc_jac_40775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 31), 'estimate_bc_jac', False)
    # Calling estimate_bc_jac(args, kwargs) (line 316)
    estimate_bc_jac_call_result_40789 = invoke(stypy.reporting.localization.Localization(__file__, 316, 31), estimate_bc_jac_40775, *[sl_bc_40776, subscript_call_result_40781, subscript_call_result_40786, p_40787], **kwargs_40788)
    
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___40790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 4), estimate_bc_jac_call_result_40789, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_40791 = invoke(stypy.reporting.localization.Localization(__file__, 316, 4), getitem___40790, int_40774)
    
    # Assigning a type to the variable 'tuple_var_assignment_38838' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'tuple_var_assignment_38838', subscript_call_result_40791)
    
    # Assigning a Subscript to a Name (line 316):
    
    # Obtaining the type of the subscript
    int_40792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 4), 'int')
    
    # Call to estimate_bc_jac(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'sl_bc' (line 316)
    sl_bc_40794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 47), 'sl_bc', False)
    
    # Obtaining the type of the subscript
    slice_40795 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 316, 54), None, None, None)
    int_40796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 59), 'int')
    # Getting the type of 'y' (line 316)
    y_40797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 54), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___40798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 54), y_40797, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_40799 = invoke(stypy.reporting.localization.Localization(__file__, 316, 54), getitem___40798, (slice_40795, int_40796))
    
    
    # Obtaining the type of the subscript
    slice_40800 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 316, 63), None, None, None)
    int_40801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 68), 'int')
    # Getting the type of 'y' (line 316)
    y_40802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 63), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___40803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 63), y_40802, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_40804 = invoke(stypy.reporting.localization.Localization(__file__, 316, 63), getitem___40803, (slice_40800, int_40801))
    
    # Getting the type of 'p' (line 316)
    p_40805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 73), 'p', False)
    # Processing the call keyword arguments (line 316)
    kwargs_40806 = {}
    # Getting the type of 'estimate_bc_jac' (line 316)
    estimate_bc_jac_40793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 31), 'estimate_bc_jac', False)
    # Calling estimate_bc_jac(args, kwargs) (line 316)
    estimate_bc_jac_call_result_40807 = invoke(stypy.reporting.localization.Localization(__file__, 316, 31), estimate_bc_jac_40793, *[sl_bc_40794, subscript_call_result_40799, subscript_call_result_40804, p_40805], **kwargs_40806)
    
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___40808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 4), estimate_bc_jac_call_result_40807, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_40809 = invoke(stypy.reporting.localization.Localization(__file__, 316, 4), getitem___40808, int_40792)
    
    # Assigning a type to the variable 'tuple_var_assignment_38839' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'tuple_var_assignment_38839', subscript_call_result_40809)
    
    # Assigning a Name to a Name (line 316):
    # Getting the type of 'tuple_var_assignment_38837' (line 316)
    tuple_var_assignment_38837_40810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'tuple_var_assignment_38837')
    # Assigning a type to the variable 'dbc_dya' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'dbc_dya', tuple_var_assignment_38837_40810)
    
    # Assigning a Name to a Name (line 316):
    # Getting the type of 'tuple_var_assignment_38838' (line 316)
    tuple_var_assignment_38838_40811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'tuple_var_assignment_38838')
    # Assigning a type to the variable 'dbc_dyb' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 13), 'dbc_dyb', tuple_var_assignment_38838_40811)
    
    # Assigning a Name to a Name (line 316):
    # Getting the type of 'tuple_var_assignment_38839' (line 316)
    tuple_var_assignment_38839_40812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'tuple_var_assignment_38839')
    # Assigning a type to the variable 'dbc_dp' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 22), 'dbc_dp', tuple_var_assignment_38839_40812)
    
    # Assigning a Call to a Name (line 317):
    
    # Assigning a Call to a Name (line 317):
    
    # Call to construct_global_jac(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 'n' (line 317)
    n_40814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 29), 'n', False)
    # Getting the type of 'm' (line 317)
    m_40815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 32), 'm', False)
    # Getting the type of 'k' (line 317)
    k_40816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 35), 'k', False)
    # Getting the type of 'i_jac' (line 317)
    i_jac_40817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 38), 'i_jac', False)
    # Getting the type of 'j_jac' (line 317)
    j_jac_40818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 45), 'j_jac', False)
    # Getting the type of 'h' (line 317)
    h_40819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 52), 'h', False)
    # Getting the type of 'df_dy' (line 317)
    df_dy_40820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 55), 'df_dy', False)
    # Getting the type of 'df_dy_middle' (line 317)
    df_dy_middle_40821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 62), 'df_dy_middle', False)
    # Getting the type of 'df_dp' (line 318)
    df_dp_40822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 29), 'df_dp', False)
    # Getting the type of 'df_dp_middle' (line 318)
    df_dp_middle_40823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 36), 'df_dp_middle', False)
    # Getting the type of 'dbc_dya' (line 318)
    dbc_dya_40824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 50), 'dbc_dya', False)
    # Getting the type of 'dbc_dyb' (line 318)
    dbc_dyb_40825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 59), 'dbc_dyb', False)
    # Getting the type of 'dbc_dp' (line 318)
    dbc_dp_40826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 68), 'dbc_dp', False)
    # Processing the call keyword arguments (line 317)
    kwargs_40827 = {}
    # Getting the type of 'construct_global_jac' (line 317)
    construct_global_jac_40813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'construct_global_jac', False)
    # Calling construct_global_jac(args, kwargs) (line 317)
    construct_global_jac_call_result_40828 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), construct_global_jac_40813, *[n_40814, m_40815, k_40816, i_jac_40817, j_jac_40818, h_40819, df_dy_40820, df_dy_middle_40821, df_dp_40822, df_dp_middle_40823, dbc_dya_40824, dbc_dyb_40825, dbc_dp_40826], **kwargs_40827)
    
    # Assigning a type to the variable 'J' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'J', construct_global_jac_call_result_40828)
    
    # Assigning a Call to a Name (line 319):
    
    # Assigning a Call to a Name (line 319):
    
    # Call to toarray(...): (line 319)
    # Processing the call keyword arguments (line 319)
    kwargs_40831 = {}
    # Getting the type of 'J' (line 319)
    J_40829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'J', False)
    # Obtaining the member 'toarray' of a type (line 319)
    toarray_40830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), J_40829, 'toarray')
    # Calling toarray(args, kwargs) (line 319)
    toarray_call_result_40832 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), toarray_40830, *[], **kwargs_40831)
    
    # Assigning a type to the variable 'J' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'J', toarray_call_result_40832)
    
    # Call to assert_allclose(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'J' (line 320)
    J_40834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 20), 'J', False)
    # Getting the type of 'J_true' (line 320)
    J_true_40835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 23), 'J_true', False)
    # Processing the call keyword arguments (line 320)
    float_40836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 36), 'float')
    keyword_40837 = float_40836
    float_40838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 47), 'float')
    keyword_40839 = float_40838
    kwargs_40840 = {'rtol': keyword_40837, 'atol': keyword_40839}
    # Getting the type of 'assert_allclose' (line 320)
    assert_allclose_40833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 320)
    assert_allclose_call_result_40841 = invoke(stypy.reporting.localization.Localization(__file__, 320, 4), assert_allclose_40833, *[J_40834, J_true_40835], **kwargs_40840)
    
    
    # ################# End of 'test_compute_global_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_compute_global_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 270)
    stypy_return_type_40842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_40842)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_compute_global_jac'
    return stypy_return_type_40842

# Assigning a type to the variable 'test_compute_global_jac' (line 270)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), 'test_compute_global_jac', test_compute_global_jac)

@norecursion
def test_parameter_validation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_parameter_validation'
    module_type_store = module_type_store.open_function_context('test_parameter_validation', 323, 0, False)
    
    # Passed parameters checking function
    test_parameter_validation.stypy_localization = localization
    test_parameter_validation.stypy_type_of_self = None
    test_parameter_validation.stypy_type_store = module_type_store
    test_parameter_validation.stypy_function_name = 'test_parameter_validation'
    test_parameter_validation.stypy_param_names_list = []
    test_parameter_validation.stypy_varargs_param_name = None
    test_parameter_validation.stypy_kwargs_param_name = None
    test_parameter_validation.stypy_call_defaults = defaults
    test_parameter_validation.stypy_call_varargs = varargs
    test_parameter_validation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_parameter_validation', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_parameter_validation', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_parameter_validation(...)' code ##################

    
    # Assigning a List to a Name (line 324):
    
    # Assigning a List to a Name (line 324):
    
    # Obtaining an instance of the builtin type 'list' (line 324)
    list_40843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 324)
    # Adding element type (line 324)
    int_40844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 8), list_40843, int_40844)
    # Adding element type (line 324)
    int_40845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 8), list_40843, int_40845)
    # Adding element type (line 324)
    float_40846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 8), list_40843, float_40846)
    
    # Assigning a type to the variable 'x' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'x', list_40843)
    
    # Assigning a Call to a Name (line 325):
    
    # Assigning a Call to a Name (line 325):
    
    # Call to zeros(...): (line 325)
    # Processing the call arguments (line 325)
    
    # Obtaining an instance of the builtin type 'tuple' (line 325)
    tuple_40849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 325)
    # Adding element type (line 325)
    int_40850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 18), tuple_40849, int_40850)
    # Adding element type (line 325)
    int_40851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 18), tuple_40849, int_40851)
    
    # Processing the call keyword arguments (line 325)
    kwargs_40852 = {}
    # Getting the type of 'np' (line 325)
    np_40847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 325)
    zeros_40848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 8), np_40847, 'zeros')
    # Calling zeros(args, kwargs) (line 325)
    zeros_call_result_40853 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), zeros_40848, *[tuple_40849], **kwargs_40852)
    
    # Assigning a type to the variable 'y' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'y', zeros_call_result_40853)
    
    # Call to assert_raises(...): (line 326)
    # Processing the call arguments (line 326)
    # Getting the type of 'ValueError' (line 326)
    ValueError_40855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 18), 'ValueError', False)
    # Getting the type of 'solve_bvp' (line 326)
    solve_bvp_40856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 30), 'solve_bvp', False)
    # Getting the type of 'exp_fun' (line 326)
    exp_fun_40857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 41), 'exp_fun', False)
    # Getting the type of 'exp_bc' (line 326)
    exp_bc_40858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 50), 'exp_bc', False)
    # Getting the type of 'x' (line 326)
    x_40859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 58), 'x', False)
    # Getting the type of 'y' (line 326)
    y_40860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 61), 'y', False)
    # Processing the call keyword arguments (line 326)
    kwargs_40861 = {}
    # Getting the type of 'assert_raises' (line 326)
    assert_raises_40854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 326)
    assert_raises_call_result_40862 = invoke(stypy.reporting.localization.Localization(__file__, 326, 4), assert_raises_40854, *[ValueError_40855, solve_bvp_40856, exp_fun_40857, exp_bc_40858, x_40859, y_40860], **kwargs_40861)
    
    
    # Assigning a Call to a Name (line 328):
    
    # Assigning a Call to a Name (line 328):
    
    # Call to linspace(...): (line 328)
    # Processing the call arguments (line 328)
    int_40865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 20), 'int')
    int_40866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 23), 'int')
    int_40867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 26), 'int')
    # Processing the call keyword arguments (line 328)
    kwargs_40868 = {}
    # Getting the type of 'np' (line 328)
    np_40863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 328)
    linspace_40864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 8), np_40863, 'linspace')
    # Calling linspace(args, kwargs) (line 328)
    linspace_call_result_40869 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), linspace_40864, *[int_40865, int_40866, int_40867], **kwargs_40868)
    
    # Assigning a type to the variable 'x' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'x', linspace_call_result_40869)
    
    # Assigning a Call to a Name (line 329):
    
    # Assigning a Call to a Name (line 329):
    
    # Call to zeros(...): (line 329)
    # Processing the call arguments (line 329)
    
    # Obtaining an instance of the builtin type 'tuple' (line 329)
    tuple_40872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 329)
    # Adding element type (line 329)
    int_40873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 18), tuple_40872, int_40873)
    # Adding element type (line 329)
    int_40874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 18), tuple_40872, int_40874)
    
    # Processing the call keyword arguments (line 329)
    kwargs_40875 = {}
    # Getting the type of 'np' (line 329)
    np_40870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 329)
    zeros_40871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), np_40870, 'zeros')
    # Calling zeros(args, kwargs) (line 329)
    zeros_call_result_40876 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), zeros_40871, *[tuple_40872], **kwargs_40875)
    
    # Assigning a type to the variable 'y' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'y', zeros_call_result_40876)
    
    # Call to assert_raises(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'ValueError' (line 330)
    ValueError_40878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 18), 'ValueError', False)
    # Getting the type of 'solve_bvp' (line 330)
    solve_bvp_40879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 30), 'solve_bvp', False)
    # Getting the type of 'exp_fun' (line 330)
    exp_fun_40880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 41), 'exp_fun', False)
    # Getting the type of 'exp_bc' (line 330)
    exp_bc_40881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 50), 'exp_bc', False)
    # Getting the type of 'x' (line 330)
    x_40882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 58), 'x', False)
    # Getting the type of 'y' (line 330)
    y_40883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 61), 'y', False)
    # Processing the call keyword arguments (line 330)
    kwargs_40884 = {}
    # Getting the type of 'assert_raises' (line 330)
    assert_raises_40877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 330)
    assert_raises_call_result_40885 = invoke(stypy.reporting.localization.Localization(__file__, 330, 4), assert_raises_40877, *[ValueError_40878, solve_bvp_40879, exp_fun_40880, exp_bc_40881, x_40882, y_40883], **kwargs_40884)
    
    
    # Assigning a Lambda to a Name (line 332):
    
    # Assigning a Lambda to a Name (line 332):

    @norecursion
    def _stypy_temp_lambda_15(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_15'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_15', 332, 10, True)
        # Passed parameters checking function
        _stypy_temp_lambda_15.stypy_localization = localization
        _stypy_temp_lambda_15.stypy_type_of_self = None
        _stypy_temp_lambda_15.stypy_type_store = module_type_store
        _stypy_temp_lambda_15.stypy_function_name = '_stypy_temp_lambda_15'
        _stypy_temp_lambda_15.stypy_param_names_list = ['x', 'y', 'p']
        _stypy_temp_lambda_15.stypy_varargs_param_name = None
        _stypy_temp_lambda_15.stypy_kwargs_param_name = None
        _stypy_temp_lambda_15.stypy_call_defaults = defaults
        _stypy_temp_lambda_15.stypy_call_varargs = varargs
        _stypy_temp_lambda_15.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_15', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_15', ['x', 'y', 'p'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to exp_fun(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'x' (line 332)
        x_40887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), 'x', False)
        # Getting the type of 'y' (line 332)
        y_40888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 37), 'y', False)
        # Processing the call keyword arguments (line 332)
        kwargs_40889 = {}
        # Getting the type of 'exp_fun' (line 332)
        exp_fun_40886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 26), 'exp_fun', False)
        # Calling exp_fun(args, kwargs) (line 332)
        exp_fun_call_result_40890 = invoke(stypy.reporting.localization.Localization(__file__, 332, 26), exp_fun_40886, *[x_40887, y_40888], **kwargs_40889)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 10), 'stypy_return_type', exp_fun_call_result_40890)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_15' in the type store
        # Getting the type of 'stypy_return_type' (line 332)
        stypy_return_type_40891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 10), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40891)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_15'
        return stypy_return_type_40891

    # Assigning a type to the variable '_stypy_temp_lambda_15' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 10), '_stypy_temp_lambda_15', _stypy_temp_lambda_15)
    # Getting the type of '_stypy_temp_lambda_15' (line 332)
    _stypy_temp_lambda_15_40892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 10), '_stypy_temp_lambda_15')
    # Assigning a type to the variable 'fun' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'fun', _stypy_temp_lambda_15_40892)
    
    # Assigning a Lambda to a Name (line 333):
    
    # Assigning a Lambda to a Name (line 333):

    @norecursion
    def _stypy_temp_lambda_16(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_16'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_16', 333, 9, True)
        # Passed parameters checking function
        _stypy_temp_lambda_16.stypy_localization = localization
        _stypy_temp_lambda_16.stypy_type_of_self = None
        _stypy_temp_lambda_16.stypy_type_store = module_type_store
        _stypy_temp_lambda_16.stypy_function_name = '_stypy_temp_lambda_16'
        _stypy_temp_lambda_16.stypy_param_names_list = ['ya', 'yb', 'p']
        _stypy_temp_lambda_16.stypy_varargs_param_name = None
        _stypy_temp_lambda_16.stypy_kwargs_param_name = None
        _stypy_temp_lambda_16.stypy_call_defaults = defaults
        _stypy_temp_lambda_16.stypy_call_varargs = varargs
        _stypy_temp_lambda_16.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_16', ['ya', 'yb', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_16', ['ya', 'yb', 'p'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to exp_bc(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'ya' (line 333)
        ya_40894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 34), 'ya', False)
        # Getting the type of 'yb' (line 333)
        yb_40895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 38), 'yb', False)
        # Processing the call keyword arguments (line 333)
        kwargs_40896 = {}
        # Getting the type of 'exp_bc' (line 333)
        exp_bc_40893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 27), 'exp_bc', False)
        # Calling exp_bc(args, kwargs) (line 333)
        exp_bc_call_result_40897 = invoke(stypy.reporting.localization.Localization(__file__, 333, 27), exp_bc_40893, *[ya_40894, yb_40895], **kwargs_40896)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 9), 'stypy_return_type', exp_bc_call_result_40897)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_16' in the type store
        # Getting the type of 'stypy_return_type' (line 333)
        stypy_return_type_40898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 9), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40898)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_16'
        return stypy_return_type_40898

    # Assigning a type to the variable '_stypy_temp_lambda_16' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 9), '_stypy_temp_lambda_16', _stypy_temp_lambda_16)
    # Getting the type of '_stypy_temp_lambda_16' (line 333)
    _stypy_temp_lambda_16_40899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 9), '_stypy_temp_lambda_16')
    # Assigning a type to the variable 'bc' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'bc', _stypy_temp_lambda_16_40899)
    
    # Assigning a Call to a Name (line 335):
    
    # Assigning a Call to a Name (line 335):
    
    # Call to zeros(...): (line 335)
    # Processing the call arguments (line 335)
    
    # Obtaining an instance of the builtin type 'tuple' (line 335)
    tuple_40902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 335)
    # Adding element type (line 335)
    int_40903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 18), tuple_40902, int_40903)
    # Adding element type (line 335)
    
    # Obtaining the type of the subscript
    int_40904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 29), 'int')
    # Getting the type of 'x' (line 335)
    x_40905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 335)
    shape_40906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 21), x_40905, 'shape')
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___40907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 21), shape_40906, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_40908 = invoke(stypy.reporting.localization.Localization(__file__, 335, 21), getitem___40907, int_40904)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 18), tuple_40902, subscript_call_result_40908)
    
    # Processing the call keyword arguments (line 335)
    kwargs_40909 = {}
    # Getting the type of 'np' (line 335)
    np_40900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 335)
    zeros_40901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), np_40900, 'zeros')
    # Calling zeros(args, kwargs) (line 335)
    zeros_call_result_40910 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), zeros_40901, *[tuple_40902], **kwargs_40909)
    
    # Assigning a type to the variable 'y' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'y', zeros_call_result_40910)
    
    # Call to assert_raises(...): (line 336)
    # Processing the call arguments (line 336)
    # Getting the type of 'ValueError' (line 336)
    ValueError_40912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 18), 'ValueError', False)
    # Getting the type of 'solve_bvp' (line 336)
    solve_bvp_40913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 30), 'solve_bvp', False)
    # Getting the type of 'fun' (line 336)
    fun_40914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 41), 'fun', False)
    # Getting the type of 'bc' (line 336)
    bc_40915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 46), 'bc', False)
    # Getting the type of 'x' (line 336)
    x_40916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 50), 'x', False)
    # Getting the type of 'y' (line 336)
    y_40917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 53), 'y', False)
    # Processing the call keyword arguments (line 336)
    
    # Obtaining an instance of the builtin type 'list' (line 336)
    list_40918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 58), 'list')
    # Adding type elements to the builtin type 'list' instance (line 336)
    # Adding element type (line 336)
    int_40919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 58), list_40918, int_40919)
    
    keyword_40920 = list_40918
    kwargs_40921 = {'p': keyword_40920}
    # Getting the type of 'assert_raises' (line 336)
    assert_raises_40911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 336)
    assert_raises_call_result_40922 = invoke(stypy.reporting.localization.Localization(__file__, 336, 4), assert_raises_40911, *[ValueError_40912, solve_bvp_40913, fun_40914, bc_40915, x_40916, y_40917], **kwargs_40921)
    

    @norecursion
    def wrong_shape_fun(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wrong_shape_fun'
        module_type_store = module_type_store.open_function_context('wrong_shape_fun', 338, 4, False)
        
        # Passed parameters checking function
        wrong_shape_fun.stypy_localization = localization
        wrong_shape_fun.stypy_type_of_self = None
        wrong_shape_fun.stypy_type_store = module_type_store
        wrong_shape_fun.stypy_function_name = 'wrong_shape_fun'
        wrong_shape_fun.stypy_param_names_list = ['x', 'y']
        wrong_shape_fun.stypy_varargs_param_name = None
        wrong_shape_fun.stypy_kwargs_param_name = None
        wrong_shape_fun.stypy_call_defaults = defaults
        wrong_shape_fun.stypy_call_varargs = varargs
        wrong_shape_fun.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'wrong_shape_fun', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wrong_shape_fun', localization, ['x', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wrong_shape_fun(...)' code ##################

        
        # Call to zeros(...): (line 339)
        # Processing the call arguments (line 339)
        int_40925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 24), 'int')
        # Processing the call keyword arguments (line 339)
        kwargs_40926 = {}
        # Getting the type of 'np' (line 339)
        np_40923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'np', False)
        # Obtaining the member 'zeros' of a type (line 339)
        zeros_40924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 15), np_40923, 'zeros')
        # Calling zeros(args, kwargs) (line 339)
        zeros_call_result_40927 = invoke(stypy.reporting.localization.Localization(__file__, 339, 15), zeros_40924, *[int_40925], **kwargs_40926)
        
        # Assigning a type to the variable 'stypy_return_type' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'stypy_return_type', zeros_call_result_40927)
        
        # ################# End of 'wrong_shape_fun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wrong_shape_fun' in the type store
        # Getting the type of 'stypy_return_type' (line 338)
        stypy_return_type_40928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40928)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wrong_shape_fun'
        return stypy_return_type_40928

    # Assigning a type to the variable 'wrong_shape_fun' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'wrong_shape_fun', wrong_shape_fun)
    
    # Call to assert_raises(...): (line 341)
    # Processing the call arguments (line 341)
    # Getting the type of 'ValueError' (line 341)
    ValueError_40930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 18), 'ValueError', False)
    # Getting the type of 'solve_bvp' (line 341)
    solve_bvp_40931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 30), 'solve_bvp', False)
    # Getting the type of 'wrong_shape_fun' (line 341)
    wrong_shape_fun_40932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 41), 'wrong_shape_fun', False)
    # Getting the type of 'bc' (line 341)
    bc_40933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 58), 'bc', False)
    # Getting the type of 'x' (line 341)
    x_40934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 62), 'x', False)
    # Getting the type of 'y' (line 341)
    y_40935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 65), 'y', False)
    # Processing the call keyword arguments (line 341)
    kwargs_40936 = {}
    # Getting the type of 'assert_raises' (line 341)
    assert_raises_40929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 341)
    assert_raises_call_result_40937 = invoke(stypy.reporting.localization.Localization(__file__, 341, 4), assert_raises_40929, *[ValueError_40930, solve_bvp_40931, wrong_shape_fun_40932, bc_40933, x_40934, y_40935], **kwargs_40936)
    
    
    # Assigning a Call to a Name (line 343):
    
    # Assigning a Call to a Name (line 343):
    
    # Call to array(...): (line 343)
    # Processing the call arguments (line 343)
    
    # Obtaining an instance of the builtin type 'list' (line 343)
    list_40940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 343)
    # Adding element type (line 343)
    
    # Obtaining an instance of the builtin type 'list' (line 343)
    list_40941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 343)
    # Adding element type (line 343)
    int_40942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 18), list_40941, int_40942)
    # Adding element type (line 343)
    int_40943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 18), list_40941, int_40943)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 17), list_40940, list_40941)
    
    # Processing the call keyword arguments (line 343)
    kwargs_40944 = {}
    # Getting the type of 'np' (line 343)
    np_40938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 343)
    array_40939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), np_40938, 'array')
    # Calling array(args, kwargs) (line 343)
    array_call_result_40945 = invoke(stypy.reporting.localization.Localization(__file__, 343, 8), array_40939, *[list_40940], **kwargs_40944)
    
    # Assigning a type to the variable 'S' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'S', array_call_result_40945)
    
    # Call to assert_raises(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'ValueError' (line 344)
    ValueError_40947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 18), 'ValueError', False)
    # Getting the type of 'solve_bvp' (line 344)
    solve_bvp_40948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 30), 'solve_bvp', False)
    # Getting the type of 'exp_fun' (line 344)
    exp_fun_40949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 41), 'exp_fun', False)
    # Getting the type of 'exp_bc' (line 344)
    exp_bc_40950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 50), 'exp_bc', False)
    # Getting the type of 'x' (line 344)
    x_40951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 58), 'x', False)
    # Getting the type of 'y' (line 344)
    y_40952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 61), 'y', False)
    # Processing the call keyword arguments (line 344)
    # Getting the type of 'S' (line 344)
    S_40953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 66), 'S', False)
    keyword_40954 = S_40953
    kwargs_40955 = {'S': keyword_40954}
    # Getting the type of 'assert_raises' (line 344)
    assert_raises_40946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 344)
    assert_raises_call_result_40956 = invoke(stypy.reporting.localization.Localization(__file__, 344, 4), assert_raises_40946, *[ValueError_40947, solve_bvp_40948, exp_fun_40949, exp_bc_40950, x_40951, y_40952], **kwargs_40955)
    
    
    # ################# End of 'test_parameter_validation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_parameter_validation' in the type store
    # Getting the type of 'stypy_return_type' (line 323)
    stypy_return_type_40957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_40957)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_parameter_validation'
    return stypy_return_type_40957

# Assigning a type to the variable 'test_parameter_validation' (line 323)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'test_parameter_validation', test_parameter_validation)

@norecursion
def test_no_params(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_no_params'
    module_type_store = module_type_store.open_function_context('test_no_params', 347, 0, False)
    
    # Passed parameters checking function
    test_no_params.stypy_localization = localization
    test_no_params.stypy_type_of_self = None
    test_no_params.stypy_type_store = module_type_store
    test_no_params.stypy_function_name = 'test_no_params'
    test_no_params.stypy_param_names_list = []
    test_no_params.stypy_varargs_param_name = None
    test_no_params.stypy_kwargs_param_name = None
    test_no_params.stypy_call_defaults = defaults
    test_no_params.stypy_call_varargs = varargs
    test_no_params.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_no_params', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_no_params', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_no_params(...)' code ##################

    
    # Assigning a Call to a Name (line 348):
    
    # Assigning a Call to a Name (line 348):
    
    # Call to linspace(...): (line 348)
    # Processing the call arguments (line 348)
    int_40960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 20), 'int')
    int_40961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 23), 'int')
    int_40962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 26), 'int')
    # Processing the call keyword arguments (line 348)
    kwargs_40963 = {}
    # Getting the type of 'np' (line 348)
    np_40958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 348)
    linspace_40959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), np_40958, 'linspace')
    # Calling linspace(args, kwargs) (line 348)
    linspace_call_result_40964 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), linspace_40959, *[int_40960, int_40961, int_40962], **kwargs_40963)
    
    # Assigning a type to the variable 'x' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'x', linspace_call_result_40964)
    
    # Assigning a Call to a Name (line 349):
    
    # Assigning a Call to a Name (line 349):
    
    # Call to linspace(...): (line 349)
    # Processing the call arguments (line 349)
    int_40967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 25), 'int')
    int_40968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 28), 'int')
    int_40969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 31), 'int')
    # Processing the call keyword arguments (line 349)
    kwargs_40970 = {}
    # Getting the type of 'np' (line 349)
    np_40965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 13), 'np', False)
    # Obtaining the member 'linspace' of a type (line 349)
    linspace_40966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 13), np_40965, 'linspace')
    # Calling linspace(args, kwargs) (line 349)
    linspace_call_result_40971 = invoke(stypy.reporting.localization.Localization(__file__, 349, 13), linspace_40966, *[int_40967, int_40968, int_40969], **kwargs_40970)
    
    # Assigning a type to the variable 'x_test' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'x_test', linspace_call_result_40971)
    
    # Assigning a Call to a Name (line 350):
    
    # Assigning a Call to a Name (line 350):
    
    # Call to zeros(...): (line 350)
    # Processing the call arguments (line 350)
    
    # Obtaining an instance of the builtin type 'tuple' (line 350)
    tuple_40974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 350)
    # Adding element type (line 350)
    int_40975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 18), tuple_40974, int_40975)
    # Adding element type (line 350)
    
    # Obtaining the type of the subscript
    int_40976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 29), 'int')
    # Getting the type of 'x' (line 350)
    x_40977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 350)
    shape_40978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 21), x_40977, 'shape')
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___40979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 21), shape_40978, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_40980 = invoke(stypy.reporting.localization.Localization(__file__, 350, 21), getitem___40979, int_40976)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 18), tuple_40974, subscript_call_result_40980)
    
    # Processing the call keyword arguments (line 350)
    kwargs_40981 = {}
    # Getting the type of 'np' (line 350)
    np_40972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 350)
    zeros_40973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), np_40972, 'zeros')
    # Calling zeros(args, kwargs) (line 350)
    zeros_call_result_40982 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), zeros_40973, *[tuple_40974], **kwargs_40981)
    
    # Assigning a type to the variable 'y' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'y', zeros_call_result_40982)
    
    
    # Obtaining an instance of the builtin type 'list' (line 351)
    list_40983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 351)
    # Adding element type (line 351)
    # Getting the type of 'None' (line 351)
    None_40984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 20), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 19), list_40983, None_40984)
    # Adding element type (line 351)
    # Getting the type of 'exp_fun_jac' (line 351)
    exp_fun_jac_40985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 26), 'exp_fun_jac')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 19), list_40983, exp_fun_jac_40985)
    
    # Testing the type of a for loop iterable (line 351)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 351, 4), list_40983)
    # Getting the type of the for loop variable (line 351)
    for_loop_var_40986 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 351, 4), list_40983)
    # Assigning a type to the variable 'fun_jac' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'fun_jac', for_loop_var_40986)
    # SSA begins for a for statement (line 351)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'list' (line 352)
    list_40987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 352)
    # Adding element type (line 352)
    # Getting the type of 'None' (line 352)
    None_40988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 23), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 22), list_40987, None_40988)
    # Adding element type (line 352)
    # Getting the type of 'exp_bc_jac' (line 352)
    exp_bc_jac_40989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 29), 'exp_bc_jac')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 22), list_40987, exp_bc_jac_40989)
    
    # Testing the type of a for loop iterable (line 352)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 352, 8), list_40987)
    # Getting the type of the for loop variable (line 352)
    for_loop_var_40990 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 352, 8), list_40987)
    # Assigning a type to the variable 'bc_jac' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'bc_jac', for_loop_var_40990)
    # SSA begins for a for statement (line 352)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 353):
    
    # Assigning a Call to a Name (line 353):
    
    # Call to solve_bvp(...): (line 353)
    # Processing the call arguments (line 353)
    # Getting the type of 'exp_fun' (line 353)
    exp_fun_40992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 28), 'exp_fun', False)
    # Getting the type of 'exp_bc' (line 353)
    exp_bc_40993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 37), 'exp_bc', False)
    # Getting the type of 'x' (line 353)
    x_40994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 45), 'x', False)
    # Getting the type of 'y' (line 353)
    y_40995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 48), 'y', False)
    # Processing the call keyword arguments (line 353)
    # Getting the type of 'fun_jac' (line 353)
    fun_jac_40996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 59), 'fun_jac', False)
    keyword_40997 = fun_jac_40996
    # Getting the type of 'bc_jac' (line 354)
    bc_jac_40998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 35), 'bc_jac', False)
    keyword_40999 = bc_jac_40998
    kwargs_41000 = {'bc_jac': keyword_40999, 'fun_jac': keyword_40997}
    # Getting the type of 'solve_bvp' (line 353)
    solve_bvp_40991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 18), 'solve_bvp', False)
    # Calling solve_bvp(args, kwargs) (line 353)
    solve_bvp_call_result_41001 = invoke(stypy.reporting.localization.Localization(__file__, 353, 18), solve_bvp_40991, *[exp_fun_40992, exp_bc_40993, x_40994, y_40995], **kwargs_41000)
    
    # Assigning a type to the variable 'sol' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'sol', solve_bvp_call_result_41001)
    
    # Call to assert_equal(...): (line 356)
    # Processing the call arguments (line 356)
    # Getting the type of 'sol' (line 356)
    sol_41003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 25), 'sol', False)
    # Obtaining the member 'status' of a type (line 356)
    status_41004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 25), sol_41003, 'status')
    int_41005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 37), 'int')
    # Processing the call keyword arguments (line 356)
    kwargs_41006 = {}
    # Getting the type of 'assert_equal' (line 356)
    assert_equal_41002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 356)
    assert_equal_call_result_41007 = invoke(stypy.reporting.localization.Localization(__file__, 356, 12), assert_equal_41002, *[status_41004, int_41005], **kwargs_41006)
    
    
    # Call to assert_(...): (line 357)
    # Processing the call arguments (line 357)
    # Getting the type of 'sol' (line 357)
    sol_41009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'sol', False)
    # Obtaining the member 'success' of a type (line 357)
    success_41010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 20), sol_41009, 'success')
    # Processing the call keyword arguments (line 357)
    kwargs_41011 = {}
    # Getting the type of 'assert_' (line 357)
    assert__41008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 357)
    assert__call_result_41012 = invoke(stypy.reporting.localization.Localization(__file__, 357, 12), assert__41008, *[success_41010], **kwargs_41011)
    
    
    # Call to assert_equal(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of 'sol' (line 359)
    sol_41014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 25), 'sol', False)
    # Obtaining the member 'x' of a type (line 359)
    x_41015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 25), sol_41014, 'x')
    # Obtaining the member 'size' of a type (line 359)
    size_41016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 25), x_41015, 'size')
    int_41017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 37), 'int')
    # Processing the call keyword arguments (line 359)
    kwargs_41018 = {}
    # Getting the type of 'assert_equal' (line 359)
    assert_equal_41013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 359)
    assert_equal_call_result_41019 = invoke(stypy.reporting.localization.Localization(__file__, 359, 12), assert_equal_41013, *[size_41016, int_41017], **kwargs_41018)
    
    
    # Assigning a Call to a Name (line 361):
    
    # Assigning a Call to a Name (line 361):
    
    # Call to sol(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'x_test' (line 361)
    x_test_41022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 31), 'x_test', False)
    # Processing the call keyword arguments (line 361)
    kwargs_41023 = {}
    # Getting the type of 'sol' (line 361)
    sol_41020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 23), 'sol', False)
    # Obtaining the member 'sol' of a type (line 361)
    sol_41021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 23), sol_41020, 'sol')
    # Calling sol(args, kwargs) (line 361)
    sol_call_result_41024 = invoke(stypy.reporting.localization.Localization(__file__, 361, 23), sol_41021, *[x_test_41022], **kwargs_41023)
    
    # Assigning a type to the variable 'sol_test' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'sol_test', sol_call_result_41024)
    
    # Call to assert_allclose(...): (line 363)
    # Processing the call arguments (line 363)
    
    # Obtaining the type of the subscript
    int_41026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 37), 'int')
    # Getting the type of 'sol_test' (line 363)
    sol_test_41027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 28), 'sol_test', False)
    # Obtaining the member '__getitem__' of a type (line 363)
    getitem___41028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 28), sol_test_41027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 363)
    subscript_call_result_41029 = invoke(stypy.reporting.localization.Localization(__file__, 363, 28), getitem___41028, int_41026)
    
    
    # Call to exp_sol(...): (line 363)
    # Processing the call arguments (line 363)
    # Getting the type of 'x_test' (line 363)
    x_test_41031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 49), 'x_test', False)
    # Processing the call keyword arguments (line 363)
    kwargs_41032 = {}
    # Getting the type of 'exp_sol' (line 363)
    exp_sol_41030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 41), 'exp_sol', False)
    # Calling exp_sol(args, kwargs) (line 363)
    exp_sol_call_result_41033 = invoke(stypy.reporting.localization.Localization(__file__, 363, 41), exp_sol_41030, *[x_test_41031], **kwargs_41032)
    
    # Processing the call keyword arguments (line 363)
    float_41034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 63), 'float')
    keyword_41035 = float_41034
    kwargs_41036 = {'atol': keyword_41035}
    # Getting the type of 'assert_allclose' (line 363)
    assert_allclose_41025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 363)
    assert_allclose_call_result_41037 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), assert_allclose_41025, *[subscript_call_result_41029, exp_sol_call_result_41033], **kwargs_41036)
    
    
    # Assigning a Call to a Name (line 365):
    
    # Assigning a Call to a Name (line 365):
    
    # Call to exp_fun(...): (line 365)
    # Processing the call arguments (line 365)
    # Getting the type of 'x_test' (line 365)
    x_test_41039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 29), 'x_test', False)
    # Getting the type of 'sol_test' (line 365)
    sol_test_41040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 37), 'sol_test', False)
    # Processing the call keyword arguments (line 365)
    kwargs_41041 = {}
    # Getting the type of 'exp_fun' (line 365)
    exp_fun_41038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 21), 'exp_fun', False)
    # Calling exp_fun(args, kwargs) (line 365)
    exp_fun_call_result_41042 = invoke(stypy.reporting.localization.Localization(__file__, 365, 21), exp_fun_41038, *[x_test_41039, sol_test_41040], **kwargs_41041)
    
    # Assigning a type to the variable 'f_test' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'f_test', exp_fun_call_result_41042)
    
    # Assigning a BinOp to a Name (line 366):
    
    # Assigning a BinOp to a Name (line 366):
    
    # Call to sol(...): (line 366)
    # Processing the call arguments (line 366)
    # Getting the type of 'x_test' (line 366)
    x_test_41045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 24), 'x_test', False)
    int_41046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 32), 'int')
    # Processing the call keyword arguments (line 366)
    kwargs_41047 = {}
    # Getting the type of 'sol' (line 366)
    sol_41043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 'sol', False)
    # Obtaining the member 'sol' of a type (line 366)
    sol_41044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 16), sol_41043, 'sol')
    # Calling sol(args, kwargs) (line 366)
    sol_call_result_41048 = invoke(stypy.reporting.localization.Localization(__file__, 366, 16), sol_41044, *[x_test_41045, int_41046], **kwargs_41047)
    
    # Getting the type of 'f_test' (line 366)
    f_test_41049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 37), 'f_test')
    # Applying the binary operator '-' (line 366)
    result_sub_41050 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 16), '-', sol_call_result_41048, f_test_41049)
    
    # Assigning a type to the variable 'r' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'r', result_sub_41050)
    
    # Assigning a BinOp to a Name (line 367):
    
    # Assigning a BinOp to a Name (line 367):
    # Getting the type of 'r' (line 367)
    r_41051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 22), 'r')
    int_41052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 27), 'int')
    
    # Call to abs(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'f_test' (line 367)
    f_test_41055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 38), 'f_test', False)
    # Processing the call keyword arguments (line 367)
    kwargs_41056 = {}
    # Getting the type of 'np' (line 367)
    np_41053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 31), 'np', False)
    # Obtaining the member 'abs' of a type (line 367)
    abs_41054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 31), np_41053, 'abs')
    # Calling abs(args, kwargs) (line 367)
    abs_call_result_41057 = invoke(stypy.reporting.localization.Localization(__file__, 367, 31), abs_41054, *[f_test_41055], **kwargs_41056)
    
    # Applying the binary operator '+' (line 367)
    result_add_41058 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 27), '+', int_41052, abs_call_result_41057)
    
    # Applying the binary operator 'div' (line 367)
    result_div_41059 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 22), 'div', r_41051, result_add_41058)
    
    # Assigning a type to the variable 'rel_res' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'rel_res', result_div_41059)
    
    # Assigning a BinOp to a Name (line 368):
    
    # Assigning a BinOp to a Name (line 368):
    
    # Call to sum(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'rel_res' (line 368)
    rel_res_41062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 30), 'rel_res', False)
    int_41063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 39), 'int')
    # Applying the binary operator '**' (line 368)
    result_pow_41064 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 30), '**', rel_res_41062, int_41063)
    
    # Processing the call keyword arguments (line 368)
    int_41065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 47), 'int')
    keyword_41066 = int_41065
    kwargs_41067 = {'axis': keyword_41066}
    # Getting the type of 'np' (line 368)
    np_41060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 23), 'np', False)
    # Obtaining the member 'sum' of a type (line 368)
    sum_41061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 23), np_41060, 'sum')
    # Calling sum(args, kwargs) (line 368)
    sum_call_result_41068 = invoke(stypy.reporting.localization.Localization(__file__, 368, 23), sum_41061, *[result_pow_41064], **kwargs_41067)
    
    float_41069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 51), 'float')
    # Applying the binary operator '**' (line 368)
    result_pow_41070 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 23), '**', sum_call_result_41068, float_41069)
    
    # Assigning a type to the variable 'norm_res' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'norm_res', result_pow_41070)
    
    # Call to assert_(...): (line 369)
    # Processing the call arguments (line 369)
    
    # Call to all(...): (line 369)
    # Processing the call arguments (line 369)
    
    # Getting the type of 'norm_res' (line 369)
    norm_res_41074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 27), 'norm_res', False)
    float_41075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 38), 'float')
    # Applying the binary operator '<' (line 369)
    result_lt_41076 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 27), '<', norm_res_41074, float_41075)
    
    # Processing the call keyword arguments (line 369)
    kwargs_41077 = {}
    # Getting the type of 'np' (line 369)
    np_41072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 20), 'np', False)
    # Obtaining the member 'all' of a type (line 369)
    all_41073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 20), np_41072, 'all')
    # Calling all(args, kwargs) (line 369)
    all_call_result_41078 = invoke(stypy.reporting.localization.Localization(__file__, 369, 20), all_41073, *[result_lt_41076], **kwargs_41077)
    
    # Processing the call keyword arguments (line 369)
    kwargs_41079 = {}
    # Getting the type of 'assert_' (line 369)
    assert__41071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 369)
    assert__call_result_41080 = invoke(stypy.reporting.localization.Localization(__file__, 369, 12), assert__41071, *[all_call_result_41078], **kwargs_41079)
    
    
    # Call to assert_(...): (line 371)
    # Processing the call arguments (line 371)
    
    # Call to all(...): (line 371)
    # Processing the call arguments (line 371)
    
    # Getting the type of 'sol' (line 371)
    sol_41084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 27), 'sol', False)
    # Obtaining the member 'rms_residuals' of a type (line 371)
    rms_residuals_41085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 27), sol_41084, 'rms_residuals')
    float_41086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 47), 'float')
    # Applying the binary operator '<' (line 371)
    result_lt_41087 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 27), '<', rms_residuals_41085, float_41086)
    
    # Processing the call keyword arguments (line 371)
    kwargs_41088 = {}
    # Getting the type of 'np' (line 371)
    np_41082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 20), 'np', False)
    # Obtaining the member 'all' of a type (line 371)
    all_41083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 20), np_41082, 'all')
    # Calling all(args, kwargs) (line 371)
    all_call_result_41089 = invoke(stypy.reporting.localization.Localization(__file__, 371, 20), all_41083, *[result_lt_41087], **kwargs_41088)
    
    # Processing the call keyword arguments (line 371)
    kwargs_41090 = {}
    # Getting the type of 'assert_' (line 371)
    assert__41081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 371)
    assert__call_result_41091 = invoke(stypy.reporting.localization.Localization(__file__, 371, 12), assert__41081, *[all_call_result_41089], **kwargs_41090)
    
    
    # Call to assert_allclose(...): (line 372)
    # Processing the call arguments (line 372)
    
    # Call to sol(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'sol' (line 372)
    sol_41095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 36), 'sol', False)
    # Obtaining the member 'x' of a type (line 372)
    x_41096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 36), sol_41095, 'x')
    # Processing the call keyword arguments (line 372)
    kwargs_41097 = {}
    # Getting the type of 'sol' (line 372)
    sol_41093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 28), 'sol', False)
    # Obtaining the member 'sol' of a type (line 372)
    sol_41094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 28), sol_41093, 'sol')
    # Calling sol(args, kwargs) (line 372)
    sol_call_result_41098 = invoke(stypy.reporting.localization.Localization(__file__, 372, 28), sol_41094, *[x_41096], **kwargs_41097)
    
    # Getting the type of 'sol' (line 372)
    sol_41099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 44), 'sol', False)
    # Obtaining the member 'y' of a type (line 372)
    y_41100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 44), sol_41099, 'y')
    # Processing the call keyword arguments (line 372)
    float_41101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 56), 'float')
    keyword_41102 = float_41101
    float_41103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 68), 'float')
    keyword_41104 = float_41103
    kwargs_41105 = {'rtol': keyword_41102, 'atol': keyword_41104}
    # Getting the type of 'assert_allclose' (line 372)
    assert_allclose_41092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 372)
    assert_allclose_call_result_41106 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), assert_allclose_41092, *[sol_call_result_41098, y_41100], **kwargs_41105)
    
    
    # Call to assert_allclose(...): (line 373)
    # Processing the call arguments (line 373)
    
    # Call to sol(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of 'sol' (line 373)
    sol_41110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 36), 'sol', False)
    # Obtaining the member 'x' of a type (line 373)
    x_41111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 36), sol_41110, 'x')
    int_41112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 43), 'int')
    # Processing the call keyword arguments (line 373)
    kwargs_41113 = {}
    # Getting the type of 'sol' (line 373)
    sol_41108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 28), 'sol', False)
    # Obtaining the member 'sol' of a type (line 373)
    sol_41109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 28), sol_41108, 'sol')
    # Calling sol(args, kwargs) (line 373)
    sol_call_result_41114 = invoke(stypy.reporting.localization.Localization(__file__, 373, 28), sol_41109, *[x_41111, int_41112], **kwargs_41113)
    
    # Getting the type of 'sol' (line 373)
    sol_41115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 47), 'sol', False)
    # Obtaining the member 'yp' of a type (line 373)
    yp_41116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 47), sol_41115, 'yp')
    # Processing the call keyword arguments (line 373)
    float_41117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 60), 'float')
    keyword_41118 = float_41117
    float_41119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 72), 'float')
    keyword_41120 = float_41119
    kwargs_41121 = {'rtol': keyword_41118, 'atol': keyword_41120}
    # Getting the type of 'assert_allclose' (line 373)
    assert_allclose_41107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 373)
    assert_allclose_call_result_41122 = invoke(stypy.reporting.localization.Localization(__file__, 373, 12), assert_allclose_41107, *[sol_call_result_41114, yp_41116], **kwargs_41121)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_no_params(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_no_params' in the type store
    # Getting the type of 'stypy_return_type' (line 347)
    stypy_return_type_41123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_41123)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_no_params'
    return stypy_return_type_41123

# Assigning a type to the variable 'test_no_params' (line 347)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 0), 'test_no_params', test_no_params)

@norecursion
def test_with_params(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_with_params'
    module_type_store = module_type_store.open_function_context('test_with_params', 376, 0, False)
    
    # Passed parameters checking function
    test_with_params.stypy_localization = localization
    test_with_params.stypy_type_of_self = None
    test_with_params.stypy_type_store = module_type_store
    test_with_params.stypy_function_name = 'test_with_params'
    test_with_params.stypy_param_names_list = []
    test_with_params.stypy_varargs_param_name = None
    test_with_params.stypy_kwargs_param_name = None
    test_with_params.stypy_call_defaults = defaults
    test_with_params.stypy_call_varargs = varargs
    test_with_params.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_with_params', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_with_params', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_with_params(...)' code ##################

    
    # Assigning a Call to a Name (line 377):
    
    # Assigning a Call to a Name (line 377):
    
    # Call to linspace(...): (line 377)
    # Processing the call arguments (line 377)
    int_41126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 20), 'int')
    # Getting the type of 'np' (line 377)
    np_41127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 23), 'np', False)
    # Obtaining the member 'pi' of a type (line 377)
    pi_41128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 23), np_41127, 'pi')
    int_41129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 30), 'int')
    # Processing the call keyword arguments (line 377)
    kwargs_41130 = {}
    # Getting the type of 'np' (line 377)
    np_41124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 377)
    linspace_41125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 8), np_41124, 'linspace')
    # Calling linspace(args, kwargs) (line 377)
    linspace_call_result_41131 = invoke(stypy.reporting.localization.Localization(__file__, 377, 8), linspace_41125, *[int_41126, pi_41128, int_41129], **kwargs_41130)
    
    # Assigning a type to the variable 'x' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'x', linspace_call_result_41131)
    
    # Assigning a Call to a Name (line 378):
    
    # Assigning a Call to a Name (line 378):
    
    # Call to linspace(...): (line 378)
    # Processing the call arguments (line 378)
    int_41134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 25), 'int')
    # Getting the type of 'np' (line 378)
    np_41135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 28), 'np', False)
    # Obtaining the member 'pi' of a type (line 378)
    pi_41136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 28), np_41135, 'pi')
    int_41137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 35), 'int')
    # Processing the call keyword arguments (line 378)
    kwargs_41138 = {}
    # Getting the type of 'np' (line 378)
    np_41132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 13), 'np', False)
    # Obtaining the member 'linspace' of a type (line 378)
    linspace_41133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 13), np_41132, 'linspace')
    # Calling linspace(args, kwargs) (line 378)
    linspace_call_result_41139 = invoke(stypy.reporting.localization.Localization(__file__, 378, 13), linspace_41133, *[int_41134, pi_41136, int_41137], **kwargs_41138)
    
    # Assigning a type to the variable 'x_test' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'x_test', linspace_call_result_41139)
    
    # Assigning a Call to a Name (line 379):
    
    # Assigning a Call to a Name (line 379):
    
    # Call to ones(...): (line 379)
    # Processing the call arguments (line 379)
    
    # Obtaining an instance of the builtin type 'tuple' (line 379)
    tuple_41142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 379)
    # Adding element type (line 379)
    int_41143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 17), tuple_41142, int_41143)
    # Adding element type (line 379)
    
    # Obtaining the type of the subscript
    int_41144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 28), 'int')
    # Getting the type of 'x' (line 379)
    x_41145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 20), 'x', False)
    # Obtaining the member 'shape' of a type (line 379)
    shape_41146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 20), x_41145, 'shape')
    # Obtaining the member '__getitem__' of a type (line 379)
    getitem___41147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 20), shape_41146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 379)
    subscript_call_result_41148 = invoke(stypy.reporting.localization.Localization(__file__, 379, 20), getitem___41147, int_41144)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 17), tuple_41142, subscript_call_result_41148)
    
    # Processing the call keyword arguments (line 379)
    kwargs_41149 = {}
    # Getting the type of 'np' (line 379)
    np_41140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 379)
    ones_41141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), np_41140, 'ones')
    # Calling ones(args, kwargs) (line 379)
    ones_call_result_41150 = invoke(stypy.reporting.localization.Localization(__file__, 379, 8), ones_41141, *[tuple_41142], **kwargs_41149)
    
    # Assigning a type to the variable 'y' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'y', ones_call_result_41150)
    
    
    # Obtaining an instance of the builtin type 'list' (line 381)
    list_41151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 381)
    # Adding element type (line 381)
    # Getting the type of 'None' (line 381)
    None_41152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 20), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 19), list_41151, None_41152)
    # Adding element type (line 381)
    # Getting the type of 'sl_fun_jac' (line 381)
    sl_fun_jac_41153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 26), 'sl_fun_jac')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 19), list_41151, sl_fun_jac_41153)
    
    # Testing the type of a for loop iterable (line 381)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 381, 4), list_41151)
    # Getting the type of the for loop variable (line 381)
    for_loop_var_41154 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 381, 4), list_41151)
    # Assigning a type to the variable 'fun_jac' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'fun_jac', for_loop_var_41154)
    # SSA begins for a for statement (line 381)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'list' (line 382)
    list_41155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 382)
    # Adding element type (line 382)
    # Getting the type of 'None' (line 382)
    None_41156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 23), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 22), list_41155, None_41156)
    # Adding element type (line 382)
    # Getting the type of 'sl_bc_jac' (line 382)
    sl_bc_jac_41157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 29), 'sl_bc_jac')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 22), list_41155, sl_bc_jac_41157)
    
    # Testing the type of a for loop iterable (line 382)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 382, 8), list_41155)
    # Getting the type of the for loop variable (line 382)
    for_loop_var_41158 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 382, 8), list_41155)
    # Assigning a type to the variable 'bc_jac' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'bc_jac', for_loop_var_41158)
    # SSA begins for a for statement (line 382)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 383):
    
    # Assigning a Call to a Name (line 383):
    
    # Call to solve_bvp(...): (line 383)
    # Processing the call arguments (line 383)
    # Getting the type of 'sl_fun' (line 383)
    sl_fun_41160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 28), 'sl_fun', False)
    # Getting the type of 'sl_bc' (line 383)
    sl_bc_41161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 36), 'sl_bc', False)
    # Getting the type of 'x' (line 383)
    x_41162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 43), 'x', False)
    # Getting the type of 'y' (line 383)
    y_41163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 46), 'y', False)
    # Processing the call keyword arguments (line 383)
    
    # Obtaining an instance of the builtin type 'list' (line 383)
    list_41164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 383)
    # Adding element type (line 383)
    float_41165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 52), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 51), list_41164, float_41165)
    
    keyword_41166 = list_41164
    # Getting the type of 'fun_jac' (line 383)
    fun_jac_41167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 66), 'fun_jac', False)
    keyword_41168 = fun_jac_41167
    # Getting the type of 'bc_jac' (line 384)
    bc_jac_41169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 35), 'bc_jac', False)
    keyword_41170 = bc_jac_41169
    kwargs_41171 = {'bc_jac': keyword_41170, 'p': keyword_41166, 'fun_jac': keyword_41168}
    # Getting the type of 'solve_bvp' (line 383)
    solve_bvp_41159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 18), 'solve_bvp', False)
    # Calling solve_bvp(args, kwargs) (line 383)
    solve_bvp_call_result_41172 = invoke(stypy.reporting.localization.Localization(__file__, 383, 18), solve_bvp_41159, *[sl_fun_41160, sl_bc_41161, x_41162, y_41163], **kwargs_41171)
    
    # Assigning a type to the variable 'sol' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'sol', solve_bvp_call_result_41172)
    
    # Call to assert_equal(...): (line 386)
    # Processing the call arguments (line 386)
    # Getting the type of 'sol' (line 386)
    sol_41174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 25), 'sol', False)
    # Obtaining the member 'status' of a type (line 386)
    status_41175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 25), sol_41174, 'status')
    int_41176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 37), 'int')
    # Processing the call keyword arguments (line 386)
    kwargs_41177 = {}
    # Getting the type of 'assert_equal' (line 386)
    assert_equal_41173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 386)
    assert_equal_call_result_41178 = invoke(stypy.reporting.localization.Localization(__file__, 386, 12), assert_equal_41173, *[status_41175, int_41176], **kwargs_41177)
    
    
    # Call to assert_(...): (line 387)
    # Processing the call arguments (line 387)
    # Getting the type of 'sol' (line 387)
    sol_41180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 20), 'sol', False)
    # Obtaining the member 'success' of a type (line 387)
    success_41181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 20), sol_41180, 'success')
    # Processing the call keyword arguments (line 387)
    kwargs_41182 = {}
    # Getting the type of 'assert_' (line 387)
    assert__41179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 387)
    assert__call_result_41183 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), assert__41179, *[success_41181], **kwargs_41182)
    
    
    # Call to assert_(...): (line 389)
    # Processing the call arguments (line 389)
    
    # Getting the type of 'sol' (line 389)
    sol_41185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 20), 'sol', False)
    # Obtaining the member 'x' of a type (line 389)
    x_41186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 20), sol_41185, 'x')
    # Obtaining the member 'size' of a type (line 389)
    size_41187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 20), x_41186, 'size')
    int_41188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 33), 'int')
    # Applying the binary operator '<' (line 389)
    result_lt_41189 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 20), '<', size_41187, int_41188)
    
    # Processing the call keyword arguments (line 389)
    kwargs_41190 = {}
    # Getting the type of 'assert_' (line 389)
    assert__41184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 389)
    assert__call_result_41191 = invoke(stypy.reporting.localization.Localization(__file__, 389, 12), assert__41184, *[result_lt_41189], **kwargs_41190)
    
    
    # Call to assert_allclose(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'sol' (line 391)
    sol_41193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 28), 'sol', False)
    # Obtaining the member 'p' of a type (line 391)
    p_41194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 28), sol_41193, 'p')
    
    # Obtaining an instance of the builtin type 'list' (line 391)
    list_41195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 391)
    # Adding element type (line 391)
    int_41196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 35), list_41195, int_41196)
    
    # Processing the call keyword arguments (line 391)
    float_41197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 45), 'float')
    keyword_41198 = float_41197
    kwargs_41199 = {'rtol': keyword_41198}
    # Getting the type of 'assert_allclose' (line 391)
    assert_allclose_41192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 391)
    assert_allclose_call_result_41200 = invoke(stypy.reporting.localization.Localization(__file__, 391, 12), assert_allclose_41192, *[p_41194, list_41195], **kwargs_41199)
    
    
    # Assigning a Call to a Name (line 393):
    
    # Assigning a Call to a Name (line 393):
    
    # Call to sol(...): (line 393)
    # Processing the call arguments (line 393)
    # Getting the type of 'x_test' (line 393)
    x_test_41203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 31), 'x_test', False)
    # Processing the call keyword arguments (line 393)
    kwargs_41204 = {}
    # Getting the type of 'sol' (line 393)
    sol_41201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 23), 'sol', False)
    # Obtaining the member 'sol' of a type (line 393)
    sol_41202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 23), sol_41201, 'sol')
    # Calling sol(args, kwargs) (line 393)
    sol_call_result_41205 = invoke(stypy.reporting.localization.Localization(__file__, 393, 23), sol_41202, *[x_test_41203], **kwargs_41204)
    
    # Assigning a type to the variable 'sol_test' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'sol_test', sol_call_result_41205)
    
    # Call to assert_allclose(...): (line 395)
    # Processing the call arguments (line 395)
    
    # Obtaining the type of the subscript
    int_41207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 37), 'int')
    # Getting the type of 'sol_test' (line 395)
    sol_test_41208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 28), 'sol_test', False)
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___41209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 28), sol_test_41208, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 395)
    subscript_call_result_41210 = invoke(stypy.reporting.localization.Localization(__file__, 395, 28), getitem___41209, int_41207)
    
    
    # Call to sl_sol(...): (line 395)
    # Processing the call arguments (line 395)
    # Getting the type of 'x_test' (line 395)
    x_test_41212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 48), 'x_test', False)
    
    # Obtaining an instance of the builtin type 'list' (line 395)
    list_41213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 56), 'list')
    # Adding type elements to the builtin type 'list' instance (line 395)
    # Adding element type (line 395)
    int_41214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 56), list_41213, int_41214)
    
    # Processing the call keyword arguments (line 395)
    kwargs_41215 = {}
    # Getting the type of 'sl_sol' (line 395)
    sl_sol_41211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 41), 'sl_sol', False)
    # Calling sl_sol(args, kwargs) (line 395)
    sl_sol_call_result_41216 = invoke(stypy.reporting.localization.Localization(__file__, 395, 41), sl_sol_41211, *[x_test_41212, list_41213], **kwargs_41215)
    
    # Processing the call keyword arguments (line 395)
    float_41217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 33), 'float')
    keyword_41218 = float_41217
    float_41219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 44), 'float')
    keyword_41220 = float_41219
    kwargs_41221 = {'rtol': keyword_41218, 'atol': keyword_41220}
    # Getting the type of 'assert_allclose' (line 395)
    assert_allclose_41206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 395)
    assert_allclose_call_result_41222 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), assert_allclose_41206, *[subscript_call_result_41210, sl_sol_call_result_41216], **kwargs_41221)
    
    
    # Assigning a Call to a Name (line 398):
    
    # Assigning a Call to a Name (line 398):
    
    # Call to sl_fun(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'x_test' (line 398)
    x_test_41224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 28), 'x_test', False)
    # Getting the type of 'sol_test' (line 398)
    sol_test_41225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 36), 'sol_test', False)
    
    # Obtaining an instance of the builtin type 'list' (line 398)
    list_41226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 398)
    # Adding element type (line 398)
    int_41227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 46), list_41226, int_41227)
    
    # Processing the call keyword arguments (line 398)
    kwargs_41228 = {}
    # Getting the type of 'sl_fun' (line 398)
    sl_fun_41223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 21), 'sl_fun', False)
    # Calling sl_fun(args, kwargs) (line 398)
    sl_fun_call_result_41229 = invoke(stypy.reporting.localization.Localization(__file__, 398, 21), sl_fun_41223, *[x_test_41224, sol_test_41225, list_41226], **kwargs_41228)
    
    # Assigning a type to the variable 'f_test' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'f_test', sl_fun_call_result_41229)
    
    # Assigning a BinOp to a Name (line 399):
    
    # Assigning a BinOp to a Name (line 399):
    
    # Call to sol(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'x_test' (line 399)
    x_test_41232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 24), 'x_test', False)
    int_41233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 32), 'int')
    # Processing the call keyword arguments (line 399)
    kwargs_41234 = {}
    # Getting the type of 'sol' (line 399)
    sol_41230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 'sol', False)
    # Obtaining the member 'sol' of a type (line 399)
    sol_41231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 16), sol_41230, 'sol')
    # Calling sol(args, kwargs) (line 399)
    sol_call_result_41235 = invoke(stypy.reporting.localization.Localization(__file__, 399, 16), sol_41231, *[x_test_41232, int_41233], **kwargs_41234)
    
    # Getting the type of 'f_test' (line 399)
    f_test_41236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 37), 'f_test')
    # Applying the binary operator '-' (line 399)
    result_sub_41237 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 16), '-', sol_call_result_41235, f_test_41236)
    
    # Assigning a type to the variable 'r' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'r', result_sub_41237)
    
    # Assigning a BinOp to a Name (line 400):
    
    # Assigning a BinOp to a Name (line 400):
    # Getting the type of 'r' (line 400)
    r_41238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 22), 'r')
    int_41239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 27), 'int')
    
    # Call to abs(...): (line 400)
    # Processing the call arguments (line 400)
    # Getting the type of 'f_test' (line 400)
    f_test_41242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 38), 'f_test', False)
    # Processing the call keyword arguments (line 400)
    kwargs_41243 = {}
    # Getting the type of 'np' (line 400)
    np_41240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 31), 'np', False)
    # Obtaining the member 'abs' of a type (line 400)
    abs_41241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 31), np_41240, 'abs')
    # Calling abs(args, kwargs) (line 400)
    abs_call_result_41244 = invoke(stypy.reporting.localization.Localization(__file__, 400, 31), abs_41241, *[f_test_41242], **kwargs_41243)
    
    # Applying the binary operator '+' (line 400)
    result_add_41245 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 27), '+', int_41239, abs_call_result_41244)
    
    # Applying the binary operator 'div' (line 400)
    result_div_41246 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 22), 'div', r_41238, result_add_41245)
    
    # Assigning a type to the variable 'rel_res' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'rel_res', result_div_41246)
    
    # Assigning a BinOp to a Name (line 401):
    
    # Assigning a BinOp to a Name (line 401):
    
    # Call to sum(...): (line 401)
    # Processing the call arguments (line 401)
    # Getting the type of 'rel_res' (line 401)
    rel_res_41249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 30), 'rel_res', False)
    int_41250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 41), 'int')
    # Applying the binary operator '**' (line 401)
    result_pow_41251 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 30), '**', rel_res_41249, int_41250)
    
    # Processing the call keyword arguments (line 401)
    int_41252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 49), 'int')
    keyword_41253 = int_41252
    kwargs_41254 = {'axis': keyword_41253}
    # Getting the type of 'np' (line 401)
    np_41247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 23), 'np', False)
    # Obtaining the member 'sum' of a type (line 401)
    sum_41248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 23), np_41247, 'sum')
    # Calling sum(args, kwargs) (line 401)
    sum_call_result_41255 = invoke(stypy.reporting.localization.Localization(__file__, 401, 23), sum_41248, *[result_pow_41251], **kwargs_41254)
    
    float_41256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 55), 'float')
    # Applying the binary operator '**' (line 401)
    result_pow_41257 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 23), '**', sum_call_result_41255, float_41256)
    
    # Assigning a type to the variable 'norm_res' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'norm_res', result_pow_41257)
    
    # Call to assert_(...): (line 402)
    # Processing the call arguments (line 402)
    
    # Call to all(...): (line 402)
    # Processing the call arguments (line 402)
    
    # Getting the type of 'norm_res' (line 402)
    norm_res_41261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 27), 'norm_res', False)
    float_41262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 38), 'float')
    # Applying the binary operator '<' (line 402)
    result_lt_41263 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 27), '<', norm_res_41261, float_41262)
    
    # Processing the call keyword arguments (line 402)
    kwargs_41264 = {}
    # Getting the type of 'np' (line 402)
    np_41259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'np', False)
    # Obtaining the member 'all' of a type (line 402)
    all_41260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 20), np_41259, 'all')
    # Calling all(args, kwargs) (line 402)
    all_call_result_41265 = invoke(stypy.reporting.localization.Localization(__file__, 402, 20), all_41260, *[result_lt_41263], **kwargs_41264)
    
    # Processing the call keyword arguments (line 402)
    kwargs_41266 = {}
    # Getting the type of 'assert_' (line 402)
    assert__41258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 402)
    assert__call_result_41267 = invoke(stypy.reporting.localization.Localization(__file__, 402, 12), assert__41258, *[all_call_result_41265], **kwargs_41266)
    
    
    # Call to assert_(...): (line 404)
    # Processing the call arguments (line 404)
    
    # Call to all(...): (line 404)
    # Processing the call arguments (line 404)
    
    # Getting the type of 'sol' (line 404)
    sol_41271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 27), 'sol', False)
    # Obtaining the member 'rms_residuals' of a type (line 404)
    rms_residuals_41272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 27), sol_41271, 'rms_residuals')
    float_41273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 47), 'float')
    # Applying the binary operator '<' (line 404)
    result_lt_41274 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 27), '<', rms_residuals_41272, float_41273)
    
    # Processing the call keyword arguments (line 404)
    kwargs_41275 = {}
    # Getting the type of 'np' (line 404)
    np_41269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 20), 'np', False)
    # Obtaining the member 'all' of a type (line 404)
    all_41270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 20), np_41269, 'all')
    # Calling all(args, kwargs) (line 404)
    all_call_result_41276 = invoke(stypy.reporting.localization.Localization(__file__, 404, 20), all_41270, *[result_lt_41274], **kwargs_41275)
    
    # Processing the call keyword arguments (line 404)
    kwargs_41277 = {}
    # Getting the type of 'assert_' (line 404)
    assert__41268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 404)
    assert__call_result_41278 = invoke(stypy.reporting.localization.Localization(__file__, 404, 12), assert__41268, *[all_call_result_41276], **kwargs_41277)
    
    
    # Call to assert_allclose(...): (line 405)
    # Processing the call arguments (line 405)
    
    # Call to sol(...): (line 405)
    # Processing the call arguments (line 405)
    # Getting the type of 'sol' (line 405)
    sol_41282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 36), 'sol', False)
    # Obtaining the member 'x' of a type (line 405)
    x_41283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 36), sol_41282, 'x')
    # Processing the call keyword arguments (line 405)
    kwargs_41284 = {}
    # Getting the type of 'sol' (line 405)
    sol_41280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 28), 'sol', False)
    # Obtaining the member 'sol' of a type (line 405)
    sol_41281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 28), sol_41280, 'sol')
    # Calling sol(args, kwargs) (line 405)
    sol_call_result_41285 = invoke(stypy.reporting.localization.Localization(__file__, 405, 28), sol_41281, *[x_41283], **kwargs_41284)
    
    # Getting the type of 'sol' (line 405)
    sol_41286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 44), 'sol', False)
    # Obtaining the member 'y' of a type (line 405)
    y_41287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 44), sol_41286, 'y')
    # Processing the call keyword arguments (line 405)
    float_41288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 56), 'float')
    keyword_41289 = float_41288
    float_41290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 68), 'float')
    keyword_41291 = float_41290
    kwargs_41292 = {'rtol': keyword_41289, 'atol': keyword_41291}
    # Getting the type of 'assert_allclose' (line 405)
    assert_allclose_41279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 405)
    assert_allclose_call_result_41293 = invoke(stypy.reporting.localization.Localization(__file__, 405, 12), assert_allclose_41279, *[sol_call_result_41285, y_41287], **kwargs_41292)
    
    
    # Call to assert_allclose(...): (line 406)
    # Processing the call arguments (line 406)
    
    # Call to sol(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'sol' (line 406)
    sol_41297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 36), 'sol', False)
    # Obtaining the member 'x' of a type (line 406)
    x_41298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 36), sol_41297, 'x')
    int_41299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 43), 'int')
    # Processing the call keyword arguments (line 406)
    kwargs_41300 = {}
    # Getting the type of 'sol' (line 406)
    sol_41295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 28), 'sol', False)
    # Obtaining the member 'sol' of a type (line 406)
    sol_41296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 28), sol_41295, 'sol')
    # Calling sol(args, kwargs) (line 406)
    sol_call_result_41301 = invoke(stypy.reporting.localization.Localization(__file__, 406, 28), sol_41296, *[x_41298, int_41299], **kwargs_41300)
    
    # Getting the type of 'sol' (line 406)
    sol_41302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 47), 'sol', False)
    # Obtaining the member 'yp' of a type (line 406)
    yp_41303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 47), sol_41302, 'yp')
    # Processing the call keyword arguments (line 406)
    float_41304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 60), 'float')
    keyword_41305 = float_41304
    float_41306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 72), 'float')
    keyword_41307 = float_41306
    kwargs_41308 = {'rtol': keyword_41305, 'atol': keyword_41307}
    # Getting the type of 'assert_allclose' (line 406)
    assert_allclose_41294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 406)
    assert_allclose_call_result_41309 = invoke(stypy.reporting.localization.Localization(__file__, 406, 12), assert_allclose_41294, *[sol_call_result_41301, yp_41303], **kwargs_41308)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_with_params(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_with_params' in the type store
    # Getting the type of 'stypy_return_type' (line 376)
    stypy_return_type_41310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_41310)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_with_params'
    return stypy_return_type_41310

# Assigning a type to the variable 'test_with_params' (line 376)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 0), 'test_with_params', test_with_params)

@norecursion
def test_singular_term(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_singular_term'
    module_type_store = module_type_store.open_function_context('test_singular_term', 409, 0, False)
    
    # Passed parameters checking function
    test_singular_term.stypy_localization = localization
    test_singular_term.stypy_type_of_self = None
    test_singular_term.stypy_type_store = module_type_store
    test_singular_term.stypy_function_name = 'test_singular_term'
    test_singular_term.stypy_param_names_list = []
    test_singular_term.stypy_varargs_param_name = None
    test_singular_term.stypy_kwargs_param_name = None
    test_singular_term.stypy_call_defaults = defaults
    test_singular_term.stypy_call_varargs = varargs
    test_singular_term.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_singular_term', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_singular_term', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_singular_term(...)' code ##################

    
    # Assigning a Call to a Name (line 410):
    
    # Assigning a Call to a Name (line 410):
    
    # Call to linspace(...): (line 410)
    # Processing the call arguments (line 410)
    int_41313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 20), 'int')
    int_41314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 23), 'int')
    int_41315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 26), 'int')
    # Processing the call keyword arguments (line 410)
    kwargs_41316 = {}
    # Getting the type of 'np' (line 410)
    np_41311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 410)
    linspace_41312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 8), np_41311, 'linspace')
    # Calling linspace(args, kwargs) (line 410)
    linspace_call_result_41317 = invoke(stypy.reporting.localization.Localization(__file__, 410, 8), linspace_41312, *[int_41313, int_41314, int_41315], **kwargs_41316)
    
    # Assigning a type to the variable 'x' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'x', linspace_call_result_41317)
    
    # Assigning a Call to a Name (line 411):
    
    # Assigning a Call to a Name (line 411):
    
    # Call to linspace(...): (line 411)
    # Processing the call arguments (line 411)
    float_41320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 25), 'float')
    int_41321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 31), 'int')
    int_41322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 34), 'int')
    # Processing the call keyword arguments (line 411)
    kwargs_41323 = {}
    # Getting the type of 'np' (line 411)
    np_41318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 13), 'np', False)
    # Obtaining the member 'linspace' of a type (line 411)
    linspace_41319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 13), np_41318, 'linspace')
    # Calling linspace(args, kwargs) (line 411)
    linspace_call_result_41324 = invoke(stypy.reporting.localization.Localization(__file__, 411, 13), linspace_41319, *[float_41320, int_41321, int_41322], **kwargs_41323)
    
    # Assigning a type to the variable 'x_test' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'x_test', linspace_call_result_41324)
    
    # Assigning a Call to a Name (line 412):
    
    # Assigning a Call to a Name (line 412):
    
    # Call to empty(...): (line 412)
    # Processing the call arguments (line 412)
    
    # Obtaining an instance of the builtin type 'tuple' (line 412)
    tuple_41327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 412)
    # Adding element type (line 412)
    int_41328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 18), tuple_41327, int_41328)
    # Adding element type (line 412)
    int_41329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 18), tuple_41327, int_41329)
    
    # Processing the call keyword arguments (line 412)
    kwargs_41330 = {}
    # Getting the type of 'np' (line 412)
    np_41325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 412)
    empty_41326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 8), np_41325, 'empty')
    # Calling empty(args, kwargs) (line 412)
    empty_call_result_41331 = invoke(stypy.reporting.localization.Localization(__file__, 412, 8), empty_41326, *[tuple_41327], **kwargs_41330)
    
    # Assigning a type to the variable 'y' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'y', empty_call_result_41331)
    
    # Assigning a BinOp to a Subscript (line 413):
    
    # Assigning a BinOp to a Subscript (line 413):
    int_41332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 12), 'int')
    int_41333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 14), 'int')
    # Applying the binary operator 'div' (line 413)
    result_div_41334 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 12), 'div', int_41332, int_41333)
    
    float_41335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 18), 'float')
    # Applying the binary operator '**' (line 413)
    result_pow_41336 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 11), '**', result_div_41334, float_41335)
    
    # Getting the type of 'y' (line 413)
    y_41337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'y')
    int_41338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 6), 'int')
    # Storing an element on a container (line 413)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 4), y_41337, (int_41338, result_pow_41336))
    
    # Assigning a Num to a Subscript (line 414):
    
    # Assigning a Num to a Subscript (line 414):
    float_41339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 11), 'float')
    # Getting the type of 'y' (line 414)
    y_41340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'y')
    int_41341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 6), 'int')
    # Storing an element on a container (line 414)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 4), y_41340, (int_41341, float_41339))
    
    # Assigning a Call to a Name (line 415):
    
    # Assigning a Call to a Name (line 415):
    
    # Call to array(...): (line 415)
    # Processing the call arguments (line 415)
    
    # Obtaining an instance of the builtin type 'list' (line 415)
    list_41344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 415)
    # Adding element type (line 415)
    
    # Obtaining an instance of the builtin type 'list' (line 415)
    list_41345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 415)
    # Adding element type (line 415)
    int_41346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 18), list_41345, int_41346)
    # Adding element type (line 415)
    int_41347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 18), list_41345, int_41347)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 17), list_41344, list_41345)
    # Adding element type (line 415)
    
    # Obtaining an instance of the builtin type 'list' (line 415)
    list_41348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 415)
    # Adding element type (line 415)
    int_41349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 26), list_41348, int_41349)
    # Adding element type (line 415)
    int_41350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 26), list_41348, int_41350)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 17), list_41344, list_41348)
    
    # Processing the call keyword arguments (line 415)
    kwargs_41351 = {}
    # Getting the type of 'np' (line 415)
    np_41342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 415)
    array_41343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 8), np_41342, 'array')
    # Calling array(args, kwargs) (line 415)
    array_call_result_41352 = invoke(stypy.reporting.localization.Localization(__file__, 415, 8), array_41343, *[list_41344], **kwargs_41351)
    
    # Assigning a type to the variable 'S' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'S', array_call_result_41352)
    
    
    # Obtaining an instance of the builtin type 'list' (line 417)
    list_41353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 417)
    # Adding element type (line 417)
    # Getting the type of 'None' (line 417)
    None_41354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 20), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 19), list_41353, None_41354)
    # Adding element type (line 417)
    # Getting the type of 'emden_fun_jac' (line 417)
    emden_fun_jac_41355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 26), 'emden_fun_jac')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 19), list_41353, emden_fun_jac_41355)
    
    # Testing the type of a for loop iterable (line 417)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 417, 4), list_41353)
    # Getting the type of the for loop variable (line 417)
    for_loop_var_41356 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 417, 4), list_41353)
    # Assigning a type to the variable 'fun_jac' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'fun_jac', for_loop_var_41356)
    # SSA begins for a for statement (line 417)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'list' (line 418)
    list_41357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 418)
    # Adding element type (line 418)
    # Getting the type of 'None' (line 418)
    None_41358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 23), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 22), list_41357, None_41358)
    # Adding element type (line 418)
    # Getting the type of 'emden_bc_jac' (line 418)
    emden_bc_jac_41359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 29), 'emden_bc_jac')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 22), list_41357, emden_bc_jac_41359)
    
    # Testing the type of a for loop iterable (line 418)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 418, 8), list_41357)
    # Getting the type of the for loop variable (line 418)
    for_loop_var_41360 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 418, 8), list_41357)
    # Assigning a type to the variable 'bc_jac' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'bc_jac', for_loop_var_41360)
    # SSA begins for a for statement (line 418)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 419):
    
    # Assigning a Call to a Name (line 419):
    
    # Call to solve_bvp(...): (line 419)
    # Processing the call arguments (line 419)
    # Getting the type of 'emden_fun' (line 419)
    emden_fun_41362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 28), 'emden_fun', False)
    # Getting the type of 'emden_bc' (line 419)
    emden_bc_41363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 39), 'emden_bc', False)
    # Getting the type of 'x' (line 419)
    x_41364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 49), 'x', False)
    # Getting the type of 'y' (line 419)
    y_41365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 52), 'y', False)
    # Processing the call keyword arguments (line 419)
    # Getting the type of 'S' (line 419)
    S_41366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 57), 'S', False)
    keyword_41367 = S_41366
    # Getting the type of 'fun_jac' (line 419)
    fun_jac_41368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 68), 'fun_jac', False)
    keyword_41369 = fun_jac_41368
    # Getting the type of 'bc_jac' (line 420)
    bc_jac_41370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 35), 'bc_jac', False)
    keyword_41371 = bc_jac_41370
    kwargs_41372 = {'bc_jac': keyword_41371, 'S': keyword_41367, 'fun_jac': keyword_41369}
    # Getting the type of 'solve_bvp' (line 419)
    solve_bvp_41361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 18), 'solve_bvp', False)
    # Calling solve_bvp(args, kwargs) (line 419)
    solve_bvp_call_result_41373 = invoke(stypy.reporting.localization.Localization(__file__, 419, 18), solve_bvp_41361, *[emden_fun_41362, emden_bc_41363, x_41364, y_41365], **kwargs_41372)
    
    # Assigning a type to the variable 'sol' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'sol', solve_bvp_call_result_41373)
    
    # Call to assert_equal(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'sol' (line 422)
    sol_41375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 25), 'sol', False)
    # Obtaining the member 'status' of a type (line 422)
    status_41376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 25), sol_41375, 'status')
    int_41377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 37), 'int')
    # Processing the call keyword arguments (line 422)
    kwargs_41378 = {}
    # Getting the type of 'assert_equal' (line 422)
    assert_equal_41374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 422)
    assert_equal_call_result_41379 = invoke(stypy.reporting.localization.Localization(__file__, 422, 12), assert_equal_41374, *[status_41376, int_41377], **kwargs_41378)
    
    
    # Call to assert_(...): (line 423)
    # Processing the call arguments (line 423)
    # Getting the type of 'sol' (line 423)
    sol_41381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 20), 'sol', False)
    # Obtaining the member 'success' of a type (line 423)
    success_41382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 20), sol_41381, 'success')
    # Processing the call keyword arguments (line 423)
    kwargs_41383 = {}
    # Getting the type of 'assert_' (line 423)
    assert__41380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 423)
    assert__call_result_41384 = invoke(stypy.reporting.localization.Localization(__file__, 423, 12), assert__41380, *[success_41382], **kwargs_41383)
    
    
    # Call to assert_equal(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'sol' (line 425)
    sol_41386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 25), 'sol', False)
    # Obtaining the member 'x' of a type (line 425)
    x_41387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 25), sol_41386, 'x')
    # Obtaining the member 'size' of a type (line 425)
    size_41388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 25), x_41387, 'size')
    int_41389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 37), 'int')
    # Processing the call keyword arguments (line 425)
    kwargs_41390 = {}
    # Getting the type of 'assert_equal' (line 425)
    assert_equal_41385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 425)
    assert_equal_call_result_41391 = invoke(stypy.reporting.localization.Localization(__file__, 425, 12), assert_equal_41385, *[size_41388, int_41389], **kwargs_41390)
    
    
    # Assigning a Call to a Name (line 427):
    
    # Assigning a Call to a Name (line 427):
    
    # Call to sol(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'x_test' (line 427)
    x_test_41394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 31), 'x_test', False)
    # Processing the call keyword arguments (line 427)
    kwargs_41395 = {}
    # Getting the type of 'sol' (line 427)
    sol_41392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 23), 'sol', False)
    # Obtaining the member 'sol' of a type (line 427)
    sol_41393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 23), sol_41392, 'sol')
    # Calling sol(args, kwargs) (line 427)
    sol_call_result_41396 = invoke(stypy.reporting.localization.Localization(__file__, 427, 23), sol_41393, *[x_test_41394], **kwargs_41395)
    
    # Assigning a type to the variable 'sol_test' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'sol_test', sol_call_result_41396)
    
    # Call to assert_allclose(...): (line 428)
    # Processing the call arguments (line 428)
    
    # Obtaining the type of the subscript
    int_41398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 37), 'int')
    # Getting the type of 'sol_test' (line 428)
    sol_test_41399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 28), 'sol_test', False)
    # Obtaining the member '__getitem__' of a type (line 428)
    getitem___41400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 28), sol_test_41399, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 428)
    subscript_call_result_41401 = invoke(stypy.reporting.localization.Localization(__file__, 428, 28), getitem___41400, int_41398)
    
    
    # Call to emden_sol(...): (line 428)
    # Processing the call arguments (line 428)
    # Getting the type of 'x_test' (line 428)
    x_test_41403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 51), 'x_test', False)
    # Processing the call keyword arguments (line 428)
    kwargs_41404 = {}
    # Getting the type of 'emden_sol' (line 428)
    emden_sol_41402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 41), 'emden_sol', False)
    # Calling emden_sol(args, kwargs) (line 428)
    emden_sol_call_result_41405 = invoke(stypy.reporting.localization.Localization(__file__, 428, 41), emden_sol_41402, *[x_test_41403], **kwargs_41404)
    
    # Processing the call keyword arguments (line 428)
    float_41406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 65), 'float')
    keyword_41407 = float_41406
    kwargs_41408 = {'atol': keyword_41407}
    # Getting the type of 'assert_allclose' (line 428)
    assert_allclose_41397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 428)
    assert_allclose_call_result_41409 = invoke(stypy.reporting.localization.Localization(__file__, 428, 12), assert_allclose_41397, *[subscript_call_result_41401, emden_sol_call_result_41405], **kwargs_41408)
    
    
    # Assigning a BinOp to a Name (line 430):
    
    # Assigning a BinOp to a Name (line 430):
    
    # Call to emden_fun(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'x_test' (line 430)
    x_test_41411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 31), 'x_test', False)
    # Getting the type of 'sol_test' (line 430)
    sol_test_41412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 39), 'sol_test', False)
    # Processing the call keyword arguments (line 430)
    kwargs_41413 = {}
    # Getting the type of 'emden_fun' (line 430)
    emden_fun_41410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 21), 'emden_fun', False)
    # Calling emden_fun(args, kwargs) (line 430)
    emden_fun_call_result_41414 = invoke(stypy.reporting.localization.Localization(__file__, 430, 21), emden_fun_41410, *[x_test_41411, sol_test_41412], **kwargs_41413)
    
    
    # Call to dot(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'sol_test' (line 430)
    sol_test_41417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 57), 'sol_test', False)
    # Processing the call keyword arguments (line 430)
    kwargs_41418 = {}
    # Getting the type of 'S' (line 430)
    S_41415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 51), 'S', False)
    # Obtaining the member 'dot' of a type (line 430)
    dot_41416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 51), S_41415, 'dot')
    # Calling dot(args, kwargs) (line 430)
    dot_call_result_41419 = invoke(stypy.reporting.localization.Localization(__file__, 430, 51), dot_41416, *[sol_test_41417], **kwargs_41418)
    
    # Getting the type of 'x_test' (line 430)
    x_test_41420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 69), 'x_test')
    # Applying the binary operator 'div' (line 430)
    result_div_41421 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 51), 'div', dot_call_result_41419, x_test_41420)
    
    # Applying the binary operator '+' (line 430)
    result_add_41422 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 21), '+', emden_fun_call_result_41414, result_div_41421)
    
    # Assigning a type to the variable 'f_test' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'f_test', result_add_41422)
    
    # Assigning a BinOp to a Name (line 431):
    
    # Assigning a BinOp to a Name (line 431):
    
    # Call to sol(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'x_test' (line 431)
    x_test_41425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 24), 'x_test', False)
    int_41426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 32), 'int')
    # Processing the call keyword arguments (line 431)
    kwargs_41427 = {}
    # Getting the type of 'sol' (line 431)
    sol_41423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 16), 'sol', False)
    # Obtaining the member 'sol' of a type (line 431)
    sol_41424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 16), sol_41423, 'sol')
    # Calling sol(args, kwargs) (line 431)
    sol_call_result_41428 = invoke(stypy.reporting.localization.Localization(__file__, 431, 16), sol_41424, *[x_test_41425, int_41426], **kwargs_41427)
    
    # Getting the type of 'f_test' (line 431)
    f_test_41429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 37), 'f_test')
    # Applying the binary operator '-' (line 431)
    result_sub_41430 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 16), '-', sol_call_result_41428, f_test_41429)
    
    # Assigning a type to the variable 'r' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'r', result_sub_41430)
    
    # Assigning a BinOp to a Name (line 432):
    
    # Assigning a BinOp to a Name (line 432):
    # Getting the type of 'r' (line 432)
    r_41431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 22), 'r')
    int_41432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 27), 'int')
    
    # Call to abs(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'f_test' (line 432)
    f_test_41435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 38), 'f_test', False)
    # Processing the call keyword arguments (line 432)
    kwargs_41436 = {}
    # Getting the type of 'np' (line 432)
    np_41433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 31), 'np', False)
    # Obtaining the member 'abs' of a type (line 432)
    abs_41434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 31), np_41433, 'abs')
    # Calling abs(args, kwargs) (line 432)
    abs_call_result_41437 = invoke(stypy.reporting.localization.Localization(__file__, 432, 31), abs_41434, *[f_test_41435], **kwargs_41436)
    
    # Applying the binary operator '+' (line 432)
    result_add_41438 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 27), '+', int_41432, abs_call_result_41437)
    
    # Applying the binary operator 'div' (line 432)
    result_div_41439 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 22), 'div', r_41431, result_add_41438)
    
    # Assigning a type to the variable 'rel_res' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'rel_res', result_div_41439)
    
    # Assigning a BinOp to a Name (line 433):
    
    # Assigning a BinOp to a Name (line 433):
    
    # Call to sum(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'rel_res' (line 433)
    rel_res_41442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 30), 'rel_res', False)
    int_41443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 41), 'int')
    # Applying the binary operator '**' (line 433)
    result_pow_41444 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 30), '**', rel_res_41442, int_41443)
    
    # Processing the call keyword arguments (line 433)
    int_41445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 49), 'int')
    keyword_41446 = int_41445
    kwargs_41447 = {'axis': keyword_41446}
    # Getting the type of 'np' (line 433)
    np_41440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 23), 'np', False)
    # Obtaining the member 'sum' of a type (line 433)
    sum_41441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 23), np_41440, 'sum')
    # Calling sum(args, kwargs) (line 433)
    sum_call_result_41448 = invoke(stypy.reporting.localization.Localization(__file__, 433, 23), sum_41441, *[result_pow_41444], **kwargs_41447)
    
    float_41449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 55), 'float')
    # Applying the binary operator '**' (line 433)
    result_pow_41450 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 23), '**', sum_call_result_41448, float_41449)
    
    # Assigning a type to the variable 'norm_res' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'norm_res', result_pow_41450)
    
    # Call to assert_(...): (line 435)
    # Processing the call arguments (line 435)
    
    # Call to all(...): (line 435)
    # Processing the call arguments (line 435)
    
    # Getting the type of 'norm_res' (line 435)
    norm_res_41454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 27), 'norm_res', False)
    float_41455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 38), 'float')
    # Applying the binary operator '<' (line 435)
    result_lt_41456 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 27), '<', norm_res_41454, float_41455)
    
    # Processing the call keyword arguments (line 435)
    kwargs_41457 = {}
    # Getting the type of 'np' (line 435)
    np_41452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'np', False)
    # Obtaining the member 'all' of a type (line 435)
    all_41453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 20), np_41452, 'all')
    # Calling all(args, kwargs) (line 435)
    all_call_result_41458 = invoke(stypy.reporting.localization.Localization(__file__, 435, 20), all_41453, *[result_lt_41456], **kwargs_41457)
    
    # Processing the call keyword arguments (line 435)
    kwargs_41459 = {}
    # Getting the type of 'assert_' (line 435)
    assert__41451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 435)
    assert__call_result_41460 = invoke(stypy.reporting.localization.Localization(__file__, 435, 12), assert__41451, *[all_call_result_41458], **kwargs_41459)
    
    
    # Call to assert_allclose(...): (line 436)
    # Processing the call arguments (line 436)
    
    # Call to sol(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'sol' (line 436)
    sol_41464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 36), 'sol', False)
    # Obtaining the member 'x' of a type (line 436)
    x_41465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 36), sol_41464, 'x')
    # Processing the call keyword arguments (line 436)
    kwargs_41466 = {}
    # Getting the type of 'sol' (line 436)
    sol_41462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 28), 'sol', False)
    # Obtaining the member 'sol' of a type (line 436)
    sol_41463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 28), sol_41462, 'sol')
    # Calling sol(args, kwargs) (line 436)
    sol_call_result_41467 = invoke(stypy.reporting.localization.Localization(__file__, 436, 28), sol_41463, *[x_41465], **kwargs_41466)
    
    # Getting the type of 'sol' (line 436)
    sol_41468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 44), 'sol', False)
    # Obtaining the member 'y' of a type (line 436)
    y_41469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 44), sol_41468, 'y')
    # Processing the call keyword arguments (line 436)
    float_41470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 56), 'float')
    keyword_41471 = float_41470
    float_41472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 68), 'float')
    keyword_41473 = float_41472
    kwargs_41474 = {'rtol': keyword_41471, 'atol': keyword_41473}
    # Getting the type of 'assert_allclose' (line 436)
    assert_allclose_41461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 436)
    assert_allclose_call_result_41475 = invoke(stypy.reporting.localization.Localization(__file__, 436, 12), assert_allclose_41461, *[sol_call_result_41467, y_41469], **kwargs_41474)
    
    
    # Call to assert_allclose(...): (line 437)
    # Processing the call arguments (line 437)
    
    # Call to sol(...): (line 437)
    # Processing the call arguments (line 437)
    # Getting the type of 'sol' (line 437)
    sol_41479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 36), 'sol', False)
    # Obtaining the member 'x' of a type (line 437)
    x_41480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 36), sol_41479, 'x')
    int_41481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 43), 'int')
    # Processing the call keyword arguments (line 437)
    kwargs_41482 = {}
    # Getting the type of 'sol' (line 437)
    sol_41477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 28), 'sol', False)
    # Obtaining the member 'sol' of a type (line 437)
    sol_41478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 28), sol_41477, 'sol')
    # Calling sol(args, kwargs) (line 437)
    sol_call_result_41483 = invoke(stypy.reporting.localization.Localization(__file__, 437, 28), sol_41478, *[x_41480, int_41481], **kwargs_41482)
    
    # Getting the type of 'sol' (line 437)
    sol_41484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 47), 'sol', False)
    # Obtaining the member 'yp' of a type (line 437)
    yp_41485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 47), sol_41484, 'yp')
    # Processing the call keyword arguments (line 437)
    float_41486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 60), 'float')
    keyword_41487 = float_41486
    float_41488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 72), 'float')
    keyword_41489 = float_41488
    kwargs_41490 = {'rtol': keyword_41487, 'atol': keyword_41489}
    # Getting the type of 'assert_allclose' (line 437)
    assert_allclose_41476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 437)
    assert_allclose_call_result_41491 = invoke(stypy.reporting.localization.Localization(__file__, 437, 12), assert_allclose_41476, *[sol_call_result_41483, yp_41485], **kwargs_41490)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_singular_term(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_singular_term' in the type store
    # Getting the type of 'stypy_return_type' (line 409)
    stypy_return_type_41492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_41492)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_singular_term'
    return stypy_return_type_41492

# Assigning a type to the variable 'test_singular_term' (line 409)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'test_singular_term', test_singular_term)

@norecursion
def test_complex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_complex'
    module_type_store = module_type_store.open_function_context('test_complex', 440, 0, False)
    
    # Passed parameters checking function
    test_complex.stypy_localization = localization
    test_complex.stypy_type_of_self = None
    test_complex.stypy_type_store = module_type_store
    test_complex.stypy_function_name = 'test_complex'
    test_complex.stypy_param_names_list = []
    test_complex.stypy_varargs_param_name = None
    test_complex.stypy_kwargs_param_name = None
    test_complex.stypy_call_defaults = defaults
    test_complex.stypy_call_varargs = varargs
    test_complex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_complex', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_complex', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_complex(...)' code ##################

    
    # Assigning a Call to a Name (line 443):
    
    # Assigning a Call to a Name (line 443):
    
    # Call to linspace(...): (line 443)
    # Processing the call arguments (line 443)
    int_41495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 20), 'int')
    int_41496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 23), 'int')
    int_41497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 26), 'int')
    # Processing the call keyword arguments (line 443)
    kwargs_41498 = {}
    # Getting the type of 'np' (line 443)
    np_41493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 443)
    linspace_41494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), np_41493, 'linspace')
    # Calling linspace(args, kwargs) (line 443)
    linspace_call_result_41499 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), linspace_41494, *[int_41495, int_41496, int_41497], **kwargs_41498)
    
    # Assigning a type to the variable 'x' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'x', linspace_call_result_41499)
    
    # Assigning a Call to a Name (line 444):
    
    # Assigning a Call to a Name (line 444):
    
    # Call to linspace(...): (line 444)
    # Processing the call arguments (line 444)
    int_41502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 25), 'int')
    int_41503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 28), 'int')
    int_41504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 31), 'int')
    # Processing the call keyword arguments (line 444)
    kwargs_41505 = {}
    # Getting the type of 'np' (line 444)
    np_41500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 13), 'np', False)
    # Obtaining the member 'linspace' of a type (line 444)
    linspace_41501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 13), np_41500, 'linspace')
    # Calling linspace(args, kwargs) (line 444)
    linspace_call_result_41506 = invoke(stypy.reporting.localization.Localization(__file__, 444, 13), linspace_41501, *[int_41502, int_41503, int_41504], **kwargs_41505)
    
    # Assigning a type to the variable 'x_test' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'x_test', linspace_call_result_41506)
    
    # Assigning a Call to a Name (line 445):
    
    # Assigning a Call to a Name (line 445):
    
    # Call to zeros(...): (line 445)
    # Processing the call arguments (line 445)
    
    # Obtaining an instance of the builtin type 'tuple' (line 445)
    tuple_41509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 445)
    # Adding element type (line 445)
    int_41510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 18), tuple_41509, int_41510)
    # Adding element type (line 445)
    
    # Obtaining the type of the subscript
    int_41511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 29), 'int')
    # Getting the type of 'x' (line 445)
    x_41512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 445)
    shape_41513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), x_41512, 'shape')
    # Obtaining the member '__getitem__' of a type (line 445)
    getitem___41514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), shape_41513, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 445)
    subscript_call_result_41515 = invoke(stypy.reporting.localization.Localization(__file__, 445, 21), getitem___41514, int_41511)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 18), tuple_41509, subscript_call_result_41515)
    
    # Processing the call keyword arguments (line 445)
    # Getting the type of 'complex' (line 445)
    complex_41516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 40), 'complex', False)
    keyword_41517 = complex_41516
    kwargs_41518 = {'dtype': keyword_41517}
    # Getting the type of 'np' (line 445)
    np_41507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 445)
    zeros_41508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), np_41507, 'zeros')
    # Calling zeros(args, kwargs) (line 445)
    zeros_call_result_41519 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), zeros_41508, *[tuple_41509], **kwargs_41518)
    
    # Assigning a type to the variable 'y' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'y', zeros_call_result_41519)
    
    
    # Obtaining an instance of the builtin type 'list' (line 446)
    list_41520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 446)
    # Adding element type (line 446)
    # Getting the type of 'None' (line 446)
    None_41521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 20), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 19), list_41520, None_41521)
    # Adding element type (line 446)
    # Getting the type of 'exp_fun_jac' (line 446)
    exp_fun_jac_41522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 26), 'exp_fun_jac')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 19), list_41520, exp_fun_jac_41522)
    
    # Testing the type of a for loop iterable (line 446)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 446, 4), list_41520)
    # Getting the type of the for loop variable (line 446)
    for_loop_var_41523 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 446, 4), list_41520)
    # Assigning a type to the variable 'fun_jac' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'fun_jac', for_loop_var_41523)
    # SSA begins for a for statement (line 446)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'list' (line 447)
    list_41524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 447)
    # Adding element type (line 447)
    # Getting the type of 'None' (line 447)
    None_41525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 23), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 22), list_41524, None_41525)
    # Adding element type (line 447)
    # Getting the type of 'exp_bc_jac' (line 447)
    exp_bc_jac_41526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 29), 'exp_bc_jac')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 22), list_41524, exp_bc_jac_41526)
    
    # Testing the type of a for loop iterable (line 447)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 447, 8), list_41524)
    # Getting the type of the for loop variable (line 447)
    for_loop_var_41527 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 447, 8), list_41524)
    # Assigning a type to the variable 'bc_jac' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'bc_jac', for_loop_var_41527)
    # SSA begins for a for statement (line 447)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 448):
    
    # Assigning a Call to a Name (line 448):
    
    # Call to solve_bvp(...): (line 448)
    # Processing the call arguments (line 448)
    # Getting the type of 'exp_fun' (line 448)
    exp_fun_41529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 28), 'exp_fun', False)
    # Getting the type of 'exp_bc_complex' (line 448)
    exp_bc_complex_41530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 37), 'exp_bc_complex', False)
    # Getting the type of 'x' (line 448)
    x_41531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 53), 'x', False)
    # Getting the type of 'y' (line 448)
    y_41532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 56), 'y', False)
    # Processing the call keyword arguments (line 448)
    # Getting the type of 'fun_jac' (line 448)
    fun_jac_41533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 67), 'fun_jac', False)
    keyword_41534 = fun_jac_41533
    # Getting the type of 'bc_jac' (line 449)
    bc_jac_41535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 35), 'bc_jac', False)
    keyword_41536 = bc_jac_41535
    kwargs_41537 = {'bc_jac': keyword_41536, 'fun_jac': keyword_41534}
    # Getting the type of 'solve_bvp' (line 448)
    solve_bvp_41528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 18), 'solve_bvp', False)
    # Calling solve_bvp(args, kwargs) (line 448)
    solve_bvp_call_result_41538 = invoke(stypy.reporting.localization.Localization(__file__, 448, 18), solve_bvp_41528, *[exp_fun_41529, exp_bc_complex_41530, x_41531, y_41532], **kwargs_41537)
    
    # Assigning a type to the variable 'sol' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'sol', solve_bvp_call_result_41538)
    
    # Call to assert_equal(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'sol' (line 451)
    sol_41540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 25), 'sol', False)
    # Obtaining the member 'status' of a type (line 451)
    status_41541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 25), sol_41540, 'status')
    int_41542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 37), 'int')
    # Processing the call keyword arguments (line 451)
    kwargs_41543 = {}
    # Getting the type of 'assert_equal' (line 451)
    assert_equal_41539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 451)
    assert_equal_call_result_41544 = invoke(stypy.reporting.localization.Localization(__file__, 451, 12), assert_equal_41539, *[status_41541, int_41542], **kwargs_41543)
    
    
    # Call to assert_(...): (line 452)
    # Processing the call arguments (line 452)
    # Getting the type of 'sol' (line 452)
    sol_41546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 20), 'sol', False)
    # Obtaining the member 'success' of a type (line 452)
    success_41547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 20), sol_41546, 'success')
    # Processing the call keyword arguments (line 452)
    kwargs_41548 = {}
    # Getting the type of 'assert_' (line 452)
    assert__41545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 452)
    assert__call_result_41549 = invoke(stypy.reporting.localization.Localization(__file__, 452, 12), assert__41545, *[success_41547], **kwargs_41548)
    
    
    # Assigning a Call to a Name (line 454):
    
    # Assigning a Call to a Name (line 454):
    
    # Call to sol(...): (line 454)
    # Processing the call arguments (line 454)
    # Getting the type of 'x_test' (line 454)
    x_test_41552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 31), 'x_test', False)
    # Processing the call keyword arguments (line 454)
    kwargs_41553 = {}
    # Getting the type of 'sol' (line 454)
    sol_41550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 23), 'sol', False)
    # Obtaining the member 'sol' of a type (line 454)
    sol_41551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 23), sol_41550, 'sol')
    # Calling sol(args, kwargs) (line 454)
    sol_call_result_41554 = invoke(stypy.reporting.localization.Localization(__file__, 454, 23), sol_41551, *[x_test_41552], **kwargs_41553)
    
    # Assigning a type to the variable 'sol_test' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'sol_test', sol_call_result_41554)
    
    # Call to assert_allclose(...): (line 456)
    # Processing the call arguments (line 456)
    
    # Obtaining the type of the subscript
    int_41556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 37), 'int')
    # Getting the type of 'sol_test' (line 456)
    sol_test_41557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 28), 'sol_test', False)
    # Obtaining the member '__getitem__' of a type (line 456)
    getitem___41558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 28), sol_test_41557, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 456)
    subscript_call_result_41559 = invoke(stypy.reporting.localization.Localization(__file__, 456, 28), getitem___41558, int_41556)
    
    # Obtaining the member 'real' of a type (line 456)
    real_41560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 28), subscript_call_result_41559, 'real')
    
    # Call to exp_sol(...): (line 456)
    # Processing the call arguments (line 456)
    # Getting the type of 'x_test' (line 456)
    x_test_41562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 54), 'x_test', False)
    # Processing the call keyword arguments (line 456)
    kwargs_41563 = {}
    # Getting the type of 'exp_sol' (line 456)
    exp_sol_41561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 46), 'exp_sol', False)
    # Calling exp_sol(args, kwargs) (line 456)
    exp_sol_call_result_41564 = invoke(stypy.reporting.localization.Localization(__file__, 456, 46), exp_sol_41561, *[x_test_41562], **kwargs_41563)
    
    # Processing the call keyword arguments (line 456)
    float_41565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 68), 'float')
    keyword_41566 = float_41565
    kwargs_41567 = {'atol': keyword_41566}
    # Getting the type of 'assert_allclose' (line 456)
    assert_allclose_41555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 456)
    assert_allclose_call_result_41568 = invoke(stypy.reporting.localization.Localization(__file__, 456, 12), assert_allclose_41555, *[real_41560, exp_sol_call_result_41564], **kwargs_41567)
    
    
    # Call to assert_allclose(...): (line 457)
    # Processing the call arguments (line 457)
    
    # Obtaining the type of the subscript
    int_41570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 37), 'int')
    # Getting the type of 'sol_test' (line 457)
    sol_test_41571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 28), 'sol_test', False)
    # Obtaining the member '__getitem__' of a type (line 457)
    getitem___41572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 28), sol_test_41571, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 457)
    subscript_call_result_41573 = invoke(stypy.reporting.localization.Localization(__file__, 457, 28), getitem___41572, int_41570)
    
    # Obtaining the member 'imag' of a type (line 457)
    imag_41574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 28), subscript_call_result_41573, 'imag')
    
    # Call to exp_sol(...): (line 457)
    # Processing the call arguments (line 457)
    # Getting the type of 'x_test' (line 457)
    x_test_41576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 54), 'x_test', False)
    # Processing the call keyword arguments (line 457)
    kwargs_41577 = {}
    # Getting the type of 'exp_sol' (line 457)
    exp_sol_41575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 46), 'exp_sol', False)
    # Calling exp_sol(args, kwargs) (line 457)
    exp_sol_call_result_41578 = invoke(stypy.reporting.localization.Localization(__file__, 457, 46), exp_sol_41575, *[x_test_41576], **kwargs_41577)
    
    # Processing the call keyword arguments (line 457)
    float_41579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 68), 'float')
    keyword_41580 = float_41579
    kwargs_41581 = {'atol': keyword_41580}
    # Getting the type of 'assert_allclose' (line 457)
    assert_allclose_41569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 457)
    assert_allclose_call_result_41582 = invoke(stypy.reporting.localization.Localization(__file__, 457, 12), assert_allclose_41569, *[imag_41574, exp_sol_call_result_41578], **kwargs_41581)
    
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Call to exp_fun(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'x_test' (line 459)
    x_test_41584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 29), 'x_test', False)
    # Getting the type of 'sol_test' (line 459)
    sol_test_41585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 37), 'sol_test', False)
    # Processing the call keyword arguments (line 459)
    kwargs_41586 = {}
    # Getting the type of 'exp_fun' (line 459)
    exp_fun_41583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 21), 'exp_fun', False)
    # Calling exp_fun(args, kwargs) (line 459)
    exp_fun_call_result_41587 = invoke(stypy.reporting.localization.Localization(__file__, 459, 21), exp_fun_41583, *[x_test_41584, sol_test_41585], **kwargs_41586)
    
    # Assigning a type to the variable 'f_test' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'f_test', exp_fun_call_result_41587)
    
    # Assigning a BinOp to a Name (line 460):
    
    # Assigning a BinOp to a Name (line 460):
    
    # Call to sol(...): (line 460)
    # Processing the call arguments (line 460)
    # Getting the type of 'x_test' (line 460)
    x_test_41590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 24), 'x_test', False)
    int_41591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 32), 'int')
    # Processing the call keyword arguments (line 460)
    kwargs_41592 = {}
    # Getting the type of 'sol' (line 460)
    sol_41588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 16), 'sol', False)
    # Obtaining the member 'sol' of a type (line 460)
    sol_41589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 16), sol_41588, 'sol')
    # Calling sol(args, kwargs) (line 460)
    sol_call_result_41593 = invoke(stypy.reporting.localization.Localization(__file__, 460, 16), sol_41589, *[x_test_41590, int_41591], **kwargs_41592)
    
    # Getting the type of 'f_test' (line 460)
    f_test_41594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 37), 'f_test')
    # Applying the binary operator '-' (line 460)
    result_sub_41595 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 16), '-', sol_call_result_41593, f_test_41594)
    
    # Assigning a type to the variable 'r' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'r', result_sub_41595)
    
    # Assigning a BinOp to a Name (line 461):
    
    # Assigning a BinOp to a Name (line 461):
    # Getting the type of 'r' (line 461)
    r_41596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 22), 'r')
    int_41597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 27), 'int')
    
    # Call to abs(...): (line 461)
    # Processing the call arguments (line 461)
    # Getting the type of 'f_test' (line 461)
    f_test_41600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 38), 'f_test', False)
    # Processing the call keyword arguments (line 461)
    kwargs_41601 = {}
    # Getting the type of 'np' (line 461)
    np_41598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 31), 'np', False)
    # Obtaining the member 'abs' of a type (line 461)
    abs_41599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 31), np_41598, 'abs')
    # Calling abs(args, kwargs) (line 461)
    abs_call_result_41602 = invoke(stypy.reporting.localization.Localization(__file__, 461, 31), abs_41599, *[f_test_41600], **kwargs_41601)
    
    # Applying the binary operator '+' (line 461)
    result_add_41603 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 27), '+', int_41597, abs_call_result_41602)
    
    # Applying the binary operator 'div' (line 461)
    result_div_41604 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 22), 'div', r_41596, result_add_41603)
    
    # Assigning a type to the variable 'rel_res' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'rel_res', result_div_41604)
    
    # Assigning a BinOp to a Name (line 462):
    
    # Assigning a BinOp to a Name (line 462):
    
    # Call to sum(...): (line 462)
    # Processing the call arguments (line 462)
    
    # Call to real(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'rel_res' (line 462)
    rel_res_41609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 38), 'rel_res', False)
    
    # Call to conj(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'rel_res' (line 462)
    rel_res_41612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 56), 'rel_res', False)
    # Processing the call keyword arguments (line 462)
    kwargs_41613 = {}
    # Getting the type of 'np' (line 462)
    np_41610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 48), 'np', False)
    # Obtaining the member 'conj' of a type (line 462)
    conj_41611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 48), np_41610, 'conj')
    # Calling conj(args, kwargs) (line 462)
    conj_call_result_41614 = invoke(stypy.reporting.localization.Localization(__file__, 462, 48), conj_41611, *[rel_res_41612], **kwargs_41613)
    
    # Applying the binary operator '*' (line 462)
    result_mul_41615 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 38), '*', rel_res_41609, conj_call_result_41614)
    
    # Processing the call keyword arguments (line 462)
    kwargs_41616 = {}
    # Getting the type of 'np' (line 462)
    np_41607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 30), 'np', False)
    # Obtaining the member 'real' of a type (line 462)
    real_41608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 30), np_41607, 'real')
    # Calling real(args, kwargs) (line 462)
    real_call_result_41617 = invoke(stypy.reporting.localization.Localization(__file__, 462, 30), real_41608, *[result_mul_41615], **kwargs_41616)
    
    # Processing the call keyword arguments (line 462)
    int_41618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 35), 'int')
    keyword_41619 = int_41618
    kwargs_41620 = {'axis': keyword_41619}
    # Getting the type of 'np' (line 462)
    np_41605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 23), 'np', False)
    # Obtaining the member 'sum' of a type (line 462)
    sum_41606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 23), np_41605, 'sum')
    # Calling sum(args, kwargs) (line 462)
    sum_call_result_41621 = invoke(stypy.reporting.localization.Localization(__file__, 462, 23), sum_41606, *[real_call_result_41617], **kwargs_41620)
    
    float_41622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 41), 'float')
    # Applying the binary operator '**' (line 462)
    result_pow_41623 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 23), '**', sum_call_result_41621, float_41622)
    
    # Assigning a type to the variable 'norm_res' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'norm_res', result_pow_41623)
    
    # Call to assert_(...): (line 464)
    # Processing the call arguments (line 464)
    
    # Call to all(...): (line 464)
    # Processing the call arguments (line 464)
    
    # Getting the type of 'norm_res' (line 464)
    norm_res_41627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 27), 'norm_res', False)
    float_41628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 38), 'float')
    # Applying the binary operator '<' (line 464)
    result_lt_41629 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 27), '<', norm_res_41627, float_41628)
    
    # Processing the call keyword arguments (line 464)
    kwargs_41630 = {}
    # Getting the type of 'np' (line 464)
    np_41625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 20), 'np', False)
    # Obtaining the member 'all' of a type (line 464)
    all_41626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 20), np_41625, 'all')
    # Calling all(args, kwargs) (line 464)
    all_call_result_41631 = invoke(stypy.reporting.localization.Localization(__file__, 464, 20), all_41626, *[result_lt_41629], **kwargs_41630)
    
    # Processing the call keyword arguments (line 464)
    kwargs_41632 = {}
    # Getting the type of 'assert_' (line 464)
    assert__41624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 464)
    assert__call_result_41633 = invoke(stypy.reporting.localization.Localization(__file__, 464, 12), assert__41624, *[all_call_result_41631], **kwargs_41632)
    
    
    # Call to assert_(...): (line 466)
    # Processing the call arguments (line 466)
    
    # Call to all(...): (line 466)
    # Processing the call arguments (line 466)
    
    # Getting the type of 'sol' (line 466)
    sol_41637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 27), 'sol', False)
    # Obtaining the member 'rms_residuals' of a type (line 466)
    rms_residuals_41638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 27), sol_41637, 'rms_residuals')
    float_41639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 47), 'float')
    # Applying the binary operator '<' (line 466)
    result_lt_41640 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 27), '<', rms_residuals_41638, float_41639)
    
    # Processing the call keyword arguments (line 466)
    kwargs_41641 = {}
    # Getting the type of 'np' (line 466)
    np_41635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 20), 'np', False)
    # Obtaining the member 'all' of a type (line 466)
    all_41636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 20), np_41635, 'all')
    # Calling all(args, kwargs) (line 466)
    all_call_result_41642 = invoke(stypy.reporting.localization.Localization(__file__, 466, 20), all_41636, *[result_lt_41640], **kwargs_41641)
    
    # Processing the call keyword arguments (line 466)
    kwargs_41643 = {}
    # Getting the type of 'assert_' (line 466)
    assert__41634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 466)
    assert__call_result_41644 = invoke(stypy.reporting.localization.Localization(__file__, 466, 12), assert__41634, *[all_call_result_41642], **kwargs_41643)
    
    
    # Call to assert_allclose(...): (line 467)
    # Processing the call arguments (line 467)
    
    # Call to sol(...): (line 467)
    # Processing the call arguments (line 467)
    # Getting the type of 'sol' (line 467)
    sol_41648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 36), 'sol', False)
    # Obtaining the member 'x' of a type (line 467)
    x_41649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 36), sol_41648, 'x')
    # Processing the call keyword arguments (line 467)
    kwargs_41650 = {}
    # Getting the type of 'sol' (line 467)
    sol_41646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 28), 'sol', False)
    # Obtaining the member 'sol' of a type (line 467)
    sol_41647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 28), sol_41646, 'sol')
    # Calling sol(args, kwargs) (line 467)
    sol_call_result_41651 = invoke(stypy.reporting.localization.Localization(__file__, 467, 28), sol_41647, *[x_41649], **kwargs_41650)
    
    # Getting the type of 'sol' (line 467)
    sol_41652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 44), 'sol', False)
    # Obtaining the member 'y' of a type (line 467)
    y_41653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 44), sol_41652, 'y')
    # Processing the call keyword arguments (line 467)
    float_41654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 56), 'float')
    keyword_41655 = float_41654
    float_41656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 68), 'float')
    keyword_41657 = float_41656
    kwargs_41658 = {'rtol': keyword_41655, 'atol': keyword_41657}
    # Getting the type of 'assert_allclose' (line 467)
    assert_allclose_41645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 467)
    assert_allclose_call_result_41659 = invoke(stypy.reporting.localization.Localization(__file__, 467, 12), assert_allclose_41645, *[sol_call_result_41651, y_41653], **kwargs_41658)
    
    
    # Call to assert_allclose(...): (line 468)
    # Processing the call arguments (line 468)
    
    # Call to sol(...): (line 468)
    # Processing the call arguments (line 468)
    # Getting the type of 'sol' (line 468)
    sol_41663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 36), 'sol', False)
    # Obtaining the member 'x' of a type (line 468)
    x_41664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 36), sol_41663, 'x')
    int_41665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 43), 'int')
    # Processing the call keyword arguments (line 468)
    kwargs_41666 = {}
    # Getting the type of 'sol' (line 468)
    sol_41661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 28), 'sol', False)
    # Obtaining the member 'sol' of a type (line 468)
    sol_41662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 28), sol_41661, 'sol')
    # Calling sol(args, kwargs) (line 468)
    sol_call_result_41667 = invoke(stypy.reporting.localization.Localization(__file__, 468, 28), sol_41662, *[x_41664, int_41665], **kwargs_41666)
    
    # Getting the type of 'sol' (line 468)
    sol_41668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 47), 'sol', False)
    # Obtaining the member 'yp' of a type (line 468)
    yp_41669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 47), sol_41668, 'yp')
    # Processing the call keyword arguments (line 468)
    float_41670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 60), 'float')
    keyword_41671 = float_41670
    float_41672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 72), 'float')
    keyword_41673 = float_41672
    kwargs_41674 = {'rtol': keyword_41671, 'atol': keyword_41673}
    # Getting the type of 'assert_allclose' (line 468)
    assert_allclose_41660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 468)
    assert_allclose_call_result_41675 = invoke(stypy.reporting.localization.Localization(__file__, 468, 12), assert_allclose_41660, *[sol_call_result_41667, yp_41669], **kwargs_41674)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_complex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_complex' in the type store
    # Getting the type of 'stypy_return_type' (line 440)
    stypy_return_type_41676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_41676)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_complex'
    return stypy_return_type_41676

# Assigning a type to the variable 'test_complex' (line 440)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 0), 'test_complex', test_complex)

@norecursion
def test_failures(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_failures'
    module_type_store = module_type_store.open_function_context('test_failures', 471, 0, False)
    
    # Passed parameters checking function
    test_failures.stypy_localization = localization
    test_failures.stypy_type_of_self = None
    test_failures.stypy_type_store = module_type_store
    test_failures.stypy_function_name = 'test_failures'
    test_failures.stypy_param_names_list = []
    test_failures.stypy_varargs_param_name = None
    test_failures.stypy_kwargs_param_name = None
    test_failures.stypy_call_defaults = defaults
    test_failures.stypy_call_varargs = varargs
    test_failures.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_failures', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_failures', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_failures(...)' code ##################

    
    # Assigning a Call to a Name (line 472):
    
    # Assigning a Call to a Name (line 472):
    
    # Call to linspace(...): (line 472)
    # Processing the call arguments (line 472)
    int_41679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 20), 'int')
    int_41680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 23), 'int')
    int_41681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 26), 'int')
    # Processing the call keyword arguments (line 472)
    kwargs_41682 = {}
    # Getting the type of 'np' (line 472)
    np_41677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 472)
    linspace_41678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), np_41677, 'linspace')
    # Calling linspace(args, kwargs) (line 472)
    linspace_call_result_41683 = invoke(stypy.reporting.localization.Localization(__file__, 472, 8), linspace_41678, *[int_41679, int_41680, int_41681], **kwargs_41682)
    
    # Assigning a type to the variable 'x' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'x', linspace_call_result_41683)
    
    # Assigning a Call to a Name (line 473):
    
    # Assigning a Call to a Name (line 473):
    
    # Call to zeros(...): (line 473)
    # Processing the call arguments (line 473)
    
    # Obtaining an instance of the builtin type 'tuple' (line 473)
    tuple_41686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 473)
    # Adding element type (line 473)
    int_41687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 18), tuple_41686, int_41687)
    # Adding element type (line 473)
    # Getting the type of 'x' (line 473)
    x_41688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 21), 'x', False)
    # Obtaining the member 'size' of a type (line 473)
    size_41689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 21), x_41688, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 18), tuple_41686, size_41689)
    
    # Processing the call keyword arguments (line 473)
    kwargs_41690 = {}
    # Getting the type of 'np' (line 473)
    np_41684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 473)
    zeros_41685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 8), np_41684, 'zeros')
    # Calling zeros(args, kwargs) (line 473)
    zeros_call_result_41691 = invoke(stypy.reporting.localization.Localization(__file__, 473, 8), zeros_41685, *[tuple_41686], **kwargs_41690)
    
    # Assigning a type to the variable 'y' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'y', zeros_call_result_41691)
    
    # Assigning a Call to a Name (line 474):
    
    # Assigning a Call to a Name (line 474):
    
    # Call to solve_bvp(...): (line 474)
    # Processing the call arguments (line 474)
    # Getting the type of 'exp_fun' (line 474)
    exp_fun_41693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 20), 'exp_fun', False)
    # Getting the type of 'exp_bc' (line 474)
    exp_bc_41694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 29), 'exp_bc', False)
    # Getting the type of 'x' (line 474)
    x_41695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 37), 'x', False)
    # Getting the type of 'y' (line 474)
    y_41696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 40), 'y', False)
    # Processing the call keyword arguments (line 474)
    float_41697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 47), 'float')
    keyword_41698 = float_41697
    int_41699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 63), 'int')
    keyword_41700 = int_41699
    kwargs_41701 = {'max_nodes': keyword_41700, 'tol': keyword_41698}
    # Getting the type of 'solve_bvp' (line 474)
    solve_bvp_41692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 10), 'solve_bvp', False)
    # Calling solve_bvp(args, kwargs) (line 474)
    solve_bvp_call_result_41702 = invoke(stypy.reporting.localization.Localization(__file__, 474, 10), solve_bvp_41692, *[exp_fun_41693, exp_bc_41694, x_41695, y_41696], **kwargs_41701)
    
    # Assigning a type to the variable 'res' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'res', solve_bvp_call_result_41702)
    
    # Call to assert_equal(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'res' (line 475)
    res_41704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 17), 'res', False)
    # Obtaining the member 'status' of a type (line 475)
    status_41705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 17), res_41704, 'status')
    int_41706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 29), 'int')
    # Processing the call keyword arguments (line 475)
    kwargs_41707 = {}
    # Getting the type of 'assert_equal' (line 475)
    assert_equal_41703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 475)
    assert_equal_call_result_41708 = invoke(stypy.reporting.localization.Localization(__file__, 475, 4), assert_equal_41703, *[status_41705, int_41706], **kwargs_41707)
    
    
    # Call to assert_(...): (line 476)
    # Processing the call arguments (line 476)
    
    # Getting the type of 'res' (line 476)
    res_41710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 16), 'res', False)
    # Obtaining the member 'success' of a type (line 476)
    success_41711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 16), res_41710, 'success')
    # Applying the 'not' unary operator (line 476)
    result_not__41712 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 12), 'not', success_41711)
    
    # Processing the call keyword arguments (line 476)
    kwargs_41713 = {}
    # Getting the type of 'assert_' (line 476)
    assert__41709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 476)
    assert__call_result_41714 = invoke(stypy.reporting.localization.Localization(__file__, 476, 4), assert__41709, *[result_not__41712], **kwargs_41713)
    
    
    # Assigning a Call to a Name (line 478):
    
    # Assigning a Call to a Name (line 478):
    
    # Call to linspace(...): (line 478)
    # Processing the call arguments (line 478)
    int_41717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 20), 'int')
    int_41718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 23), 'int')
    int_41719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 26), 'int')
    # Processing the call keyword arguments (line 478)
    kwargs_41720 = {}
    # Getting the type of 'np' (line 478)
    np_41715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 478)
    linspace_41716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 8), np_41715, 'linspace')
    # Calling linspace(args, kwargs) (line 478)
    linspace_call_result_41721 = invoke(stypy.reporting.localization.Localization(__file__, 478, 8), linspace_41716, *[int_41717, int_41718, int_41719], **kwargs_41720)
    
    # Assigning a type to the variable 'x' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'x', linspace_call_result_41721)
    
    # Assigning a Call to a Name (line 479):
    
    # Assigning a Call to a Name (line 479):
    
    # Call to zeros(...): (line 479)
    # Processing the call arguments (line 479)
    
    # Obtaining an instance of the builtin type 'tuple' (line 479)
    tuple_41724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 479)
    # Adding element type (line 479)
    int_41725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 18), tuple_41724, int_41725)
    # Adding element type (line 479)
    # Getting the type of 'x' (line 479)
    x_41726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 21), 'x', False)
    # Obtaining the member 'size' of a type (line 479)
    size_41727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 21), x_41726, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 18), tuple_41724, size_41727)
    
    # Processing the call keyword arguments (line 479)
    kwargs_41728 = {}
    # Getting the type of 'np' (line 479)
    np_41722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 479)
    zeros_41723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 8), np_41722, 'zeros')
    # Calling zeros(args, kwargs) (line 479)
    zeros_call_result_41729 = invoke(stypy.reporting.localization.Localization(__file__, 479, 8), zeros_41723, *[tuple_41724], **kwargs_41728)
    
    # Assigning a type to the variable 'y' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'y', zeros_call_result_41729)
    
    # Assigning a Call to a Name (line 480):
    
    # Assigning a Call to a Name (line 480):
    
    # Call to solve_bvp(...): (line 480)
    # Processing the call arguments (line 480)
    # Getting the type of 'undefined_fun' (line 480)
    undefined_fun_41731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 20), 'undefined_fun', False)
    # Getting the type of 'undefined_bc' (line 480)
    undefined_bc_41732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 35), 'undefined_bc', False)
    # Getting the type of 'x' (line 480)
    x_41733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 49), 'x', False)
    # Getting the type of 'y' (line 480)
    y_41734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 52), 'y', False)
    # Processing the call keyword arguments (line 480)
    kwargs_41735 = {}
    # Getting the type of 'solve_bvp' (line 480)
    solve_bvp_41730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 10), 'solve_bvp', False)
    # Calling solve_bvp(args, kwargs) (line 480)
    solve_bvp_call_result_41736 = invoke(stypy.reporting.localization.Localization(__file__, 480, 10), solve_bvp_41730, *[undefined_fun_41731, undefined_bc_41732, x_41733, y_41734], **kwargs_41735)
    
    # Assigning a type to the variable 'res' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'res', solve_bvp_call_result_41736)
    
    # Call to assert_equal(...): (line 481)
    # Processing the call arguments (line 481)
    # Getting the type of 'res' (line 481)
    res_41738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 17), 'res', False)
    # Obtaining the member 'status' of a type (line 481)
    status_41739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 17), res_41738, 'status')
    int_41740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 29), 'int')
    # Processing the call keyword arguments (line 481)
    kwargs_41741 = {}
    # Getting the type of 'assert_equal' (line 481)
    assert_equal_41737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 481)
    assert_equal_call_result_41742 = invoke(stypy.reporting.localization.Localization(__file__, 481, 4), assert_equal_41737, *[status_41739, int_41740], **kwargs_41741)
    
    
    # Call to assert_(...): (line 482)
    # Processing the call arguments (line 482)
    
    # Getting the type of 'res' (line 482)
    res_41744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 16), 'res', False)
    # Obtaining the member 'success' of a type (line 482)
    success_41745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 16), res_41744, 'success')
    # Applying the 'not' unary operator (line 482)
    result_not__41746 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 12), 'not', success_41745)
    
    # Processing the call keyword arguments (line 482)
    kwargs_41747 = {}
    # Getting the type of 'assert_' (line 482)
    assert__41743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 482)
    assert__call_result_41748 = invoke(stypy.reporting.localization.Localization(__file__, 482, 4), assert__41743, *[result_not__41746], **kwargs_41747)
    
    
    # ################# End of 'test_failures(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_failures' in the type store
    # Getting the type of 'stypy_return_type' (line 471)
    stypy_return_type_41749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_41749)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_failures'
    return stypy_return_type_41749

# Assigning a type to the variable 'test_failures' (line 471)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 0), 'test_failures', test_failures)

@norecursion
def test_big_problem(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_big_problem'
    module_type_store = module_type_store.open_function_context('test_big_problem', 485, 0, False)
    
    # Passed parameters checking function
    test_big_problem.stypy_localization = localization
    test_big_problem.stypy_type_of_self = None
    test_big_problem.stypy_type_store = module_type_store
    test_big_problem.stypy_function_name = 'test_big_problem'
    test_big_problem.stypy_param_names_list = []
    test_big_problem.stypy_varargs_param_name = None
    test_big_problem.stypy_kwargs_param_name = None
    test_big_problem.stypy_call_defaults = defaults
    test_big_problem.stypy_call_varargs = varargs
    test_big_problem.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_big_problem', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_big_problem', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_big_problem(...)' code ##################

    
    # Assigning a Num to a Name (line 486):
    
    # Assigning a Num to a Name (line 486):
    int_41750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 8), 'int')
    # Assigning a type to the variable 'n' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'n', int_41750)
    
    # Assigning a Call to a Name (line 487):
    
    # Assigning a Call to a Name (line 487):
    
    # Call to linspace(...): (line 487)
    # Processing the call arguments (line 487)
    int_41753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 20), 'int')
    int_41754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 23), 'int')
    int_41755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 26), 'int')
    # Processing the call keyword arguments (line 487)
    kwargs_41756 = {}
    # Getting the type of 'np' (line 487)
    np_41751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 487)
    linspace_41752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 8), np_41751, 'linspace')
    # Calling linspace(args, kwargs) (line 487)
    linspace_call_result_41757 = invoke(stypy.reporting.localization.Localization(__file__, 487, 8), linspace_41752, *[int_41753, int_41754, int_41755], **kwargs_41756)
    
    # Assigning a type to the variable 'x' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'x', linspace_call_result_41757)
    
    # Assigning a Call to a Name (line 488):
    
    # Assigning a Call to a Name (line 488):
    
    # Call to zeros(...): (line 488)
    # Processing the call arguments (line 488)
    
    # Obtaining an instance of the builtin type 'tuple' (line 488)
    tuple_41760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 488)
    # Adding element type (line 488)
    int_41761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 18), 'int')
    # Getting the type of 'n' (line 488)
    n_41762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 22), 'n', False)
    # Applying the binary operator '*' (line 488)
    result_mul_41763 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 18), '*', int_41761, n_41762)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 18), tuple_41760, result_mul_41763)
    # Adding element type (line 488)
    # Getting the type of 'x' (line 488)
    x_41764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 25), 'x', False)
    # Obtaining the member 'size' of a type (line 488)
    size_41765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 25), x_41764, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 18), tuple_41760, size_41765)
    
    # Processing the call keyword arguments (line 488)
    kwargs_41766 = {}
    # Getting the type of 'np' (line 488)
    np_41758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 488)
    zeros_41759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 8), np_41758, 'zeros')
    # Calling zeros(args, kwargs) (line 488)
    zeros_call_result_41767 = invoke(stypy.reporting.localization.Localization(__file__, 488, 8), zeros_41759, *[tuple_41760], **kwargs_41766)
    
    # Assigning a type to the variable 'y' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'y', zeros_call_result_41767)
    
    # Assigning a Call to a Name (line 489):
    
    # Assigning a Call to a Name (line 489):
    
    # Call to solve_bvp(...): (line 489)
    # Processing the call arguments (line 489)
    # Getting the type of 'big_fun' (line 489)
    big_fun_41769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 20), 'big_fun', False)
    # Getting the type of 'big_bc' (line 489)
    big_bc_41770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 29), 'big_bc', False)
    # Getting the type of 'x' (line 489)
    x_41771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 37), 'x', False)
    # Getting the type of 'y' (line 489)
    y_41772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 40), 'y', False)
    # Processing the call keyword arguments (line 489)
    kwargs_41773 = {}
    # Getting the type of 'solve_bvp' (line 489)
    solve_bvp_41768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 10), 'solve_bvp', False)
    # Calling solve_bvp(args, kwargs) (line 489)
    solve_bvp_call_result_41774 = invoke(stypy.reporting.localization.Localization(__file__, 489, 10), solve_bvp_41768, *[big_fun_41769, big_bc_41770, x_41771, y_41772], **kwargs_41773)
    
    # Assigning a type to the variable 'sol' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'sol', solve_bvp_call_result_41774)
    
    # Call to assert_equal(...): (line 491)
    # Processing the call arguments (line 491)
    # Getting the type of 'sol' (line 491)
    sol_41776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 17), 'sol', False)
    # Obtaining the member 'status' of a type (line 491)
    status_41777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 17), sol_41776, 'status')
    int_41778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 29), 'int')
    # Processing the call keyword arguments (line 491)
    kwargs_41779 = {}
    # Getting the type of 'assert_equal' (line 491)
    assert_equal_41775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 491)
    assert_equal_call_result_41780 = invoke(stypy.reporting.localization.Localization(__file__, 491, 4), assert_equal_41775, *[status_41777, int_41778], **kwargs_41779)
    
    
    # Call to assert_(...): (line 492)
    # Processing the call arguments (line 492)
    # Getting the type of 'sol' (line 492)
    sol_41782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'sol', False)
    # Obtaining the member 'success' of a type (line 492)
    success_41783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 12), sol_41782, 'success')
    # Processing the call keyword arguments (line 492)
    kwargs_41784 = {}
    # Getting the type of 'assert_' (line 492)
    assert__41781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 492)
    assert__call_result_41785 = invoke(stypy.reporting.localization.Localization(__file__, 492, 4), assert__41781, *[success_41783], **kwargs_41784)
    
    
    # Assigning a Call to a Name (line 494):
    
    # Assigning a Call to a Name (line 494):
    
    # Call to sol(...): (line 494)
    # Processing the call arguments (line 494)
    # Getting the type of 'x' (line 494)
    x_41788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 23), 'x', False)
    # Processing the call keyword arguments (line 494)
    kwargs_41789 = {}
    # Getting the type of 'sol' (line 494)
    sol_41786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 15), 'sol', False)
    # Obtaining the member 'sol' of a type (line 494)
    sol_41787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 15), sol_41786, 'sol')
    # Calling sol(args, kwargs) (line 494)
    sol_call_result_41790 = invoke(stypy.reporting.localization.Localization(__file__, 494, 15), sol_41787, *[x_41788], **kwargs_41789)
    
    # Assigning a type to the variable 'sol_test' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'sol_test', sol_call_result_41790)
    
    # Call to assert_allclose(...): (line 496)
    # Processing the call arguments (line 496)
    
    # Obtaining the type of the subscript
    int_41792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 29), 'int')
    # Getting the type of 'sol_test' (line 496)
    sol_test_41793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 20), 'sol_test', False)
    # Obtaining the member '__getitem__' of a type (line 496)
    getitem___41794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 20), sol_test_41793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 496)
    subscript_call_result_41795 = invoke(stypy.reporting.localization.Localization(__file__, 496, 20), getitem___41794, int_41792)
    
    
    # Call to big_sol(...): (line 496)
    # Processing the call arguments (line 496)
    # Getting the type of 'x' (line 496)
    x_41797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 41), 'x', False)
    # Getting the type of 'n' (line 496)
    n_41798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 44), 'n', False)
    # Processing the call keyword arguments (line 496)
    kwargs_41799 = {}
    # Getting the type of 'big_sol' (line 496)
    big_sol_41796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 33), 'big_sol', False)
    # Calling big_sol(args, kwargs) (line 496)
    big_sol_call_result_41800 = invoke(stypy.reporting.localization.Localization(__file__, 496, 33), big_sol_41796, *[x_41797, n_41798], **kwargs_41799)
    
    # Processing the call keyword arguments (line 496)
    kwargs_41801 = {}
    # Getting the type of 'assert_allclose' (line 496)
    assert_allclose_41791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 496)
    assert_allclose_call_result_41802 = invoke(stypy.reporting.localization.Localization(__file__, 496, 4), assert_allclose_41791, *[subscript_call_result_41795, big_sol_call_result_41800], **kwargs_41801)
    
    
    # Assigning a Call to a Name (line 498):
    
    # Assigning a Call to a Name (line 498):
    
    # Call to big_fun(...): (line 498)
    # Processing the call arguments (line 498)
    # Getting the type of 'x' (line 498)
    x_41804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 21), 'x', False)
    # Getting the type of 'sol_test' (line 498)
    sol_test_41805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 24), 'sol_test', False)
    # Processing the call keyword arguments (line 498)
    kwargs_41806 = {}
    # Getting the type of 'big_fun' (line 498)
    big_fun_41803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 13), 'big_fun', False)
    # Calling big_fun(args, kwargs) (line 498)
    big_fun_call_result_41807 = invoke(stypy.reporting.localization.Localization(__file__, 498, 13), big_fun_41803, *[x_41804, sol_test_41805], **kwargs_41806)
    
    # Assigning a type to the variable 'f_test' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'f_test', big_fun_call_result_41807)
    
    # Assigning a BinOp to a Name (line 499):
    
    # Assigning a BinOp to a Name (line 499):
    
    # Call to sol(...): (line 499)
    # Processing the call arguments (line 499)
    # Getting the type of 'x' (line 499)
    x_41810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'x', False)
    int_41811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 19), 'int')
    # Processing the call keyword arguments (line 499)
    kwargs_41812 = {}
    # Getting the type of 'sol' (line 499)
    sol_41808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'sol', False)
    # Obtaining the member 'sol' of a type (line 499)
    sol_41809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 8), sol_41808, 'sol')
    # Calling sol(args, kwargs) (line 499)
    sol_call_result_41813 = invoke(stypy.reporting.localization.Localization(__file__, 499, 8), sol_41809, *[x_41810, int_41811], **kwargs_41812)
    
    # Getting the type of 'f_test' (line 499)
    f_test_41814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 24), 'f_test')
    # Applying the binary operator '-' (line 499)
    result_sub_41815 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 8), '-', sol_call_result_41813, f_test_41814)
    
    # Assigning a type to the variable 'r' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'r', result_sub_41815)
    
    # Assigning a BinOp to a Name (line 500):
    
    # Assigning a BinOp to a Name (line 500):
    # Getting the type of 'r' (line 500)
    r_41816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 14), 'r')
    int_41817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 19), 'int')
    
    # Call to abs(...): (line 500)
    # Processing the call arguments (line 500)
    # Getting the type of 'f_test' (line 500)
    f_test_41820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 30), 'f_test', False)
    # Processing the call keyword arguments (line 500)
    kwargs_41821 = {}
    # Getting the type of 'np' (line 500)
    np_41818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 23), 'np', False)
    # Obtaining the member 'abs' of a type (line 500)
    abs_41819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 23), np_41818, 'abs')
    # Calling abs(args, kwargs) (line 500)
    abs_call_result_41822 = invoke(stypy.reporting.localization.Localization(__file__, 500, 23), abs_41819, *[f_test_41820], **kwargs_41821)
    
    # Applying the binary operator '+' (line 500)
    result_add_41823 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 19), '+', int_41817, abs_call_result_41822)
    
    # Applying the binary operator 'div' (line 500)
    result_div_41824 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 14), 'div', r_41816, result_add_41823)
    
    # Assigning a type to the variable 'rel_res' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'rel_res', result_div_41824)
    
    # Assigning a BinOp to a Name (line 501):
    
    # Assigning a BinOp to a Name (line 501):
    
    # Call to sum(...): (line 501)
    # Processing the call arguments (line 501)
    
    # Call to real(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'rel_res' (line 501)
    rel_res_41829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 30), 'rel_res', False)
    
    # Call to conj(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'rel_res' (line 501)
    rel_res_41832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 48), 'rel_res', False)
    # Processing the call keyword arguments (line 501)
    kwargs_41833 = {}
    # Getting the type of 'np' (line 501)
    np_41830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 40), 'np', False)
    # Obtaining the member 'conj' of a type (line 501)
    conj_41831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 40), np_41830, 'conj')
    # Calling conj(args, kwargs) (line 501)
    conj_call_result_41834 = invoke(stypy.reporting.localization.Localization(__file__, 501, 40), conj_41831, *[rel_res_41832], **kwargs_41833)
    
    # Applying the binary operator '*' (line 501)
    result_mul_41835 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 30), '*', rel_res_41829, conj_call_result_41834)
    
    # Processing the call keyword arguments (line 501)
    kwargs_41836 = {}
    # Getting the type of 'np' (line 501)
    np_41827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 22), 'np', False)
    # Obtaining the member 'real' of a type (line 501)
    real_41828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 22), np_41827, 'real')
    # Calling real(args, kwargs) (line 501)
    real_call_result_41837 = invoke(stypy.reporting.localization.Localization(__file__, 501, 22), real_41828, *[result_mul_41835], **kwargs_41836)
    
    # Processing the call keyword arguments (line 501)
    int_41838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 64), 'int')
    keyword_41839 = int_41838
    kwargs_41840 = {'axis': keyword_41839}
    # Getting the type of 'np' (line 501)
    np_41825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 15), 'np', False)
    # Obtaining the member 'sum' of a type (line 501)
    sum_41826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 15), np_41825, 'sum')
    # Calling sum(args, kwargs) (line 501)
    sum_call_result_41841 = invoke(stypy.reporting.localization.Localization(__file__, 501, 15), sum_41826, *[real_call_result_41837], **kwargs_41840)
    
    float_41842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 70), 'float')
    # Applying the binary operator '**' (line 501)
    result_pow_41843 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 15), '**', sum_call_result_41841, float_41842)
    
    # Assigning a type to the variable 'norm_res' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'norm_res', result_pow_41843)
    
    # Call to assert_(...): (line 502)
    # Processing the call arguments (line 502)
    
    # Call to all(...): (line 502)
    # Processing the call arguments (line 502)
    
    # Getting the type of 'norm_res' (line 502)
    norm_res_41847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 19), 'norm_res', False)
    float_41848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 30), 'float')
    # Applying the binary operator '<' (line 502)
    result_lt_41849 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 19), '<', norm_res_41847, float_41848)
    
    # Processing the call keyword arguments (line 502)
    kwargs_41850 = {}
    # Getting the type of 'np' (line 502)
    np_41845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'np', False)
    # Obtaining the member 'all' of a type (line 502)
    all_41846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 12), np_41845, 'all')
    # Calling all(args, kwargs) (line 502)
    all_call_result_41851 = invoke(stypy.reporting.localization.Localization(__file__, 502, 12), all_41846, *[result_lt_41849], **kwargs_41850)
    
    # Processing the call keyword arguments (line 502)
    kwargs_41852 = {}
    # Getting the type of 'assert_' (line 502)
    assert__41844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 502)
    assert__call_result_41853 = invoke(stypy.reporting.localization.Localization(__file__, 502, 4), assert__41844, *[all_call_result_41851], **kwargs_41852)
    
    
    # Call to assert_(...): (line 504)
    # Processing the call arguments (line 504)
    
    # Call to all(...): (line 504)
    # Processing the call arguments (line 504)
    
    # Getting the type of 'sol' (line 504)
    sol_41857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 19), 'sol', False)
    # Obtaining the member 'rms_residuals' of a type (line 504)
    rms_residuals_41858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 19), sol_41857, 'rms_residuals')
    float_41859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 39), 'float')
    # Applying the binary operator '<' (line 504)
    result_lt_41860 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 19), '<', rms_residuals_41858, float_41859)
    
    # Processing the call keyword arguments (line 504)
    kwargs_41861 = {}
    # Getting the type of 'np' (line 504)
    np_41855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'np', False)
    # Obtaining the member 'all' of a type (line 504)
    all_41856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 12), np_41855, 'all')
    # Calling all(args, kwargs) (line 504)
    all_call_result_41862 = invoke(stypy.reporting.localization.Localization(__file__, 504, 12), all_41856, *[result_lt_41860], **kwargs_41861)
    
    # Processing the call keyword arguments (line 504)
    kwargs_41863 = {}
    # Getting the type of 'assert_' (line 504)
    assert__41854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 504)
    assert__call_result_41864 = invoke(stypy.reporting.localization.Localization(__file__, 504, 4), assert__41854, *[all_call_result_41862], **kwargs_41863)
    
    
    # Call to assert_allclose(...): (line 505)
    # Processing the call arguments (line 505)
    
    # Call to sol(...): (line 505)
    # Processing the call arguments (line 505)
    # Getting the type of 'sol' (line 505)
    sol_41868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 28), 'sol', False)
    # Obtaining the member 'x' of a type (line 505)
    x_41869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 28), sol_41868, 'x')
    # Processing the call keyword arguments (line 505)
    kwargs_41870 = {}
    # Getting the type of 'sol' (line 505)
    sol_41866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 20), 'sol', False)
    # Obtaining the member 'sol' of a type (line 505)
    sol_41867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 20), sol_41866, 'sol')
    # Calling sol(args, kwargs) (line 505)
    sol_call_result_41871 = invoke(stypy.reporting.localization.Localization(__file__, 505, 20), sol_41867, *[x_41869], **kwargs_41870)
    
    # Getting the type of 'sol' (line 505)
    sol_41872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 36), 'sol', False)
    # Obtaining the member 'y' of a type (line 505)
    y_41873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 36), sol_41872, 'y')
    # Processing the call keyword arguments (line 505)
    float_41874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 48), 'float')
    keyword_41875 = float_41874
    float_41876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 60), 'float')
    keyword_41877 = float_41876
    kwargs_41878 = {'rtol': keyword_41875, 'atol': keyword_41877}
    # Getting the type of 'assert_allclose' (line 505)
    assert_allclose_41865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 505)
    assert_allclose_call_result_41879 = invoke(stypy.reporting.localization.Localization(__file__, 505, 4), assert_allclose_41865, *[sol_call_result_41871, y_41873], **kwargs_41878)
    
    
    # Call to assert_allclose(...): (line 506)
    # Processing the call arguments (line 506)
    
    # Call to sol(...): (line 506)
    # Processing the call arguments (line 506)
    # Getting the type of 'sol' (line 506)
    sol_41883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 28), 'sol', False)
    # Obtaining the member 'x' of a type (line 506)
    x_41884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 28), sol_41883, 'x')
    int_41885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 35), 'int')
    # Processing the call keyword arguments (line 506)
    kwargs_41886 = {}
    # Getting the type of 'sol' (line 506)
    sol_41881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 20), 'sol', False)
    # Obtaining the member 'sol' of a type (line 506)
    sol_41882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 20), sol_41881, 'sol')
    # Calling sol(args, kwargs) (line 506)
    sol_call_result_41887 = invoke(stypy.reporting.localization.Localization(__file__, 506, 20), sol_41882, *[x_41884, int_41885], **kwargs_41886)
    
    # Getting the type of 'sol' (line 506)
    sol_41888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 39), 'sol', False)
    # Obtaining the member 'yp' of a type (line 506)
    yp_41889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 39), sol_41888, 'yp')
    # Processing the call keyword arguments (line 506)
    float_41890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 52), 'float')
    keyword_41891 = float_41890
    float_41892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 64), 'float')
    keyword_41893 = float_41892
    kwargs_41894 = {'rtol': keyword_41891, 'atol': keyword_41893}
    # Getting the type of 'assert_allclose' (line 506)
    assert_allclose_41880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 506)
    assert_allclose_call_result_41895 = invoke(stypy.reporting.localization.Localization(__file__, 506, 4), assert_allclose_41880, *[sol_call_result_41887, yp_41889], **kwargs_41894)
    
    
    # ################# End of 'test_big_problem(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_big_problem' in the type store
    # Getting the type of 'stypy_return_type' (line 485)
    stypy_return_type_41896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_41896)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_big_problem'
    return stypy_return_type_41896

# Assigning a type to the variable 'test_big_problem' (line 485)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 0), 'test_big_problem', test_big_problem)

@norecursion
def test_shock_layer(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_shock_layer'
    module_type_store = module_type_store.open_function_context('test_shock_layer', 509, 0, False)
    
    # Passed parameters checking function
    test_shock_layer.stypy_localization = localization
    test_shock_layer.stypy_type_of_self = None
    test_shock_layer.stypy_type_store = module_type_store
    test_shock_layer.stypy_function_name = 'test_shock_layer'
    test_shock_layer.stypy_param_names_list = []
    test_shock_layer.stypy_varargs_param_name = None
    test_shock_layer.stypy_kwargs_param_name = None
    test_shock_layer.stypy_call_defaults = defaults
    test_shock_layer.stypy_call_varargs = varargs
    test_shock_layer.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_shock_layer', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_shock_layer', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_shock_layer(...)' code ##################

    
    # Assigning a Call to a Name (line 510):
    
    # Assigning a Call to a Name (line 510):
    
    # Call to linspace(...): (line 510)
    # Processing the call arguments (line 510)
    int_41899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 20), 'int')
    int_41900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 24), 'int')
    int_41901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 27), 'int')
    # Processing the call keyword arguments (line 510)
    kwargs_41902 = {}
    # Getting the type of 'np' (line 510)
    np_41897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 510)
    linspace_41898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 8), np_41897, 'linspace')
    # Calling linspace(args, kwargs) (line 510)
    linspace_call_result_41903 = invoke(stypy.reporting.localization.Localization(__file__, 510, 8), linspace_41898, *[int_41899, int_41900, int_41901], **kwargs_41902)
    
    # Assigning a type to the variable 'x' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'x', linspace_call_result_41903)
    
    # Assigning a Call to a Name (line 511):
    
    # Assigning a Call to a Name (line 511):
    
    # Call to linspace(...): (line 511)
    # Processing the call arguments (line 511)
    int_41906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 25), 'int')
    int_41907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 29), 'int')
    int_41908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 32), 'int')
    # Processing the call keyword arguments (line 511)
    kwargs_41909 = {}
    # Getting the type of 'np' (line 511)
    np_41904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 13), 'np', False)
    # Obtaining the member 'linspace' of a type (line 511)
    linspace_41905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 13), np_41904, 'linspace')
    # Calling linspace(args, kwargs) (line 511)
    linspace_call_result_41910 = invoke(stypy.reporting.localization.Localization(__file__, 511, 13), linspace_41905, *[int_41906, int_41907, int_41908], **kwargs_41909)
    
    # Assigning a type to the variable 'x_test' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'x_test', linspace_call_result_41910)
    
    # Assigning a Call to a Name (line 512):
    
    # Assigning a Call to a Name (line 512):
    
    # Call to zeros(...): (line 512)
    # Processing the call arguments (line 512)
    
    # Obtaining an instance of the builtin type 'tuple' (line 512)
    tuple_41913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 512)
    # Adding element type (line 512)
    int_41914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 18), tuple_41913, int_41914)
    # Adding element type (line 512)
    # Getting the type of 'x' (line 512)
    x_41915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 21), 'x', False)
    # Obtaining the member 'size' of a type (line 512)
    size_41916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 21), x_41915, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 18), tuple_41913, size_41916)
    
    # Processing the call keyword arguments (line 512)
    kwargs_41917 = {}
    # Getting the type of 'np' (line 512)
    np_41911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 512)
    zeros_41912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 8), np_41911, 'zeros')
    # Calling zeros(args, kwargs) (line 512)
    zeros_call_result_41918 = invoke(stypy.reporting.localization.Localization(__file__, 512, 8), zeros_41912, *[tuple_41913], **kwargs_41917)
    
    # Assigning a type to the variable 'y' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'y', zeros_call_result_41918)
    
    # Assigning a Call to a Name (line 513):
    
    # Assigning a Call to a Name (line 513):
    
    # Call to solve_bvp(...): (line 513)
    # Processing the call arguments (line 513)
    # Getting the type of 'shock_fun' (line 513)
    shock_fun_41920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 20), 'shock_fun', False)
    # Getting the type of 'shock_bc' (line 513)
    shock_bc_41921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 31), 'shock_bc', False)
    # Getting the type of 'x' (line 513)
    x_41922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 41), 'x', False)
    # Getting the type of 'y' (line 513)
    y_41923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 44), 'y', False)
    # Processing the call keyword arguments (line 513)
    kwargs_41924 = {}
    # Getting the type of 'solve_bvp' (line 513)
    solve_bvp_41919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 10), 'solve_bvp', False)
    # Calling solve_bvp(args, kwargs) (line 513)
    solve_bvp_call_result_41925 = invoke(stypy.reporting.localization.Localization(__file__, 513, 10), solve_bvp_41919, *[shock_fun_41920, shock_bc_41921, x_41922, y_41923], **kwargs_41924)
    
    # Assigning a type to the variable 'sol' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'sol', solve_bvp_call_result_41925)
    
    # Call to assert_equal(...): (line 515)
    # Processing the call arguments (line 515)
    # Getting the type of 'sol' (line 515)
    sol_41927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 17), 'sol', False)
    # Obtaining the member 'status' of a type (line 515)
    status_41928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 17), sol_41927, 'status')
    int_41929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 29), 'int')
    # Processing the call keyword arguments (line 515)
    kwargs_41930 = {}
    # Getting the type of 'assert_equal' (line 515)
    assert_equal_41926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 515)
    assert_equal_call_result_41931 = invoke(stypy.reporting.localization.Localization(__file__, 515, 4), assert_equal_41926, *[status_41928, int_41929], **kwargs_41930)
    
    
    # Call to assert_(...): (line 516)
    # Processing the call arguments (line 516)
    # Getting the type of 'sol' (line 516)
    sol_41933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'sol', False)
    # Obtaining the member 'success' of a type (line 516)
    success_41934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 12), sol_41933, 'success')
    # Processing the call keyword arguments (line 516)
    kwargs_41935 = {}
    # Getting the type of 'assert_' (line 516)
    assert__41932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 516)
    assert__call_result_41936 = invoke(stypy.reporting.localization.Localization(__file__, 516, 4), assert__41932, *[success_41934], **kwargs_41935)
    
    
    # Call to assert_(...): (line 518)
    # Processing the call arguments (line 518)
    
    # Getting the type of 'sol' (line 518)
    sol_41938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'sol', False)
    # Obtaining the member 'x' of a type (line 518)
    x_41939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 12), sol_41938, 'x')
    # Obtaining the member 'size' of a type (line 518)
    size_41940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 12), x_41939, 'size')
    int_41941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 25), 'int')
    # Applying the binary operator '<' (line 518)
    result_lt_41942 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 12), '<', size_41940, int_41941)
    
    # Processing the call keyword arguments (line 518)
    kwargs_41943 = {}
    # Getting the type of 'assert_' (line 518)
    assert__41937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 518)
    assert__call_result_41944 = invoke(stypy.reporting.localization.Localization(__file__, 518, 4), assert__41937, *[result_lt_41942], **kwargs_41943)
    
    
    # Assigning a Call to a Name (line 520):
    
    # Assigning a Call to a Name (line 520):
    
    # Call to sol(...): (line 520)
    # Processing the call arguments (line 520)
    # Getting the type of 'x_test' (line 520)
    x_test_41947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 23), 'x_test', False)
    # Processing the call keyword arguments (line 520)
    kwargs_41948 = {}
    # Getting the type of 'sol' (line 520)
    sol_41945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 15), 'sol', False)
    # Obtaining the member 'sol' of a type (line 520)
    sol_41946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 15), sol_41945, 'sol')
    # Calling sol(args, kwargs) (line 520)
    sol_call_result_41949 = invoke(stypy.reporting.localization.Localization(__file__, 520, 15), sol_41946, *[x_test_41947], **kwargs_41948)
    
    # Assigning a type to the variable 'sol_test' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'sol_test', sol_call_result_41949)
    
    # Call to assert_allclose(...): (line 521)
    # Processing the call arguments (line 521)
    
    # Obtaining the type of the subscript
    int_41951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 29), 'int')
    # Getting the type of 'sol_test' (line 521)
    sol_test_41952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 20), 'sol_test', False)
    # Obtaining the member '__getitem__' of a type (line 521)
    getitem___41953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 20), sol_test_41952, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 521)
    subscript_call_result_41954 = invoke(stypy.reporting.localization.Localization(__file__, 521, 20), getitem___41953, int_41951)
    
    
    # Call to shock_sol(...): (line 521)
    # Processing the call arguments (line 521)
    # Getting the type of 'x_test' (line 521)
    x_test_41956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 43), 'x_test', False)
    # Processing the call keyword arguments (line 521)
    kwargs_41957 = {}
    # Getting the type of 'shock_sol' (line 521)
    shock_sol_41955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 33), 'shock_sol', False)
    # Calling shock_sol(args, kwargs) (line 521)
    shock_sol_call_result_41958 = invoke(stypy.reporting.localization.Localization(__file__, 521, 33), shock_sol_41955, *[x_test_41956], **kwargs_41957)
    
    # Processing the call keyword arguments (line 521)
    float_41959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 57), 'float')
    keyword_41960 = float_41959
    float_41961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 68), 'float')
    keyword_41962 = float_41961
    kwargs_41963 = {'rtol': keyword_41960, 'atol': keyword_41962}
    # Getting the type of 'assert_allclose' (line 521)
    assert_allclose_41950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 521)
    assert_allclose_call_result_41964 = invoke(stypy.reporting.localization.Localization(__file__, 521, 4), assert_allclose_41950, *[subscript_call_result_41954, shock_sol_call_result_41958], **kwargs_41963)
    
    
    # Assigning a Call to a Name (line 523):
    
    # Assigning a Call to a Name (line 523):
    
    # Call to shock_fun(...): (line 523)
    # Processing the call arguments (line 523)
    # Getting the type of 'x_test' (line 523)
    x_test_41966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 23), 'x_test', False)
    # Getting the type of 'sol_test' (line 523)
    sol_test_41967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 31), 'sol_test', False)
    # Processing the call keyword arguments (line 523)
    kwargs_41968 = {}
    # Getting the type of 'shock_fun' (line 523)
    shock_fun_41965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 13), 'shock_fun', False)
    # Calling shock_fun(args, kwargs) (line 523)
    shock_fun_call_result_41969 = invoke(stypy.reporting.localization.Localization(__file__, 523, 13), shock_fun_41965, *[x_test_41966, sol_test_41967], **kwargs_41968)
    
    # Assigning a type to the variable 'f_test' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'f_test', shock_fun_call_result_41969)
    
    # Assigning a BinOp to a Name (line 524):
    
    # Assigning a BinOp to a Name (line 524):
    
    # Call to sol(...): (line 524)
    # Processing the call arguments (line 524)
    # Getting the type of 'x_test' (line 524)
    x_test_41972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), 'x_test', False)
    int_41973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 24), 'int')
    # Processing the call keyword arguments (line 524)
    kwargs_41974 = {}
    # Getting the type of 'sol' (line 524)
    sol_41970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'sol', False)
    # Obtaining the member 'sol' of a type (line 524)
    sol_41971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 8), sol_41970, 'sol')
    # Calling sol(args, kwargs) (line 524)
    sol_call_result_41975 = invoke(stypy.reporting.localization.Localization(__file__, 524, 8), sol_41971, *[x_test_41972, int_41973], **kwargs_41974)
    
    # Getting the type of 'f_test' (line 524)
    f_test_41976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 29), 'f_test')
    # Applying the binary operator '-' (line 524)
    result_sub_41977 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 8), '-', sol_call_result_41975, f_test_41976)
    
    # Assigning a type to the variable 'r' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'r', result_sub_41977)
    
    # Assigning a BinOp to a Name (line 525):
    
    # Assigning a BinOp to a Name (line 525):
    # Getting the type of 'r' (line 525)
    r_41978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 14), 'r')
    int_41979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 19), 'int')
    
    # Call to abs(...): (line 525)
    # Processing the call arguments (line 525)
    # Getting the type of 'f_test' (line 525)
    f_test_41982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 30), 'f_test', False)
    # Processing the call keyword arguments (line 525)
    kwargs_41983 = {}
    # Getting the type of 'np' (line 525)
    np_41980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 23), 'np', False)
    # Obtaining the member 'abs' of a type (line 525)
    abs_41981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 23), np_41980, 'abs')
    # Calling abs(args, kwargs) (line 525)
    abs_call_result_41984 = invoke(stypy.reporting.localization.Localization(__file__, 525, 23), abs_41981, *[f_test_41982], **kwargs_41983)
    
    # Applying the binary operator '+' (line 525)
    result_add_41985 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 19), '+', int_41979, abs_call_result_41984)
    
    # Applying the binary operator 'div' (line 525)
    result_div_41986 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 14), 'div', r_41978, result_add_41985)
    
    # Assigning a type to the variable 'rel_res' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'rel_res', result_div_41986)
    
    # Assigning a BinOp to a Name (line 526):
    
    # Assigning a BinOp to a Name (line 526):
    
    # Call to sum(...): (line 526)
    # Processing the call arguments (line 526)
    # Getting the type of 'rel_res' (line 526)
    rel_res_41989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 22), 'rel_res', False)
    int_41990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 33), 'int')
    # Applying the binary operator '**' (line 526)
    result_pow_41991 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 22), '**', rel_res_41989, int_41990)
    
    # Processing the call keyword arguments (line 526)
    int_41992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 41), 'int')
    keyword_41993 = int_41992
    kwargs_41994 = {'axis': keyword_41993}
    # Getting the type of 'np' (line 526)
    np_41987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 15), 'np', False)
    # Obtaining the member 'sum' of a type (line 526)
    sum_41988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 15), np_41987, 'sum')
    # Calling sum(args, kwargs) (line 526)
    sum_call_result_41995 = invoke(stypy.reporting.localization.Localization(__file__, 526, 15), sum_41988, *[result_pow_41991], **kwargs_41994)
    
    float_41996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 47), 'float')
    # Applying the binary operator '**' (line 526)
    result_pow_41997 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 15), '**', sum_call_result_41995, float_41996)
    
    # Assigning a type to the variable 'norm_res' (line 526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 4), 'norm_res', result_pow_41997)
    
    # Call to assert_(...): (line 528)
    # Processing the call arguments (line 528)
    
    # Call to all(...): (line 528)
    # Processing the call arguments (line 528)
    
    # Getting the type of 'norm_res' (line 528)
    norm_res_42001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 19), 'norm_res', False)
    float_42002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 30), 'float')
    # Applying the binary operator '<' (line 528)
    result_lt_42003 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 19), '<', norm_res_42001, float_42002)
    
    # Processing the call keyword arguments (line 528)
    kwargs_42004 = {}
    # Getting the type of 'np' (line 528)
    np_41999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'np', False)
    # Obtaining the member 'all' of a type (line 528)
    all_42000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 12), np_41999, 'all')
    # Calling all(args, kwargs) (line 528)
    all_call_result_42005 = invoke(stypy.reporting.localization.Localization(__file__, 528, 12), all_42000, *[result_lt_42003], **kwargs_42004)
    
    # Processing the call keyword arguments (line 528)
    kwargs_42006 = {}
    # Getting the type of 'assert_' (line 528)
    assert__41998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 528)
    assert__call_result_42007 = invoke(stypy.reporting.localization.Localization(__file__, 528, 4), assert__41998, *[all_call_result_42005], **kwargs_42006)
    
    
    # Call to assert_allclose(...): (line 529)
    # Processing the call arguments (line 529)
    
    # Call to sol(...): (line 529)
    # Processing the call arguments (line 529)
    # Getting the type of 'sol' (line 529)
    sol_42011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 28), 'sol', False)
    # Obtaining the member 'x' of a type (line 529)
    x_42012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 28), sol_42011, 'x')
    # Processing the call keyword arguments (line 529)
    kwargs_42013 = {}
    # Getting the type of 'sol' (line 529)
    sol_42009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 20), 'sol', False)
    # Obtaining the member 'sol' of a type (line 529)
    sol_42010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 20), sol_42009, 'sol')
    # Calling sol(args, kwargs) (line 529)
    sol_call_result_42014 = invoke(stypy.reporting.localization.Localization(__file__, 529, 20), sol_42010, *[x_42012], **kwargs_42013)
    
    # Getting the type of 'sol' (line 529)
    sol_42015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 36), 'sol', False)
    # Obtaining the member 'y' of a type (line 529)
    y_42016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 36), sol_42015, 'y')
    # Processing the call keyword arguments (line 529)
    float_42017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 48), 'float')
    keyword_42018 = float_42017
    float_42019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 60), 'float')
    keyword_42020 = float_42019
    kwargs_42021 = {'rtol': keyword_42018, 'atol': keyword_42020}
    # Getting the type of 'assert_allclose' (line 529)
    assert_allclose_42008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 529)
    assert_allclose_call_result_42022 = invoke(stypy.reporting.localization.Localization(__file__, 529, 4), assert_allclose_42008, *[sol_call_result_42014, y_42016], **kwargs_42021)
    
    
    # Call to assert_allclose(...): (line 530)
    # Processing the call arguments (line 530)
    
    # Call to sol(...): (line 530)
    # Processing the call arguments (line 530)
    # Getting the type of 'sol' (line 530)
    sol_42026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 28), 'sol', False)
    # Obtaining the member 'x' of a type (line 530)
    x_42027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 28), sol_42026, 'x')
    int_42028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 35), 'int')
    # Processing the call keyword arguments (line 530)
    kwargs_42029 = {}
    # Getting the type of 'sol' (line 530)
    sol_42024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 20), 'sol', False)
    # Obtaining the member 'sol' of a type (line 530)
    sol_42025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 20), sol_42024, 'sol')
    # Calling sol(args, kwargs) (line 530)
    sol_call_result_42030 = invoke(stypy.reporting.localization.Localization(__file__, 530, 20), sol_42025, *[x_42027, int_42028], **kwargs_42029)
    
    # Getting the type of 'sol' (line 530)
    sol_42031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 39), 'sol', False)
    # Obtaining the member 'yp' of a type (line 530)
    yp_42032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 39), sol_42031, 'yp')
    # Processing the call keyword arguments (line 530)
    float_42033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 52), 'float')
    keyword_42034 = float_42033
    float_42035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 64), 'float')
    keyword_42036 = float_42035
    kwargs_42037 = {'rtol': keyword_42034, 'atol': keyword_42036}
    # Getting the type of 'assert_allclose' (line 530)
    assert_allclose_42023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 530)
    assert_allclose_call_result_42038 = invoke(stypy.reporting.localization.Localization(__file__, 530, 4), assert_allclose_42023, *[sol_call_result_42030, yp_42032], **kwargs_42037)
    
    
    # ################# End of 'test_shock_layer(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_shock_layer' in the type store
    # Getting the type of 'stypy_return_type' (line 509)
    stypy_return_type_42039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_42039)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_shock_layer'
    return stypy_return_type_42039

# Assigning a type to the variable 'test_shock_layer' (line 509)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 0), 'test_shock_layer', test_shock_layer)

@norecursion
def test_verbose(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_verbose'
    module_type_store = module_type_store.open_function_context('test_verbose', 533, 0, False)
    
    # Passed parameters checking function
    test_verbose.stypy_localization = localization
    test_verbose.stypy_type_of_self = None
    test_verbose.stypy_type_store = module_type_store
    test_verbose.stypy_function_name = 'test_verbose'
    test_verbose.stypy_param_names_list = []
    test_verbose.stypy_varargs_param_name = None
    test_verbose.stypy_kwargs_param_name = None
    test_verbose.stypy_call_defaults = defaults
    test_verbose.stypy_call_varargs = varargs
    test_verbose.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_verbose', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_verbose', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_verbose(...)' code ##################

    
    # Assigning a Call to a Name (line 535):
    
    # Assigning a Call to a Name (line 535):
    
    # Call to linspace(...): (line 535)
    # Processing the call arguments (line 535)
    int_42042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 20), 'int')
    int_42043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 23), 'int')
    int_42044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 26), 'int')
    # Processing the call keyword arguments (line 535)
    kwargs_42045 = {}
    # Getting the type of 'np' (line 535)
    np_42040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 535)
    linspace_42041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 8), np_42040, 'linspace')
    # Calling linspace(args, kwargs) (line 535)
    linspace_call_result_42046 = invoke(stypy.reporting.localization.Localization(__file__, 535, 8), linspace_42041, *[int_42042, int_42043, int_42044], **kwargs_42045)
    
    # Assigning a type to the variable 'x' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'x', linspace_call_result_42046)
    
    # Assigning a Call to a Name (line 536):
    
    # Assigning a Call to a Name (line 536):
    
    # Call to zeros(...): (line 536)
    # Processing the call arguments (line 536)
    
    # Obtaining an instance of the builtin type 'tuple' (line 536)
    tuple_42049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 536)
    # Adding element type (line 536)
    int_42050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 18), tuple_42049, int_42050)
    # Adding element type (line 536)
    
    # Obtaining the type of the subscript
    int_42051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 29), 'int')
    # Getting the type of 'x' (line 536)
    x_42052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 536)
    shape_42053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 21), x_42052, 'shape')
    # Obtaining the member '__getitem__' of a type (line 536)
    getitem___42054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 21), shape_42053, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 536)
    subscript_call_result_42055 = invoke(stypy.reporting.localization.Localization(__file__, 536, 21), getitem___42054, int_42051)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 18), tuple_42049, subscript_call_result_42055)
    
    # Processing the call keyword arguments (line 536)
    kwargs_42056 = {}
    # Getting the type of 'np' (line 536)
    np_42047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 536)
    zeros_42048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), np_42047, 'zeros')
    # Calling zeros(args, kwargs) (line 536)
    zeros_call_result_42057 = invoke(stypy.reporting.localization.Localization(__file__, 536, 8), zeros_42048, *[tuple_42049], **kwargs_42056)
    
    # Assigning a type to the variable 'y' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'y', zeros_call_result_42057)
    
    
    # Obtaining an instance of the builtin type 'list' (line 537)
    list_42058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 537)
    # Adding element type (line 537)
    int_42059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 19), list_42058, int_42059)
    # Adding element type (line 537)
    int_42060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 19), list_42058, int_42060)
    # Adding element type (line 537)
    int_42061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 19), list_42058, int_42061)
    
    # Testing the type of a for loop iterable (line 537)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 537, 4), list_42058)
    # Getting the type of the for loop variable (line 537)
    for_loop_var_42062 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 537, 4), list_42058)
    # Assigning a type to the variable 'verbose' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'verbose', for_loop_var_42062)
    # SSA begins for a for statement (line 537)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Attribute to a Name (line 538):
    
    # Assigning a Attribute to a Name (line 538):
    # Getting the type of 'sys' (line 538)
    sys_42063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 21), 'sys')
    # Obtaining the member 'stdout' of a type (line 538)
    stdout_42064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 21), sys_42063, 'stdout')
    # Assigning a type to the variable 'old_stdout' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'old_stdout', stdout_42064)
    
    # Assigning a Call to a Attribute (line 539):
    
    # Assigning a Call to a Attribute (line 539):
    
    # Call to StringIO(...): (line 539)
    # Processing the call keyword arguments (line 539)
    kwargs_42066 = {}
    # Getting the type of 'StringIO' (line 539)
    StringIO_42065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 21), 'StringIO', False)
    # Calling StringIO(args, kwargs) (line 539)
    StringIO_call_result_42067 = invoke(stypy.reporting.localization.Localization(__file__, 539, 21), StringIO_42065, *[], **kwargs_42066)
    
    # Getting the type of 'sys' (line 539)
    sys_42068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'sys')
    # Setting the type of the member 'stdout' of a type (line 539)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 8), sys_42068, 'stdout', StringIO_call_result_42067)
    
    # Try-finally block (line 540)
    
    # Assigning a Call to a Name (line 541):
    
    # Assigning a Call to a Name (line 541):
    
    # Call to solve_bvp(...): (line 541)
    # Processing the call arguments (line 541)
    # Getting the type of 'exp_fun' (line 541)
    exp_fun_42070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 28), 'exp_fun', False)
    # Getting the type of 'exp_bc' (line 541)
    exp_bc_42071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 37), 'exp_bc', False)
    # Getting the type of 'x' (line 541)
    x_42072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 45), 'x', False)
    # Getting the type of 'y' (line 541)
    y_42073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 48), 'y', False)
    # Processing the call keyword arguments (line 541)
    # Getting the type of 'verbose' (line 541)
    verbose_42074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 59), 'verbose', False)
    keyword_42075 = verbose_42074
    kwargs_42076 = {'verbose': keyword_42075}
    # Getting the type of 'solve_bvp' (line 541)
    solve_bvp_42069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 18), 'solve_bvp', False)
    # Calling solve_bvp(args, kwargs) (line 541)
    solve_bvp_call_result_42077 = invoke(stypy.reporting.localization.Localization(__file__, 541, 18), solve_bvp_42069, *[exp_fun_42070, exp_bc_42071, x_42072, y_42073], **kwargs_42076)
    
    # Assigning a type to the variable 'sol' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'sol', solve_bvp_call_result_42077)
    
    # Assigning a Call to a Name (line 542):
    
    # Assigning a Call to a Name (line 542):
    
    # Call to getvalue(...): (line 542)
    # Processing the call keyword arguments (line 542)
    kwargs_42081 = {}
    # Getting the type of 'sys' (line 542)
    sys_42078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 19), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 542)
    stdout_42079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 19), sys_42078, 'stdout')
    # Obtaining the member 'getvalue' of a type (line 542)
    getvalue_42080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 19), stdout_42079, 'getvalue')
    # Calling getvalue(args, kwargs) (line 542)
    getvalue_call_result_42082 = invoke(stypy.reporting.localization.Localization(__file__, 542, 19), getvalue_42080, *[], **kwargs_42081)
    
    # Assigning a type to the variable 'text' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'text', getvalue_call_result_42082)
    
    # finally branch of the try-finally block (line 540)
    
    # Assigning a Name to a Attribute (line 544):
    
    # Assigning a Name to a Attribute (line 544):
    # Getting the type of 'old_stdout' (line 544)
    old_stdout_42083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 25), 'old_stdout')
    # Getting the type of 'sys' (line 544)
    sys_42084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'sys')
    # Setting the type of the member 'stdout' of a type (line 544)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 12), sys_42084, 'stdout', old_stdout_42083)
    
    
    # Call to assert_(...): (line 546)
    # Processing the call arguments (line 546)
    # Getting the type of 'sol' (line 546)
    sol_42086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'sol', False)
    # Obtaining the member 'success' of a type (line 546)
    success_42087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 16), sol_42086, 'success')
    # Processing the call keyword arguments (line 546)
    kwargs_42088 = {}
    # Getting the type of 'assert_' (line 546)
    assert__42085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 546)
    assert__call_result_42089 = invoke(stypy.reporting.localization.Localization(__file__, 546, 8), assert__42085, *[success_42087], **kwargs_42088)
    
    
    
    # Getting the type of 'verbose' (line 547)
    verbose_42090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 11), 'verbose')
    int_42091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 22), 'int')
    # Applying the binary operator '==' (line 547)
    result_eq_42092 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 11), '==', verbose_42090, int_42091)
    
    # Testing the type of an if condition (line 547)
    if_condition_42093 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 547, 8), result_eq_42092)
    # Assigning a type to the variable 'if_condition_42093' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'if_condition_42093', if_condition_42093)
    # SSA begins for if statement (line 547)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_(...): (line 548)
    # Processing the call arguments (line 548)
    
    # Getting the type of 'text' (line 548)
    text_42095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 24), 'text', False)
    # Applying the 'not' unary operator (line 548)
    result_not__42096 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 20), 'not', text_42095)
    
    # Getting the type of 'text' (line 548)
    text_42097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 30), 'text', False)
    # Processing the call keyword arguments (line 548)
    kwargs_42098 = {}
    # Getting the type of 'assert_' (line 548)
    assert__42094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 548)
    assert__call_result_42099 = invoke(stypy.reporting.localization.Localization(__file__, 548, 12), assert__42094, *[result_not__42096, text_42097], **kwargs_42098)
    
    # SSA join for if statement (line 547)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbose' (line 549)
    verbose_42100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 11), 'verbose')
    int_42101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 22), 'int')
    # Applying the binary operator '>=' (line 549)
    result_ge_42102 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 11), '>=', verbose_42100, int_42101)
    
    # Testing the type of an if condition (line 549)
    if_condition_42103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 549, 8), result_ge_42102)
    # Assigning a type to the variable 'if_condition_42103' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'if_condition_42103', if_condition_42103)
    # SSA begins for if statement (line 549)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_(...): (line 550)
    # Processing the call arguments (line 550)
    
    str_42105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 20), 'str', 'Solved in')
    # Getting the type of 'text' (line 550)
    text_42106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 35), 'text', False)
    # Applying the binary operator 'in' (line 550)
    result_contains_42107 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 20), 'in', str_42105, text_42106)
    
    # Getting the type of 'text' (line 550)
    text_42108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 41), 'text', False)
    # Processing the call keyword arguments (line 550)
    kwargs_42109 = {}
    # Getting the type of 'assert_' (line 550)
    assert__42104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 550)
    assert__call_result_42110 = invoke(stypy.reporting.localization.Localization(__file__, 550, 12), assert__42104, *[result_contains_42107, text_42108], **kwargs_42109)
    
    # SSA join for if statement (line 549)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbose' (line 551)
    verbose_42111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 11), 'verbose')
    int_42112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 22), 'int')
    # Applying the binary operator '>=' (line 551)
    result_ge_42113 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 11), '>=', verbose_42111, int_42112)
    
    # Testing the type of an if condition (line 551)
    if_condition_42114 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 551, 8), result_ge_42113)
    # Assigning a type to the variable 'if_condition_42114' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'if_condition_42114', if_condition_42114)
    # SSA begins for if statement (line 551)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_(...): (line 552)
    # Processing the call arguments (line 552)
    
    str_42116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 20), 'str', 'Max residual')
    # Getting the type of 'text' (line 552)
    text_42117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 38), 'text', False)
    # Applying the binary operator 'in' (line 552)
    result_contains_42118 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 20), 'in', str_42116, text_42117)
    
    # Getting the type of 'text' (line 552)
    text_42119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 44), 'text', False)
    # Processing the call keyword arguments (line 552)
    kwargs_42120 = {}
    # Getting the type of 'assert_' (line 552)
    assert__42115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 552)
    assert__call_result_42121 = invoke(stypy.reporting.localization.Localization(__file__, 552, 12), assert__42115, *[result_contains_42118, text_42119], **kwargs_42120)
    
    # SSA join for if statement (line 551)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_verbose(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_verbose' in the type store
    # Getting the type of 'stypy_return_type' (line 533)
    stypy_return_type_42122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_42122)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_verbose'
    return stypy_return_type_42122

# Assigning a type to the variable 'test_verbose' (line 533)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 0), 'test_verbose', test_verbose)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
