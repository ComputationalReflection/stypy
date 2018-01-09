
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division
2: 
3: import math
4: from itertools import product
5: 
6: import numpy as np
7: from numpy.testing import assert_allclose, assert_equal, assert_
8: from pytest import raises as assert_raises
9: 
10: from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
11: 
12: from scipy.optimize._numdiff import (
13:     _adjust_scheme_to_bounds, approx_derivative, check_derivative,
14:     group_columns)
15: 
16: 
17: def test_group_columns():
18:     structure = [
19:         [1, 1, 0, 0, 0, 0],
20:         [1, 1, 1, 0, 0, 0],
21:         [0, 1, 1, 1, 0, 0],
22:         [0, 0, 1, 1, 1, 0],
23:         [0, 0, 0, 1, 1, 1],
24:         [0, 0, 0, 0, 1, 1],
25:         [0, 0, 0, 0, 0, 0]
26:     ]
27:     for transform in [np.asarray, csr_matrix, csc_matrix, lil_matrix]:
28:         A = transform(structure)
29:         order = np.arange(6)
30:         groups_true = np.array([0, 1, 2, 0, 1, 2])
31:         groups = group_columns(A, order)
32:         assert_equal(groups, groups_true)
33: 
34:         order = [1, 2, 4, 3, 5, 0]
35:         groups_true = np.array([2, 0, 1, 2, 0, 1])
36:         groups = group_columns(A, order)
37:         assert_equal(groups, groups_true)
38: 
39:     # Test repeatability.
40:     groups_1 = group_columns(A)
41:     groups_2 = group_columns(A)
42:     assert_equal(groups_1, groups_2)
43: 
44: 
45: class TestAdjustSchemeToBounds(object):
46:     def test_no_bounds(self):
47:         x0 = np.zeros(3)
48:         h = np.ones(3) * 1e-2
49:         inf_lower = np.empty_like(x0)
50:         inf_upper = np.empty_like(x0)
51:         inf_lower.fill(-np.inf)
52:         inf_upper.fill(np.inf)
53: 
54:         h_adjusted, one_sided = _adjust_scheme_to_bounds(
55:             x0, h, 1, '1-sided', inf_lower, inf_upper)
56:         assert_allclose(h_adjusted, h)
57:         assert_(np.all(one_sided))
58: 
59:         h_adjusted, one_sided = _adjust_scheme_to_bounds(
60:             x0, h, 2, '1-sided', inf_lower, inf_upper)
61:         assert_allclose(h_adjusted, h)
62:         assert_(np.all(one_sided))
63: 
64:         h_adjusted, one_sided = _adjust_scheme_to_bounds(
65:             x0, h, 1, '2-sided', inf_lower, inf_upper)
66:         assert_allclose(h_adjusted, h)
67:         assert_(np.all(~one_sided))
68: 
69:         h_adjusted, one_sided = _adjust_scheme_to_bounds(
70:             x0, h, 2, '2-sided', inf_lower, inf_upper)
71:         assert_allclose(h_adjusted, h)
72:         assert_(np.all(~one_sided))
73: 
74:     def test_with_bound(self):
75:         x0 = np.array([0.0, 0.85, -0.85])
76:         lb = -np.ones(3)
77:         ub = np.ones(3)
78:         h = np.array([1, 1, -1]) * 1e-1
79: 
80:         h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 1, '1-sided', lb, ub)
81:         assert_allclose(h_adjusted, h)
82: 
83:         h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 2, '1-sided', lb, ub)
84:         assert_allclose(h_adjusted, np.array([1, -1, 1]) * 1e-1)
85: 
86:         h_adjusted, one_sided = _adjust_scheme_to_bounds(
87:             x0, h, 1, '2-sided', lb, ub)
88:         assert_allclose(h_adjusted, np.abs(h))
89:         assert_(np.all(~one_sided))
90: 
91:         h_adjusted, one_sided = _adjust_scheme_to_bounds(
92:             x0, h, 2, '2-sided', lb, ub)
93:         assert_allclose(h_adjusted, np.array([1, -1, 1]) * 1e-1)
94:         assert_equal(one_sided, np.array([False, True, True]))
95: 
96:     def test_tight_bounds(self):
97:         lb = np.array([-0.03, -0.03])
98:         ub = np.array([0.05, 0.05])
99:         x0 = np.array([0.0, 0.03])
100:         h = np.array([-0.1, -0.1])
101: 
102:         h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 1, '1-sided', lb, ub)
103:         assert_allclose(h_adjusted, np.array([0.05, -0.06]))
104: 
105:         h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 2, '1-sided', lb, ub)
106:         assert_allclose(h_adjusted, np.array([0.025, -0.03]))
107: 
108:         h_adjusted, one_sided = _adjust_scheme_to_bounds(
109:             x0, h, 1, '2-sided', lb, ub)
110:         assert_allclose(h_adjusted, np.array([0.03, -0.03]))
111:         assert_equal(one_sided, np.array([False, True]))
112: 
113:         h_adjusted, one_sided = _adjust_scheme_to_bounds(
114:             x0, h, 2, '2-sided', lb, ub)
115:         assert_allclose(h_adjusted, np.array([0.015, -0.015]))
116:         assert_equal(one_sided, np.array([False, True]))
117: 
118: 
119: class TestApproxDerivativesDense(object):
120:     def fun_scalar_scalar(self, x):
121:         return np.sinh(x)
122: 
123:     def jac_scalar_scalar(self, x):
124:         return np.cosh(x)
125: 
126:     def fun_scalar_vector(self, x):
127:         return np.array([x[0]**2, np.tan(x[0]), np.exp(x[0])])
128: 
129:     def jac_scalar_vector(self, x):
130:         return np.array(
131:             [2 * x[0], np.cos(x[0]) ** -2, np.exp(x[0])]).reshape(-1, 1)
132: 
133:     def fun_vector_scalar(self, x):
134:         return np.sin(x[0] * x[1]) * np.log(x[0])
135: 
136:     def wrong_dimensions_fun(self, x):
137:         return np.array([x**2, np.tan(x), np.exp(x)])
138: 
139:     def jac_vector_scalar(self, x):
140:         return np.array([
141:             x[1] * np.cos(x[0] * x[1]) * np.log(x[0]) +
142:             np.sin(x[0] * x[1]) / x[0],
143:             x[0] * np.cos(x[0] * x[1]) * np.log(x[0])
144:         ])
145: 
146:     def fun_vector_vector(self, x):
147:         return np.array([
148:             x[0] * np.sin(x[1]),
149:             x[1] * np.cos(x[0]),
150:             x[0] ** 3 * x[1] ** -0.5
151:         ])
152: 
153:     def jac_vector_vector(self, x):
154:         return np.array([
155:             [np.sin(x[1]), x[0] * np.cos(x[1])],
156:             [-x[1] * np.sin(x[0]), np.cos(x[0])],
157:             [3 * x[0] ** 2 * x[1] ** -0.5, -0.5 * x[0] ** 3 * x[1] ** -1.5]
158:         ])
159: 
160:     def fun_parametrized(self, x, c0, c1=1.0):
161:         return np.array([np.exp(c0 * x[0]), np.exp(c1 * x[1])])
162: 
163:     def jac_parametrized(self, x, c0, c1=0.1):
164:         return np.array([
165:             [c0 * np.exp(c0 * x[0]), 0],
166:             [0, c1 * np.exp(c1 * x[1])]
167:         ])
168: 
169:     def fun_with_nan(self, x):
170:         return x if np.abs(x) <= 1e-8 else np.nan
171: 
172:     def jac_with_nan(self, x):
173:         return 1.0 if np.abs(x) <= 1e-8 else np.nan
174: 
175:     def fun_zero_jacobian(self, x):
176:         return np.array([x[0] * x[1], np.cos(x[0] * x[1])])
177: 
178:     def jac_zero_jacobian(self, x):
179:         return np.array([
180:             [x[1], x[0]],
181:             [-x[1] * np.sin(x[0] * x[1]), -x[0] * np.sin(x[0] * x[1])]
182:         ])
183: 
184:     def fun_non_numpy(self, x):
185:         return math.exp(x)
186: 
187:     def jac_non_numpy(self, x):
188:         return math.exp(x)
189: 
190:     def test_scalar_scalar(self):
191:         x0 = 1.0
192:         jac_diff_2 = approx_derivative(self.fun_scalar_scalar, x0,
193:                                        method='2-point')
194:         jac_diff_3 = approx_derivative(self.fun_scalar_scalar, x0)
195:         jac_diff_4 = approx_derivative(self.fun_scalar_scalar, x0,
196:                                        method='cs')
197:         jac_true = self.jac_scalar_scalar(x0)
198:         assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
199:         assert_allclose(jac_diff_3, jac_true, rtol=1e-9)
200:         assert_allclose(jac_diff_4, jac_true, rtol=1e-12)
201: 
202:     def test_scalar_vector(self):
203:         x0 = 0.5
204:         jac_diff_2 = approx_derivative(self.fun_scalar_vector, x0,
205:                                        method='2-point')
206:         jac_diff_3 = approx_derivative(self.fun_scalar_vector, x0)
207:         jac_diff_4 = approx_derivative(self.fun_scalar_vector, x0,
208:                                        method='cs')
209:         jac_true = self.jac_scalar_vector(np.atleast_1d(x0))
210:         assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
211:         assert_allclose(jac_diff_3, jac_true, rtol=1e-9)
212:         assert_allclose(jac_diff_4, jac_true, rtol=1e-12)
213: 
214:     def test_vector_scalar(self):
215:         x0 = np.array([100.0, -0.5])
216:         jac_diff_2 = approx_derivative(self.fun_vector_scalar, x0,
217:                                        method='2-point')
218:         jac_diff_3 = approx_derivative(self.fun_vector_scalar, x0)
219:         jac_diff_4 = approx_derivative(self.fun_vector_scalar, x0,
220:                                        method='cs')
221:         jac_true = self.jac_vector_scalar(x0)
222:         assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
223:         assert_allclose(jac_diff_3, jac_true, rtol=1e-7)
224:         assert_allclose(jac_diff_4, jac_true, rtol=1e-12)
225: 
226:     def test_vector_vector(self):
227:         x0 = np.array([-100.0, 0.2])
228:         jac_diff_2 = approx_derivative(self.fun_vector_vector, x0,
229:                                        method='2-point')
230:         jac_diff_3 = approx_derivative(self.fun_vector_vector, x0)
231:         jac_diff_4 = approx_derivative(self.fun_vector_vector, x0,
232:                                        method='cs')
233:         jac_true = self.jac_vector_vector(x0)
234:         assert_allclose(jac_diff_2, jac_true, rtol=1e-5)
235:         assert_allclose(jac_diff_3, jac_true, rtol=1e-6)
236:         assert_allclose(jac_diff_4, jac_true, rtol=1e-12)
237: 
238:     def test_wrong_dimensions(self):
239:         x0 = 1.0
240:         assert_raises(RuntimeError, approx_derivative,
241:                       self.wrong_dimensions_fun, x0)
242:         f0 = self.wrong_dimensions_fun(np.atleast_1d(x0))
243:         assert_raises(ValueError, approx_derivative,
244:                       self.wrong_dimensions_fun, x0, f0=f0)
245: 
246:     def test_custom_rel_step(self):
247:         x0 = np.array([-0.1, 0.1])
248:         jac_diff_2 = approx_derivative(self.fun_vector_vector, x0,
249:                                        method='2-point', rel_step=1e-4)
250:         jac_diff_3 = approx_derivative(self.fun_vector_vector, x0,
251:                                        rel_step=1e-4)
252:         jac_true = self.jac_vector_vector(x0)
253:         assert_allclose(jac_diff_2, jac_true, rtol=1e-2)
254:         assert_allclose(jac_diff_3, jac_true, rtol=1e-4)
255: 
256:     def test_options(self):
257:         x0 = np.array([1.0, 1.0])
258:         c0 = -1.0
259:         c1 = 1.0
260:         lb = 0.0
261:         ub = 2.0
262:         f0 = self.fun_parametrized(x0, c0, c1=c1)
263:         rel_step = np.array([-1e-6, 1e-7])
264:         jac_true = self.jac_parametrized(x0, c0, c1)
265:         jac_diff_2 = approx_derivative(
266:             self.fun_parametrized, x0, method='2-point', rel_step=rel_step,
267:             f0=f0, args=(c0,), kwargs=dict(c1=c1), bounds=(lb, ub))
268:         jac_diff_3 = approx_derivative(
269:             self.fun_parametrized, x0, rel_step=rel_step,
270:             f0=f0, args=(c0,), kwargs=dict(c1=c1), bounds=(lb, ub))
271:         assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
272:         assert_allclose(jac_diff_3, jac_true, rtol=1e-9)
273: 
274:     def test_with_bounds_2_point(self):
275:         lb = -np.ones(2)
276:         ub = np.ones(2)
277: 
278:         x0 = np.array([-2.0, 0.2])
279:         assert_raises(ValueError, approx_derivative,
280:                       self.fun_vector_vector, x0, bounds=(lb, ub))
281: 
282:         x0 = np.array([-1.0, 1.0])
283:         jac_diff = approx_derivative(self.fun_vector_vector, x0,
284:                                      method='2-point', bounds=(lb, ub))
285:         jac_true = self.jac_vector_vector(x0)
286:         assert_allclose(jac_diff, jac_true, rtol=1e-6)
287: 
288:     def test_with_bounds_3_point(self):
289:         lb = np.array([1.0, 1.0])
290:         ub = np.array([2.0, 2.0])
291: 
292:         x0 = np.array([1.0, 2.0])
293:         jac_true = self.jac_vector_vector(x0)
294: 
295:         jac_diff = approx_derivative(self.fun_vector_vector, x0)
296:         assert_allclose(jac_diff, jac_true, rtol=1e-9)
297: 
298:         jac_diff = approx_derivative(self.fun_vector_vector, x0,
299:                                      bounds=(lb, np.inf))
300:         assert_allclose(jac_diff, jac_true, rtol=1e-9)
301: 
302:         jac_diff = approx_derivative(self.fun_vector_vector, x0,
303:                                      bounds=(-np.inf, ub))
304:         assert_allclose(jac_diff, jac_true, rtol=1e-9)
305: 
306:         jac_diff = approx_derivative(self.fun_vector_vector, x0,
307:                                      bounds=(lb, ub))
308:         assert_allclose(jac_diff, jac_true, rtol=1e-9)
309: 
310:     def test_tight_bounds(self):
311:         x0 = np.array([10.0, 10.0])
312:         lb = x0 - 3e-9
313:         ub = x0 + 2e-9
314:         jac_true = self.jac_vector_vector(x0)
315:         jac_diff = approx_derivative(
316:             self.fun_vector_vector, x0, method='2-point', bounds=(lb, ub))
317:         assert_allclose(jac_diff, jac_true, rtol=1e-6)
318:         jac_diff = approx_derivative(
319:             self.fun_vector_vector, x0, method='2-point',
320:             rel_step=1e-6, bounds=(lb, ub))
321:         assert_allclose(jac_diff, jac_true, rtol=1e-6)
322: 
323:         jac_diff = approx_derivative(
324:             self.fun_vector_vector, x0, bounds=(lb, ub))
325:         assert_allclose(jac_diff, jac_true, rtol=1e-6)
326:         jac_diff = approx_derivative(
327:             self.fun_vector_vector, x0, rel_step=1e-6, bounds=(lb, ub))
328:         assert_allclose(jac_true, jac_diff, rtol=1e-6)
329: 
330:     def test_bound_switches(self):
331:         lb = -1e-8
332:         ub = 1e-8
333:         x0 = 0.0
334:         jac_true = self.jac_with_nan(x0)
335:         jac_diff_2 = approx_derivative(
336:             self.fun_with_nan, x0, method='2-point', rel_step=1e-6,
337:             bounds=(lb, ub))
338:         jac_diff_3 = approx_derivative(
339:             self.fun_with_nan, x0, rel_step=1e-6, bounds=(lb, ub))
340:         assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
341:         assert_allclose(jac_diff_3, jac_true, rtol=1e-9)
342: 
343:         x0 = 1e-8
344:         jac_true = self.jac_with_nan(x0)
345:         jac_diff_2 = approx_derivative(
346:             self.fun_with_nan, x0, method='2-point', rel_step=1e-6,
347:             bounds=(lb, ub))
348:         jac_diff_3 = approx_derivative(
349:             self.fun_with_nan, x0, rel_step=1e-6, bounds=(lb, ub))
350:         assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
351:         assert_allclose(jac_diff_3, jac_true, rtol=1e-9)
352: 
353:     def test_non_numpy(self):
354:         x0 = 1.0
355:         jac_true = self.jac_non_numpy(x0)
356:         jac_diff_2 = approx_derivative(self.jac_non_numpy, x0,
357:                                        method='2-point')
358:         jac_diff_3 = approx_derivative(self.jac_non_numpy, x0)
359:         assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
360:         assert_allclose(jac_diff_3, jac_true, rtol=1e-8)
361: 
362:         # math.exp cannot handle complex arguments, hence this raises
363:         assert_raises(TypeError, approx_derivative, self.jac_non_numpy, x0,
364:                                        **dict(method='cs'))
365: 
366:     def test_check_derivative(self):
367:         x0 = np.array([-10.0, 10])
368:         accuracy = check_derivative(self.fun_vector_vector,
369:                                     self.jac_vector_vector, x0)
370:         assert_(accuracy < 1e-9)
371:         accuracy = check_derivative(self.fun_vector_vector,
372:                                     self.jac_vector_vector, x0)
373:         assert_(accuracy < 1e-6)
374: 
375:         x0 = np.array([0.0, 0.0])
376:         accuracy = check_derivative(self.fun_zero_jacobian,
377:                                     self.jac_zero_jacobian, x0)
378:         assert_(accuracy == 0)
379:         accuracy = check_derivative(self.fun_zero_jacobian,
380:                                     self.jac_zero_jacobian, x0)
381:         assert_(accuracy == 0)
382: 
383: 
384: class TestApproxDerivativeSparse(object):
385:     # Example from Numerical Optimization 2nd edition, p. 198.
386:     def setup_method(self):
387:         np.random.seed(0)
388:         self.n = 50
389:         self.lb = -0.1 * (1 + np.arange(self.n))
390:         self.ub = 0.1 * (1 + np.arange(self.n))
391:         self.x0 = np.empty(self.n)
392:         self.x0[::2] = (1 - 1e-7) * self.lb[::2]
393:         self.x0[1::2] = (1 - 1e-7) * self.ub[1::2]
394: 
395:         self.J_true = self.jac(self.x0)
396: 
397:     def fun(self, x):
398:         e = x[1:]**3 - x[:-1]**2
399:         return np.hstack((0, 3 * e)) + np.hstack((2 * e, 0))
400: 
401:     def jac(self, x):
402:         n = x.size
403:         J = np.zeros((n, n))
404:         J[0, 0] = -4 * x[0]
405:         J[0, 1] = 6 * x[1]**2
406:         for i in range(1, n - 1):
407:             J[i, i - 1] = -6 * x[i-1]
408:             J[i, i] = 9 * x[i]**2 - 4 * x[i]
409:             J[i, i + 1] = 6 * x[i+1]**2
410:         J[-1, -1] = 9 * x[-1]**2
411:         J[-1, -2] = -6 * x[-2]
412: 
413:         return J
414: 
415:     def structure(self, n):
416:         A = np.zeros((n, n), dtype=int)
417:         A[0, 0] = 1
418:         A[0, 1] = 1
419:         for i in range(1, n - 1):
420:             A[i, i - 1: i + 2] = 1
421:         A[-1, -1] = 1
422:         A[-1, -2] = 1
423: 
424:         return A
425: 
426:     def test_all(self):
427:         A = self.structure(self.n)
428:         order = np.arange(self.n)
429:         groups_1 = group_columns(A, order)
430:         np.random.shuffle(order)
431:         groups_2 = group_columns(A, order)
432: 
433:         for method, groups, l, u in product(
434:                 ['2-point', '3-point', 'cs'], [groups_1, groups_2],
435:                 [-np.inf, self.lb], [np.inf, self.ub]):
436:             J = approx_derivative(self.fun, self.x0, method=method,
437:                                   bounds=(l, u), sparsity=(A, groups))
438:             assert_(isinstance(J, csr_matrix))
439:             assert_allclose(J.toarray(), self.J_true, rtol=1e-6)
440: 
441:             rel_step = 1e-8 * np.ones_like(self.x0)
442:             rel_step[::2] *= -1
443:             J = approx_derivative(self.fun, self.x0, method=method,
444:                                   rel_step=rel_step, sparsity=(A, groups))
445:             assert_allclose(J.toarray(), self.J_true, rtol=1e-5)
446: 
447:     def test_no_precomputed_groups(self):
448:         A = self.structure(self.n)
449:         J = approx_derivative(self.fun, self.x0, sparsity=A)
450:         assert_allclose(J.toarray(), self.J_true, rtol=1e-6)
451: 
452:     def test_equivalence(self):
453:         structure = np.ones((self.n, self.n), dtype=int)
454:         groups = np.arange(self.n)
455:         for method in ['2-point', '3-point', 'cs']:
456:             J_dense = approx_derivative(self.fun, self.x0, method=method)
457:             J_sparse = approx_derivative(
458:                 self.fun, self.x0, sparsity=(structure, groups), method=method)
459:             assert_equal(J_dense, J_sparse.toarray())
460: 
461:     def test_check_derivative(self):
462:         def jac(x):
463:             return csr_matrix(self.jac(x))
464: 
465:         accuracy = check_derivative(self.fun, jac, self.x0,
466:                                     bounds=(self.lb, self.ub))
467:         assert_(accuracy < 1e-9)
468: 
469:         accuracy = check_derivative(self.fun, jac, self.x0,
470:                                     bounds=(self.lb, self.ub))
471:         assert_(accuracy < 1e-9)
472: 
473: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import math' statement (line 3)
import math

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from itertools import product' statement (line 4)
try:
    from itertools import product

except:
    product = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'itertools', None, module_type_store, ['product'], [product])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_241395 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_241395) is not StypyTypeError):

    if (import_241395 != 'pyd_module'):
        __import__(import_241395)
        sys_modules_241396 = sys.modules[import_241395]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_241396.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_241395)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_allclose, assert_equal, assert_' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_241397 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_241397) is not StypyTypeError):

    if (import_241397 != 'pyd_module'):
        __import__(import_241397)
        sys_modules_241398 = sys.modules[import_241397]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_241398.module_type_store, module_type_store, ['assert_allclose', 'assert_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_241398, sys_modules_241398.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_equal', 'assert_'], [assert_allclose, assert_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_241397)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from pytest import assert_raises' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_241399 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_241399) is not StypyTypeError):

    if (import_241399 != 'pyd_module'):
        __import__(import_241399)
        sys_modules_241400 = sys.modules[import_241399]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_241400.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_241400, sys_modules_241400.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_241399)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.sparse import csr_matrix, csc_matrix, lil_matrix' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_241401 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse')

if (type(import_241401) is not StypyTypeError):

    if (import_241401 != 'pyd_module'):
        __import__(import_241401)
        sys_modules_241402 = sys.modules[import_241401]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse', sys_modules_241402.module_type_store, module_type_store, ['csr_matrix', 'csc_matrix', 'lil_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_241402, sys_modules_241402.module_type_store, module_type_store)
    else:
        from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse', None, module_type_store, ['csr_matrix', 'csc_matrix', 'lil_matrix'], [csr_matrix, csc_matrix, lil_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse', import_241401)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.optimize._numdiff import _adjust_scheme_to_bounds, approx_derivative, check_derivative, group_columns' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_241403 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._numdiff')

if (type(import_241403) is not StypyTypeError):

    if (import_241403 != 'pyd_module'):
        __import__(import_241403)
        sys_modules_241404 = sys.modules[import_241403]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._numdiff', sys_modules_241404.module_type_store, module_type_store, ['_adjust_scheme_to_bounds', 'approx_derivative', 'check_derivative', 'group_columns'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_241404, sys_modules_241404.module_type_store, module_type_store)
    else:
        from scipy.optimize._numdiff import _adjust_scheme_to_bounds, approx_derivative, check_derivative, group_columns

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._numdiff', None, module_type_store, ['_adjust_scheme_to_bounds', 'approx_derivative', 'check_derivative', 'group_columns'], [_adjust_scheme_to_bounds, approx_derivative, check_derivative, group_columns])

else:
    # Assigning a type to the variable 'scipy.optimize._numdiff' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._numdiff', import_241403)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')


@norecursion
def test_group_columns(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_group_columns'
    module_type_store = module_type_store.open_function_context('test_group_columns', 17, 0, False)
    
    # Passed parameters checking function
    test_group_columns.stypy_localization = localization
    test_group_columns.stypy_type_of_self = None
    test_group_columns.stypy_type_store = module_type_store
    test_group_columns.stypy_function_name = 'test_group_columns'
    test_group_columns.stypy_param_names_list = []
    test_group_columns.stypy_varargs_param_name = None
    test_group_columns.stypy_kwargs_param_name = None
    test_group_columns.stypy_call_defaults = defaults
    test_group_columns.stypy_call_varargs = varargs
    test_group_columns.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_group_columns', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_group_columns', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_group_columns(...)' code ##################

    
    # Assigning a List to a Name (line 18):
    
    # Assigning a List to a Name (line 18):
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_241405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_241406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    int_241407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 8), list_241406, int_241407)
    # Adding element type (line 19)
    int_241408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 8), list_241406, int_241408)
    # Adding element type (line 19)
    int_241409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 8), list_241406, int_241409)
    # Adding element type (line 19)
    int_241410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 8), list_241406, int_241410)
    # Adding element type (line 19)
    int_241411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 8), list_241406, int_241411)
    # Adding element type (line 19)
    int_241412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 8), list_241406, int_241412)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_241405, list_241406)
    # Adding element type (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 20)
    list_241413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 20)
    # Adding element type (line 20)
    int_241414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 8), list_241413, int_241414)
    # Adding element type (line 20)
    int_241415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 8), list_241413, int_241415)
    # Adding element type (line 20)
    int_241416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 8), list_241413, int_241416)
    # Adding element type (line 20)
    int_241417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 8), list_241413, int_241417)
    # Adding element type (line 20)
    int_241418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 8), list_241413, int_241418)
    # Adding element type (line 20)
    int_241419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 8), list_241413, int_241419)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_241405, list_241413)
    # Adding element type (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_241420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    int_241421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 8), list_241420, int_241421)
    # Adding element type (line 21)
    int_241422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 8), list_241420, int_241422)
    # Adding element type (line 21)
    int_241423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 8), list_241420, int_241423)
    # Adding element type (line 21)
    int_241424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 8), list_241420, int_241424)
    # Adding element type (line 21)
    int_241425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 8), list_241420, int_241425)
    # Adding element type (line 21)
    int_241426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 8), list_241420, int_241426)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_241405, list_241420)
    # Adding element type (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_241427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    int_241428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 8), list_241427, int_241428)
    # Adding element type (line 22)
    int_241429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 8), list_241427, int_241429)
    # Adding element type (line 22)
    int_241430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 8), list_241427, int_241430)
    # Adding element type (line 22)
    int_241431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 8), list_241427, int_241431)
    # Adding element type (line 22)
    int_241432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 8), list_241427, int_241432)
    # Adding element type (line 22)
    int_241433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 8), list_241427, int_241433)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_241405, list_241427)
    # Adding element type (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_241434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    int_241435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), list_241434, int_241435)
    # Adding element type (line 23)
    int_241436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), list_241434, int_241436)
    # Adding element type (line 23)
    int_241437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), list_241434, int_241437)
    # Adding element type (line 23)
    int_241438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), list_241434, int_241438)
    # Adding element type (line 23)
    int_241439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), list_241434, int_241439)
    # Adding element type (line 23)
    int_241440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), list_241434, int_241440)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_241405, list_241434)
    # Adding element type (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 24)
    list_241441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 24)
    # Adding element type (line 24)
    int_241442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 8), list_241441, int_241442)
    # Adding element type (line 24)
    int_241443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 8), list_241441, int_241443)
    # Adding element type (line 24)
    int_241444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 8), list_241441, int_241444)
    # Adding element type (line 24)
    int_241445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 8), list_241441, int_241445)
    # Adding element type (line 24)
    int_241446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 8), list_241441, int_241446)
    # Adding element type (line 24)
    int_241447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 8), list_241441, int_241447)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_241405, list_241441)
    # Adding element type (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_241448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    int_241449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), list_241448, int_241449)
    # Adding element type (line 25)
    int_241450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), list_241448, int_241450)
    # Adding element type (line 25)
    int_241451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), list_241448, int_241451)
    # Adding element type (line 25)
    int_241452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), list_241448, int_241452)
    # Adding element type (line 25)
    int_241453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), list_241448, int_241453)
    # Adding element type (line 25)
    int_241454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), list_241448, int_241454)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_241405, list_241448)
    
    # Assigning a type to the variable 'structure' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'structure', list_241405)
    
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_241455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    # Getting the type of 'np' (line 27)
    np_241456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'np')
    # Obtaining the member 'asarray' of a type (line 27)
    asarray_241457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 22), np_241456, 'asarray')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_241455, asarray_241457)
    # Adding element type (line 27)
    # Getting the type of 'csr_matrix' (line 27)
    csr_matrix_241458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 34), 'csr_matrix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_241455, csr_matrix_241458)
    # Adding element type (line 27)
    # Getting the type of 'csc_matrix' (line 27)
    csc_matrix_241459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 46), 'csc_matrix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_241455, csc_matrix_241459)
    # Adding element type (line 27)
    # Getting the type of 'lil_matrix' (line 27)
    lil_matrix_241460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 58), 'lil_matrix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_241455, lil_matrix_241460)
    
    # Testing the type of a for loop iterable (line 27)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 4), list_241455)
    # Getting the type of the for loop variable (line 27)
    for_loop_var_241461 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 4), list_241455)
    # Assigning a type to the variable 'transform' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'transform', for_loop_var_241461)
    # SSA begins for a for statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 28):
    
    # Assigning a Call to a Name (line 28):
    
    # Call to transform(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'structure' (line 28)
    structure_241463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'structure', False)
    # Processing the call keyword arguments (line 28)
    kwargs_241464 = {}
    # Getting the type of 'transform' (line 28)
    transform_241462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'transform', False)
    # Calling transform(args, kwargs) (line 28)
    transform_call_result_241465 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), transform_241462, *[structure_241463], **kwargs_241464)
    
    # Assigning a type to the variable 'A' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'A', transform_call_result_241465)
    
    # Assigning a Call to a Name (line 29):
    
    # Assigning a Call to a Name (line 29):
    
    # Call to arange(...): (line 29)
    # Processing the call arguments (line 29)
    int_241468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'int')
    # Processing the call keyword arguments (line 29)
    kwargs_241469 = {}
    # Getting the type of 'np' (line 29)
    np_241466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'np', False)
    # Obtaining the member 'arange' of a type (line 29)
    arange_241467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 16), np_241466, 'arange')
    # Calling arange(args, kwargs) (line 29)
    arange_call_result_241470 = invoke(stypy.reporting.localization.Localization(__file__, 29, 16), arange_241467, *[int_241468], **kwargs_241469)
    
    # Assigning a type to the variable 'order' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'order', arange_call_result_241470)
    
    # Assigning a Call to a Name (line 30):
    
    # Assigning a Call to a Name (line 30):
    
    # Call to array(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_241473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    int_241474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 31), list_241473, int_241474)
    # Adding element type (line 30)
    int_241475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 31), list_241473, int_241475)
    # Adding element type (line 30)
    int_241476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 31), list_241473, int_241476)
    # Adding element type (line 30)
    int_241477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 31), list_241473, int_241477)
    # Adding element type (line 30)
    int_241478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 31), list_241473, int_241478)
    # Adding element type (line 30)
    int_241479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 31), list_241473, int_241479)
    
    # Processing the call keyword arguments (line 30)
    kwargs_241480 = {}
    # Getting the type of 'np' (line 30)
    np_241471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'np', False)
    # Obtaining the member 'array' of a type (line 30)
    array_241472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 22), np_241471, 'array')
    # Calling array(args, kwargs) (line 30)
    array_call_result_241481 = invoke(stypy.reporting.localization.Localization(__file__, 30, 22), array_241472, *[list_241473], **kwargs_241480)
    
    # Assigning a type to the variable 'groups_true' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'groups_true', array_call_result_241481)
    
    # Assigning a Call to a Name (line 31):
    
    # Assigning a Call to a Name (line 31):
    
    # Call to group_columns(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'A' (line 31)
    A_241483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'A', False)
    # Getting the type of 'order' (line 31)
    order_241484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 34), 'order', False)
    # Processing the call keyword arguments (line 31)
    kwargs_241485 = {}
    # Getting the type of 'group_columns' (line 31)
    group_columns_241482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'group_columns', False)
    # Calling group_columns(args, kwargs) (line 31)
    group_columns_call_result_241486 = invoke(stypy.reporting.localization.Localization(__file__, 31, 17), group_columns_241482, *[A_241483, order_241484], **kwargs_241485)
    
    # Assigning a type to the variable 'groups' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'groups', group_columns_call_result_241486)
    
    # Call to assert_equal(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'groups' (line 32)
    groups_241488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 21), 'groups', False)
    # Getting the type of 'groups_true' (line 32)
    groups_true_241489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 29), 'groups_true', False)
    # Processing the call keyword arguments (line 32)
    kwargs_241490 = {}
    # Getting the type of 'assert_equal' (line 32)
    assert_equal_241487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 32)
    assert_equal_call_result_241491 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), assert_equal_241487, *[groups_241488, groups_true_241489], **kwargs_241490)
    
    
    # Assigning a List to a Name (line 34):
    
    # Assigning a List to a Name (line 34):
    
    # Obtaining an instance of the builtin type 'list' (line 34)
    list_241492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 34)
    # Adding element type (line 34)
    int_241493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 16), list_241492, int_241493)
    # Adding element type (line 34)
    int_241494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 16), list_241492, int_241494)
    # Adding element type (line 34)
    int_241495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 16), list_241492, int_241495)
    # Adding element type (line 34)
    int_241496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 16), list_241492, int_241496)
    # Adding element type (line 34)
    int_241497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 16), list_241492, int_241497)
    # Adding element type (line 34)
    int_241498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 16), list_241492, int_241498)
    
    # Assigning a type to the variable 'order' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'order', list_241492)
    
    # Assigning a Call to a Name (line 35):
    
    # Assigning a Call to a Name (line 35):
    
    # Call to array(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Obtaining an instance of the builtin type 'list' (line 35)
    list_241501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 35)
    # Adding element type (line 35)
    int_241502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 31), list_241501, int_241502)
    # Adding element type (line 35)
    int_241503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 31), list_241501, int_241503)
    # Adding element type (line 35)
    int_241504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 31), list_241501, int_241504)
    # Adding element type (line 35)
    int_241505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 31), list_241501, int_241505)
    # Adding element type (line 35)
    int_241506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 31), list_241501, int_241506)
    # Adding element type (line 35)
    int_241507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 31), list_241501, int_241507)
    
    # Processing the call keyword arguments (line 35)
    kwargs_241508 = {}
    # Getting the type of 'np' (line 35)
    np_241499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'np', False)
    # Obtaining the member 'array' of a type (line 35)
    array_241500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 22), np_241499, 'array')
    # Calling array(args, kwargs) (line 35)
    array_call_result_241509 = invoke(stypy.reporting.localization.Localization(__file__, 35, 22), array_241500, *[list_241501], **kwargs_241508)
    
    # Assigning a type to the variable 'groups_true' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'groups_true', array_call_result_241509)
    
    # Assigning a Call to a Name (line 36):
    
    # Assigning a Call to a Name (line 36):
    
    # Call to group_columns(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'A' (line 36)
    A_241511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'A', False)
    # Getting the type of 'order' (line 36)
    order_241512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'order', False)
    # Processing the call keyword arguments (line 36)
    kwargs_241513 = {}
    # Getting the type of 'group_columns' (line 36)
    group_columns_241510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'group_columns', False)
    # Calling group_columns(args, kwargs) (line 36)
    group_columns_call_result_241514 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), group_columns_241510, *[A_241511, order_241512], **kwargs_241513)
    
    # Assigning a type to the variable 'groups' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'groups', group_columns_call_result_241514)
    
    # Call to assert_equal(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'groups' (line 37)
    groups_241516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'groups', False)
    # Getting the type of 'groups_true' (line 37)
    groups_true_241517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'groups_true', False)
    # Processing the call keyword arguments (line 37)
    kwargs_241518 = {}
    # Getting the type of 'assert_equal' (line 37)
    assert_equal_241515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 37)
    assert_equal_call_result_241519 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), assert_equal_241515, *[groups_241516, groups_true_241517], **kwargs_241518)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Call to a Name (line 40):
    
    # Call to group_columns(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'A' (line 40)
    A_241521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 29), 'A', False)
    # Processing the call keyword arguments (line 40)
    kwargs_241522 = {}
    # Getting the type of 'group_columns' (line 40)
    group_columns_241520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'group_columns', False)
    # Calling group_columns(args, kwargs) (line 40)
    group_columns_call_result_241523 = invoke(stypy.reporting.localization.Localization(__file__, 40, 15), group_columns_241520, *[A_241521], **kwargs_241522)
    
    # Assigning a type to the variable 'groups_1' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'groups_1', group_columns_call_result_241523)
    
    # Assigning a Call to a Name (line 41):
    
    # Assigning a Call to a Name (line 41):
    
    # Call to group_columns(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'A' (line 41)
    A_241525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 29), 'A', False)
    # Processing the call keyword arguments (line 41)
    kwargs_241526 = {}
    # Getting the type of 'group_columns' (line 41)
    group_columns_241524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'group_columns', False)
    # Calling group_columns(args, kwargs) (line 41)
    group_columns_call_result_241527 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), group_columns_241524, *[A_241525], **kwargs_241526)
    
    # Assigning a type to the variable 'groups_2' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'groups_2', group_columns_call_result_241527)
    
    # Call to assert_equal(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'groups_1' (line 42)
    groups_1_241529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'groups_1', False)
    # Getting the type of 'groups_2' (line 42)
    groups_2_241530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'groups_2', False)
    # Processing the call keyword arguments (line 42)
    kwargs_241531 = {}
    # Getting the type of 'assert_equal' (line 42)
    assert_equal_241528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 42)
    assert_equal_call_result_241532 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), assert_equal_241528, *[groups_1_241529, groups_2_241530], **kwargs_241531)
    
    
    # ################# End of 'test_group_columns(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_group_columns' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_241533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_241533)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_group_columns'
    return stypy_return_type_241533

# Assigning a type to the variable 'test_group_columns' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'test_group_columns', test_group_columns)
# Declaration of the 'TestAdjustSchemeToBounds' class

class TestAdjustSchemeToBounds(object, ):

    @norecursion
    def test_no_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_no_bounds'
        module_type_store = module_type_store.open_function_context('test_no_bounds', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestAdjustSchemeToBounds.test_no_bounds.__dict__.__setitem__('stypy_localization', localization)
        TestAdjustSchemeToBounds.test_no_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestAdjustSchemeToBounds.test_no_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestAdjustSchemeToBounds.test_no_bounds.__dict__.__setitem__('stypy_function_name', 'TestAdjustSchemeToBounds.test_no_bounds')
        TestAdjustSchemeToBounds.test_no_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        TestAdjustSchemeToBounds.test_no_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestAdjustSchemeToBounds.test_no_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestAdjustSchemeToBounds.test_no_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestAdjustSchemeToBounds.test_no_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestAdjustSchemeToBounds.test_no_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestAdjustSchemeToBounds.test_no_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestAdjustSchemeToBounds.test_no_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_no_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_no_bounds(...)' code ##################

        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to zeros(...): (line 47)
        # Processing the call arguments (line 47)
        int_241536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'int')
        # Processing the call keyword arguments (line 47)
        kwargs_241537 = {}
        # Getting the type of 'np' (line 47)
        np_241534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'np', False)
        # Obtaining the member 'zeros' of a type (line 47)
        zeros_241535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 13), np_241534, 'zeros')
        # Calling zeros(args, kwargs) (line 47)
        zeros_call_result_241538 = invoke(stypy.reporting.localization.Localization(__file__, 47, 13), zeros_241535, *[int_241536], **kwargs_241537)
        
        # Assigning a type to the variable 'x0' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'x0', zeros_call_result_241538)
        
        # Assigning a BinOp to a Name (line 48):
        
        # Assigning a BinOp to a Name (line 48):
        
        # Call to ones(...): (line 48)
        # Processing the call arguments (line 48)
        int_241541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 20), 'int')
        # Processing the call keyword arguments (line 48)
        kwargs_241542 = {}
        # Getting the type of 'np' (line 48)
        np_241539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'np', False)
        # Obtaining the member 'ones' of a type (line 48)
        ones_241540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), np_241539, 'ones')
        # Calling ones(args, kwargs) (line 48)
        ones_call_result_241543 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), ones_241540, *[int_241541], **kwargs_241542)
        
        float_241544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 25), 'float')
        # Applying the binary operator '*' (line 48)
        result_mul_241545 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 12), '*', ones_call_result_241543, float_241544)
        
        # Assigning a type to the variable 'h' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'h', result_mul_241545)
        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to empty_like(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'x0' (line 49)
        x0_241548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'x0', False)
        # Processing the call keyword arguments (line 49)
        kwargs_241549 = {}
        # Getting the type of 'np' (line 49)
        np_241546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 20), 'np', False)
        # Obtaining the member 'empty_like' of a type (line 49)
        empty_like_241547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 20), np_241546, 'empty_like')
        # Calling empty_like(args, kwargs) (line 49)
        empty_like_call_result_241550 = invoke(stypy.reporting.localization.Localization(__file__, 49, 20), empty_like_241547, *[x0_241548], **kwargs_241549)
        
        # Assigning a type to the variable 'inf_lower' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'inf_lower', empty_like_call_result_241550)
        
        # Assigning a Call to a Name (line 50):
        
        # Assigning a Call to a Name (line 50):
        
        # Call to empty_like(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'x0' (line 50)
        x0_241553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 34), 'x0', False)
        # Processing the call keyword arguments (line 50)
        kwargs_241554 = {}
        # Getting the type of 'np' (line 50)
        np_241551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'np', False)
        # Obtaining the member 'empty_like' of a type (line 50)
        empty_like_241552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 20), np_241551, 'empty_like')
        # Calling empty_like(args, kwargs) (line 50)
        empty_like_call_result_241555 = invoke(stypy.reporting.localization.Localization(__file__, 50, 20), empty_like_241552, *[x0_241553], **kwargs_241554)
        
        # Assigning a type to the variable 'inf_upper' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'inf_upper', empty_like_call_result_241555)
        
        # Call to fill(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Getting the type of 'np' (line 51)
        np_241558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 24), 'np', False)
        # Obtaining the member 'inf' of a type (line 51)
        inf_241559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 24), np_241558, 'inf')
        # Applying the 'usub' unary operator (line 51)
        result___neg___241560 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 23), 'usub', inf_241559)
        
        # Processing the call keyword arguments (line 51)
        kwargs_241561 = {}
        # Getting the type of 'inf_lower' (line 51)
        inf_lower_241556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'inf_lower', False)
        # Obtaining the member 'fill' of a type (line 51)
        fill_241557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), inf_lower_241556, 'fill')
        # Calling fill(args, kwargs) (line 51)
        fill_call_result_241562 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), fill_241557, *[result___neg___241560], **kwargs_241561)
        
        
        # Call to fill(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'np' (line 52)
        np_241565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 23), 'np', False)
        # Obtaining the member 'inf' of a type (line 52)
        inf_241566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 23), np_241565, 'inf')
        # Processing the call keyword arguments (line 52)
        kwargs_241567 = {}
        # Getting the type of 'inf_upper' (line 52)
        inf_upper_241563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'inf_upper', False)
        # Obtaining the member 'fill' of a type (line 52)
        fill_241564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), inf_upper_241563, 'fill')
        # Calling fill(args, kwargs) (line 52)
        fill_call_result_241568 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), fill_241564, *[inf_241566], **kwargs_241567)
        
        
        # Assigning a Call to a Tuple (line 54):
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_241569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'x0' (line 55)
        x0_241571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'x0', False)
        # Getting the type of 'h' (line 55)
        h_241572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'h', False)
        int_241573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'int')
        str_241574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'str', '1-sided')
        # Getting the type of 'inf_lower' (line 55)
        inf_lower_241575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 33), 'inf_lower', False)
        # Getting the type of 'inf_upper' (line 55)
        inf_upper_241576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 44), 'inf_upper', False)
        # Processing the call keyword arguments (line 54)
        kwargs_241577 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 54)
        _adjust_scheme_to_bounds_241570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 54)
        _adjust_scheme_to_bounds_call_result_241578 = invoke(stypy.reporting.localization.Localization(__file__, 54, 32), _adjust_scheme_to_bounds_241570, *[x0_241571, h_241572, int_241573, str_241574, inf_lower_241575, inf_upper_241576], **kwargs_241577)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___241579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), _adjust_scheme_to_bounds_call_result_241578, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_241580 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), getitem___241579, int_241569)
        
        # Assigning a type to the variable 'tuple_var_assignment_241371' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_241371', subscript_call_result_241580)
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_241581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'x0' (line 55)
        x0_241583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'x0', False)
        # Getting the type of 'h' (line 55)
        h_241584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'h', False)
        int_241585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'int')
        str_241586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'str', '1-sided')
        # Getting the type of 'inf_lower' (line 55)
        inf_lower_241587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 33), 'inf_lower', False)
        # Getting the type of 'inf_upper' (line 55)
        inf_upper_241588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 44), 'inf_upper', False)
        # Processing the call keyword arguments (line 54)
        kwargs_241589 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 54)
        _adjust_scheme_to_bounds_241582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 54)
        _adjust_scheme_to_bounds_call_result_241590 = invoke(stypy.reporting.localization.Localization(__file__, 54, 32), _adjust_scheme_to_bounds_241582, *[x0_241583, h_241584, int_241585, str_241586, inf_lower_241587, inf_upper_241588], **kwargs_241589)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___241591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), _adjust_scheme_to_bounds_call_result_241590, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_241592 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), getitem___241591, int_241581)
        
        # Assigning a type to the variable 'tuple_var_assignment_241372' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_241372', subscript_call_result_241592)
        
        # Assigning a Name to a Name (line 54):
        # Getting the type of 'tuple_var_assignment_241371' (line 54)
        tuple_var_assignment_241371_241593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_241371')
        # Assigning a type to the variable 'h_adjusted' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'h_adjusted', tuple_var_assignment_241371_241593)
        
        # Assigning a Name to a Name (line 54):
        # Getting the type of 'tuple_var_assignment_241372' (line 54)
        tuple_var_assignment_241372_241594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_241372')
        # Assigning a type to the variable 'one_sided' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'one_sided', tuple_var_assignment_241372_241594)
        
        # Call to assert_allclose(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'h_adjusted' (line 56)
        h_adjusted_241596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'h_adjusted', False)
        # Getting the type of 'h' (line 56)
        h_241597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'h', False)
        # Processing the call keyword arguments (line 56)
        kwargs_241598 = {}
        # Getting the type of 'assert_allclose' (line 56)
        assert_allclose_241595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 56)
        assert_allclose_call_result_241599 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), assert_allclose_241595, *[h_adjusted_241596, h_241597], **kwargs_241598)
        
        
        # Call to assert_(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Call to all(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'one_sided' (line 57)
        one_sided_241603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'one_sided', False)
        # Processing the call keyword arguments (line 57)
        kwargs_241604 = {}
        # Getting the type of 'np' (line 57)
        np_241601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 57)
        all_241602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 16), np_241601, 'all')
        # Calling all(args, kwargs) (line 57)
        all_call_result_241605 = invoke(stypy.reporting.localization.Localization(__file__, 57, 16), all_241602, *[one_sided_241603], **kwargs_241604)
        
        # Processing the call keyword arguments (line 57)
        kwargs_241606 = {}
        # Getting the type of 'assert_' (line 57)
        assert__241600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 57)
        assert__call_result_241607 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), assert__241600, *[all_call_result_241605], **kwargs_241606)
        
        
        # Assigning a Call to a Tuple (line 59):
        
        # Assigning a Subscript to a Name (line 59):
        
        # Obtaining the type of the subscript
        int_241608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'x0' (line 60)
        x0_241610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'x0', False)
        # Getting the type of 'h' (line 60)
        h_241611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'h', False)
        int_241612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'int')
        str_241613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'str', '1-sided')
        # Getting the type of 'inf_lower' (line 60)
        inf_lower_241614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 33), 'inf_lower', False)
        # Getting the type of 'inf_upper' (line 60)
        inf_upper_241615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), 'inf_upper', False)
        # Processing the call keyword arguments (line 59)
        kwargs_241616 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 59)
        _adjust_scheme_to_bounds_241609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 59)
        _adjust_scheme_to_bounds_call_result_241617 = invoke(stypy.reporting.localization.Localization(__file__, 59, 32), _adjust_scheme_to_bounds_241609, *[x0_241610, h_241611, int_241612, str_241613, inf_lower_241614, inf_upper_241615], **kwargs_241616)
        
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___241618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), _adjust_scheme_to_bounds_call_result_241617, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_241619 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), getitem___241618, int_241608)
        
        # Assigning a type to the variable 'tuple_var_assignment_241373' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'tuple_var_assignment_241373', subscript_call_result_241619)
        
        # Assigning a Subscript to a Name (line 59):
        
        # Obtaining the type of the subscript
        int_241620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'x0' (line 60)
        x0_241622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'x0', False)
        # Getting the type of 'h' (line 60)
        h_241623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'h', False)
        int_241624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'int')
        str_241625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'str', '1-sided')
        # Getting the type of 'inf_lower' (line 60)
        inf_lower_241626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 33), 'inf_lower', False)
        # Getting the type of 'inf_upper' (line 60)
        inf_upper_241627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), 'inf_upper', False)
        # Processing the call keyword arguments (line 59)
        kwargs_241628 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 59)
        _adjust_scheme_to_bounds_241621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 59)
        _adjust_scheme_to_bounds_call_result_241629 = invoke(stypy.reporting.localization.Localization(__file__, 59, 32), _adjust_scheme_to_bounds_241621, *[x0_241622, h_241623, int_241624, str_241625, inf_lower_241626, inf_upper_241627], **kwargs_241628)
        
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___241630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), _adjust_scheme_to_bounds_call_result_241629, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_241631 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), getitem___241630, int_241620)
        
        # Assigning a type to the variable 'tuple_var_assignment_241374' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'tuple_var_assignment_241374', subscript_call_result_241631)
        
        # Assigning a Name to a Name (line 59):
        # Getting the type of 'tuple_var_assignment_241373' (line 59)
        tuple_var_assignment_241373_241632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'tuple_var_assignment_241373')
        # Assigning a type to the variable 'h_adjusted' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'h_adjusted', tuple_var_assignment_241373_241632)
        
        # Assigning a Name to a Name (line 59):
        # Getting the type of 'tuple_var_assignment_241374' (line 59)
        tuple_var_assignment_241374_241633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'tuple_var_assignment_241374')
        # Assigning a type to the variable 'one_sided' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'one_sided', tuple_var_assignment_241374_241633)
        
        # Call to assert_allclose(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'h_adjusted' (line 61)
        h_adjusted_241635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'h_adjusted', False)
        # Getting the type of 'h' (line 61)
        h_241636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 36), 'h', False)
        # Processing the call keyword arguments (line 61)
        kwargs_241637 = {}
        # Getting the type of 'assert_allclose' (line 61)
        assert_allclose_241634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 61)
        assert_allclose_call_result_241638 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assert_allclose_241634, *[h_adjusted_241635, h_241636], **kwargs_241637)
        
        
        # Call to assert_(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to all(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'one_sided' (line 62)
        one_sided_241642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'one_sided', False)
        # Processing the call keyword arguments (line 62)
        kwargs_241643 = {}
        # Getting the type of 'np' (line 62)
        np_241640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 62)
        all_241641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), np_241640, 'all')
        # Calling all(args, kwargs) (line 62)
        all_call_result_241644 = invoke(stypy.reporting.localization.Localization(__file__, 62, 16), all_241641, *[one_sided_241642], **kwargs_241643)
        
        # Processing the call keyword arguments (line 62)
        kwargs_241645 = {}
        # Getting the type of 'assert_' (line 62)
        assert__241639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 62)
        assert__call_result_241646 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), assert__241639, *[all_call_result_241644], **kwargs_241645)
        
        
        # Assigning a Call to a Tuple (line 64):
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_241647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'x0' (line 65)
        x0_241649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'x0', False)
        # Getting the type of 'h' (line 65)
        h_241650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'h', False)
        int_241651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'int')
        str_241652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'str', '2-sided')
        # Getting the type of 'inf_lower' (line 65)
        inf_lower_241653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 33), 'inf_lower', False)
        # Getting the type of 'inf_upper' (line 65)
        inf_upper_241654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 44), 'inf_upper', False)
        # Processing the call keyword arguments (line 64)
        kwargs_241655 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 64)
        _adjust_scheme_to_bounds_241648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 64)
        _adjust_scheme_to_bounds_call_result_241656 = invoke(stypy.reporting.localization.Localization(__file__, 64, 32), _adjust_scheme_to_bounds_241648, *[x0_241649, h_241650, int_241651, str_241652, inf_lower_241653, inf_upper_241654], **kwargs_241655)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___241657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), _adjust_scheme_to_bounds_call_result_241656, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_241658 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___241657, int_241647)
        
        # Assigning a type to the variable 'tuple_var_assignment_241375' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_241375', subscript_call_result_241658)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_241659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'x0' (line 65)
        x0_241661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'x0', False)
        # Getting the type of 'h' (line 65)
        h_241662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'h', False)
        int_241663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'int')
        str_241664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'str', '2-sided')
        # Getting the type of 'inf_lower' (line 65)
        inf_lower_241665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 33), 'inf_lower', False)
        # Getting the type of 'inf_upper' (line 65)
        inf_upper_241666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 44), 'inf_upper', False)
        # Processing the call keyword arguments (line 64)
        kwargs_241667 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 64)
        _adjust_scheme_to_bounds_241660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 64)
        _adjust_scheme_to_bounds_call_result_241668 = invoke(stypy.reporting.localization.Localization(__file__, 64, 32), _adjust_scheme_to_bounds_241660, *[x0_241661, h_241662, int_241663, str_241664, inf_lower_241665, inf_upper_241666], **kwargs_241667)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___241669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), _adjust_scheme_to_bounds_call_result_241668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_241670 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___241669, int_241659)
        
        # Assigning a type to the variable 'tuple_var_assignment_241376' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_241376', subscript_call_result_241670)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_241375' (line 64)
        tuple_var_assignment_241375_241671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_241375')
        # Assigning a type to the variable 'h_adjusted' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'h_adjusted', tuple_var_assignment_241375_241671)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_241376' (line 64)
        tuple_var_assignment_241376_241672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_241376')
        # Assigning a type to the variable 'one_sided' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'one_sided', tuple_var_assignment_241376_241672)
        
        # Call to assert_allclose(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'h_adjusted' (line 66)
        h_adjusted_241674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'h_adjusted', False)
        # Getting the type of 'h' (line 66)
        h_241675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 36), 'h', False)
        # Processing the call keyword arguments (line 66)
        kwargs_241676 = {}
        # Getting the type of 'assert_allclose' (line 66)
        assert_allclose_241673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 66)
        assert_allclose_call_result_241677 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assert_allclose_241673, *[h_adjusted_241674, h_241675], **kwargs_241676)
        
        
        # Call to assert_(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to all(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Getting the type of 'one_sided' (line 67)
        one_sided_241681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'one_sided', False)
        # Applying the '~' unary operator (line 67)
        result_inv_241682 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 23), '~', one_sided_241681)
        
        # Processing the call keyword arguments (line 67)
        kwargs_241683 = {}
        # Getting the type of 'np' (line 67)
        np_241679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 67)
        all_241680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 16), np_241679, 'all')
        # Calling all(args, kwargs) (line 67)
        all_call_result_241684 = invoke(stypy.reporting.localization.Localization(__file__, 67, 16), all_241680, *[result_inv_241682], **kwargs_241683)
        
        # Processing the call keyword arguments (line 67)
        kwargs_241685 = {}
        # Getting the type of 'assert_' (line 67)
        assert__241678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 67)
        assert__call_result_241686 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), assert__241678, *[all_call_result_241684], **kwargs_241685)
        
        
        # Assigning a Call to a Tuple (line 69):
        
        # Assigning a Subscript to a Name (line 69):
        
        # Obtaining the type of the subscript
        int_241687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'x0' (line 70)
        x0_241689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'x0', False)
        # Getting the type of 'h' (line 70)
        h_241690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'h', False)
        int_241691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 19), 'int')
        str_241692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 22), 'str', '2-sided')
        # Getting the type of 'inf_lower' (line 70)
        inf_lower_241693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 33), 'inf_lower', False)
        # Getting the type of 'inf_upper' (line 70)
        inf_upper_241694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 44), 'inf_upper', False)
        # Processing the call keyword arguments (line 69)
        kwargs_241695 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 69)
        _adjust_scheme_to_bounds_241688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 69)
        _adjust_scheme_to_bounds_call_result_241696 = invoke(stypy.reporting.localization.Localization(__file__, 69, 32), _adjust_scheme_to_bounds_241688, *[x0_241689, h_241690, int_241691, str_241692, inf_lower_241693, inf_upper_241694], **kwargs_241695)
        
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___241697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), _adjust_scheme_to_bounds_call_result_241696, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_241698 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), getitem___241697, int_241687)
        
        # Assigning a type to the variable 'tuple_var_assignment_241377' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tuple_var_assignment_241377', subscript_call_result_241698)
        
        # Assigning a Subscript to a Name (line 69):
        
        # Obtaining the type of the subscript
        int_241699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'x0' (line 70)
        x0_241701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'x0', False)
        # Getting the type of 'h' (line 70)
        h_241702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'h', False)
        int_241703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 19), 'int')
        str_241704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 22), 'str', '2-sided')
        # Getting the type of 'inf_lower' (line 70)
        inf_lower_241705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 33), 'inf_lower', False)
        # Getting the type of 'inf_upper' (line 70)
        inf_upper_241706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 44), 'inf_upper', False)
        # Processing the call keyword arguments (line 69)
        kwargs_241707 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 69)
        _adjust_scheme_to_bounds_241700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 69)
        _adjust_scheme_to_bounds_call_result_241708 = invoke(stypy.reporting.localization.Localization(__file__, 69, 32), _adjust_scheme_to_bounds_241700, *[x0_241701, h_241702, int_241703, str_241704, inf_lower_241705, inf_upper_241706], **kwargs_241707)
        
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___241709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), _adjust_scheme_to_bounds_call_result_241708, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_241710 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), getitem___241709, int_241699)
        
        # Assigning a type to the variable 'tuple_var_assignment_241378' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tuple_var_assignment_241378', subscript_call_result_241710)
        
        # Assigning a Name to a Name (line 69):
        # Getting the type of 'tuple_var_assignment_241377' (line 69)
        tuple_var_assignment_241377_241711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tuple_var_assignment_241377')
        # Assigning a type to the variable 'h_adjusted' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'h_adjusted', tuple_var_assignment_241377_241711)
        
        # Assigning a Name to a Name (line 69):
        # Getting the type of 'tuple_var_assignment_241378' (line 69)
        tuple_var_assignment_241378_241712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tuple_var_assignment_241378')
        # Assigning a type to the variable 'one_sided' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'one_sided', tuple_var_assignment_241378_241712)
        
        # Call to assert_allclose(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'h_adjusted' (line 71)
        h_adjusted_241714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'h_adjusted', False)
        # Getting the type of 'h' (line 71)
        h_241715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 36), 'h', False)
        # Processing the call keyword arguments (line 71)
        kwargs_241716 = {}
        # Getting the type of 'assert_allclose' (line 71)
        assert_allclose_241713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 71)
        assert_allclose_call_result_241717 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), assert_allclose_241713, *[h_adjusted_241714, h_241715], **kwargs_241716)
        
        
        # Call to assert_(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Call to all(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Getting the type of 'one_sided' (line 72)
        one_sided_241721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'one_sided', False)
        # Applying the '~' unary operator (line 72)
        result_inv_241722 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 23), '~', one_sided_241721)
        
        # Processing the call keyword arguments (line 72)
        kwargs_241723 = {}
        # Getting the type of 'np' (line 72)
        np_241719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 72)
        all_241720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 16), np_241719, 'all')
        # Calling all(args, kwargs) (line 72)
        all_call_result_241724 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), all_241720, *[result_inv_241722], **kwargs_241723)
        
        # Processing the call keyword arguments (line 72)
        kwargs_241725 = {}
        # Getting the type of 'assert_' (line 72)
        assert__241718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 72)
        assert__call_result_241726 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), assert__241718, *[all_call_result_241724], **kwargs_241725)
        
        
        # ################# End of 'test_no_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_no_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_241727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_241727)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_no_bounds'
        return stypy_return_type_241727


    @norecursion
    def test_with_bound(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_with_bound'
        module_type_store = module_type_store.open_function_context('test_with_bound', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestAdjustSchemeToBounds.test_with_bound.__dict__.__setitem__('stypy_localization', localization)
        TestAdjustSchemeToBounds.test_with_bound.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestAdjustSchemeToBounds.test_with_bound.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestAdjustSchemeToBounds.test_with_bound.__dict__.__setitem__('stypy_function_name', 'TestAdjustSchemeToBounds.test_with_bound')
        TestAdjustSchemeToBounds.test_with_bound.__dict__.__setitem__('stypy_param_names_list', [])
        TestAdjustSchemeToBounds.test_with_bound.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestAdjustSchemeToBounds.test_with_bound.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestAdjustSchemeToBounds.test_with_bound.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestAdjustSchemeToBounds.test_with_bound.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestAdjustSchemeToBounds.test_with_bound.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestAdjustSchemeToBounds.test_with_bound.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestAdjustSchemeToBounds.test_with_bound', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_with_bound', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_with_bound(...)' code ##################

        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to array(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_241730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        float_241731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_241730, float_241731)
        # Adding element type (line 75)
        float_241732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_241730, float_241732)
        # Adding element type (line 75)
        float_241733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_241730, float_241733)
        
        # Processing the call keyword arguments (line 75)
        kwargs_241734 = {}
        # Getting the type of 'np' (line 75)
        np_241728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 75)
        array_241729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 13), np_241728, 'array')
        # Calling array(args, kwargs) (line 75)
        array_call_result_241735 = invoke(stypy.reporting.localization.Localization(__file__, 75, 13), array_241729, *[list_241730], **kwargs_241734)
        
        # Assigning a type to the variable 'x0' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'x0', array_call_result_241735)
        
        # Assigning a UnaryOp to a Name (line 76):
        
        # Assigning a UnaryOp to a Name (line 76):
        
        
        # Call to ones(...): (line 76)
        # Processing the call arguments (line 76)
        int_241738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 22), 'int')
        # Processing the call keyword arguments (line 76)
        kwargs_241739 = {}
        # Getting the type of 'np' (line 76)
        np_241736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'np', False)
        # Obtaining the member 'ones' of a type (line 76)
        ones_241737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 14), np_241736, 'ones')
        # Calling ones(args, kwargs) (line 76)
        ones_call_result_241740 = invoke(stypy.reporting.localization.Localization(__file__, 76, 14), ones_241737, *[int_241738], **kwargs_241739)
        
        # Applying the 'usub' unary operator (line 76)
        result___neg___241741 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 13), 'usub', ones_call_result_241740)
        
        # Assigning a type to the variable 'lb' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'lb', result___neg___241741)
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to ones(...): (line 77)
        # Processing the call arguments (line 77)
        int_241744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 21), 'int')
        # Processing the call keyword arguments (line 77)
        kwargs_241745 = {}
        # Getting the type of 'np' (line 77)
        np_241742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'np', False)
        # Obtaining the member 'ones' of a type (line 77)
        ones_241743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), np_241742, 'ones')
        # Calling ones(args, kwargs) (line 77)
        ones_call_result_241746 = invoke(stypy.reporting.localization.Localization(__file__, 77, 13), ones_241743, *[int_241744], **kwargs_241745)
        
        # Assigning a type to the variable 'ub' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'ub', ones_call_result_241746)
        
        # Assigning a BinOp to a Name (line 78):
        
        # Assigning a BinOp to a Name (line 78):
        
        # Call to array(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_241749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        int_241750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 21), list_241749, int_241750)
        # Adding element type (line 78)
        int_241751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 21), list_241749, int_241751)
        # Adding element type (line 78)
        int_241752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 21), list_241749, int_241752)
        
        # Processing the call keyword arguments (line 78)
        kwargs_241753 = {}
        # Getting the type of 'np' (line 78)
        np_241747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 78)
        array_241748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), np_241747, 'array')
        # Calling array(args, kwargs) (line 78)
        array_call_result_241754 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), array_241748, *[list_241749], **kwargs_241753)
        
        float_241755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 35), 'float')
        # Applying the binary operator '*' (line 78)
        result_mul_241756 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 12), '*', array_call_result_241754, float_241755)
        
        # Assigning a type to the variable 'h' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'h', result_mul_241756)
        
        # Assigning a Call to a Tuple (line 80):
        
        # Assigning a Subscript to a Name (line 80):
        
        # Obtaining the type of the subscript
        int_241757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'x0' (line 80)
        x0_241759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 49), 'x0', False)
        # Getting the type of 'h' (line 80)
        h_241760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 53), 'h', False)
        int_241761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 56), 'int')
        str_241762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 59), 'str', '1-sided')
        # Getting the type of 'lb' (line 80)
        lb_241763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 70), 'lb', False)
        # Getting the type of 'ub' (line 80)
        ub_241764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 74), 'ub', False)
        # Processing the call keyword arguments (line 80)
        kwargs_241765 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 80)
        _adjust_scheme_to_bounds_241758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 80)
        _adjust_scheme_to_bounds_call_result_241766 = invoke(stypy.reporting.localization.Localization(__file__, 80, 24), _adjust_scheme_to_bounds_241758, *[x0_241759, h_241760, int_241761, str_241762, lb_241763, ub_241764], **kwargs_241765)
        
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___241767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), _adjust_scheme_to_bounds_call_result_241766, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_241768 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), getitem___241767, int_241757)
        
        # Assigning a type to the variable 'tuple_var_assignment_241379' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'tuple_var_assignment_241379', subscript_call_result_241768)
        
        # Assigning a Subscript to a Name (line 80):
        
        # Obtaining the type of the subscript
        int_241769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'x0' (line 80)
        x0_241771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 49), 'x0', False)
        # Getting the type of 'h' (line 80)
        h_241772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 53), 'h', False)
        int_241773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 56), 'int')
        str_241774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 59), 'str', '1-sided')
        # Getting the type of 'lb' (line 80)
        lb_241775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 70), 'lb', False)
        # Getting the type of 'ub' (line 80)
        ub_241776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 74), 'ub', False)
        # Processing the call keyword arguments (line 80)
        kwargs_241777 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 80)
        _adjust_scheme_to_bounds_241770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 80)
        _adjust_scheme_to_bounds_call_result_241778 = invoke(stypy.reporting.localization.Localization(__file__, 80, 24), _adjust_scheme_to_bounds_241770, *[x0_241771, h_241772, int_241773, str_241774, lb_241775, ub_241776], **kwargs_241777)
        
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___241779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), _adjust_scheme_to_bounds_call_result_241778, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_241780 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), getitem___241779, int_241769)
        
        # Assigning a type to the variable 'tuple_var_assignment_241380' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'tuple_var_assignment_241380', subscript_call_result_241780)
        
        # Assigning a Name to a Name (line 80):
        # Getting the type of 'tuple_var_assignment_241379' (line 80)
        tuple_var_assignment_241379_241781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'tuple_var_assignment_241379')
        # Assigning a type to the variable 'h_adjusted' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'h_adjusted', tuple_var_assignment_241379_241781)
        
        # Assigning a Name to a Name (line 80):
        # Getting the type of 'tuple_var_assignment_241380' (line 80)
        tuple_var_assignment_241380_241782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'tuple_var_assignment_241380')
        # Assigning a type to the variable '_' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), '_', tuple_var_assignment_241380_241782)
        
        # Call to assert_allclose(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'h_adjusted' (line 81)
        h_adjusted_241784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 24), 'h_adjusted', False)
        # Getting the type of 'h' (line 81)
        h_241785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 36), 'h', False)
        # Processing the call keyword arguments (line 81)
        kwargs_241786 = {}
        # Getting the type of 'assert_allclose' (line 81)
        assert_allclose_241783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 81)
        assert_allclose_call_result_241787 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), assert_allclose_241783, *[h_adjusted_241784, h_241785], **kwargs_241786)
        
        
        # Assigning a Call to a Tuple (line 83):
        
        # Assigning a Subscript to a Name (line 83):
        
        # Obtaining the type of the subscript
        int_241788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'x0' (line 83)
        x0_241790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 49), 'x0', False)
        # Getting the type of 'h' (line 83)
        h_241791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 53), 'h', False)
        int_241792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 56), 'int')
        str_241793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 59), 'str', '1-sided')
        # Getting the type of 'lb' (line 83)
        lb_241794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 70), 'lb', False)
        # Getting the type of 'ub' (line 83)
        ub_241795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 74), 'ub', False)
        # Processing the call keyword arguments (line 83)
        kwargs_241796 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 83)
        _adjust_scheme_to_bounds_241789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 83)
        _adjust_scheme_to_bounds_call_result_241797 = invoke(stypy.reporting.localization.Localization(__file__, 83, 24), _adjust_scheme_to_bounds_241789, *[x0_241790, h_241791, int_241792, str_241793, lb_241794, ub_241795], **kwargs_241796)
        
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___241798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), _adjust_scheme_to_bounds_call_result_241797, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_241799 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), getitem___241798, int_241788)
        
        # Assigning a type to the variable 'tuple_var_assignment_241381' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_var_assignment_241381', subscript_call_result_241799)
        
        # Assigning a Subscript to a Name (line 83):
        
        # Obtaining the type of the subscript
        int_241800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'x0' (line 83)
        x0_241802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 49), 'x0', False)
        # Getting the type of 'h' (line 83)
        h_241803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 53), 'h', False)
        int_241804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 56), 'int')
        str_241805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 59), 'str', '1-sided')
        # Getting the type of 'lb' (line 83)
        lb_241806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 70), 'lb', False)
        # Getting the type of 'ub' (line 83)
        ub_241807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 74), 'ub', False)
        # Processing the call keyword arguments (line 83)
        kwargs_241808 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 83)
        _adjust_scheme_to_bounds_241801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 83)
        _adjust_scheme_to_bounds_call_result_241809 = invoke(stypy.reporting.localization.Localization(__file__, 83, 24), _adjust_scheme_to_bounds_241801, *[x0_241802, h_241803, int_241804, str_241805, lb_241806, ub_241807], **kwargs_241808)
        
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___241810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), _adjust_scheme_to_bounds_call_result_241809, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_241811 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), getitem___241810, int_241800)
        
        # Assigning a type to the variable 'tuple_var_assignment_241382' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_var_assignment_241382', subscript_call_result_241811)
        
        # Assigning a Name to a Name (line 83):
        # Getting the type of 'tuple_var_assignment_241381' (line 83)
        tuple_var_assignment_241381_241812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_var_assignment_241381')
        # Assigning a type to the variable 'h_adjusted' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'h_adjusted', tuple_var_assignment_241381_241812)
        
        # Assigning a Name to a Name (line 83):
        # Getting the type of 'tuple_var_assignment_241382' (line 83)
        tuple_var_assignment_241382_241813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_var_assignment_241382')
        # Assigning a type to the variable '_' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), '_', tuple_var_assignment_241382_241813)
        
        # Call to assert_allclose(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'h_adjusted' (line 84)
        h_adjusted_241815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 24), 'h_adjusted', False)
        
        # Call to array(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_241818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        int_241819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 45), list_241818, int_241819)
        # Adding element type (line 84)
        int_241820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 45), list_241818, int_241820)
        # Adding element type (line 84)
        int_241821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 45), list_241818, int_241821)
        
        # Processing the call keyword arguments (line 84)
        kwargs_241822 = {}
        # Getting the type of 'np' (line 84)
        np_241816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 36), 'np', False)
        # Obtaining the member 'array' of a type (line 84)
        array_241817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 36), np_241816, 'array')
        # Calling array(args, kwargs) (line 84)
        array_call_result_241823 = invoke(stypy.reporting.localization.Localization(__file__, 84, 36), array_241817, *[list_241818], **kwargs_241822)
        
        float_241824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 59), 'float')
        # Applying the binary operator '*' (line 84)
        result_mul_241825 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 36), '*', array_call_result_241823, float_241824)
        
        # Processing the call keyword arguments (line 84)
        kwargs_241826 = {}
        # Getting the type of 'assert_allclose' (line 84)
        assert_allclose_241814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 84)
        assert_allclose_call_result_241827 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), assert_allclose_241814, *[h_adjusted_241815, result_mul_241825], **kwargs_241826)
        
        
        # Assigning a Call to a Tuple (line 86):
        
        # Assigning a Subscript to a Name (line 86):
        
        # Obtaining the type of the subscript
        int_241828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'x0' (line 87)
        x0_241830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'x0', False)
        # Getting the type of 'h' (line 87)
        h_241831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'h', False)
        int_241832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 19), 'int')
        str_241833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 22), 'str', '2-sided')
        # Getting the type of 'lb' (line 87)
        lb_241834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 33), 'lb', False)
        # Getting the type of 'ub' (line 87)
        ub_241835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 37), 'ub', False)
        # Processing the call keyword arguments (line 86)
        kwargs_241836 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 86)
        _adjust_scheme_to_bounds_241829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 86)
        _adjust_scheme_to_bounds_call_result_241837 = invoke(stypy.reporting.localization.Localization(__file__, 86, 32), _adjust_scheme_to_bounds_241829, *[x0_241830, h_241831, int_241832, str_241833, lb_241834, ub_241835], **kwargs_241836)
        
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___241838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), _adjust_scheme_to_bounds_call_result_241837, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_241839 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), getitem___241838, int_241828)
        
        # Assigning a type to the variable 'tuple_var_assignment_241383' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_241383', subscript_call_result_241839)
        
        # Assigning a Subscript to a Name (line 86):
        
        # Obtaining the type of the subscript
        int_241840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'x0' (line 87)
        x0_241842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'x0', False)
        # Getting the type of 'h' (line 87)
        h_241843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'h', False)
        int_241844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 19), 'int')
        str_241845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 22), 'str', '2-sided')
        # Getting the type of 'lb' (line 87)
        lb_241846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 33), 'lb', False)
        # Getting the type of 'ub' (line 87)
        ub_241847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 37), 'ub', False)
        # Processing the call keyword arguments (line 86)
        kwargs_241848 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 86)
        _adjust_scheme_to_bounds_241841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 86)
        _adjust_scheme_to_bounds_call_result_241849 = invoke(stypy.reporting.localization.Localization(__file__, 86, 32), _adjust_scheme_to_bounds_241841, *[x0_241842, h_241843, int_241844, str_241845, lb_241846, ub_241847], **kwargs_241848)
        
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___241850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), _adjust_scheme_to_bounds_call_result_241849, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_241851 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), getitem___241850, int_241840)
        
        # Assigning a type to the variable 'tuple_var_assignment_241384' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_241384', subscript_call_result_241851)
        
        # Assigning a Name to a Name (line 86):
        # Getting the type of 'tuple_var_assignment_241383' (line 86)
        tuple_var_assignment_241383_241852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_241383')
        # Assigning a type to the variable 'h_adjusted' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'h_adjusted', tuple_var_assignment_241383_241852)
        
        # Assigning a Name to a Name (line 86):
        # Getting the type of 'tuple_var_assignment_241384' (line 86)
        tuple_var_assignment_241384_241853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_241384')
        # Assigning a type to the variable 'one_sided' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'one_sided', tuple_var_assignment_241384_241853)
        
        # Call to assert_allclose(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'h_adjusted' (line 88)
        h_adjusted_241855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'h_adjusted', False)
        
        # Call to abs(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'h' (line 88)
        h_241858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 43), 'h', False)
        # Processing the call keyword arguments (line 88)
        kwargs_241859 = {}
        # Getting the type of 'np' (line 88)
        np_241856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 36), 'np', False)
        # Obtaining the member 'abs' of a type (line 88)
        abs_241857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 36), np_241856, 'abs')
        # Calling abs(args, kwargs) (line 88)
        abs_call_result_241860 = invoke(stypy.reporting.localization.Localization(__file__, 88, 36), abs_241857, *[h_241858], **kwargs_241859)
        
        # Processing the call keyword arguments (line 88)
        kwargs_241861 = {}
        # Getting the type of 'assert_allclose' (line 88)
        assert_allclose_241854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 88)
        assert_allclose_call_result_241862 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), assert_allclose_241854, *[h_adjusted_241855, abs_call_result_241860], **kwargs_241861)
        
        
        # Call to assert_(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to all(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Getting the type of 'one_sided' (line 89)
        one_sided_241866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'one_sided', False)
        # Applying the '~' unary operator (line 89)
        result_inv_241867 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 23), '~', one_sided_241866)
        
        # Processing the call keyword arguments (line 89)
        kwargs_241868 = {}
        # Getting the type of 'np' (line 89)
        np_241864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 89)
        all_241865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 16), np_241864, 'all')
        # Calling all(args, kwargs) (line 89)
        all_call_result_241869 = invoke(stypy.reporting.localization.Localization(__file__, 89, 16), all_241865, *[result_inv_241867], **kwargs_241868)
        
        # Processing the call keyword arguments (line 89)
        kwargs_241870 = {}
        # Getting the type of 'assert_' (line 89)
        assert__241863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 89)
        assert__call_result_241871 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assert__241863, *[all_call_result_241869], **kwargs_241870)
        
        
        # Assigning a Call to a Tuple (line 91):
        
        # Assigning a Subscript to a Name (line 91):
        
        # Obtaining the type of the subscript
        int_241872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'x0' (line 92)
        x0_241874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'x0', False)
        # Getting the type of 'h' (line 92)
        h_241875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'h', False)
        int_241876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 19), 'int')
        str_241877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'str', '2-sided')
        # Getting the type of 'lb' (line 92)
        lb_241878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'lb', False)
        # Getting the type of 'ub' (line 92)
        ub_241879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 37), 'ub', False)
        # Processing the call keyword arguments (line 91)
        kwargs_241880 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 91)
        _adjust_scheme_to_bounds_241873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 91)
        _adjust_scheme_to_bounds_call_result_241881 = invoke(stypy.reporting.localization.Localization(__file__, 91, 32), _adjust_scheme_to_bounds_241873, *[x0_241874, h_241875, int_241876, str_241877, lb_241878, ub_241879], **kwargs_241880)
        
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___241882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), _adjust_scheme_to_bounds_call_result_241881, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_241883 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), getitem___241882, int_241872)
        
        # Assigning a type to the variable 'tuple_var_assignment_241385' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_241385', subscript_call_result_241883)
        
        # Assigning a Subscript to a Name (line 91):
        
        # Obtaining the type of the subscript
        int_241884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'x0' (line 92)
        x0_241886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'x0', False)
        # Getting the type of 'h' (line 92)
        h_241887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'h', False)
        int_241888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 19), 'int')
        str_241889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'str', '2-sided')
        # Getting the type of 'lb' (line 92)
        lb_241890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'lb', False)
        # Getting the type of 'ub' (line 92)
        ub_241891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 37), 'ub', False)
        # Processing the call keyword arguments (line 91)
        kwargs_241892 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 91)
        _adjust_scheme_to_bounds_241885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 91)
        _adjust_scheme_to_bounds_call_result_241893 = invoke(stypy.reporting.localization.Localization(__file__, 91, 32), _adjust_scheme_to_bounds_241885, *[x0_241886, h_241887, int_241888, str_241889, lb_241890, ub_241891], **kwargs_241892)
        
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___241894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), _adjust_scheme_to_bounds_call_result_241893, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_241895 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), getitem___241894, int_241884)
        
        # Assigning a type to the variable 'tuple_var_assignment_241386' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_241386', subscript_call_result_241895)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'tuple_var_assignment_241385' (line 91)
        tuple_var_assignment_241385_241896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_241385')
        # Assigning a type to the variable 'h_adjusted' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'h_adjusted', tuple_var_assignment_241385_241896)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'tuple_var_assignment_241386' (line 91)
        tuple_var_assignment_241386_241897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_241386')
        # Assigning a type to the variable 'one_sided' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'one_sided', tuple_var_assignment_241386_241897)
        
        # Call to assert_allclose(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'h_adjusted' (line 93)
        h_adjusted_241899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'h_adjusted', False)
        
        # Call to array(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_241902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        int_241903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 45), list_241902, int_241903)
        # Adding element type (line 93)
        int_241904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 45), list_241902, int_241904)
        # Adding element type (line 93)
        int_241905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 45), list_241902, int_241905)
        
        # Processing the call keyword arguments (line 93)
        kwargs_241906 = {}
        # Getting the type of 'np' (line 93)
        np_241900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 36), 'np', False)
        # Obtaining the member 'array' of a type (line 93)
        array_241901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 36), np_241900, 'array')
        # Calling array(args, kwargs) (line 93)
        array_call_result_241907 = invoke(stypy.reporting.localization.Localization(__file__, 93, 36), array_241901, *[list_241902], **kwargs_241906)
        
        float_241908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 59), 'float')
        # Applying the binary operator '*' (line 93)
        result_mul_241909 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 36), '*', array_call_result_241907, float_241908)
        
        # Processing the call keyword arguments (line 93)
        kwargs_241910 = {}
        # Getting the type of 'assert_allclose' (line 93)
        assert_allclose_241898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 93)
        assert_allclose_call_result_241911 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), assert_allclose_241898, *[h_adjusted_241899, result_mul_241909], **kwargs_241910)
        
        
        # Call to assert_equal(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'one_sided' (line 94)
        one_sided_241913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'one_sided', False)
        
        # Call to array(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_241916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        # Adding element type (line 94)
        # Getting the type of 'False' (line 94)
        False_241917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 42), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 41), list_241916, False_241917)
        # Adding element type (line 94)
        # Getting the type of 'True' (line 94)
        True_241918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 49), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 41), list_241916, True_241918)
        # Adding element type (line 94)
        # Getting the type of 'True' (line 94)
        True_241919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 55), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 41), list_241916, True_241919)
        
        # Processing the call keyword arguments (line 94)
        kwargs_241920 = {}
        # Getting the type of 'np' (line 94)
        np_241914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 32), 'np', False)
        # Obtaining the member 'array' of a type (line 94)
        array_241915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 32), np_241914, 'array')
        # Calling array(args, kwargs) (line 94)
        array_call_result_241921 = invoke(stypy.reporting.localization.Localization(__file__, 94, 32), array_241915, *[list_241916], **kwargs_241920)
        
        # Processing the call keyword arguments (line 94)
        kwargs_241922 = {}
        # Getting the type of 'assert_equal' (line 94)
        assert_equal_241912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 94)
        assert_equal_call_result_241923 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), assert_equal_241912, *[one_sided_241913, array_call_result_241921], **kwargs_241922)
        
        
        # ################# End of 'test_with_bound(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_with_bound' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_241924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_241924)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_with_bound'
        return stypy_return_type_241924


    @norecursion
    def test_tight_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tight_bounds'
        module_type_store = module_type_store.open_function_context('test_tight_bounds', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestAdjustSchemeToBounds.test_tight_bounds.__dict__.__setitem__('stypy_localization', localization)
        TestAdjustSchemeToBounds.test_tight_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestAdjustSchemeToBounds.test_tight_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestAdjustSchemeToBounds.test_tight_bounds.__dict__.__setitem__('stypy_function_name', 'TestAdjustSchemeToBounds.test_tight_bounds')
        TestAdjustSchemeToBounds.test_tight_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        TestAdjustSchemeToBounds.test_tight_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestAdjustSchemeToBounds.test_tight_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestAdjustSchemeToBounds.test_tight_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestAdjustSchemeToBounds.test_tight_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestAdjustSchemeToBounds.test_tight_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestAdjustSchemeToBounds.test_tight_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestAdjustSchemeToBounds.test_tight_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tight_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tight_bounds(...)' code ##################

        
        # Assigning a Call to a Name (line 97):
        
        # Assigning a Call to a Name (line 97):
        
        # Call to array(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_241927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        float_241928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 22), list_241927, float_241928)
        # Adding element type (line 97)
        float_241929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 22), list_241927, float_241929)
        
        # Processing the call keyword arguments (line 97)
        kwargs_241930 = {}
        # Getting the type of 'np' (line 97)
        np_241925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 97)
        array_241926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 13), np_241925, 'array')
        # Calling array(args, kwargs) (line 97)
        array_call_result_241931 = invoke(stypy.reporting.localization.Localization(__file__, 97, 13), array_241926, *[list_241927], **kwargs_241930)
        
        # Assigning a type to the variable 'lb' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'lb', array_call_result_241931)
        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to array(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_241934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        # Adding element type (line 98)
        float_241935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 22), list_241934, float_241935)
        # Adding element type (line 98)
        float_241936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 22), list_241934, float_241936)
        
        # Processing the call keyword arguments (line 98)
        kwargs_241937 = {}
        # Getting the type of 'np' (line 98)
        np_241932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 98)
        array_241933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 13), np_241932, 'array')
        # Calling array(args, kwargs) (line 98)
        array_call_result_241938 = invoke(stypy.reporting.localization.Localization(__file__, 98, 13), array_241933, *[list_241934], **kwargs_241937)
        
        # Assigning a type to the variable 'ub' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'ub', array_call_result_241938)
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to array(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Obtaining an instance of the builtin type 'list' (line 99)
        list_241941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 99)
        # Adding element type (line 99)
        float_241942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 22), list_241941, float_241942)
        # Adding element type (line 99)
        float_241943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 22), list_241941, float_241943)
        
        # Processing the call keyword arguments (line 99)
        kwargs_241944 = {}
        # Getting the type of 'np' (line 99)
        np_241939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 99)
        array_241940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 13), np_241939, 'array')
        # Calling array(args, kwargs) (line 99)
        array_call_result_241945 = invoke(stypy.reporting.localization.Localization(__file__, 99, 13), array_241940, *[list_241941], **kwargs_241944)
        
        # Assigning a type to the variable 'x0' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'x0', array_call_result_241945)
        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to array(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_241948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        # Adding element type (line 100)
        float_241949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 21), list_241948, float_241949)
        # Adding element type (line 100)
        float_241950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 21), list_241948, float_241950)
        
        # Processing the call keyword arguments (line 100)
        kwargs_241951 = {}
        # Getting the type of 'np' (line 100)
        np_241946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 100)
        array_241947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), np_241946, 'array')
        # Calling array(args, kwargs) (line 100)
        array_call_result_241952 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), array_241947, *[list_241948], **kwargs_241951)
        
        # Assigning a type to the variable 'h' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'h', array_call_result_241952)
        
        # Assigning a Call to a Tuple (line 102):
        
        # Assigning a Subscript to a Name (line 102):
        
        # Obtaining the type of the subscript
        int_241953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'x0' (line 102)
        x0_241955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 49), 'x0', False)
        # Getting the type of 'h' (line 102)
        h_241956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 53), 'h', False)
        int_241957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 56), 'int')
        str_241958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 59), 'str', '1-sided')
        # Getting the type of 'lb' (line 102)
        lb_241959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 70), 'lb', False)
        # Getting the type of 'ub' (line 102)
        ub_241960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 74), 'ub', False)
        # Processing the call keyword arguments (line 102)
        kwargs_241961 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 102)
        _adjust_scheme_to_bounds_241954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 102)
        _adjust_scheme_to_bounds_call_result_241962 = invoke(stypy.reporting.localization.Localization(__file__, 102, 24), _adjust_scheme_to_bounds_241954, *[x0_241955, h_241956, int_241957, str_241958, lb_241959, ub_241960], **kwargs_241961)
        
        # Obtaining the member '__getitem__' of a type (line 102)
        getitem___241963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), _adjust_scheme_to_bounds_call_result_241962, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 102)
        subscript_call_result_241964 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), getitem___241963, int_241953)
        
        # Assigning a type to the variable 'tuple_var_assignment_241387' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'tuple_var_assignment_241387', subscript_call_result_241964)
        
        # Assigning a Subscript to a Name (line 102):
        
        # Obtaining the type of the subscript
        int_241965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'x0' (line 102)
        x0_241967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 49), 'x0', False)
        # Getting the type of 'h' (line 102)
        h_241968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 53), 'h', False)
        int_241969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 56), 'int')
        str_241970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 59), 'str', '1-sided')
        # Getting the type of 'lb' (line 102)
        lb_241971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 70), 'lb', False)
        # Getting the type of 'ub' (line 102)
        ub_241972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 74), 'ub', False)
        # Processing the call keyword arguments (line 102)
        kwargs_241973 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 102)
        _adjust_scheme_to_bounds_241966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 102)
        _adjust_scheme_to_bounds_call_result_241974 = invoke(stypy.reporting.localization.Localization(__file__, 102, 24), _adjust_scheme_to_bounds_241966, *[x0_241967, h_241968, int_241969, str_241970, lb_241971, ub_241972], **kwargs_241973)
        
        # Obtaining the member '__getitem__' of a type (line 102)
        getitem___241975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), _adjust_scheme_to_bounds_call_result_241974, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 102)
        subscript_call_result_241976 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), getitem___241975, int_241965)
        
        # Assigning a type to the variable 'tuple_var_assignment_241388' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'tuple_var_assignment_241388', subscript_call_result_241976)
        
        # Assigning a Name to a Name (line 102):
        # Getting the type of 'tuple_var_assignment_241387' (line 102)
        tuple_var_assignment_241387_241977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'tuple_var_assignment_241387')
        # Assigning a type to the variable 'h_adjusted' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'h_adjusted', tuple_var_assignment_241387_241977)
        
        # Assigning a Name to a Name (line 102):
        # Getting the type of 'tuple_var_assignment_241388' (line 102)
        tuple_var_assignment_241388_241978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'tuple_var_assignment_241388')
        # Assigning a type to the variable '_' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), '_', tuple_var_assignment_241388_241978)
        
        # Call to assert_allclose(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'h_adjusted' (line 103)
        h_adjusted_241980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'h_adjusted', False)
        
        # Call to array(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining an instance of the builtin type 'list' (line 103)
        list_241983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 103)
        # Adding element type (line 103)
        float_241984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 45), list_241983, float_241984)
        # Adding element type (line 103)
        float_241985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 45), list_241983, float_241985)
        
        # Processing the call keyword arguments (line 103)
        kwargs_241986 = {}
        # Getting the type of 'np' (line 103)
        np_241981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 36), 'np', False)
        # Obtaining the member 'array' of a type (line 103)
        array_241982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 36), np_241981, 'array')
        # Calling array(args, kwargs) (line 103)
        array_call_result_241987 = invoke(stypy.reporting.localization.Localization(__file__, 103, 36), array_241982, *[list_241983], **kwargs_241986)
        
        # Processing the call keyword arguments (line 103)
        kwargs_241988 = {}
        # Getting the type of 'assert_allclose' (line 103)
        assert_allclose_241979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 103)
        assert_allclose_call_result_241989 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), assert_allclose_241979, *[h_adjusted_241980, array_call_result_241987], **kwargs_241988)
        
        
        # Assigning a Call to a Tuple (line 105):
        
        # Assigning a Subscript to a Name (line 105):
        
        # Obtaining the type of the subscript
        int_241990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'x0' (line 105)
        x0_241992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 49), 'x0', False)
        # Getting the type of 'h' (line 105)
        h_241993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 53), 'h', False)
        int_241994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 56), 'int')
        str_241995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 59), 'str', '1-sided')
        # Getting the type of 'lb' (line 105)
        lb_241996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 70), 'lb', False)
        # Getting the type of 'ub' (line 105)
        ub_241997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 74), 'ub', False)
        # Processing the call keyword arguments (line 105)
        kwargs_241998 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 105)
        _adjust_scheme_to_bounds_241991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 105)
        _adjust_scheme_to_bounds_call_result_241999 = invoke(stypy.reporting.localization.Localization(__file__, 105, 24), _adjust_scheme_to_bounds_241991, *[x0_241992, h_241993, int_241994, str_241995, lb_241996, ub_241997], **kwargs_241998)
        
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___242000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), _adjust_scheme_to_bounds_call_result_241999, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_242001 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), getitem___242000, int_241990)
        
        # Assigning a type to the variable 'tuple_var_assignment_241389' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tuple_var_assignment_241389', subscript_call_result_242001)
        
        # Assigning a Subscript to a Name (line 105):
        
        # Obtaining the type of the subscript
        int_242002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'x0' (line 105)
        x0_242004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 49), 'x0', False)
        # Getting the type of 'h' (line 105)
        h_242005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 53), 'h', False)
        int_242006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 56), 'int')
        str_242007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 59), 'str', '1-sided')
        # Getting the type of 'lb' (line 105)
        lb_242008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 70), 'lb', False)
        # Getting the type of 'ub' (line 105)
        ub_242009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 74), 'ub', False)
        # Processing the call keyword arguments (line 105)
        kwargs_242010 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 105)
        _adjust_scheme_to_bounds_242003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 105)
        _adjust_scheme_to_bounds_call_result_242011 = invoke(stypy.reporting.localization.Localization(__file__, 105, 24), _adjust_scheme_to_bounds_242003, *[x0_242004, h_242005, int_242006, str_242007, lb_242008, ub_242009], **kwargs_242010)
        
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___242012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), _adjust_scheme_to_bounds_call_result_242011, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_242013 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), getitem___242012, int_242002)
        
        # Assigning a type to the variable 'tuple_var_assignment_241390' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tuple_var_assignment_241390', subscript_call_result_242013)
        
        # Assigning a Name to a Name (line 105):
        # Getting the type of 'tuple_var_assignment_241389' (line 105)
        tuple_var_assignment_241389_242014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tuple_var_assignment_241389')
        # Assigning a type to the variable 'h_adjusted' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'h_adjusted', tuple_var_assignment_241389_242014)
        
        # Assigning a Name to a Name (line 105):
        # Getting the type of 'tuple_var_assignment_241390' (line 105)
        tuple_var_assignment_241390_242015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tuple_var_assignment_241390')
        # Assigning a type to the variable '_' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), '_', tuple_var_assignment_241390_242015)
        
        # Call to assert_allclose(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'h_adjusted' (line 106)
        h_adjusted_242017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'h_adjusted', False)
        
        # Call to array(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Obtaining an instance of the builtin type 'list' (line 106)
        list_242020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 106)
        # Adding element type (line 106)
        float_242021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 45), list_242020, float_242021)
        # Adding element type (line 106)
        float_242022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 45), list_242020, float_242022)
        
        # Processing the call keyword arguments (line 106)
        kwargs_242023 = {}
        # Getting the type of 'np' (line 106)
        np_242018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 36), 'np', False)
        # Obtaining the member 'array' of a type (line 106)
        array_242019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 36), np_242018, 'array')
        # Calling array(args, kwargs) (line 106)
        array_call_result_242024 = invoke(stypy.reporting.localization.Localization(__file__, 106, 36), array_242019, *[list_242020], **kwargs_242023)
        
        # Processing the call keyword arguments (line 106)
        kwargs_242025 = {}
        # Getting the type of 'assert_allclose' (line 106)
        assert_allclose_242016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 106)
        assert_allclose_call_result_242026 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), assert_allclose_242016, *[h_adjusted_242017, array_call_result_242024], **kwargs_242025)
        
        
        # Assigning a Call to a Tuple (line 108):
        
        # Assigning a Subscript to a Name (line 108):
        
        # Obtaining the type of the subscript
        int_242027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'x0' (line 109)
        x0_242029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'x0', False)
        # Getting the type of 'h' (line 109)
        h_242030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'h', False)
        int_242031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 19), 'int')
        str_242032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'str', '2-sided')
        # Getting the type of 'lb' (line 109)
        lb_242033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 33), 'lb', False)
        # Getting the type of 'ub' (line 109)
        ub_242034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 37), 'ub', False)
        # Processing the call keyword arguments (line 108)
        kwargs_242035 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 108)
        _adjust_scheme_to_bounds_242028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 108)
        _adjust_scheme_to_bounds_call_result_242036 = invoke(stypy.reporting.localization.Localization(__file__, 108, 32), _adjust_scheme_to_bounds_242028, *[x0_242029, h_242030, int_242031, str_242032, lb_242033, ub_242034], **kwargs_242035)
        
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___242037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), _adjust_scheme_to_bounds_call_result_242036, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_242038 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), getitem___242037, int_242027)
        
        # Assigning a type to the variable 'tuple_var_assignment_241391' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_241391', subscript_call_result_242038)
        
        # Assigning a Subscript to a Name (line 108):
        
        # Obtaining the type of the subscript
        int_242039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'x0' (line 109)
        x0_242041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'x0', False)
        # Getting the type of 'h' (line 109)
        h_242042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'h', False)
        int_242043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 19), 'int')
        str_242044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'str', '2-sided')
        # Getting the type of 'lb' (line 109)
        lb_242045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 33), 'lb', False)
        # Getting the type of 'ub' (line 109)
        ub_242046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 37), 'ub', False)
        # Processing the call keyword arguments (line 108)
        kwargs_242047 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 108)
        _adjust_scheme_to_bounds_242040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 108)
        _adjust_scheme_to_bounds_call_result_242048 = invoke(stypy.reporting.localization.Localization(__file__, 108, 32), _adjust_scheme_to_bounds_242040, *[x0_242041, h_242042, int_242043, str_242044, lb_242045, ub_242046], **kwargs_242047)
        
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___242049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), _adjust_scheme_to_bounds_call_result_242048, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_242050 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), getitem___242049, int_242039)
        
        # Assigning a type to the variable 'tuple_var_assignment_241392' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_241392', subscript_call_result_242050)
        
        # Assigning a Name to a Name (line 108):
        # Getting the type of 'tuple_var_assignment_241391' (line 108)
        tuple_var_assignment_241391_242051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_241391')
        # Assigning a type to the variable 'h_adjusted' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'h_adjusted', tuple_var_assignment_241391_242051)
        
        # Assigning a Name to a Name (line 108):
        # Getting the type of 'tuple_var_assignment_241392' (line 108)
        tuple_var_assignment_241392_242052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_241392')
        # Assigning a type to the variable 'one_sided' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'one_sided', tuple_var_assignment_241392_242052)
        
        # Call to assert_allclose(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'h_adjusted' (line 110)
        h_adjusted_242054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 24), 'h_adjusted', False)
        
        # Call to array(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_242057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        float_242058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 45), list_242057, float_242058)
        # Adding element type (line 110)
        float_242059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 45), list_242057, float_242059)
        
        # Processing the call keyword arguments (line 110)
        kwargs_242060 = {}
        # Getting the type of 'np' (line 110)
        np_242055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 36), 'np', False)
        # Obtaining the member 'array' of a type (line 110)
        array_242056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 36), np_242055, 'array')
        # Calling array(args, kwargs) (line 110)
        array_call_result_242061 = invoke(stypy.reporting.localization.Localization(__file__, 110, 36), array_242056, *[list_242057], **kwargs_242060)
        
        # Processing the call keyword arguments (line 110)
        kwargs_242062 = {}
        # Getting the type of 'assert_allclose' (line 110)
        assert_allclose_242053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 110)
        assert_allclose_call_result_242063 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), assert_allclose_242053, *[h_adjusted_242054, array_call_result_242061], **kwargs_242062)
        
        
        # Call to assert_equal(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'one_sided' (line 111)
        one_sided_242065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 21), 'one_sided', False)
        
        # Call to array(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_242068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        # Getting the type of 'False' (line 111)
        False_242069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 42), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 41), list_242068, False_242069)
        # Adding element type (line 111)
        # Getting the type of 'True' (line 111)
        True_242070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 49), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 41), list_242068, True_242070)
        
        # Processing the call keyword arguments (line 111)
        kwargs_242071 = {}
        # Getting the type of 'np' (line 111)
        np_242066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 32), 'np', False)
        # Obtaining the member 'array' of a type (line 111)
        array_242067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 32), np_242066, 'array')
        # Calling array(args, kwargs) (line 111)
        array_call_result_242072 = invoke(stypy.reporting.localization.Localization(__file__, 111, 32), array_242067, *[list_242068], **kwargs_242071)
        
        # Processing the call keyword arguments (line 111)
        kwargs_242073 = {}
        # Getting the type of 'assert_equal' (line 111)
        assert_equal_242064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 111)
        assert_equal_call_result_242074 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), assert_equal_242064, *[one_sided_242065, array_call_result_242072], **kwargs_242073)
        
        
        # Assigning a Call to a Tuple (line 113):
        
        # Assigning a Subscript to a Name (line 113):
        
        # Obtaining the type of the subscript
        int_242075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'x0' (line 114)
        x0_242077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'x0', False)
        # Getting the type of 'h' (line 114)
        h_242078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'h', False)
        int_242079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 19), 'int')
        str_242080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 22), 'str', '2-sided')
        # Getting the type of 'lb' (line 114)
        lb_242081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 33), 'lb', False)
        # Getting the type of 'ub' (line 114)
        ub_242082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 37), 'ub', False)
        # Processing the call keyword arguments (line 113)
        kwargs_242083 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 113)
        _adjust_scheme_to_bounds_242076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 113)
        _adjust_scheme_to_bounds_call_result_242084 = invoke(stypy.reporting.localization.Localization(__file__, 113, 32), _adjust_scheme_to_bounds_242076, *[x0_242077, h_242078, int_242079, str_242080, lb_242081, ub_242082], **kwargs_242083)
        
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___242085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), _adjust_scheme_to_bounds_call_result_242084, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
        subscript_call_result_242086 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), getitem___242085, int_242075)
        
        # Assigning a type to the variable 'tuple_var_assignment_241393' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'tuple_var_assignment_241393', subscript_call_result_242086)
        
        # Assigning a Subscript to a Name (line 113):
        
        # Obtaining the type of the subscript
        int_242087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 8), 'int')
        
        # Call to _adjust_scheme_to_bounds(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'x0' (line 114)
        x0_242089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'x0', False)
        # Getting the type of 'h' (line 114)
        h_242090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'h', False)
        int_242091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 19), 'int')
        str_242092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 22), 'str', '2-sided')
        # Getting the type of 'lb' (line 114)
        lb_242093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 33), 'lb', False)
        # Getting the type of 'ub' (line 114)
        ub_242094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 37), 'ub', False)
        # Processing the call keyword arguments (line 113)
        kwargs_242095 = {}
        # Getting the type of '_adjust_scheme_to_bounds' (line 113)
        _adjust_scheme_to_bounds_242088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 32), '_adjust_scheme_to_bounds', False)
        # Calling _adjust_scheme_to_bounds(args, kwargs) (line 113)
        _adjust_scheme_to_bounds_call_result_242096 = invoke(stypy.reporting.localization.Localization(__file__, 113, 32), _adjust_scheme_to_bounds_242088, *[x0_242089, h_242090, int_242091, str_242092, lb_242093, ub_242094], **kwargs_242095)
        
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___242097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), _adjust_scheme_to_bounds_call_result_242096, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
        subscript_call_result_242098 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), getitem___242097, int_242087)
        
        # Assigning a type to the variable 'tuple_var_assignment_241394' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'tuple_var_assignment_241394', subscript_call_result_242098)
        
        # Assigning a Name to a Name (line 113):
        # Getting the type of 'tuple_var_assignment_241393' (line 113)
        tuple_var_assignment_241393_242099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'tuple_var_assignment_241393')
        # Assigning a type to the variable 'h_adjusted' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'h_adjusted', tuple_var_assignment_241393_242099)
        
        # Assigning a Name to a Name (line 113):
        # Getting the type of 'tuple_var_assignment_241394' (line 113)
        tuple_var_assignment_241394_242100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'tuple_var_assignment_241394')
        # Assigning a type to the variable 'one_sided' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'one_sided', tuple_var_assignment_241394_242100)
        
        # Call to assert_allclose(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'h_adjusted' (line 115)
        h_adjusted_242102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 24), 'h_adjusted', False)
        
        # Call to array(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_242105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        # Adding element type (line 115)
        float_242106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 45), list_242105, float_242106)
        # Adding element type (line 115)
        float_242107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 45), list_242105, float_242107)
        
        # Processing the call keyword arguments (line 115)
        kwargs_242108 = {}
        # Getting the type of 'np' (line 115)
        np_242103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 36), 'np', False)
        # Obtaining the member 'array' of a type (line 115)
        array_242104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 36), np_242103, 'array')
        # Calling array(args, kwargs) (line 115)
        array_call_result_242109 = invoke(stypy.reporting.localization.Localization(__file__, 115, 36), array_242104, *[list_242105], **kwargs_242108)
        
        # Processing the call keyword arguments (line 115)
        kwargs_242110 = {}
        # Getting the type of 'assert_allclose' (line 115)
        assert_allclose_242101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 115)
        assert_allclose_call_result_242111 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), assert_allclose_242101, *[h_adjusted_242102, array_call_result_242109], **kwargs_242110)
        
        
        # Call to assert_equal(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'one_sided' (line 116)
        one_sided_242113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 21), 'one_sided', False)
        
        # Call to array(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_242116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        # Adding element type (line 116)
        # Getting the type of 'False' (line 116)
        False_242117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 42), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 41), list_242116, False_242117)
        # Adding element type (line 116)
        # Getting the type of 'True' (line 116)
        True_242118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 49), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 41), list_242116, True_242118)
        
        # Processing the call keyword arguments (line 116)
        kwargs_242119 = {}
        # Getting the type of 'np' (line 116)
        np_242114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 32), 'np', False)
        # Obtaining the member 'array' of a type (line 116)
        array_242115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 32), np_242114, 'array')
        # Calling array(args, kwargs) (line 116)
        array_call_result_242120 = invoke(stypy.reporting.localization.Localization(__file__, 116, 32), array_242115, *[list_242116], **kwargs_242119)
        
        # Processing the call keyword arguments (line 116)
        kwargs_242121 = {}
        # Getting the type of 'assert_equal' (line 116)
        assert_equal_242112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 116)
        assert_equal_call_result_242122 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), assert_equal_242112, *[one_sided_242113, array_call_result_242120], **kwargs_242121)
        
        
        # ################# End of 'test_tight_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tight_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_242123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242123)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tight_bounds'
        return stypy_return_type_242123


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 45, 0, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestAdjustSchemeToBounds.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestAdjustSchemeToBounds' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'TestAdjustSchemeToBounds', TestAdjustSchemeToBounds)
# Declaration of the 'TestApproxDerivativesDense' class

class TestApproxDerivativesDense(object, ):

    @norecursion
    def fun_scalar_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun_scalar_scalar'
        module_type_store = module_type_store.open_function_context('fun_scalar_scalar', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.fun_scalar_scalar.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.fun_scalar_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.fun_scalar_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.fun_scalar_scalar.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.fun_scalar_scalar')
        TestApproxDerivativesDense.fun_scalar_scalar.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.fun_scalar_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.fun_scalar_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.fun_scalar_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.fun_scalar_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.fun_scalar_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.fun_scalar_scalar.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.fun_scalar_scalar', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun_scalar_scalar', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun_scalar_scalar(...)' code ##################

        
        # Call to sinh(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'x' (line 121)
        x_242126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'x', False)
        # Processing the call keyword arguments (line 121)
        kwargs_242127 = {}
        # Getting the type of 'np' (line 121)
        np_242124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'np', False)
        # Obtaining the member 'sinh' of a type (line 121)
        sinh_242125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 15), np_242124, 'sinh')
        # Calling sinh(args, kwargs) (line 121)
        sinh_call_result_242128 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), sinh_242125, *[x_242126], **kwargs_242127)
        
        # Assigning a type to the variable 'stypy_return_type' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'stypy_return_type', sinh_call_result_242128)
        
        # ################# End of 'fun_scalar_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_scalar_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_242129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242129)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_scalar_scalar'
        return stypy_return_type_242129


    @norecursion
    def jac_scalar_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac_scalar_scalar'
        module_type_store = module_type_store.open_function_context('jac_scalar_scalar', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.jac_scalar_scalar.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.jac_scalar_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.jac_scalar_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.jac_scalar_scalar.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.jac_scalar_scalar')
        TestApproxDerivativesDense.jac_scalar_scalar.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.jac_scalar_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.jac_scalar_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.jac_scalar_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.jac_scalar_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.jac_scalar_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.jac_scalar_scalar.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.jac_scalar_scalar', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_scalar_scalar', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_scalar_scalar(...)' code ##################

        
        # Call to cosh(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'x' (line 124)
        x_242132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'x', False)
        # Processing the call keyword arguments (line 124)
        kwargs_242133 = {}
        # Getting the type of 'np' (line 124)
        np_242130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'np', False)
        # Obtaining the member 'cosh' of a type (line 124)
        cosh_242131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 15), np_242130, 'cosh')
        # Calling cosh(args, kwargs) (line 124)
        cosh_call_result_242134 = invoke(stypy.reporting.localization.Localization(__file__, 124, 15), cosh_242131, *[x_242132], **kwargs_242133)
        
        # Assigning a type to the variable 'stypy_return_type' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'stypy_return_type', cosh_call_result_242134)
        
        # ################# End of 'jac_scalar_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_scalar_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_242135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242135)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_scalar_scalar'
        return stypy_return_type_242135


    @norecursion
    def fun_scalar_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun_scalar_vector'
        module_type_store = module_type_store.open_function_context('fun_scalar_vector', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.fun_scalar_vector.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.fun_scalar_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.fun_scalar_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.fun_scalar_vector.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.fun_scalar_vector')
        TestApproxDerivativesDense.fun_scalar_vector.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.fun_scalar_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.fun_scalar_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.fun_scalar_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.fun_scalar_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.fun_scalar_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.fun_scalar_vector.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.fun_scalar_vector', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun_scalar_vector', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun_scalar_vector(...)' code ##################

        
        # Call to array(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Obtaining an instance of the builtin type 'list' (line 127)
        list_242138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 127)
        # Adding element type (line 127)
        
        # Obtaining the type of the subscript
        int_242139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 27), 'int')
        # Getting the type of 'x' (line 127)
        x_242140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 25), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___242141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 25), x_242140, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_242142 = invoke(stypy.reporting.localization.Localization(__file__, 127, 25), getitem___242141, int_242139)
        
        int_242143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 31), 'int')
        # Applying the binary operator '**' (line 127)
        result_pow_242144 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 25), '**', subscript_call_result_242142, int_242143)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 24), list_242138, result_pow_242144)
        # Adding element type (line 127)
        
        # Call to tan(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Obtaining the type of the subscript
        int_242147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 43), 'int')
        # Getting the type of 'x' (line 127)
        x_242148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 41), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___242149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 41), x_242148, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_242150 = invoke(stypy.reporting.localization.Localization(__file__, 127, 41), getitem___242149, int_242147)
        
        # Processing the call keyword arguments (line 127)
        kwargs_242151 = {}
        # Getting the type of 'np' (line 127)
        np_242145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 34), 'np', False)
        # Obtaining the member 'tan' of a type (line 127)
        tan_242146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 34), np_242145, 'tan')
        # Calling tan(args, kwargs) (line 127)
        tan_call_result_242152 = invoke(stypy.reporting.localization.Localization(__file__, 127, 34), tan_242146, *[subscript_call_result_242150], **kwargs_242151)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 24), list_242138, tan_call_result_242152)
        # Adding element type (line 127)
        
        # Call to exp(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Obtaining the type of the subscript
        int_242155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 57), 'int')
        # Getting the type of 'x' (line 127)
        x_242156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 55), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___242157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 55), x_242156, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_242158 = invoke(stypy.reporting.localization.Localization(__file__, 127, 55), getitem___242157, int_242155)
        
        # Processing the call keyword arguments (line 127)
        kwargs_242159 = {}
        # Getting the type of 'np' (line 127)
        np_242153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 48), 'np', False)
        # Obtaining the member 'exp' of a type (line 127)
        exp_242154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 48), np_242153, 'exp')
        # Calling exp(args, kwargs) (line 127)
        exp_call_result_242160 = invoke(stypy.reporting.localization.Localization(__file__, 127, 48), exp_242154, *[subscript_call_result_242158], **kwargs_242159)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 24), list_242138, exp_call_result_242160)
        
        # Processing the call keyword arguments (line 127)
        kwargs_242161 = {}
        # Getting the type of 'np' (line 127)
        np_242136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 127)
        array_242137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 15), np_242136, 'array')
        # Calling array(args, kwargs) (line 127)
        array_call_result_242162 = invoke(stypy.reporting.localization.Localization(__file__, 127, 15), array_242137, *[list_242138], **kwargs_242161)
        
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'stypy_return_type', array_call_result_242162)
        
        # ################# End of 'fun_scalar_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_scalar_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_242163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242163)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_scalar_vector'
        return stypy_return_type_242163


    @norecursion
    def jac_scalar_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac_scalar_vector'
        module_type_store = module_type_store.open_function_context('jac_scalar_vector', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.jac_scalar_vector.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.jac_scalar_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.jac_scalar_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.jac_scalar_vector.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.jac_scalar_vector')
        TestApproxDerivativesDense.jac_scalar_vector.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.jac_scalar_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.jac_scalar_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.jac_scalar_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.jac_scalar_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.jac_scalar_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.jac_scalar_vector.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.jac_scalar_vector', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_scalar_vector', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_scalar_vector(...)' code ##################

        
        # Call to reshape(...): (line 130)
        # Processing the call arguments (line 130)
        int_242194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 66), 'int')
        int_242195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 70), 'int')
        # Processing the call keyword arguments (line 130)
        kwargs_242196 = {}
        
        # Call to array(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Obtaining an instance of the builtin type 'list' (line 131)
        list_242166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 131)
        # Adding element type (line 131)
        int_242167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 13), 'int')
        
        # Obtaining the type of the subscript
        int_242168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 19), 'int')
        # Getting the type of 'x' (line 131)
        x_242169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 17), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___242170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 17), x_242169, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_242171 = invoke(stypy.reporting.localization.Localization(__file__, 131, 17), getitem___242170, int_242168)
        
        # Applying the binary operator '*' (line 131)
        result_mul_242172 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 13), '*', int_242167, subscript_call_result_242171)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 12), list_242166, result_mul_242172)
        # Adding element type (line 131)
        
        # Call to cos(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Obtaining the type of the subscript
        int_242175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 32), 'int')
        # Getting the type of 'x' (line 131)
        x_242176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___242177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 30), x_242176, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_242178 = invoke(stypy.reporting.localization.Localization(__file__, 131, 30), getitem___242177, int_242175)
        
        # Processing the call keyword arguments (line 131)
        kwargs_242179 = {}
        # Getting the type of 'np' (line 131)
        np_242173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 23), 'np', False)
        # Obtaining the member 'cos' of a type (line 131)
        cos_242174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 23), np_242173, 'cos')
        # Calling cos(args, kwargs) (line 131)
        cos_call_result_242180 = invoke(stypy.reporting.localization.Localization(__file__, 131, 23), cos_242174, *[subscript_call_result_242178], **kwargs_242179)
        
        int_242181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 39), 'int')
        # Applying the binary operator '**' (line 131)
        result_pow_242182 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 23), '**', cos_call_result_242180, int_242181)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 12), list_242166, result_pow_242182)
        # Adding element type (line 131)
        
        # Call to exp(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Obtaining the type of the subscript
        int_242185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 52), 'int')
        # Getting the type of 'x' (line 131)
        x_242186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 50), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___242187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 50), x_242186, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_242188 = invoke(stypy.reporting.localization.Localization(__file__, 131, 50), getitem___242187, int_242185)
        
        # Processing the call keyword arguments (line 131)
        kwargs_242189 = {}
        # Getting the type of 'np' (line 131)
        np_242183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 43), 'np', False)
        # Obtaining the member 'exp' of a type (line 131)
        exp_242184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 43), np_242183, 'exp')
        # Calling exp(args, kwargs) (line 131)
        exp_call_result_242190 = invoke(stypy.reporting.localization.Localization(__file__, 131, 43), exp_242184, *[subscript_call_result_242188], **kwargs_242189)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 12), list_242166, exp_call_result_242190)
        
        # Processing the call keyword arguments (line 130)
        kwargs_242191 = {}
        # Getting the type of 'np' (line 130)
        np_242164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 130)
        array_242165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 15), np_242164, 'array')
        # Calling array(args, kwargs) (line 130)
        array_call_result_242192 = invoke(stypy.reporting.localization.Localization(__file__, 130, 15), array_242165, *[list_242166], **kwargs_242191)
        
        # Obtaining the member 'reshape' of a type (line 130)
        reshape_242193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 15), array_call_result_242192, 'reshape')
        # Calling reshape(args, kwargs) (line 130)
        reshape_call_result_242197 = invoke(stypy.reporting.localization.Localization(__file__, 130, 15), reshape_242193, *[int_242194, int_242195], **kwargs_242196)
        
        # Assigning a type to the variable 'stypy_return_type' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'stypy_return_type', reshape_call_result_242197)
        
        # ################# End of 'jac_scalar_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_scalar_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_242198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242198)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_scalar_vector'
        return stypy_return_type_242198


    @norecursion
    def fun_vector_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun_vector_scalar'
        module_type_store = module_type_store.open_function_context('fun_vector_scalar', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.fun_vector_scalar.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.fun_vector_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.fun_vector_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.fun_vector_scalar.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.fun_vector_scalar')
        TestApproxDerivativesDense.fun_vector_scalar.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.fun_vector_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.fun_vector_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.fun_vector_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.fun_vector_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.fun_vector_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.fun_vector_scalar.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.fun_vector_scalar', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun_vector_scalar', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun_vector_scalar(...)' code ##################

        
        # Call to sin(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Obtaining the type of the subscript
        int_242201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 24), 'int')
        # Getting the type of 'x' (line 134)
        x_242202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___242203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 22), x_242202, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_242204 = invoke(stypy.reporting.localization.Localization(__file__, 134, 22), getitem___242203, int_242201)
        
        
        # Obtaining the type of the subscript
        int_242205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 31), 'int')
        # Getting the type of 'x' (line 134)
        x_242206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 29), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___242207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 29), x_242206, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_242208 = invoke(stypy.reporting.localization.Localization(__file__, 134, 29), getitem___242207, int_242205)
        
        # Applying the binary operator '*' (line 134)
        result_mul_242209 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 22), '*', subscript_call_result_242204, subscript_call_result_242208)
        
        # Processing the call keyword arguments (line 134)
        kwargs_242210 = {}
        # Getting the type of 'np' (line 134)
        np_242199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'np', False)
        # Obtaining the member 'sin' of a type (line 134)
        sin_242200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 15), np_242199, 'sin')
        # Calling sin(args, kwargs) (line 134)
        sin_call_result_242211 = invoke(stypy.reporting.localization.Localization(__file__, 134, 15), sin_242200, *[result_mul_242209], **kwargs_242210)
        
        
        # Call to log(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Obtaining the type of the subscript
        int_242214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 46), 'int')
        # Getting the type of 'x' (line 134)
        x_242215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 44), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___242216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 44), x_242215, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_242217 = invoke(stypy.reporting.localization.Localization(__file__, 134, 44), getitem___242216, int_242214)
        
        # Processing the call keyword arguments (line 134)
        kwargs_242218 = {}
        # Getting the type of 'np' (line 134)
        np_242212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 37), 'np', False)
        # Obtaining the member 'log' of a type (line 134)
        log_242213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 37), np_242212, 'log')
        # Calling log(args, kwargs) (line 134)
        log_call_result_242219 = invoke(stypy.reporting.localization.Localization(__file__, 134, 37), log_242213, *[subscript_call_result_242217], **kwargs_242218)
        
        # Applying the binary operator '*' (line 134)
        result_mul_242220 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), '*', sin_call_result_242211, log_call_result_242219)
        
        # Assigning a type to the variable 'stypy_return_type' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type', result_mul_242220)
        
        # ################# End of 'fun_vector_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_vector_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_242221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242221)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_vector_scalar'
        return stypy_return_type_242221


    @norecursion
    def wrong_dimensions_fun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wrong_dimensions_fun'
        module_type_store = module_type_store.open_function_context('wrong_dimensions_fun', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.wrong_dimensions_fun.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.wrong_dimensions_fun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.wrong_dimensions_fun.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.wrong_dimensions_fun.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.wrong_dimensions_fun')
        TestApproxDerivativesDense.wrong_dimensions_fun.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.wrong_dimensions_fun.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.wrong_dimensions_fun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.wrong_dimensions_fun.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.wrong_dimensions_fun.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.wrong_dimensions_fun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.wrong_dimensions_fun.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.wrong_dimensions_fun', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wrong_dimensions_fun', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wrong_dimensions_fun(...)' code ##################

        
        # Call to array(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_242224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        # Getting the type of 'x' (line 137)
        x_242225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 25), 'x', False)
        int_242226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 28), 'int')
        # Applying the binary operator '**' (line 137)
        result_pow_242227 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 25), '**', x_242225, int_242226)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 24), list_242224, result_pow_242227)
        # Adding element type (line 137)
        
        # Call to tan(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'x' (line 137)
        x_242230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 38), 'x', False)
        # Processing the call keyword arguments (line 137)
        kwargs_242231 = {}
        # Getting the type of 'np' (line 137)
        np_242228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 31), 'np', False)
        # Obtaining the member 'tan' of a type (line 137)
        tan_242229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 31), np_242228, 'tan')
        # Calling tan(args, kwargs) (line 137)
        tan_call_result_242232 = invoke(stypy.reporting.localization.Localization(__file__, 137, 31), tan_242229, *[x_242230], **kwargs_242231)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 24), list_242224, tan_call_result_242232)
        # Adding element type (line 137)
        
        # Call to exp(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'x' (line 137)
        x_242235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 49), 'x', False)
        # Processing the call keyword arguments (line 137)
        kwargs_242236 = {}
        # Getting the type of 'np' (line 137)
        np_242233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), 'np', False)
        # Obtaining the member 'exp' of a type (line 137)
        exp_242234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 42), np_242233, 'exp')
        # Calling exp(args, kwargs) (line 137)
        exp_call_result_242237 = invoke(stypy.reporting.localization.Localization(__file__, 137, 42), exp_242234, *[x_242235], **kwargs_242236)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 24), list_242224, exp_call_result_242237)
        
        # Processing the call keyword arguments (line 137)
        kwargs_242238 = {}
        # Getting the type of 'np' (line 137)
        np_242222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 137)
        array_242223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 15), np_242222, 'array')
        # Calling array(args, kwargs) (line 137)
        array_call_result_242239 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), array_242223, *[list_242224], **kwargs_242238)
        
        # Assigning a type to the variable 'stypy_return_type' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'stypy_return_type', array_call_result_242239)
        
        # ################# End of 'wrong_dimensions_fun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wrong_dimensions_fun' in the type store
        # Getting the type of 'stypy_return_type' (line 136)
        stypy_return_type_242240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242240)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wrong_dimensions_fun'
        return stypy_return_type_242240


    @norecursion
    def jac_vector_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac_vector_scalar'
        module_type_store = module_type_store.open_function_context('jac_vector_scalar', 139, 4, False)
        # Assigning a type to the variable 'self' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.jac_vector_scalar.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.jac_vector_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.jac_vector_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.jac_vector_scalar.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.jac_vector_scalar')
        TestApproxDerivativesDense.jac_vector_scalar.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.jac_vector_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.jac_vector_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.jac_vector_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.jac_vector_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.jac_vector_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.jac_vector_scalar.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.jac_vector_scalar', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_vector_scalar', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_vector_scalar(...)' code ##################

        
        # Call to array(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_242243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        # Adding element type (line 140)
        
        # Obtaining the type of the subscript
        int_242244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 14), 'int')
        # Getting the type of 'x' (line 141)
        x_242245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___242246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), x_242245, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_242247 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), getitem___242246, int_242244)
        
        
        # Call to cos(...): (line 141)
        # Processing the call arguments (line 141)
        
        # Obtaining the type of the subscript
        int_242250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 28), 'int')
        # Getting the type of 'x' (line 141)
        x_242251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 26), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___242252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 26), x_242251, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_242253 = invoke(stypy.reporting.localization.Localization(__file__, 141, 26), getitem___242252, int_242250)
        
        
        # Obtaining the type of the subscript
        int_242254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 35), 'int')
        # Getting the type of 'x' (line 141)
        x_242255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___242256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 33), x_242255, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_242257 = invoke(stypy.reporting.localization.Localization(__file__, 141, 33), getitem___242256, int_242254)
        
        # Applying the binary operator '*' (line 141)
        result_mul_242258 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 26), '*', subscript_call_result_242253, subscript_call_result_242257)
        
        # Processing the call keyword arguments (line 141)
        kwargs_242259 = {}
        # Getting the type of 'np' (line 141)
        np_242248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 19), 'np', False)
        # Obtaining the member 'cos' of a type (line 141)
        cos_242249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 19), np_242248, 'cos')
        # Calling cos(args, kwargs) (line 141)
        cos_call_result_242260 = invoke(stypy.reporting.localization.Localization(__file__, 141, 19), cos_242249, *[result_mul_242258], **kwargs_242259)
        
        # Applying the binary operator '*' (line 141)
        result_mul_242261 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 12), '*', subscript_call_result_242247, cos_call_result_242260)
        
        
        # Call to log(...): (line 141)
        # Processing the call arguments (line 141)
        
        # Obtaining the type of the subscript
        int_242264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 50), 'int')
        # Getting the type of 'x' (line 141)
        x_242265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 48), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___242266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 48), x_242265, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_242267 = invoke(stypy.reporting.localization.Localization(__file__, 141, 48), getitem___242266, int_242264)
        
        # Processing the call keyword arguments (line 141)
        kwargs_242268 = {}
        # Getting the type of 'np' (line 141)
        np_242262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 41), 'np', False)
        # Obtaining the member 'log' of a type (line 141)
        log_242263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 41), np_242262, 'log')
        # Calling log(args, kwargs) (line 141)
        log_call_result_242269 = invoke(stypy.reporting.localization.Localization(__file__, 141, 41), log_242263, *[subscript_call_result_242267], **kwargs_242268)
        
        # Applying the binary operator '*' (line 141)
        result_mul_242270 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 39), '*', result_mul_242261, log_call_result_242269)
        
        
        # Call to sin(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Obtaining the type of the subscript
        int_242273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 21), 'int')
        # Getting the type of 'x' (line 142)
        x_242274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___242275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 19), x_242274, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_242276 = invoke(stypy.reporting.localization.Localization(__file__, 142, 19), getitem___242275, int_242273)
        
        
        # Obtaining the type of the subscript
        int_242277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 28), 'int')
        # Getting the type of 'x' (line 142)
        x_242278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___242279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 26), x_242278, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_242280 = invoke(stypy.reporting.localization.Localization(__file__, 142, 26), getitem___242279, int_242277)
        
        # Applying the binary operator '*' (line 142)
        result_mul_242281 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 19), '*', subscript_call_result_242276, subscript_call_result_242280)
        
        # Processing the call keyword arguments (line 142)
        kwargs_242282 = {}
        # Getting the type of 'np' (line 142)
        np_242271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'np', False)
        # Obtaining the member 'sin' of a type (line 142)
        sin_242272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), np_242271, 'sin')
        # Calling sin(args, kwargs) (line 142)
        sin_call_result_242283 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), sin_242272, *[result_mul_242281], **kwargs_242282)
        
        
        # Obtaining the type of the subscript
        int_242284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 36), 'int')
        # Getting the type of 'x' (line 142)
        x_242285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 34), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___242286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 34), x_242285, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_242287 = invoke(stypy.reporting.localization.Localization(__file__, 142, 34), getitem___242286, int_242284)
        
        # Applying the binary operator 'div' (line 142)
        result_div_242288 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 12), 'div', sin_call_result_242283, subscript_call_result_242287)
        
        # Applying the binary operator '+' (line 141)
        result_add_242289 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 12), '+', result_mul_242270, result_div_242288)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 24), list_242243, result_add_242289)
        # Adding element type (line 140)
        
        # Obtaining the type of the subscript
        int_242290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 14), 'int')
        # Getting the type of 'x' (line 143)
        x_242291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___242292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 12), x_242291, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_242293 = invoke(stypy.reporting.localization.Localization(__file__, 143, 12), getitem___242292, int_242290)
        
        
        # Call to cos(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Obtaining the type of the subscript
        int_242296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 28), 'int')
        # Getting the type of 'x' (line 143)
        x_242297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 26), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___242298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 26), x_242297, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_242299 = invoke(stypy.reporting.localization.Localization(__file__, 143, 26), getitem___242298, int_242296)
        
        
        # Obtaining the type of the subscript
        int_242300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 35), 'int')
        # Getting the type of 'x' (line 143)
        x_242301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 33), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___242302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 33), x_242301, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_242303 = invoke(stypy.reporting.localization.Localization(__file__, 143, 33), getitem___242302, int_242300)
        
        # Applying the binary operator '*' (line 143)
        result_mul_242304 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 26), '*', subscript_call_result_242299, subscript_call_result_242303)
        
        # Processing the call keyword arguments (line 143)
        kwargs_242305 = {}
        # Getting the type of 'np' (line 143)
        np_242294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 19), 'np', False)
        # Obtaining the member 'cos' of a type (line 143)
        cos_242295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 19), np_242294, 'cos')
        # Calling cos(args, kwargs) (line 143)
        cos_call_result_242306 = invoke(stypy.reporting.localization.Localization(__file__, 143, 19), cos_242295, *[result_mul_242304], **kwargs_242305)
        
        # Applying the binary operator '*' (line 143)
        result_mul_242307 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 12), '*', subscript_call_result_242293, cos_call_result_242306)
        
        
        # Call to log(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Obtaining the type of the subscript
        int_242310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 50), 'int')
        # Getting the type of 'x' (line 143)
        x_242311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 48), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___242312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 48), x_242311, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_242313 = invoke(stypy.reporting.localization.Localization(__file__, 143, 48), getitem___242312, int_242310)
        
        # Processing the call keyword arguments (line 143)
        kwargs_242314 = {}
        # Getting the type of 'np' (line 143)
        np_242308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 41), 'np', False)
        # Obtaining the member 'log' of a type (line 143)
        log_242309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 41), np_242308, 'log')
        # Calling log(args, kwargs) (line 143)
        log_call_result_242315 = invoke(stypy.reporting.localization.Localization(__file__, 143, 41), log_242309, *[subscript_call_result_242313], **kwargs_242314)
        
        # Applying the binary operator '*' (line 143)
        result_mul_242316 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 39), '*', result_mul_242307, log_call_result_242315)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 24), list_242243, result_mul_242316)
        
        # Processing the call keyword arguments (line 140)
        kwargs_242317 = {}
        # Getting the type of 'np' (line 140)
        np_242241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 140)
        array_242242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 15), np_242241, 'array')
        # Calling array(args, kwargs) (line 140)
        array_call_result_242318 = invoke(stypy.reporting.localization.Localization(__file__, 140, 15), array_242242, *[list_242243], **kwargs_242317)
        
        # Assigning a type to the variable 'stypy_return_type' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'stypy_return_type', array_call_result_242318)
        
        # ################# End of 'jac_vector_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_vector_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_242319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242319)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_vector_scalar'
        return stypy_return_type_242319


    @norecursion
    def fun_vector_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun_vector_vector'
        module_type_store = module_type_store.open_function_context('fun_vector_vector', 146, 4, False)
        # Assigning a type to the variable 'self' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.fun_vector_vector.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.fun_vector_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.fun_vector_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.fun_vector_vector.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.fun_vector_vector')
        TestApproxDerivativesDense.fun_vector_vector.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.fun_vector_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.fun_vector_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.fun_vector_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.fun_vector_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.fun_vector_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.fun_vector_vector.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.fun_vector_vector', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun_vector_vector', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun_vector_vector(...)' code ##################

        
        # Call to array(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_242322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        
        # Obtaining the type of the subscript
        int_242323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 14), 'int')
        # Getting the type of 'x' (line 148)
        x_242324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___242325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), x_242324, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_242326 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), getitem___242325, int_242323)
        
        
        # Call to sin(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Obtaining the type of the subscript
        int_242329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 28), 'int')
        # Getting the type of 'x' (line 148)
        x_242330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 26), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___242331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 26), x_242330, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_242332 = invoke(stypy.reporting.localization.Localization(__file__, 148, 26), getitem___242331, int_242329)
        
        # Processing the call keyword arguments (line 148)
        kwargs_242333 = {}
        # Getting the type of 'np' (line 148)
        np_242327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'np', False)
        # Obtaining the member 'sin' of a type (line 148)
        sin_242328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 19), np_242327, 'sin')
        # Calling sin(args, kwargs) (line 148)
        sin_call_result_242334 = invoke(stypy.reporting.localization.Localization(__file__, 148, 19), sin_242328, *[subscript_call_result_242332], **kwargs_242333)
        
        # Applying the binary operator '*' (line 148)
        result_mul_242335 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 12), '*', subscript_call_result_242326, sin_call_result_242334)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 24), list_242322, result_mul_242335)
        # Adding element type (line 147)
        
        # Obtaining the type of the subscript
        int_242336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 14), 'int')
        # Getting the type of 'x' (line 149)
        x_242337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___242338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), x_242337, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_242339 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), getitem___242338, int_242336)
        
        
        # Call to cos(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Obtaining the type of the subscript
        int_242342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 28), 'int')
        # Getting the type of 'x' (line 149)
        x_242343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___242344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 26), x_242343, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_242345 = invoke(stypy.reporting.localization.Localization(__file__, 149, 26), getitem___242344, int_242342)
        
        # Processing the call keyword arguments (line 149)
        kwargs_242346 = {}
        # Getting the type of 'np' (line 149)
        np_242340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 19), 'np', False)
        # Obtaining the member 'cos' of a type (line 149)
        cos_242341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 19), np_242340, 'cos')
        # Calling cos(args, kwargs) (line 149)
        cos_call_result_242347 = invoke(stypy.reporting.localization.Localization(__file__, 149, 19), cos_242341, *[subscript_call_result_242345], **kwargs_242346)
        
        # Applying the binary operator '*' (line 149)
        result_mul_242348 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 12), '*', subscript_call_result_242339, cos_call_result_242347)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 24), list_242322, result_mul_242348)
        # Adding element type (line 147)
        
        # Obtaining the type of the subscript
        int_242349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 14), 'int')
        # Getting the type of 'x' (line 150)
        x_242350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___242351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 12), x_242350, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_242352 = invoke(stypy.reporting.localization.Localization(__file__, 150, 12), getitem___242351, int_242349)
        
        int_242353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 20), 'int')
        # Applying the binary operator '**' (line 150)
        result_pow_242354 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), '**', subscript_call_result_242352, int_242353)
        
        
        # Obtaining the type of the subscript
        int_242355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 26), 'int')
        # Getting the type of 'x' (line 150)
        x_242356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___242357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 24), x_242356, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_242358 = invoke(stypy.reporting.localization.Localization(__file__, 150, 24), getitem___242357, int_242355)
        
        float_242359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 32), 'float')
        # Applying the binary operator '**' (line 150)
        result_pow_242360 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 24), '**', subscript_call_result_242358, float_242359)
        
        # Applying the binary operator '*' (line 150)
        result_mul_242361 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), '*', result_pow_242354, result_pow_242360)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 24), list_242322, result_mul_242361)
        
        # Processing the call keyword arguments (line 147)
        kwargs_242362 = {}
        # Getting the type of 'np' (line 147)
        np_242320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 147)
        array_242321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 15), np_242320, 'array')
        # Calling array(args, kwargs) (line 147)
        array_call_result_242363 = invoke(stypy.reporting.localization.Localization(__file__, 147, 15), array_242321, *[list_242322], **kwargs_242362)
        
        # Assigning a type to the variable 'stypy_return_type' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'stypy_return_type', array_call_result_242363)
        
        # ################# End of 'fun_vector_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_vector_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_242364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242364)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_vector_vector'
        return stypy_return_type_242364


    @norecursion
    def jac_vector_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac_vector_vector'
        module_type_store = module_type_store.open_function_context('jac_vector_vector', 153, 4, False)
        # Assigning a type to the variable 'self' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.jac_vector_vector.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.jac_vector_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.jac_vector_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.jac_vector_vector.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.jac_vector_vector')
        TestApproxDerivativesDense.jac_vector_vector.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.jac_vector_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.jac_vector_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.jac_vector_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.jac_vector_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.jac_vector_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.jac_vector_vector.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.jac_vector_vector', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_vector_vector', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_vector_vector(...)' code ##################

        
        # Call to array(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_242367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        # Adding element type (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 155)
        list_242368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 155)
        # Adding element type (line 155)
        
        # Call to sin(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Obtaining the type of the subscript
        int_242371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 22), 'int')
        # Getting the type of 'x' (line 155)
        x_242372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___242373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 20), x_242372, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_242374 = invoke(stypy.reporting.localization.Localization(__file__, 155, 20), getitem___242373, int_242371)
        
        # Processing the call keyword arguments (line 155)
        kwargs_242375 = {}
        # Getting the type of 'np' (line 155)
        np_242369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 13), 'np', False)
        # Obtaining the member 'sin' of a type (line 155)
        sin_242370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 13), np_242369, 'sin')
        # Calling sin(args, kwargs) (line 155)
        sin_call_result_242376 = invoke(stypy.reporting.localization.Localization(__file__, 155, 13), sin_242370, *[subscript_call_result_242374], **kwargs_242375)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 12), list_242368, sin_call_result_242376)
        # Adding element type (line 155)
        
        # Obtaining the type of the subscript
        int_242377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 29), 'int')
        # Getting the type of 'x' (line 155)
        x_242378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 27), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___242379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 27), x_242378, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_242380 = invoke(stypy.reporting.localization.Localization(__file__, 155, 27), getitem___242379, int_242377)
        
        
        # Call to cos(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Obtaining the type of the subscript
        int_242383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 43), 'int')
        # Getting the type of 'x' (line 155)
        x_242384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 41), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___242385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 41), x_242384, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_242386 = invoke(stypy.reporting.localization.Localization(__file__, 155, 41), getitem___242385, int_242383)
        
        # Processing the call keyword arguments (line 155)
        kwargs_242387 = {}
        # Getting the type of 'np' (line 155)
        np_242381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'np', False)
        # Obtaining the member 'cos' of a type (line 155)
        cos_242382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 34), np_242381, 'cos')
        # Calling cos(args, kwargs) (line 155)
        cos_call_result_242388 = invoke(stypy.reporting.localization.Localization(__file__, 155, 34), cos_242382, *[subscript_call_result_242386], **kwargs_242387)
        
        # Applying the binary operator '*' (line 155)
        result_mul_242389 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 27), '*', subscript_call_result_242380, cos_call_result_242388)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 12), list_242368, result_mul_242389)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 24), list_242367, list_242368)
        # Adding element type (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_242390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        # Adding element type (line 156)
        
        
        # Obtaining the type of the subscript
        int_242391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 16), 'int')
        # Getting the type of 'x' (line 156)
        x_242392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 14), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 156)
        getitem___242393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 14), x_242392, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 156)
        subscript_call_result_242394 = invoke(stypy.reporting.localization.Localization(__file__, 156, 14), getitem___242393, int_242391)
        
        # Applying the 'usub' unary operator (line 156)
        result___neg___242395 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 13), 'usub', subscript_call_result_242394)
        
        
        # Call to sin(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Obtaining the type of the subscript
        int_242398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 30), 'int')
        # Getting the type of 'x' (line 156)
        x_242399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 28), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 156)
        getitem___242400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 28), x_242399, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 156)
        subscript_call_result_242401 = invoke(stypy.reporting.localization.Localization(__file__, 156, 28), getitem___242400, int_242398)
        
        # Processing the call keyword arguments (line 156)
        kwargs_242402 = {}
        # Getting the type of 'np' (line 156)
        np_242396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'np', False)
        # Obtaining the member 'sin' of a type (line 156)
        sin_242397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 21), np_242396, 'sin')
        # Calling sin(args, kwargs) (line 156)
        sin_call_result_242403 = invoke(stypy.reporting.localization.Localization(__file__, 156, 21), sin_242397, *[subscript_call_result_242401], **kwargs_242402)
        
        # Applying the binary operator '*' (line 156)
        result_mul_242404 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 13), '*', result___neg___242395, sin_call_result_242403)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_242390, result_mul_242404)
        # Adding element type (line 156)
        
        # Call to cos(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Obtaining the type of the subscript
        int_242407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 44), 'int')
        # Getting the type of 'x' (line 156)
        x_242408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 42), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 156)
        getitem___242409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 42), x_242408, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 156)
        subscript_call_result_242410 = invoke(stypy.reporting.localization.Localization(__file__, 156, 42), getitem___242409, int_242407)
        
        # Processing the call keyword arguments (line 156)
        kwargs_242411 = {}
        # Getting the type of 'np' (line 156)
        np_242405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 35), 'np', False)
        # Obtaining the member 'cos' of a type (line 156)
        cos_242406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 35), np_242405, 'cos')
        # Calling cos(args, kwargs) (line 156)
        cos_call_result_242412 = invoke(stypy.reporting.localization.Localization(__file__, 156, 35), cos_242406, *[subscript_call_result_242410], **kwargs_242411)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_242390, cos_call_result_242412)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 24), list_242367, list_242390)
        # Adding element type (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 157)
        list_242413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 157)
        # Adding element type (line 157)
        int_242414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 13), 'int')
        
        # Obtaining the type of the subscript
        int_242415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 19), 'int')
        # Getting the type of 'x' (line 157)
        x_242416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 17), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___242417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 17), x_242416, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_242418 = invoke(stypy.reporting.localization.Localization(__file__, 157, 17), getitem___242417, int_242415)
        
        int_242419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 25), 'int')
        # Applying the binary operator '**' (line 157)
        result_pow_242420 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 17), '**', subscript_call_result_242418, int_242419)
        
        # Applying the binary operator '*' (line 157)
        result_mul_242421 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 13), '*', int_242414, result_pow_242420)
        
        
        # Obtaining the type of the subscript
        int_242422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 31), 'int')
        # Getting the type of 'x' (line 157)
        x_242423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 29), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___242424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 29), x_242423, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_242425 = invoke(stypy.reporting.localization.Localization(__file__, 157, 29), getitem___242424, int_242422)
        
        float_242426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 37), 'float')
        # Applying the binary operator '**' (line 157)
        result_pow_242427 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 29), '**', subscript_call_result_242425, float_242426)
        
        # Applying the binary operator '*' (line 157)
        result_mul_242428 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 27), '*', result_mul_242421, result_pow_242427)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 12), list_242413, result_mul_242428)
        # Adding element type (line 157)
        float_242429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 43), 'float')
        
        # Obtaining the type of the subscript
        int_242430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 52), 'int')
        # Getting the type of 'x' (line 157)
        x_242431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 50), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___242432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 50), x_242431, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_242433 = invoke(stypy.reporting.localization.Localization(__file__, 157, 50), getitem___242432, int_242430)
        
        int_242434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 58), 'int')
        # Applying the binary operator '**' (line 157)
        result_pow_242435 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 50), '**', subscript_call_result_242433, int_242434)
        
        # Applying the binary operator '*' (line 157)
        result_mul_242436 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 43), '*', float_242429, result_pow_242435)
        
        
        # Obtaining the type of the subscript
        int_242437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 64), 'int')
        # Getting the type of 'x' (line 157)
        x_242438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 62), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___242439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 62), x_242438, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_242440 = invoke(stypy.reporting.localization.Localization(__file__, 157, 62), getitem___242439, int_242437)
        
        float_242441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 70), 'float')
        # Applying the binary operator '**' (line 157)
        result_pow_242442 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 62), '**', subscript_call_result_242440, float_242441)
        
        # Applying the binary operator '*' (line 157)
        result_mul_242443 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 60), '*', result_mul_242436, result_pow_242442)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 12), list_242413, result_mul_242443)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 24), list_242367, list_242413)
        
        # Processing the call keyword arguments (line 154)
        kwargs_242444 = {}
        # Getting the type of 'np' (line 154)
        np_242365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 154)
        array_242366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 15), np_242365, 'array')
        # Calling array(args, kwargs) (line 154)
        array_call_result_242445 = invoke(stypy.reporting.localization.Localization(__file__, 154, 15), array_242366, *[list_242367], **kwargs_242444)
        
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', array_call_result_242445)
        
        # ################# End of 'jac_vector_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_vector_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_242446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242446)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_vector_vector'
        return stypy_return_type_242446


    @norecursion
    def fun_parametrized(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_242447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 41), 'float')
        defaults = [float_242447]
        # Create a new context for function 'fun_parametrized'
        module_type_store = module_type_store.open_function_context('fun_parametrized', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.fun_parametrized.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.fun_parametrized.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.fun_parametrized.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.fun_parametrized.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.fun_parametrized')
        TestApproxDerivativesDense.fun_parametrized.__dict__.__setitem__('stypy_param_names_list', ['x', 'c0', 'c1'])
        TestApproxDerivativesDense.fun_parametrized.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.fun_parametrized.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.fun_parametrized.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.fun_parametrized.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.fun_parametrized.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.fun_parametrized.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.fun_parametrized', ['x', 'c0', 'c1'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun_parametrized', localization, ['x', 'c0', 'c1'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun_parametrized(...)' code ##################

        
        # Call to array(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Obtaining an instance of the builtin type 'list' (line 161)
        list_242450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 161)
        # Adding element type (line 161)
        
        # Call to exp(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'c0' (line 161)
        c0_242453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 32), 'c0', False)
        
        # Obtaining the type of the subscript
        int_242454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 39), 'int')
        # Getting the type of 'x' (line 161)
        x_242455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 161)
        getitem___242456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 37), x_242455, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 161)
        subscript_call_result_242457 = invoke(stypy.reporting.localization.Localization(__file__, 161, 37), getitem___242456, int_242454)
        
        # Applying the binary operator '*' (line 161)
        result_mul_242458 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 32), '*', c0_242453, subscript_call_result_242457)
        
        # Processing the call keyword arguments (line 161)
        kwargs_242459 = {}
        # Getting the type of 'np' (line 161)
        np_242451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 25), 'np', False)
        # Obtaining the member 'exp' of a type (line 161)
        exp_242452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 25), np_242451, 'exp')
        # Calling exp(args, kwargs) (line 161)
        exp_call_result_242460 = invoke(stypy.reporting.localization.Localization(__file__, 161, 25), exp_242452, *[result_mul_242458], **kwargs_242459)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 24), list_242450, exp_call_result_242460)
        # Adding element type (line 161)
        
        # Call to exp(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'c1' (line 161)
        c1_242463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 51), 'c1', False)
        
        # Obtaining the type of the subscript
        int_242464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 58), 'int')
        # Getting the type of 'x' (line 161)
        x_242465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 56), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 161)
        getitem___242466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 56), x_242465, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 161)
        subscript_call_result_242467 = invoke(stypy.reporting.localization.Localization(__file__, 161, 56), getitem___242466, int_242464)
        
        # Applying the binary operator '*' (line 161)
        result_mul_242468 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 51), '*', c1_242463, subscript_call_result_242467)
        
        # Processing the call keyword arguments (line 161)
        kwargs_242469 = {}
        # Getting the type of 'np' (line 161)
        np_242461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 44), 'np', False)
        # Obtaining the member 'exp' of a type (line 161)
        exp_242462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 44), np_242461, 'exp')
        # Calling exp(args, kwargs) (line 161)
        exp_call_result_242470 = invoke(stypy.reporting.localization.Localization(__file__, 161, 44), exp_242462, *[result_mul_242468], **kwargs_242469)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 24), list_242450, exp_call_result_242470)
        
        # Processing the call keyword arguments (line 161)
        kwargs_242471 = {}
        # Getting the type of 'np' (line 161)
        np_242448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 161)
        array_242449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 15), np_242448, 'array')
        # Calling array(args, kwargs) (line 161)
        array_call_result_242472 = invoke(stypy.reporting.localization.Localization(__file__, 161, 15), array_242449, *[list_242450], **kwargs_242471)
        
        # Assigning a type to the variable 'stypy_return_type' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'stypy_return_type', array_call_result_242472)
        
        # ################# End of 'fun_parametrized(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_parametrized' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_242473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242473)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_parametrized'
        return stypy_return_type_242473


    @norecursion
    def jac_parametrized(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_242474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 41), 'float')
        defaults = [float_242474]
        # Create a new context for function 'jac_parametrized'
        module_type_store = module_type_store.open_function_context('jac_parametrized', 163, 4, False)
        # Assigning a type to the variable 'self' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.jac_parametrized.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.jac_parametrized.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.jac_parametrized.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.jac_parametrized.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.jac_parametrized')
        TestApproxDerivativesDense.jac_parametrized.__dict__.__setitem__('stypy_param_names_list', ['x', 'c0', 'c1'])
        TestApproxDerivativesDense.jac_parametrized.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.jac_parametrized.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.jac_parametrized.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.jac_parametrized.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.jac_parametrized.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.jac_parametrized.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.jac_parametrized', ['x', 'c0', 'c1'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_parametrized', localization, ['x', 'c0', 'c1'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_parametrized(...)' code ##################

        
        # Call to array(...): (line 164)
        # Processing the call arguments (line 164)
        
        # Obtaining an instance of the builtin type 'list' (line 164)
        list_242477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 164)
        # Adding element type (line 164)
        
        # Obtaining an instance of the builtin type 'list' (line 165)
        list_242478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 165)
        # Adding element type (line 165)
        # Getting the type of 'c0' (line 165)
        c0_242479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 13), 'c0', False)
        
        # Call to exp(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'c0' (line 165)
        c0_242482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'c0', False)
        
        # Obtaining the type of the subscript
        int_242483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 32), 'int')
        # Getting the type of 'x' (line 165)
        x_242484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 30), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___242485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 30), x_242484, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_242486 = invoke(stypy.reporting.localization.Localization(__file__, 165, 30), getitem___242485, int_242483)
        
        # Applying the binary operator '*' (line 165)
        result_mul_242487 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 25), '*', c0_242482, subscript_call_result_242486)
        
        # Processing the call keyword arguments (line 165)
        kwargs_242488 = {}
        # Getting the type of 'np' (line 165)
        np_242480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), 'np', False)
        # Obtaining the member 'exp' of a type (line 165)
        exp_242481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 18), np_242480, 'exp')
        # Calling exp(args, kwargs) (line 165)
        exp_call_result_242489 = invoke(stypy.reporting.localization.Localization(__file__, 165, 18), exp_242481, *[result_mul_242487], **kwargs_242488)
        
        # Applying the binary operator '*' (line 165)
        result_mul_242490 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 13), '*', c0_242479, exp_call_result_242489)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 12), list_242478, result_mul_242490)
        # Adding element type (line 165)
        int_242491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 12), list_242478, int_242491)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 24), list_242477, list_242478)
        # Adding element type (line 164)
        
        # Obtaining an instance of the builtin type 'list' (line 166)
        list_242492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 166)
        # Adding element type (line 166)
        int_242493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 12), list_242492, int_242493)
        # Adding element type (line 166)
        # Getting the type of 'c1' (line 166)
        c1_242494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'c1', False)
        
        # Call to exp(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'c1' (line 166)
        c1_242497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'c1', False)
        
        # Obtaining the type of the subscript
        int_242498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 35), 'int')
        # Getting the type of 'x' (line 166)
        x_242499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 33), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___242500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 33), x_242499, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_242501 = invoke(stypy.reporting.localization.Localization(__file__, 166, 33), getitem___242500, int_242498)
        
        # Applying the binary operator '*' (line 166)
        result_mul_242502 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 28), '*', c1_242497, subscript_call_result_242501)
        
        # Processing the call keyword arguments (line 166)
        kwargs_242503 = {}
        # Getting the type of 'np' (line 166)
        np_242495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'np', False)
        # Obtaining the member 'exp' of a type (line 166)
        exp_242496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 21), np_242495, 'exp')
        # Calling exp(args, kwargs) (line 166)
        exp_call_result_242504 = invoke(stypy.reporting.localization.Localization(__file__, 166, 21), exp_242496, *[result_mul_242502], **kwargs_242503)
        
        # Applying the binary operator '*' (line 166)
        result_mul_242505 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 16), '*', c1_242494, exp_call_result_242504)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 12), list_242492, result_mul_242505)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 24), list_242477, list_242492)
        
        # Processing the call keyword arguments (line 164)
        kwargs_242506 = {}
        # Getting the type of 'np' (line 164)
        np_242475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 164)
        array_242476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 15), np_242475, 'array')
        # Calling array(args, kwargs) (line 164)
        array_call_result_242507 = invoke(stypy.reporting.localization.Localization(__file__, 164, 15), array_242476, *[list_242477], **kwargs_242506)
        
        # Assigning a type to the variable 'stypy_return_type' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'stypy_return_type', array_call_result_242507)
        
        # ################# End of 'jac_parametrized(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_parametrized' in the type store
        # Getting the type of 'stypy_return_type' (line 163)
        stypy_return_type_242508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242508)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_parametrized'
        return stypy_return_type_242508


    @norecursion
    def fun_with_nan(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun_with_nan'
        module_type_store = module_type_store.open_function_context('fun_with_nan', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.fun_with_nan.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.fun_with_nan.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.fun_with_nan.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.fun_with_nan.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.fun_with_nan')
        TestApproxDerivativesDense.fun_with_nan.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.fun_with_nan.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.fun_with_nan.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.fun_with_nan.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.fun_with_nan.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.fun_with_nan.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.fun_with_nan.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.fun_with_nan', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun_with_nan', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun_with_nan(...)' code ##################

        
        
        
        # Call to abs(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'x' (line 170)
        x_242511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'x', False)
        # Processing the call keyword arguments (line 170)
        kwargs_242512 = {}
        # Getting the type of 'np' (line 170)
        np_242509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'np', False)
        # Obtaining the member 'abs' of a type (line 170)
        abs_242510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 20), np_242509, 'abs')
        # Calling abs(args, kwargs) (line 170)
        abs_call_result_242513 = invoke(stypy.reporting.localization.Localization(__file__, 170, 20), abs_242510, *[x_242511], **kwargs_242512)
        
        float_242514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 33), 'float')
        # Applying the binary operator '<=' (line 170)
        result_le_242515 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 20), '<=', abs_call_result_242513, float_242514)
        
        # Testing the type of an if expression (line 170)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 15), result_le_242515)
        # SSA begins for if expression (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'x' (line 170)
        x_242516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'x')
        # SSA branch for the else part of an if expression (line 170)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'np' (line 170)
        np_242517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 43), 'np')
        # Obtaining the member 'nan' of a type (line 170)
        nan_242518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 43), np_242517, 'nan')
        # SSA join for if expression (line 170)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_242519 = union_type.UnionType.add(x_242516, nan_242518)
        
        # Assigning a type to the variable 'stypy_return_type' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'stypy_return_type', if_exp_242519)
        
        # ################# End of 'fun_with_nan(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_with_nan' in the type store
        # Getting the type of 'stypy_return_type' (line 169)
        stypy_return_type_242520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242520)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_with_nan'
        return stypy_return_type_242520


    @norecursion
    def jac_with_nan(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac_with_nan'
        module_type_store = module_type_store.open_function_context('jac_with_nan', 172, 4, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.jac_with_nan.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.jac_with_nan.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.jac_with_nan.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.jac_with_nan.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.jac_with_nan')
        TestApproxDerivativesDense.jac_with_nan.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.jac_with_nan.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.jac_with_nan.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.jac_with_nan.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.jac_with_nan.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.jac_with_nan.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.jac_with_nan.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.jac_with_nan', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_with_nan', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_with_nan(...)' code ##################

        
        
        
        # Call to abs(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'x' (line 173)
        x_242523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 29), 'x', False)
        # Processing the call keyword arguments (line 173)
        kwargs_242524 = {}
        # Getting the type of 'np' (line 173)
        np_242521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'np', False)
        # Obtaining the member 'abs' of a type (line 173)
        abs_242522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 22), np_242521, 'abs')
        # Calling abs(args, kwargs) (line 173)
        abs_call_result_242525 = invoke(stypy.reporting.localization.Localization(__file__, 173, 22), abs_242522, *[x_242523], **kwargs_242524)
        
        float_242526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 35), 'float')
        # Applying the binary operator '<=' (line 173)
        result_le_242527 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 22), '<=', abs_call_result_242525, float_242526)
        
        # Testing the type of an if expression (line 173)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 15), result_le_242527)
        # SSA begins for if expression (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        float_242528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 15), 'float')
        # SSA branch for the else part of an if expression (line 173)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'np' (line 173)
        np_242529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 45), 'np')
        # Obtaining the member 'nan' of a type (line 173)
        nan_242530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 45), np_242529, 'nan')
        # SSA join for if expression (line 173)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_242531 = union_type.UnionType.add(float_242528, nan_242530)
        
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', if_exp_242531)
        
        # ################# End of 'jac_with_nan(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_with_nan' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_242532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242532)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_with_nan'
        return stypy_return_type_242532


    @norecursion
    def fun_zero_jacobian(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun_zero_jacobian'
        module_type_store = module_type_store.open_function_context('fun_zero_jacobian', 175, 4, False)
        # Assigning a type to the variable 'self' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.fun_zero_jacobian.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.fun_zero_jacobian.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.fun_zero_jacobian.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.fun_zero_jacobian.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.fun_zero_jacobian')
        TestApproxDerivativesDense.fun_zero_jacobian.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.fun_zero_jacobian.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.fun_zero_jacobian.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.fun_zero_jacobian.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.fun_zero_jacobian.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.fun_zero_jacobian.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.fun_zero_jacobian.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.fun_zero_jacobian', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun_zero_jacobian', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun_zero_jacobian(...)' code ##################

        
        # Call to array(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_242535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        
        # Obtaining the type of the subscript
        int_242536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 27), 'int')
        # Getting the type of 'x' (line 176)
        x_242537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___242538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 25), x_242537, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_242539 = invoke(stypy.reporting.localization.Localization(__file__, 176, 25), getitem___242538, int_242536)
        
        
        # Obtaining the type of the subscript
        int_242540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 34), 'int')
        # Getting the type of 'x' (line 176)
        x_242541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___242542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 32), x_242541, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_242543 = invoke(stypy.reporting.localization.Localization(__file__, 176, 32), getitem___242542, int_242540)
        
        # Applying the binary operator '*' (line 176)
        result_mul_242544 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 25), '*', subscript_call_result_242539, subscript_call_result_242543)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 24), list_242535, result_mul_242544)
        # Adding element type (line 176)
        
        # Call to cos(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Obtaining the type of the subscript
        int_242547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 47), 'int')
        # Getting the type of 'x' (line 176)
        x_242548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 45), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___242549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 45), x_242548, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_242550 = invoke(stypy.reporting.localization.Localization(__file__, 176, 45), getitem___242549, int_242547)
        
        
        # Obtaining the type of the subscript
        int_242551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 54), 'int')
        # Getting the type of 'x' (line 176)
        x_242552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 52), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___242553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 52), x_242552, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_242554 = invoke(stypy.reporting.localization.Localization(__file__, 176, 52), getitem___242553, int_242551)
        
        # Applying the binary operator '*' (line 176)
        result_mul_242555 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 45), '*', subscript_call_result_242550, subscript_call_result_242554)
        
        # Processing the call keyword arguments (line 176)
        kwargs_242556 = {}
        # Getting the type of 'np' (line 176)
        np_242545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 38), 'np', False)
        # Obtaining the member 'cos' of a type (line 176)
        cos_242546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 38), np_242545, 'cos')
        # Calling cos(args, kwargs) (line 176)
        cos_call_result_242557 = invoke(stypy.reporting.localization.Localization(__file__, 176, 38), cos_242546, *[result_mul_242555], **kwargs_242556)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 24), list_242535, cos_call_result_242557)
        
        # Processing the call keyword arguments (line 176)
        kwargs_242558 = {}
        # Getting the type of 'np' (line 176)
        np_242533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 176)
        array_242534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 15), np_242533, 'array')
        # Calling array(args, kwargs) (line 176)
        array_call_result_242559 = invoke(stypy.reporting.localization.Localization(__file__, 176, 15), array_242534, *[list_242535], **kwargs_242558)
        
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type', array_call_result_242559)
        
        # ################# End of 'fun_zero_jacobian(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_zero_jacobian' in the type store
        # Getting the type of 'stypy_return_type' (line 175)
        stypy_return_type_242560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242560)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_zero_jacobian'
        return stypy_return_type_242560


    @norecursion
    def jac_zero_jacobian(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac_zero_jacobian'
        module_type_store = module_type_store.open_function_context('jac_zero_jacobian', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.jac_zero_jacobian.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.jac_zero_jacobian.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.jac_zero_jacobian.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.jac_zero_jacobian.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.jac_zero_jacobian')
        TestApproxDerivativesDense.jac_zero_jacobian.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.jac_zero_jacobian.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.jac_zero_jacobian.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.jac_zero_jacobian.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.jac_zero_jacobian.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.jac_zero_jacobian.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.jac_zero_jacobian.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.jac_zero_jacobian', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_zero_jacobian', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_zero_jacobian(...)' code ##################

        
        # Call to array(...): (line 179)
        # Processing the call arguments (line 179)
        
        # Obtaining an instance of the builtin type 'list' (line 179)
        list_242563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 179)
        # Adding element type (line 179)
        
        # Obtaining an instance of the builtin type 'list' (line 180)
        list_242564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 180)
        # Adding element type (line 180)
        
        # Obtaining the type of the subscript
        int_242565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 15), 'int')
        # Getting the type of 'x' (line 180)
        x_242566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 13), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___242567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 13), x_242566, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_242568 = invoke(stypy.reporting.localization.Localization(__file__, 180, 13), getitem___242567, int_242565)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 12), list_242564, subscript_call_result_242568)
        # Adding element type (line 180)
        
        # Obtaining the type of the subscript
        int_242569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 21), 'int')
        # Getting the type of 'x' (line 180)
        x_242570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___242571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 19), x_242570, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_242572 = invoke(stypy.reporting.localization.Localization(__file__, 180, 19), getitem___242571, int_242569)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 12), list_242564, subscript_call_result_242572)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 24), list_242563, list_242564)
        # Adding element type (line 179)
        
        # Obtaining an instance of the builtin type 'list' (line 181)
        list_242573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 181)
        # Adding element type (line 181)
        
        
        # Obtaining the type of the subscript
        int_242574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 16), 'int')
        # Getting the type of 'x' (line 181)
        x_242575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 14), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___242576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 14), x_242575, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_242577 = invoke(stypy.reporting.localization.Localization(__file__, 181, 14), getitem___242576, int_242574)
        
        # Applying the 'usub' unary operator (line 181)
        result___neg___242578 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 13), 'usub', subscript_call_result_242577)
        
        
        # Call to sin(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Obtaining the type of the subscript
        int_242581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 30), 'int')
        # Getting the type of 'x' (line 181)
        x_242582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___242583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 28), x_242582, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_242584 = invoke(stypy.reporting.localization.Localization(__file__, 181, 28), getitem___242583, int_242581)
        
        
        # Obtaining the type of the subscript
        int_242585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 37), 'int')
        # Getting the type of 'x' (line 181)
        x_242586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 35), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___242587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 35), x_242586, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_242588 = invoke(stypy.reporting.localization.Localization(__file__, 181, 35), getitem___242587, int_242585)
        
        # Applying the binary operator '*' (line 181)
        result_mul_242589 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 28), '*', subscript_call_result_242584, subscript_call_result_242588)
        
        # Processing the call keyword arguments (line 181)
        kwargs_242590 = {}
        # Getting the type of 'np' (line 181)
        np_242579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 21), 'np', False)
        # Obtaining the member 'sin' of a type (line 181)
        sin_242580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 21), np_242579, 'sin')
        # Calling sin(args, kwargs) (line 181)
        sin_call_result_242591 = invoke(stypy.reporting.localization.Localization(__file__, 181, 21), sin_242580, *[result_mul_242589], **kwargs_242590)
        
        # Applying the binary operator '*' (line 181)
        result_mul_242592 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 13), '*', result___neg___242578, sin_call_result_242591)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 12), list_242573, result_mul_242592)
        # Adding element type (line 181)
        
        
        # Obtaining the type of the subscript
        int_242593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 45), 'int')
        # Getting the type of 'x' (line 181)
        x_242594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 43), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___242595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 43), x_242594, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_242596 = invoke(stypy.reporting.localization.Localization(__file__, 181, 43), getitem___242595, int_242593)
        
        # Applying the 'usub' unary operator (line 181)
        result___neg___242597 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 42), 'usub', subscript_call_result_242596)
        
        
        # Call to sin(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Obtaining the type of the subscript
        int_242600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 59), 'int')
        # Getting the type of 'x' (line 181)
        x_242601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 57), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___242602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 57), x_242601, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_242603 = invoke(stypy.reporting.localization.Localization(__file__, 181, 57), getitem___242602, int_242600)
        
        
        # Obtaining the type of the subscript
        int_242604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 66), 'int')
        # Getting the type of 'x' (line 181)
        x_242605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 64), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___242606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 64), x_242605, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_242607 = invoke(stypy.reporting.localization.Localization(__file__, 181, 64), getitem___242606, int_242604)
        
        # Applying the binary operator '*' (line 181)
        result_mul_242608 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 57), '*', subscript_call_result_242603, subscript_call_result_242607)
        
        # Processing the call keyword arguments (line 181)
        kwargs_242609 = {}
        # Getting the type of 'np' (line 181)
        np_242598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 50), 'np', False)
        # Obtaining the member 'sin' of a type (line 181)
        sin_242599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 50), np_242598, 'sin')
        # Calling sin(args, kwargs) (line 181)
        sin_call_result_242610 = invoke(stypy.reporting.localization.Localization(__file__, 181, 50), sin_242599, *[result_mul_242608], **kwargs_242609)
        
        # Applying the binary operator '*' (line 181)
        result_mul_242611 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 42), '*', result___neg___242597, sin_call_result_242610)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 12), list_242573, result_mul_242611)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 24), list_242563, list_242573)
        
        # Processing the call keyword arguments (line 179)
        kwargs_242612 = {}
        # Getting the type of 'np' (line 179)
        np_242561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 179)
        array_242562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 15), np_242561, 'array')
        # Calling array(args, kwargs) (line 179)
        array_call_result_242613 = invoke(stypy.reporting.localization.Localization(__file__, 179, 15), array_242562, *[list_242563], **kwargs_242612)
        
        # Assigning a type to the variable 'stypy_return_type' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'stypy_return_type', array_call_result_242613)
        
        # ################# End of 'jac_zero_jacobian(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_zero_jacobian' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_242614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242614)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_zero_jacobian'
        return stypy_return_type_242614


    @norecursion
    def fun_non_numpy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun_non_numpy'
        module_type_store = module_type_store.open_function_context('fun_non_numpy', 184, 4, False)
        # Assigning a type to the variable 'self' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.fun_non_numpy.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.fun_non_numpy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.fun_non_numpy.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.fun_non_numpy.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.fun_non_numpy')
        TestApproxDerivativesDense.fun_non_numpy.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.fun_non_numpy.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.fun_non_numpy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.fun_non_numpy.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.fun_non_numpy.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.fun_non_numpy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.fun_non_numpy.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.fun_non_numpy', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun_non_numpy', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun_non_numpy(...)' code ##################

        
        # Call to exp(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'x' (line 185)
        x_242617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'x', False)
        # Processing the call keyword arguments (line 185)
        kwargs_242618 = {}
        # Getting the type of 'math' (line 185)
        math_242615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'math', False)
        # Obtaining the member 'exp' of a type (line 185)
        exp_242616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 15), math_242615, 'exp')
        # Calling exp(args, kwargs) (line 185)
        exp_call_result_242619 = invoke(stypy.reporting.localization.Localization(__file__, 185, 15), exp_242616, *[x_242617], **kwargs_242618)
        
        # Assigning a type to the variable 'stypy_return_type' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type', exp_call_result_242619)
        
        # ################# End of 'fun_non_numpy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_non_numpy' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_242620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242620)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_non_numpy'
        return stypy_return_type_242620


    @norecursion
    def jac_non_numpy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac_non_numpy'
        module_type_store = module_type_store.open_function_context('jac_non_numpy', 187, 4, False)
        # Assigning a type to the variable 'self' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.jac_non_numpy.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.jac_non_numpy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.jac_non_numpy.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.jac_non_numpy.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.jac_non_numpy')
        TestApproxDerivativesDense.jac_non_numpy.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativesDense.jac_non_numpy.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.jac_non_numpy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.jac_non_numpy.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.jac_non_numpy.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.jac_non_numpy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.jac_non_numpy.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.jac_non_numpy', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_non_numpy', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_non_numpy(...)' code ##################

        
        # Call to exp(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'x' (line 188)
        x_242623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'x', False)
        # Processing the call keyword arguments (line 188)
        kwargs_242624 = {}
        # Getting the type of 'math' (line 188)
        math_242621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'math', False)
        # Obtaining the member 'exp' of a type (line 188)
        exp_242622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 15), math_242621, 'exp')
        # Calling exp(args, kwargs) (line 188)
        exp_call_result_242625 = invoke(stypy.reporting.localization.Localization(__file__, 188, 15), exp_242622, *[x_242623], **kwargs_242624)
        
        # Assigning a type to the variable 'stypy_return_type' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'stypy_return_type', exp_call_result_242625)
        
        # ################# End of 'jac_non_numpy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_non_numpy' in the type store
        # Getting the type of 'stypy_return_type' (line 187)
        stypy_return_type_242626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242626)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_non_numpy'
        return stypy_return_type_242626


    @norecursion
    def test_scalar_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalar_scalar'
        module_type_store = module_type_store.open_function_context('test_scalar_scalar', 190, 4, False)
        # Assigning a type to the variable 'self' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.test_scalar_scalar.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.test_scalar_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.test_scalar_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.test_scalar_scalar.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.test_scalar_scalar')
        TestApproxDerivativesDense.test_scalar_scalar.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativesDense.test_scalar_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.test_scalar_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.test_scalar_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.test_scalar_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.test_scalar_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.test_scalar_scalar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.test_scalar_scalar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalar_scalar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalar_scalar(...)' code ##################

        
        # Assigning a Num to a Name (line 191):
        
        # Assigning a Num to a Name (line 191):
        float_242627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 13), 'float')
        # Assigning a type to the variable 'x0' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'x0', float_242627)
        
        # Assigning a Call to a Name (line 192):
        
        # Assigning a Call to a Name (line 192):
        
        # Call to approx_derivative(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'self' (line 192)
        self_242629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 39), 'self', False)
        # Obtaining the member 'fun_scalar_scalar' of a type (line 192)
        fun_scalar_scalar_242630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 39), self_242629, 'fun_scalar_scalar')
        # Getting the type of 'x0' (line 192)
        x0_242631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 63), 'x0', False)
        # Processing the call keyword arguments (line 192)
        str_242632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 46), 'str', '2-point')
        keyword_242633 = str_242632
        kwargs_242634 = {'method': keyword_242633}
        # Getting the type of 'approx_derivative' (line 192)
        approx_derivative_242628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 192)
        approx_derivative_call_result_242635 = invoke(stypy.reporting.localization.Localization(__file__, 192, 21), approx_derivative_242628, *[fun_scalar_scalar_242630, x0_242631], **kwargs_242634)
        
        # Assigning a type to the variable 'jac_diff_2' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'jac_diff_2', approx_derivative_call_result_242635)
        
        # Assigning a Call to a Name (line 194):
        
        # Assigning a Call to a Name (line 194):
        
        # Call to approx_derivative(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'self' (line 194)
        self_242637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 39), 'self', False)
        # Obtaining the member 'fun_scalar_scalar' of a type (line 194)
        fun_scalar_scalar_242638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 39), self_242637, 'fun_scalar_scalar')
        # Getting the type of 'x0' (line 194)
        x0_242639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 63), 'x0', False)
        # Processing the call keyword arguments (line 194)
        kwargs_242640 = {}
        # Getting the type of 'approx_derivative' (line 194)
        approx_derivative_242636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 194)
        approx_derivative_call_result_242641 = invoke(stypy.reporting.localization.Localization(__file__, 194, 21), approx_derivative_242636, *[fun_scalar_scalar_242638, x0_242639], **kwargs_242640)
        
        # Assigning a type to the variable 'jac_diff_3' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'jac_diff_3', approx_derivative_call_result_242641)
        
        # Assigning a Call to a Name (line 195):
        
        # Assigning a Call to a Name (line 195):
        
        # Call to approx_derivative(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'self' (line 195)
        self_242643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 39), 'self', False)
        # Obtaining the member 'fun_scalar_scalar' of a type (line 195)
        fun_scalar_scalar_242644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 39), self_242643, 'fun_scalar_scalar')
        # Getting the type of 'x0' (line 195)
        x0_242645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 63), 'x0', False)
        # Processing the call keyword arguments (line 195)
        str_242646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 46), 'str', 'cs')
        keyword_242647 = str_242646
        kwargs_242648 = {'method': keyword_242647}
        # Getting the type of 'approx_derivative' (line 195)
        approx_derivative_242642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 195)
        approx_derivative_call_result_242649 = invoke(stypy.reporting.localization.Localization(__file__, 195, 21), approx_derivative_242642, *[fun_scalar_scalar_242644, x0_242645], **kwargs_242648)
        
        # Assigning a type to the variable 'jac_diff_4' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'jac_diff_4', approx_derivative_call_result_242649)
        
        # Assigning a Call to a Name (line 197):
        
        # Assigning a Call to a Name (line 197):
        
        # Call to jac_scalar_scalar(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'x0' (line 197)
        x0_242652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 42), 'x0', False)
        # Processing the call keyword arguments (line 197)
        kwargs_242653 = {}
        # Getting the type of 'self' (line 197)
        self_242650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'self', False)
        # Obtaining the member 'jac_scalar_scalar' of a type (line 197)
        jac_scalar_scalar_242651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 19), self_242650, 'jac_scalar_scalar')
        # Calling jac_scalar_scalar(args, kwargs) (line 197)
        jac_scalar_scalar_call_result_242654 = invoke(stypy.reporting.localization.Localization(__file__, 197, 19), jac_scalar_scalar_242651, *[x0_242652], **kwargs_242653)
        
        # Assigning a type to the variable 'jac_true' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'jac_true', jac_scalar_scalar_call_result_242654)
        
        # Call to assert_allclose(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'jac_diff_2' (line 198)
        jac_diff_2_242656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 24), 'jac_diff_2', False)
        # Getting the type of 'jac_true' (line 198)
        jac_true_242657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 198)
        float_242658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 51), 'float')
        keyword_242659 = float_242658
        kwargs_242660 = {'rtol': keyword_242659}
        # Getting the type of 'assert_allclose' (line 198)
        assert_allclose_242655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 198)
        assert_allclose_call_result_242661 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), assert_allclose_242655, *[jac_diff_2_242656, jac_true_242657], **kwargs_242660)
        
        
        # Call to assert_allclose(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'jac_diff_3' (line 199)
        jac_diff_3_242663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 24), 'jac_diff_3', False)
        # Getting the type of 'jac_true' (line 199)
        jac_true_242664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 199)
        float_242665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 51), 'float')
        keyword_242666 = float_242665
        kwargs_242667 = {'rtol': keyword_242666}
        # Getting the type of 'assert_allclose' (line 199)
        assert_allclose_242662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 199)
        assert_allclose_call_result_242668 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), assert_allclose_242662, *[jac_diff_3_242663, jac_true_242664], **kwargs_242667)
        
        
        # Call to assert_allclose(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'jac_diff_4' (line 200)
        jac_diff_4_242670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 24), 'jac_diff_4', False)
        # Getting the type of 'jac_true' (line 200)
        jac_true_242671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 200)
        float_242672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 51), 'float')
        keyword_242673 = float_242672
        kwargs_242674 = {'rtol': keyword_242673}
        # Getting the type of 'assert_allclose' (line 200)
        assert_allclose_242669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 200)
        assert_allclose_call_result_242675 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), assert_allclose_242669, *[jac_diff_4_242670, jac_true_242671], **kwargs_242674)
        
        
        # ################# End of 'test_scalar_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalar_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 190)
        stypy_return_type_242676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242676)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalar_scalar'
        return stypy_return_type_242676


    @norecursion
    def test_scalar_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalar_vector'
        module_type_store = module_type_store.open_function_context('test_scalar_vector', 202, 4, False)
        # Assigning a type to the variable 'self' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.test_scalar_vector.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.test_scalar_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.test_scalar_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.test_scalar_vector.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.test_scalar_vector')
        TestApproxDerivativesDense.test_scalar_vector.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativesDense.test_scalar_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.test_scalar_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.test_scalar_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.test_scalar_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.test_scalar_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.test_scalar_vector.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.test_scalar_vector', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalar_vector', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalar_vector(...)' code ##################

        
        # Assigning a Num to a Name (line 203):
        
        # Assigning a Num to a Name (line 203):
        float_242677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 13), 'float')
        # Assigning a type to the variable 'x0' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'x0', float_242677)
        
        # Assigning a Call to a Name (line 204):
        
        # Assigning a Call to a Name (line 204):
        
        # Call to approx_derivative(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'self' (line 204)
        self_242679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 39), 'self', False)
        # Obtaining the member 'fun_scalar_vector' of a type (line 204)
        fun_scalar_vector_242680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 39), self_242679, 'fun_scalar_vector')
        # Getting the type of 'x0' (line 204)
        x0_242681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 63), 'x0', False)
        # Processing the call keyword arguments (line 204)
        str_242682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 46), 'str', '2-point')
        keyword_242683 = str_242682
        kwargs_242684 = {'method': keyword_242683}
        # Getting the type of 'approx_derivative' (line 204)
        approx_derivative_242678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 204)
        approx_derivative_call_result_242685 = invoke(stypy.reporting.localization.Localization(__file__, 204, 21), approx_derivative_242678, *[fun_scalar_vector_242680, x0_242681], **kwargs_242684)
        
        # Assigning a type to the variable 'jac_diff_2' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'jac_diff_2', approx_derivative_call_result_242685)
        
        # Assigning a Call to a Name (line 206):
        
        # Assigning a Call to a Name (line 206):
        
        # Call to approx_derivative(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'self' (line 206)
        self_242687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 39), 'self', False)
        # Obtaining the member 'fun_scalar_vector' of a type (line 206)
        fun_scalar_vector_242688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 39), self_242687, 'fun_scalar_vector')
        # Getting the type of 'x0' (line 206)
        x0_242689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 63), 'x0', False)
        # Processing the call keyword arguments (line 206)
        kwargs_242690 = {}
        # Getting the type of 'approx_derivative' (line 206)
        approx_derivative_242686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 206)
        approx_derivative_call_result_242691 = invoke(stypy.reporting.localization.Localization(__file__, 206, 21), approx_derivative_242686, *[fun_scalar_vector_242688, x0_242689], **kwargs_242690)
        
        # Assigning a type to the variable 'jac_diff_3' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'jac_diff_3', approx_derivative_call_result_242691)
        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to approx_derivative(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'self' (line 207)
        self_242693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 39), 'self', False)
        # Obtaining the member 'fun_scalar_vector' of a type (line 207)
        fun_scalar_vector_242694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 39), self_242693, 'fun_scalar_vector')
        # Getting the type of 'x0' (line 207)
        x0_242695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 63), 'x0', False)
        # Processing the call keyword arguments (line 207)
        str_242696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 46), 'str', 'cs')
        keyword_242697 = str_242696
        kwargs_242698 = {'method': keyword_242697}
        # Getting the type of 'approx_derivative' (line 207)
        approx_derivative_242692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 207)
        approx_derivative_call_result_242699 = invoke(stypy.reporting.localization.Localization(__file__, 207, 21), approx_derivative_242692, *[fun_scalar_vector_242694, x0_242695], **kwargs_242698)
        
        # Assigning a type to the variable 'jac_diff_4' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'jac_diff_4', approx_derivative_call_result_242699)
        
        # Assigning a Call to a Name (line 209):
        
        # Assigning a Call to a Name (line 209):
        
        # Call to jac_scalar_vector(...): (line 209)
        # Processing the call arguments (line 209)
        
        # Call to atleast_1d(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'x0' (line 209)
        x0_242704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 56), 'x0', False)
        # Processing the call keyword arguments (line 209)
        kwargs_242705 = {}
        # Getting the type of 'np' (line 209)
        np_242702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 42), 'np', False)
        # Obtaining the member 'atleast_1d' of a type (line 209)
        atleast_1d_242703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 42), np_242702, 'atleast_1d')
        # Calling atleast_1d(args, kwargs) (line 209)
        atleast_1d_call_result_242706 = invoke(stypy.reporting.localization.Localization(__file__, 209, 42), atleast_1d_242703, *[x0_242704], **kwargs_242705)
        
        # Processing the call keyword arguments (line 209)
        kwargs_242707 = {}
        # Getting the type of 'self' (line 209)
        self_242700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 19), 'self', False)
        # Obtaining the member 'jac_scalar_vector' of a type (line 209)
        jac_scalar_vector_242701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 19), self_242700, 'jac_scalar_vector')
        # Calling jac_scalar_vector(args, kwargs) (line 209)
        jac_scalar_vector_call_result_242708 = invoke(stypy.reporting.localization.Localization(__file__, 209, 19), jac_scalar_vector_242701, *[atleast_1d_call_result_242706], **kwargs_242707)
        
        # Assigning a type to the variable 'jac_true' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'jac_true', jac_scalar_vector_call_result_242708)
        
        # Call to assert_allclose(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'jac_diff_2' (line 210)
        jac_diff_2_242710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 24), 'jac_diff_2', False)
        # Getting the type of 'jac_true' (line 210)
        jac_true_242711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 210)
        float_242712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 51), 'float')
        keyword_242713 = float_242712
        kwargs_242714 = {'rtol': keyword_242713}
        # Getting the type of 'assert_allclose' (line 210)
        assert_allclose_242709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 210)
        assert_allclose_call_result_242715 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), assert_allclose_242709, *[jac_diff_2_242710, jac_true_242711], **kwargs_242714)
        
        
        # Call to assert_allclose(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'jac_diff_3' (line 211)
        jac_diff_3_242717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 24), 'jac_diff_3', False)
        # Getting the type of 'jac_true' (line 211)
        jac_true_242718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 211)
        float_242719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 51), 'float')
        keyword_242720 = float_242719
        kwargs_242721 = {'rtol': keyword_242720}
        # Getting the type of 'assert_allclose' (line 211)
        assert_allclose_242716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 211)
        assert_allclose_call_result_242722 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), assert_allclose_242716, *[jac_diff_3_242717, jac_true_242718], **kwargs_242721)
        
        
        # Call to assert_allclose(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'jac_diff_4' (line 212)
        jac_diff_4_242724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 24), 'jac_diff_4', False)
        # Getting the type of 'jac_true' (line 212)
        jac_true_242725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 212)
        float_242726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 51), 'float')
        keyword_242727 = float_242726
        kwargs_242728 = {'rtol': keyword_242727}
        # Getting the type of 'assert_allclose' (line 212)
        assert_allclose_242723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 212)
        assert_allclose_call_result_242729 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), assert_allclose_242723, *[jac_diff_4_242724, jac_true_242725], **kwargs_242728)
        
        
        # ################# End of 'test_scalar_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalar_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 202)
        stypy_return_type_242730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242730)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalar_vector'
        return stypy_return_type_242730


    @norecursion
    def test_vector_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vector_scalar'
        module_type_store = module_type_store.open_function_context('test_vector_scalar', 214, 4, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.test_vector_scalar.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.test_vector_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.test_vector_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.test_vector_scalar.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.test_vector_scalar')
        TestApproxDerivativesDense.test_vector_scalar.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativesDense.test_vector_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.test_vector_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.test_vector_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.test_vector_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.test_vector_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.test_vector_scalar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.test_vector_scalar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vector_scalar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vector_scalar(...)' code ##################

        
        # Assigning a Call to a Name (line 215):
        
        # Assigning a Call to a Name (line 215):
        
        # Call to array(...): (line 215)
        # Processing the call arguments (line 215)
        
        # Obtaining an instance of the builtin type 'list' (line 215)
        list_242733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 215)
        # Adding element type (line 215)
        float_242734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 22), list_242733, float_242734)
        # Adding element type (line 215)
        float_242735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 22), list_242733, float_242735)
        
        # Processing the call keyword arguments (line 215)
        kwargs_242736 = {}
        # Getting the type of 'np' (line 215)
        np_242731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 215)
        array_242732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 13), np_242731, 'array')
        # Calling array(args, kwargs) (line 215)
        array_call_result_242737 = invoke(stypy.reporting.localization.Localization(__file__, 215, 13), array_242732, *[list_242733], **kwargs_242736)
        
        # Assigning a type to the variable 'x0' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'x0', array_call_result_242737)
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Call to approx_derivative(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'self' (line 216)
        self_242739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 39), 'self', False)
        # Obtaining the member 'fun_vector_scalar' of a type (line 216)
        fun_vector_scalar_242740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 39), self_242739, 'fun_vector_scalar')
        # Getting the type of 'x0' (line 216)
        x0_242741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 63), 'x0', False)
        # Processing the call keyword arguments (line 216)
        str_242742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 46), 'str', '2-point')
        keyword_242743 = str_242742
        kwargs_242744 = {'method': keyword_242743}
        # Getting the type of 'approx_derivative' (line 216)
        approx_derivative_242738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 216)
        approx_derivative_call_result_242745 = invoke(stypy.reporting.localization.Localization(__file__, 216, 21), approx_derivative_242738, *[fun_vector_scalar_242740, x0_242741], **kwargs_242744)
        
        # Assigning a type to the variable 'jac_diff_2' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'jac_diff_2', approx_derivative_call_result_242745)
        
        # Assigning a Call to a Name (line 218):
        
        # Assigning a Call to a Name (line 218):
        
        # Call to approx_derivative(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'self' (line 218)
        self_242747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 39), 'self', False)
        # Obtaining the member 'fun_vector_scalar' of a type (line 218)
        fun_vector_scalar_242748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 39), self_242747, 'fun_vector_scalar')
        # Getting the type of 'x0' (line 218)
        x0_242749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 63), 'x0', False)
        # Processing the call keyword arguments (line 218)
        kwargs_242750 = {}
        # Getting the type of 'approx_derivative' (line 218)
        approx_derivative_242746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 218)
        approx_derivative_call_result_242751 = invoke(stypy.reporting.localization.Localization(__file__, 218, 21), approx_derivative_242746, *[fun_vector_scalar_242748, x0_242749], **kwargs_242750)
        
        # Assigning a type to the variable 'jac_diff_3' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'jac_diff_3', approx_derivative_call_result_242751)
        
        # Assigning a Call to a Name (line 219):
        
        # Assigning a Call to a Name (line 219):
        
        # Call to approx_derivative(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'self' (line 219)
        self_242753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 39), 'self', False)
        # Obtaining the member 'fun_vector_scalar' of a type (line 219)
        fun_vector_scalar_242754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 39), self_242753, 'fun_vector_scalar')
        # Getting the type of 'x0' (line 219)
        x0_242755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 63), 'x0', False)
        # Processing the call keyword arguments (line 219)
        str_242756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 46), 'str', 'cs')
        keyword_242757 = str_242756
        kwargs_242758 = {'method': keyword_242757}
        # Getting the type of 'approx_derivative' (line 219)
        approx_derivative_242752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 219)
        approx_derivative_call_result_242759 = invoke(stypy.reporting.localization.Localization(__file__, 219, 21), approx_derivative_242752, *[fun_vector_scalar_242754, x0_242755], **kwargs_242758)
        
        # Assigning a type to the variable 'jac_diff_4' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'jac_diff_4', approx_derivative_call_result_242759)
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to jac_vector_scalar(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'x0' (line 221)
        x0_242762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 42), 'x0', False)
        # Processing the call keyword arguments (line 221)
        kwargs_242763 = {}
        # Getting the type of 'self' (line 221)
        self_242760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 19), 'self', False)
        # Obtaining the member 'jac_vector_scalar' of a type (line 221)
        jac_vector_scalar_242761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 19), self_242760, 'jac_vector_scalar')
        # Calling jac_vector_scalar(args, kwargs) (line 221)
        jac_vector_scalar_call_result_242764 = invoke(stypy.reporting.localization.Localization(__file__, 221, 19), jac_vector_scalar_242761, *[x0_242762], **kwargs_242763)
        
        # Assigning a type to the variable 'jac_true' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'jac_true', jac_vector_scalar_call_result_242764)
        
        # Call to assert_allclose(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'jac_diff_2' (line 222)
        jac_diff_2_242766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 24), 'jac_diff_2', False)
        # Getting the type of 'jac_true' (line 222)
        jac_true_242767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 222)
        float_242768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 51), 'float')
        keyword_242769 = float_242768
        kwargs_242770 = {'rtol': keyword_242769}
        # Getting the type of 'assert_allclose' (line 222)
        assert_allclose_242765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 222)
        assert_allclose_call_result_242771 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), assert_allclose_242765, *[jac_diff_2_242766, jac_true_242767], **kwargs_242770)
        
        
        # Call to assert_allclose(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'jac_diff_3' (line 223)
        jac_diff_3_242773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 24), 'jac_diff_3', False)
        # Getting the type of 'jac_true' (line 223)
        jac_true_242774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 223)
        float_242775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 51), 'float')
        keyword_242776 = float_242775
        kwargs_242777 = {'rtol': keyword_242776}
        # Getting the type of 'assert_allclose' (line 223)
        assert_allclose_242772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 223)
        assert_allclose_call_result_242778 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), assert_allclose_242772, *[jac_diff_3_242773, jac_true_242774], **kwargs_242777)
        
        
        # Call to assert_allclose(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'jac_diff_4' (line 224)
        jac_diff_4_242780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 24), 'jac_diff_4', False)
        # Getting the type of 'jac_true' (line 224)
        jac_true_242781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 224)
        float_242782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 51), 'float')
        keyword_242783 = float_242782
        kwargs_242784 = {'rtol': keyword_242783}
        # Getting the type of 'assert_allclose' (line 224)
        assert_allclose_242779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 224)
        assert_allclose_call_result_242785 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), assert_allclose_242779, *[jac_diff_4_242780, jac_true_242781], **kwargs_242784)
        
        
        # ################# End of 'test_vector_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vector_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 214)
        stypy_return_type_242786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242786)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vector_scalar'
        return stypy_return_type_242786


    @norecursion
    def test_vector_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vector_vector'
        module_type_store = module_type_store.open_function_context('test_vector_vector', 226, 4, False)
        # Assigning a type to the variable 'self' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.test_vector_vector.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.test_vector_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.test_vector_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.test_vector_vector.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.test_vector_vector')
        TestApproxDerivativesDense.test_vector_vector.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativesDense.test_vector_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.test_vector_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.test_vector_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.test_vector_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.test_vector_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.test_vector_vector.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.test_vector_vector', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vector_vector', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vector_vector(...)' code ##################

        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to array(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_242789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        float_242790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 22), list_242789, float_242790)
        # Adding element type (line 227)
        float_242791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 22), list_242789, float_242791)
        
        # Processing the call keyword arguments (line 227)
        kwargs_242792 = {}
        # Getting the type of 'np' (line 227)
        np_242787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 227)
        array_242788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 13), np_242787, 'array')
        # Calling array(args, kwargs) (line 227)
        array_call_result_242793 = invoke(stypy.reporting.localization.Localization(__file__, 227, 13), array_242788, *[list_242789], **kwargs_242792)
        
        # Assigning a type to the variable 'x0' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'x0', array_call_result_242793)
        
        # Assigning a Call to a Name (line 228):
        
        # Assigning a Call to a Name (line 228):
        
        # Call to approx_derivative(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'self' (line 228)
        self_242795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 39), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 228)
        fun_vector_vector_242796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 39), self_242795, 'fun_vector_vector')
        # Getting the type of 'x0' (line 228)
        x0_242797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 63), 'x0', False)
        # Processing the call keyword arguments (line 228)
        str_242798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 46), 'str', '2-point')
        keyword_242799 = str_242798
        kwargs_242800 = {'method': keyword_242799}
        # Getting the type of 'approx_derivative' (line 228)
        approx_derivative_242794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 228)
        approx_derivative_call_result_242801 = invoke(stypy.reporting.localization.Localization(__file__, 228, 21), approx_derivative_242794, *[fun_vector_vector_242796, x0_242797], **kwargs_242800)
        
        # Assigning a type to the variable 'jac_diff_2' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'jac_diff_2', approx_derivative_call_result_242801)
        
        # Assigning a Call to a Name (line 230):
        
        # Assigning a Call to a Name (line 230):
        
        # Call to approx_derivative(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'self' (line 230)
        self_242803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 39), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 230)
        fun_vector_vector_242804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 39), self_242803, 'fun_vector_vector')
        # Getting the type of 'x0' (line 230)
        x0_242805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 63), 'x0', False)
        # Processing the call keyword arguments (line 230)
        kwargs_242806 = {}
        # Getting the type of 'approx_derivative' (line 230)
        approx_derivative_242802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 230)
        approx_derivative_call_result_242807 = invoke(stypy.reporting.localization.Localization(__file__, 230, 21), approx_derivative_242802, *[fun_vector_vector_242804, x0_242805], **kwargs_242806)
        
        # Assigning a type to the variable 'jac_diff_3' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'jac_diff_3', approx_derivative_call_result_242807)
        
        # Assigning a Call to a Name (line 231):
        
        # Assigning a Call to a Name (line 231):
        
        # Call to approx_derivative(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'self' (line 231)
        self_242809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 39), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 231)
        fun_vector_vector_242810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 39), self_242809, 'fun_vector_vector')
        # Getting the type of 'x0' (line 231)
        x0_242811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 63), 'x0', False)
        # Processing the call keyword arguments (line 231)
        str_242812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 46), 'str', 'cs')
        keyword_242813 = str_242812
        kwargs_242814 = {'method': keyword_242813}
        # Getting the type of 'approx_derivative' (line 231)
        approx_derivative_242808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 231)
        approx_derivative_call_result_242815 = invoke(stypy.reporting.localization.Localization(__file__, 231, 21), approx_derivative_242808, *[fun_vector_vector_242810, x0_242811], **kwargs_242814)
        
        # Assigning a type to the variable 'jac_diff_4' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'jac_diff_4', approx_derivative_call_result_242815)
        
        # Assigning a Call to a Name (line 233):
        
        # Assigning a Call to a Name (line 233):
        
        # Call to jac_vector_vector(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'x0' (line 233)
        x0_242818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 42), 'x0', False)
        # Processing the call keyword arguments (line 233)
        kwargs_242819 = {}
        # Getting the type of 'self' (line 233)
        self_242816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 19), 'self', False)
        # Obtaining the member 'jac_vector_vector' of a type (line 233)
        jac_vector_vector_242817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 19), self_242816, 'jac_vector_vector')
        # Calling jac_vector_vector(args, kwargs) (line 233)
        jac_vector_vector_call_result_242820 = invoke(stypy.reporting.localization.Localization(__file__, 233, 19), jac_vector_vector_242817, *[x0_242818], **kwargs_242819)
        
        # Assigning a type to the variable 'jac_true' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'jac_true', jac_vector_vector_call_result_242820)
        
        # Call to assert_allclose(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'jac_diff_2' (line 234)
        jac_diff_2_242822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 24), 'jac_diff_2', False)
        # Getting the type of 'jac_true' (line 234)
        jac_true_242823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 234)
        float_242824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 51), 'float')
        keyword_242825 = float_242824
        kwargs_242826 = {'rtol': keyword_242825}
        # Getting the type of 'assert_allclose' (line 234)
        assert_allclose_242821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 234)
        assert_allclose_call_result_242827 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), assert_allclose_242821, *[jac_diff_2_242822, jac_true_242823], **kwargs_242826)
        
        
        # Call to assert_allclose(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'jac_diff_3' (line 235)
        jac_diff_3_242829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 24), 'jac_diff_3', False)
        # Getting the type of 'jac_true' (line 235)
        jac_true_242830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 235)
        float_242831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 51), 'float')
        keyword_242832 = float_242831
        kwargs_242833 = {'rtol': keyword_242832}
        # Getting the type of 'assert_allclose' (line 235)
        assert_allclose_242828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 235)
        assert_allclose_call_result_242834 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), assert_allclose_242828, *[jac_diff_3_242829, jac_true_242830], **kwargs_242833)
        
        
        # Call to assert_allclose(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'jac_diff_4' (line 236)
        jac_diff_4_242836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 24), 'jac_diff_4', False)
        # Getting the type of 'jac_true' (line 236)
        jac_true_242837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 236)
        float_242838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 51), 'float')
        keyword_242839 = float_242838
        kwargs_242840 = {'rtol': keyword_242839}
        # Getting the type of 'assert_allclose' (line 236)
        assert_allclose_242835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 236)
        assert_allclose_call_result_242841 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), assert_allclose_242835, *[jac_diff_4_242836, jac_true_242837], **kwargs_242840)
        
        
        # ################# End of 'test_vector_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vector_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 226)
        stypy_return_type_242842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242842)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vector_vector'
        return stypy_return_type_242842


    @norecursion
    def test_wrong_dimensions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_wrong_dimensions'
        module_type_store = module_type_store.open_function_context('test_wrong_dimensions', 238, 4, False)
        # Assigning a type to the variable 'self' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.test_wrong_dimensions.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.test_wrong_dimensions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.test_wrong_dimensions.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.test_wrong_dimensions.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.test_wrong_dimensions')
        TestApproxDerivativesDense.test_wrong_dimensions.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativesDense.test_wrong_dimensions.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.test_wrong_dimensions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.test_wrong_dimensions.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.test_wrong_dimensions.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.test_wrong_dimensions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.test_wrong_dimensions.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.test_wrong_dimensions', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_wrong_dimensions', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_wrong_dimensions(...)' code ##################

        
        # Assigning a Num to a Name (line 239):
        
        # Assigning a Num to a Name (line 239):
        float_242843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 13), 'float')
        # Assigning a type to the variable 'x0' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'x0', float_242843)
        
        # Call to assert_raises(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'RuntimeError' (line 240)
        RuntimeError_242845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 22), 'RuntimeError', False)
        # Getting the type of 'approx_derivative' (line 240)
        approx_derivative_242846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 36), 'approx_derivative', False)
        # Getting the type of 'self' (line 241)
        self_242847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 22), 'self', False)
        # Obtaining the member 'wrong_dimensions_fun' of a type (line 241)
        wrong_dimensions_fun_242848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 22), self_242847, 'wrong_dimensions_fun')
        # Getting the type of 'x0' (line 241)
        x0_242849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 49), 'x0', False)
        # Processing the call keyword arguments (line 240)
        kwargs_242850 = {}
        # Getting the type of 'assert_raises' (line 240)
        assert_raises_242844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 240)
        assert_raises_call_result_242851 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), assert_raises_242844, *[RuntimeError_242845, approx_derivative_242846, wrong_dimensions_fun_242848, x0_242849], **kwargs_242850)
        
        
        # Assigning a Call to a Name (line 242):
        
        # Assigning a Call to a Name (line 242):
        
        # Call to wrong_dimensions_fun(...): (line 242)
        # Processing the call arguments (line 242)
        
        # Call to atleast_1d(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'x0' (line 242)
        x0_242856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 53), 'x0', False)
        # Processing the call keyword arguments (line 242)
        kwargs_242857 = {}
        # Getting the type of 'np' (line 242)
        np_242854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 39), 'np', False)
        # Obtaining the member 'atleast_1d' of a type (line 242)
        atleast_1d_242855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 39), np_242854, 'atleast_1d')
        # Calling atleast_1d(args, kwargs) (line 242)
        atleast_1d_call_result_242858 = invoke(stypy.reporting.localization.Localization(__file__, 242, 39), atleast_1d_242855, *[x0_242856], **kwargs_242857)
        
        # Processing the call keyword arguments (line 242)
        kwargs_242859 = {}
        # Getting the type of 'self' (line 242)
        self_242852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 13), 'self', False)
        # Obtaining the member 'wrong_dimensions_fun' of a type (line 242)
        wrong_dimensions_fun_242853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 13), self_242852, 'wrong_dimensions_fun')
        # Calling wrong_dimensions_fun(args, kwargs) (line 242)
        wrong_dimensions_fun_call_result_242860 = invoke(stypy.reporting.localization.Localization(__file__, 242, 13), wrong_dimensions_fun_242853, *[atleast_1d_call_result_242858], **kwargs_242859)
        
        # Assigning a type to the variable 'f0' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'f0', wrong_dimensions_fun_call_result_242860)
        
        # Call to assert_raises(...): (line 243)
        # Processing the call arguments (line 243)
        # Getting the type of 'ValueError' (line 243)
        ValueError_242862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 22), 'ValueError', False)
        # Getting the type of 'approx_derivative' (line 243)
        approx_derivative_242863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 34), 'approx_derivative', False)
        # Getting the type of 'self' (line 244)
        self_242864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 22), 'self', False)
        # Obtaining the member 'wrong_dimensions_fun' of a type (line 244)
        wrong_dimensions_fun_242865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 22), self_242864, 'wrong_dimensions_fun')
        # Getting the type of 'x0' (line 244)
        x0_242866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 49), 'x0', False)
        # Processing the call keyword arguments (line 243)
        # Getting the type of 'f0' (line 244)
        f0_242867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 56), 'f0', False)
        keyword_242868 = f0_242867
        kwargs_242869 = {'f0': keyword_242868}
        # Getting the type of 'assert_raises' (line 243)
        assert_raises_242861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 243)
        assert_raises_call_result_242870 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), assert_raises_242861, *[ValueError_242862, approx_derivative_242863, wrong_dimensions_fun_242865, x0_242866], **kwargs_242869)
        
        
        # ################# End of 'test_wrong_dimensions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_wrong_dimensions' in the type store
        # Getting the type of 'stypy_return_type' (line 238)
        stypy_return_type_242871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242871)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_wrong_dimensions'
        return stypy_return_type_242871


    @norecursion
    def test_custom_rel_step(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_custom_rel_step'
        module_type_store = module_type_store.open_function_context('test_custom_rel_step', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.test_custom_rel_step.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.test_custom_rel_step.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.test_custom_rel_step.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.test_custom_rel_step.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.test_custom_rel_step')
        TestApproxDerivativesDense.test_custom_rel_step.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativesDense.test_custom_rel_step.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.test_custom_rel_step.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.test_custom_rel_step.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.test_custom_rel_step.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.test_custom_rel_step.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.test_custom_rel_step.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.test_custom_rel_step', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_custom_rel_step', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_custom_rel_step(...)' code ##################

        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Call to array(...): (line 247)
        # Processing the call arguments (line 247)
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_242874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        # Adding element type (line 247)
        float_242875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 22), list_242874, float_242875)
        # Adding element type (line 247)
        float_242876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 22), list_242874, float_242876)
        
        # Processing the call keyword arguments (line 247)
        kwargs_242877 = {}
        # Getting the type of 'np' (line 247)
        np_242872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 247)
        array_242873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 13), np_242872, 'array')
        # Calling array(args, kwargs) (line 247)
        array_call_result_242878 = invoke(stypy.reporting.localization.Localization(__file__, 247, 13), array_242873, *[list_242874], **kwargs_242877)
        
        # Assigning a type to the variable 'x0' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'x0', array_call_result_242878)
        
        # Assigning a Call to a Name (line 248):
        
        # Assigning a Call to a Name (line 248):
        
        # Call to approx_derivative(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'self' (line 248)
        self_242880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 39), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 248)
        fun_vector_vector_242881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 39), self_242880, 'fun_vector_vector')
        # Getting the type of 'x0' (line 248)
        x0_242882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 63), 'x0', False)
        # Processing the call keyword arguments (line 248)
        str_242883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 46), 'str', '2-point')
        keyword_242884 = str_242883
        float_242885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 66), 'float')
        keyword_242886 = float_242885
        kwargs_242887 = {'method': keyword_242884, 'rel_step': keyword_242886}
        # Getting the type of 'approx_derivative' (line 248)
        approx_derivative_242879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 248)
        approx_derivative_call_result_242888 = invoke(stypy.reporting.localization.Localization(__file__, 248, 21), approx_derivative_242879, *[fun_vector_vector_242881, x0_242882], **kwargs_242887)
        
        # Assigning a type to the variable 'jac_diff_2' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'jac_diff_2', approx_derivative_call_result_242888)
        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to approx_derivative(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'self' (line 250)
        self_242890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 39), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 250)
        fun_vector_vector_242891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 39), self_242890, 'fun_vector_vector')
        # Getting the type of 'x0' (line 250)
        x0_242892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 63), 'x0', False)
        # Processing the call keyword arguments (line 250)
        float_242893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 48), 'float')
        keyword_242894 = float_242893
        kwargs_242895 = {'rel_step': keyword_242894}
        # Getting the type of 'approx_derivative' (line 250)
        approx_derivative_242889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 250)
        approx_derivative_call_result_242896 = invoke(stypy.reporting.localization.Localization(__file__, 250, 21), approx_derivative_242889, *[fun_vector_vector_242891, x0_242892], **kwargs_242895)
        
        # Assigning a type to the variable 'jac_diff_3' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'jac_diff_3', approx_derivative_call_result_242896)
        
        # Assigning a Call to a Name (line 252):
        
        # Assigning a Call to a Name (line 252):
        
        # Call to jac_vector_vector(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'x0' (line 252)
        x0_242899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 42), 'x0', False)
        # Processing the call keyword arguments (line 252)
        kwargs_242900 = {}
        # Getting the type of 'self' (line 252)
        self_242897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 19), 'self', False)
        # Obtaining the member 'jac_vector_vector' of a type (line 252)
        jac_vector_vector_242898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 19), self_242897, 'jac_vector_vector')
        # Calling jac_vector_vector(args, kwargs) (line 252)
        jac_vector_vector_call_result_242901 = invoke(stypy.reporting.localization.Localization(__file__, 252, 19), jac_vector_vector_242898, *[x0_242899], **kwargs_242900)
        
        # Assigning a type to the variable 'jac_true' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'jac_true', jac_vector_vector_call_result_242901)
        
        # Call to assert_allclose(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'jac_diff_2' (line 253)
        jac_diff_2_242903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 24), 'jac_diff_2', False)
        # Getting the type of 'jac_true' (line 253)
        jac_true_242904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 253)
        float_242905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 51), 'float')
        keyword_242906 = float_242905
        kwargs_242907 = {'rtol': keyword_242906}
        # Getting the type of 'assert_allclose' (line 253)
        assert_allclose_242902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 253)
        assert_allclose_call_result_242908 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), assert_allclose_242902, *[jac_diff_2_242903, jac_true_242904], **kwargs_242907)
        
        
        # Call to assert_allclose(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'jac_diff_3' (line 254)
        jac_diff_3_242910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 24), 'jac_diff_3', False)
        # Getting the type of 'jac_true' (line 254)
        jac_true_242911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 254)
        float_242912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 51), 'float')
        keyword_242913 = float_242912
        kwargs_242914 = {'rtol': keyword_242913}
        # Getting the type of 'assert_allclose' (line 254)
        assert_allclose_242909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 254)
        assert_allclose_call_result_242915 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), assert_allclose_242909, *[jac_diff_3_242910, jac_true_242911], **kwargs_242914)
        
        
        # ################# End of 'test_custom_rel_step(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_custom_rel_step' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_242916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242916)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_custom_rel_step'
        return stypy_return_type_242916


    @norecursion
    def test_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_options'
        module_type_store = module_type_store.open_function_context('test_options', 256, 4, False)
        # Assigning a type to the variable 'self' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.test_options.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.test_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.test_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.test_options.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.test_options')
        TestApproxDerivativesDense.test_options.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativesDense.test_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.test_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.test_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.test_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.test_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.test_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.test_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Call to array(...): (line 257)
        # Processing the call arguments (line 257)
        
        # Obtaining an instance of the builtin type 'list' (line 257)
        list_242919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 257)
        # Adding element type (line 257)
        float_242920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 22), list_242919, float_242920)
        # Adding element type (line 257)
        float_242921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 22), list_242919, float_242921)
        
        # Processing the call keyword arguments (line 257)
        kwargs_242922 = {}
        # Getting the type of 'np' (line 257)
        np_242917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 257)
        array_242918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 13), np_242917, 'array')
        # Calling array(args, kwargs) (line 257)
        array_call_result_242923 = invoke(stypy.reporting.localization.Localization(__file__, 257, 13), array_242918, *[list_242919], **kwargs_242922)
        
        # Assigning a type to the variable 'x0' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'x0', array_call_result_242923)
        
        # Assigning a Num to a Name (line 258):
        
        # Assigning a Num to a Name (line 258):
        float_242924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 13), 'float')
        # Assigning a type to the variable 'c0' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'c0', float_242924)
        
        # Assigning a Num to a Name (line 259):
        
        # Assigning a Num to a Name (line 259):
        float_242925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 13), 'float')
        # Assigning a type to the variable 'c1' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'c1', float_242925)
        
        # Assigning a Num to a Name (line 260):
        
        # Assigning a Num to a Name (line 260):
        float_242926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 13), 'float')
        # Assigning a type to the variable 'lb' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'lb', float_242926)
        
        # Assigning a Num to a Name (line 261):
        
        # Assigning a Num to a Name (line 261):
        float_242927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 13), 'float')
        # Assigning a type to the variable 'ub' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'ub', float_242927)
        
        # Assigning a Call to a Name (line 262):
        
        # Assigning a Call to a Name (line 262):
        
        # Call to fun_parametrized(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'x0' (line 262)
        x0_242930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 35), 'x0', False)
        # Getting the type of 'c0' (line 262)
        c0_242931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 39), 'c0', False)
        # Processing the call keyword arguments (line 262)
        # Getting the type of 'c1' (line 262)
        c1_242932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 46), 'c1', False)
        keyword_242933 = c1_242932
        kwargs_242934 = {'c1': keyword_242933}
        # Getting the type of 'self' (line 262)
        self_242928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 13), 'self', False)
        # Obtaining the member 'fun_parametrized' of a type (line 262)
        fun_parametrized_242929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 13), self_242928, 'fun_parametrized')
        # Calling fun_parametrized(args, kwargs) (line 262)
        fun_parametrized_call_result_242935 = invoke(stypy.reporting.localization.Localization(__file__, 262, 13), fun_parametrized_242929, *[x0_242930, c0_242931], **kwargs_242934)
        
        # Assigning a type to the variable 'f0' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'f0', fun_parametrized_call_result_242935)
        
        # Assigning a Call to a Name (line 263):
        
        # Assigning a Call to a Name (line 263):
        
        # Call to array(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Obtaining an instance of the builtin type 'list' (line 263)
        list_242938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 263)
        # Adding element type (line 263)
        float_242939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 28), list_242938, float_242939)
        # Adding element type (line 263)
        float_242940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 28), list_242938, float_242940)
        
        # Processing the call keyword arguments (line 263)
        kwargs_242941 = {}
        # Getting the type of 'np' (line 263)
        np_242936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 263)
        array_242937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 19), np_242936, 'array')
        # Calling array(args, kwargs) (line 263)
        array_call_result_242942 = invoke(stypy.reporting.localization.Localization(__file__, 263, 19), array_242937, *[list_242938], **kwargs_242941)
        
        # Assigning a type to the variable 'rel_step' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'rel_step', array_call_result_242942)
        
        # Assigning a Call to a Name (line 264):
        
        # Assigning a Call to a Name (line 264):
        
        # Call to jac_parametrized(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'x0' (line 264)
        x0_242945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 41), 'x0', False)
        # Getting the type of 'c0' (line 264)
        c0_242946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 45), 'c0', False)
        # Getting the type of 'c1' (line 264)
        c1_242947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 49), 'c1', False)
        # Processing the call keyword arguments (line 264)
        kwargs_242948 = {}
        # Getting the type of 'self' (line 264)
        self_242943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 19), 'self', False)
        # Obtaining the member 'jac_parametrized' of a type (line 264)
        jac_parametrized_242944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 19), self_242943, 'jac_parametrized')
        # Calling jac_parametrized(args, kwargs) (line 264)
        jac_parametrized_call_result_242949 = invoke(stypy.reporting.localization.Localization(__file__, 264, 19), jac_parametrized_242944, *[x0_242945, c0_242946, c1_242947], **kwargs_242948)
        
        # Assigning a type to the variable 'jac_true' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'jac_true', jac_parametrized_call_result_242949)
        
        # Assigning a Call to a Name (line 265):
        
        # Assigning a Call to a Name (line 265):
        
        # Call to approx_derivative(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'self' (line 266)
        self_242951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'self', False)
        # Obtaining the member 'fun_parametrized' of a type (line 266)
        fun_parametrized_242952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), self_242951, 'fun_parametrized')
        # Getting the type of 'x0' (line 266)
        x0_242953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 35), 'x0', False)
        # Processing the call keyword arguments (line 265)
        str_242954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 46), 'str', '2-point')
        keyword_242955 = str_242954
        # Getting the type of 'rel_step' (line 266)
        rel_step_242956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 66), 'rel_step', False)
        keyword_242957 = rel_step_242956
        # Getting the type of 'f0' (line 267)
        f0_242958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 15), 'f0', False)
        keyword_242959 = f0_242958
        
        # Obtaining an instance of the builtin type 'tuple' (line 267)
        tuple_242960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 267)
        # Adding element type (line 267)
        # Getting the type of 'c0' (line 267)
        c0_242961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 25), 'c0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 25), tuple_242960, c0_242961)
        
        keyword_242962 = tuple_242960
        
        # Call to dict(...): (line 267)
        # Processing the call keyword arguments (line 267)
        # Getting the type of 'c1' (line 267)
        c1_242964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 46), 'c1', False)
        keyword_242965 = c1_242964
        kwargs_242966 = {'c1': keyword_242965}
        # Getting the type of 'dict' (line 267)
        dict_242963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 38), 'dict', False)
        # Calling dict(args, kwargs) (line 267)
        dict_call_result_242967 = invoke(stypy.reporting.localization.Localization(__file__, 267, 38), dict_242963, *[], **kwargs_242966)
        
        keyword_242968 = dict_call_result_242967
        
        # Obtaining an instance of the builtin type 'tuple' (line 267)
        tuple_242969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 267)
        # Adding element type (line 267)
        # Getting the type of 'lb' (line 267)
        lb_242970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 59), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 59), tuple_242969, lb_242970)
        # Adding element type (line 267)
        # Getting the type of 'ub' (line 267)
        ub_242971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 63), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 59), tuple_242969, ub_242971)
        
        keyword_242972 = tuple_242969
        kwargs_242973 = {'f0': keyword_242959, 'args': keyword_242962, 'bounds': keyword_242972, 'kwargs': keyword_242968, 'method': keyword_242955, 'rel_step': keyword_242957}
        # Getting the type of 'approx_derivative' (line 265)
        approx_derivative_242950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 265)
        approx_derivative_call_result_242974 = invoke(stypy.reporting.localization.Localization(__file__, 265, 21), approx_derivative_242950, *[fun_parametrized_242952, x0_242953], **kwargs_242973)
        
        # Assigning a type to the variable 'jac_diff_2' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'jac_diff_2', approx_derivative_call_result_242974)
        
        # Assigning a Call to a Name (line 268):
        
        # Assigning a Call to a Name (line 268):
        
        # Call to approx_derivative(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'self' (line 269)
        self_242976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'self', False)
        # Obtaining the member 'fun_parametrized' of a type (line 269)
        fun_parametrized_242977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), self_242976, 'fun_parametrized')
        # Getting the type of 'x0' (line 269)
        x0_242978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 35), 'x0', False)
        # Processing the call keyword arguments (line 268)
        # Getting the type of 'rel_step' (line 269)
        rel_step_242979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 48), 'rel_step', False)
        keyword_242980 = rel_step_242979
        # Getting the type of 'f0' (line 270)
        f0_242981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 15), 'f0', False)
        keyword_242982 = f0_242981
        
        # Obtaining an instance of the builtin type 'tuple' (line 270)
        tuple_242983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 270)
        # Adding element type (line 270)
        # Getting the type of 'c0' (line 270)
        c0_242984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 25), 'c0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 25), tuple_242983, c0_242984)
        
        keyword_242985 = tuple_242983
        
        # Call to dict(...): (line 270)
        # Processing the call keyword arguments (line 270)
        # Getting the type of 'c1' (line 270)
        c1_242987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 46), 'c1', False)
        keyword_242988 = c1_242987
        kwargs_242989 = {'c1': keyword_242988}
        # Getting the type of 'dict' (line 270)
        dict_242986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 38), 'dict', False)
        # Calling dict(args, kwargs) (line 270)
        dict_call_result_242990 = invoke(stypy.reporting.localization.Localization(__file__, 270, 38), dict_242986, *[], **kwargs_242989)
        
        keyword_242991 = dict_call_result_242990
        
        # Obtaining an instance of the builtin type 'tuple' (line 270)
        tuple_242992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 270)
        # Adding element type (line 270)
        # Getting the type of 'lb' (line 270)
        lb_242993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 59), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 59), tuple_242992, lb_242993)
        # Adding element type (line 270)
        # Getting the type of 'ub' (line 270)
        ub_242994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 63), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 59), tuple_242992, ub_242994)
        
        keyword_242995 = tuple_242992
        kwargs_242996 = {'f0': keyword_242982, 'args': keyword_242985, 'bounds': keyword_242995, 'rel_step': keyword_242980, 'kwargs': keyword_242991}
        # Getting the type of 'approx_derivative' (line 268)
        approx_derivative_242975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 268)
        approx_derivative_call_result_242997 = invoke(stypy.reporting.localization.Localization(__file__, 268, 21), approx_derivative_242975, *[fun_parametrized_242977, x0_242978], **kwargs_242996)
        
        # Assigning a type to the variable 'jac_diff_3' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'jac_diff_3', approx_derivative_call_result_242997)
        
        # Call to assert_allclose(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'jac_diff_2' (line 271)
        jac_diff_2_242999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 24), 'jac_diff_2', False)
        # Getting the type of 'jac_true' (line 271)
        jac_true_243000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 271)
        float_243001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 51), 'float')
        keyword_243002 = float_243001
        kwargs_243003 = {'rtol': keyword_243002}
        # Getting the type of 'assert_allclose' (line 271)
        assert_allclose_242998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 271)
        assert_allclose_call_result_243004 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), assert_allclose_242998, *[jac_diff_2_242999, jac_true_243000], **kwargs_243003)
        
        
        # Call to assert_allclose(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'jac_diff_3' (line 272)
        jac_diff_3_243006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 24), 'jac_diff_3', False)
        # Getting the type of 'jac_true' (line 272)
        jac_true_243007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 272)
        float_243008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 51), 'float')
        keyword_243009 = float_243008
        kwargs_243010 = {'rtol': keyword_243009}
        # Getting the type of 'assert_allclose' (line 272)
        assert_allclose_243005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 272)
        assert_allclose_call_result_243011 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), assert_allclose_243005, *[jac_diff_3_243006, jac_true_243007], **kwargs_243010)
        
        
        # ################# End of 'test_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_options' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_243012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243012)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_options'
        return stypy_return_type_243012


    @norecursion
    def test_with_bounds_2_point(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_with_bounds_2_point'
        module_type_store = module_type_store.open_function_context('test_with_bounds_2_point', 274, 4, False)
        # Assigning a type to the variable 'self' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.test_with_bounds_2_point.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.test_with_bounds_2_point.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.test_with_bounds_2_point.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.test_with_bounds_2_point.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.test_with_bounds_2_point')
        TestApproxDerivativesDense.test_with_bounds_2_point.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativesDense.test_with_bounds_2_point.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.test_with_bounds_2_point.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.test_with_bounds_2_point.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.test_with_bounds_2_point.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.test_with_bounds_2_point.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.test_with_bounds_2_point.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.test_with_bounds_2_point', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_with_bounds_2_point', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_with_bounds_2_point(...)' code ##################

        
        # Assigning a UnaryOp to a Name (line 275):
        
        # Assigning a UnaryOp to a Name (line 275):
        
        
        # Call to ones(...): (line 275)
        # Processing the call arguments (line 275)
        int_243015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 22), 'int')
        # Processing the call keyword arguments (line 275)
        kwargs_243016 = {}
        # Getting the type of 'np' (line 275)
        np_243013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 14), 'np', False)
        # Obtaining the member 'ones' of a type (line 275)
        ones_243014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 14), np_243013, 'ones')
        # Calling ones(args, kwargs) (line 275)
        ones_call_result_243017 = invoke(stypy.reporting.localization.Localization(__file__, 275, 14), ones_243014, *[int_243015], **kwargs_243016)
        
        # Applying the 'usub' unary operator (line 275)
        result___neg___243018 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 13), 'usub', ones_call_result_243017)
        
        # Assigning a type to the variable 'lb' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'lb', result___neg___243018)
        
        # Assigning a Call to a Name (line 276):
        
        # Assigning a Call to a Name (line 276):
        
        # Call to ones(...): (line 276)
        # Processing the call arguments (line 276)
        int_243021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 21), 'int')
        # Processing the call keyword arguments (line 276)
        kwargs_243022 = {}
        # Getting the type of 'np' (line 276)
        np_243019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 13), 'np', False)
        # Obtaining the member 'ones' of a type (line 276)
        ones_243020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 13), np_243019, 'ones')
        # Calling ones(args, kwargs) (line 276)
        ones_call_result_243023 = invoke(stypy.reporting.localization.Localization(__file__, 276, 13), ones_243020, *[int_243021], **kwargs_243022)
        
        # Assigning a type to the variable 'ub' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'ub', ones_call_result_243023)
        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to array(...): (line 278)
        # Processing the call arguments (line 278)
        
        # Obtaining an instance of the builtin type 'list' (line 278)
        list_243026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 278)
        # Adding element type (line 278)
        float_243027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 22), list_243026, float_243027)
        # Adding element type (line 278)
        float_243028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 22), list_243026, float_243028)
        
        # Processing the call keyword arguments (line 278)
        kwargs_243029 = {}
        # Getting the type of 'np' (line 278)
        np_243024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 278)
        array_243025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 13), np_243024, 'array')
        # Calling array(args, kwargs) (line 278)
        array_call_result_243030 = invoke(stypy.reporting.localization.Localization(__file__, 278, 13), array_243025, *[list_243026], **kwargs_243029)
        
        # Assigning a type to the variable 'x0' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'x0', array_call_result_243030)
        
        # Call to assert_raises(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'ValueError' (line 279)
        ValueError_243032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 22), 'ValueError', False)
        # Getting the type of 'approx_derivative' (line 279)
        approx_derivative_243033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 34), 'approx_derivative', False)
        # Getting the type of 'self' (line 280)
        self_243034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 22), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 280)
        fun_vector_vector_243035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 22), self_243034, 'fun_vector_vector')
        # Getting the type of 'x0' (line 280)
        x0_243036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 46), 'x0', False)
        # Processing the call keyword arguments (line 279)
        
        # Obtaining an instance of the builtin type 'tuple' (line 280)
        tuple_243037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 280)
        # Adding element type (line 280)
        # Getting the type of 'lb' (line 280)
        lb_243038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 58), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 58), tuple_243037, lb_243038)
        # Adding element type (line 280)
        # Getting the type of 'ub' (line 280)
        ub_243039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 62), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 58), tuple_243037, ub_243039)
        
        keyword_243040 = tuple_243037
        kwargs_243041 = {'bounds': keyword_243040}
        # Getting the type of 'assert_raises' (line 279)
        assert_raises_243031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 279)
        assert_raises_call_result_243042 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), assert_raises_243031, *[ValueError_243032, approx_derivative_243033, fun_vector_vector_243035, x0_243036], **kwargs_243041)
        
        
        # Assigning a Call to a Name (line 282):
        
        # Assigning a Call to a Name (line 282):
        
        # Call to array(...): (line 282)
        # Processing the call arguments (line 282)
        
        # Obtaining an instance of the builtin type 'list' (line 282)
        list_243045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 282)
        # Adding element type (line 282)
        float_243046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 22), list_243045, float_243046)
        # Adding element type (line 282)
        float_243047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 22), list_243045, float_243047)
        
        # Processing the call keyword arguments (line 282)
        kwargs_243048 = {}
        # Getting the type of 'np' (line 282)
        np_243043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 282)
        array_243044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 13), np_243043, 'array')
        # Calling array(args, kwargs) (line 282)
        array_call_result_243049 = invoke(stypy.reporting.localization.Localization(__file__, 282, 13), array_243044, *[list_243045], **kwargs_243048)
        
        # Assigning a type to the variable 'x0' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'x0', array_call_result_243049)
        
        # Assigning a Call to a Name (line 283):
        
        # Assigning a Call to a Name (line 283):
        
        # Call to approx_derivative(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'self' (line 283)
        self_243051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 37), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 283)
        fun_vector_vector_243052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 37), self_243051, 'fun_vector_vector')
        # Getting the type of 'x0' (line 283)
        x0_243053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 61), 'x0', False)
        # Processing the call keyword arguments (line 283)
        str_243054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 44), 'str', '2-point')
        keyword_243055 = str_243054
        
        # Obtaining an instance of the builtin type 'tuple' (line 284)
        tuple_243056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 284)
        # Adding element type (line 284)
        # Getting the type of 'lb' (line 284)
        lb_243057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 63), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 63), tuple_243056, lb_243057)
        # Adding element type (line 284)
        # Getting the type of 'ub' (line 284)
        ub_243058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 67), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 63), tuple_243056, ub_243058)
        
        keyword_243059 = tuple_243056
        kwargs_243060 = {'method': keyword_243055, 'bounds': keyword_243059}
        # Getting the type of 'approx_derivative' (line 283)
        approx_derivative_243050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 283)
        approx_derivative_call_result_243061 = invoke(stypy.reporting.localization.Localization(__file__, 283, 19), approx_derivative_243050, *[fun_vector_vector_243052, x0_243053], **kwargs_243060)
        
        # Assigning a type to the variable 'jac_diff' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'jac_diff', approx_derivative_call_result_243061)
        
        # Assigning a Call to a Name (line 285):
        
        # Assigning a Call to a Name (line 285):
        
        # Call to jac_vector_vector(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'x0' (line 285)
        x0_243064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 42), 'x0', False)
        # Processing the call keyword arguments (line 285)
        kwargs_243065 = {}
        # Getting the type of 'self' (line 285)
        self_243062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 19), 'self', False)
        # Obtaining the member 'jac_vector_vector' of a type (line 285)
        jac_vector_vector_243063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 19), self_243062, 'jac_vector_vector')
        # Calling jac_vector_vector(args, kwargs) (line 285)
        jac_vector_vector_call_result_243066 = invoke(stypy.reporting.localization.Localization(__file__, 285, 19), jac_vector_vector_243063, *[x0_243064], **kwargs_243065)
        
        # Assigning a type to the variable 'jac_true' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'jac_true', jac_vector_vector_call_result_243066)
        
        # Call to assert_allclose(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'jac_diff' (line 286)
        jac_diff_243068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 24), 'jac_diff', False)
        # Getting the type of 'jac_true' (line 286)
        jac_true_243069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 34), 'jac_true', False)
        # Processing the call keyword arguments (line 286)
        float_243070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 49), 'float')
        keyword_243071 = float_243070
        kwargs_243072 = {'rtol': keyword_243071}
        # Getting the type of 'assert_allclose' (line 286)
        assert_allclose_243067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 286)
        assert_allclose_call_result_243073 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), assert_allclose_243067, *[jac_diff_243068, jac_true_243069], **kwargs_243072)
        
        
        # ################# End of 'test_with_bounds_2_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_with_bounds_2_point' in the type store
        # Getting the type of 'stypy_return_type' (line 274)
        stypy_return_type_243074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243074)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_with_bounds_2_point'
        return stypy_return_type_243074


    @norecursion
    def test_with_bounds_3_point(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_with_bounds_3_point'
        module_type_store = module_type_store.open_function_context('test_with_bounds_3_point', 288, 4, False)
        # Assigning a type to the variable 'self' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.test_with_bounds_3_point.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.test_with_bounds_3_point.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.test_with_bounds_3_point.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.test_with_bounds_3_point.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.test_with_bounds_3_point')
        TestApproxDerivativesDense.test_with_bounds_3_point.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativesDense.test_with_bounds_3_point.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.test_with_bounds_3_point.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.test_with_bounds_3_point.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.test_with_bounds_3_point.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.test_with_bounds_3_point.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.test_with_bounds_3_point.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.test_with_bounds_3_point', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_with_bounds_3_point', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_with_bounds_3_point(...)' code ##################

        
        # Assigning a Call to a Name (line 289):
        
        # Assigning a Call to a Name (line 289):
        
        # Call to array(...): (line 289)
        # Processing the call arguments (line 289)
        
        # Obtaining an instance of the builtin type 'list' (line 289)
        list_243077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 289)
        # Adding element type (line 289)
        float_243078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 22), list_243077, float_243078)
        # Adding element type (line 289)
        float_243079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 22), list_243077, float_243079)
        
        # Processing the call keyword arguments (line 289)
        kwargs_243080 = {}
        # Getting the type of 'np' (line 289)
        np_243075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 289)
        array_243076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 13), np_243075, 'array')
        # Calling array(args, kwargs) (line 289)
        array_call_result_243081 = invoke(stypy.reporting.localization.Localization(__file__, 289, 13), array_243076, *[list_243077], **kwargs_243080)
        
        # Assigning a type to the variable 'lb' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'lb', array_call_result_243081)
        
        # Assigning a Call to a Name (line 290):
        
        # Assigning a Call to a Name (line 290):
        
        # Call to array(...): (line 290)
        # Processing the call arguments (line 290)
        
        # Obtaining an instance of the builtin type 'list' (line 290)
        list_243084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 290)
        # Adding element type (line 290)
        float_243085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 22), list_243084, float_243085)
        # Adding element type (line 290)
        float_243086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 22), list_243084, float_243086)
        
        # Processing the call keyword arguments (line 290)
        kwargs_243087 = {}
        # Getting the type of 'np' (line 290)
        np_243082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 290)
        array_243083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 13), np_243082, 'array')
        # Calling array(args, kwargs) (line 290)
        array_call_result_243088 = invoke(stypy.reporting.localization.Localization(__file__, 290, 13), array_243083, *[list_243084], **kwargs_243087)
        
        # Assigning a type to the variable 'ub' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'ub', array_call_result_243088)
        
        # Assigning a Call to a Name (line 292):
        
        # Assigning a Call to a Name (line 292):
        
        # Call to array(...): (line 292)
        # Processing the call arguments (line 292)
        
        # Obtaining an instance of the builtin type 'list' (line 292)
        list_243091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 292)
        # Adding element type (line 292)
        float_243092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_243091, float_243092)
        # Adding element type (line 292)
        float_243093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_243091, float_243093)
        
        # Processing the call keyword arguments (line 292)
        kwargs_243094 = {}
        # Getting the type of 'np' (line 292)
        np_243089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 292)
        array_243090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 13), np_243089, 'array')
        # Calling array(args, kwargs) (line 292)
        array_call_result_243095 = invoke(stypy.reporting.localization.Localization(__file__, 292, 13), array_243090, *[list_243091], **kwargs_243094)
        
        # Assigning a type to the variable 'x0' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'x0', array_call_result_243095)
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to jac_vector_vector(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'x0' (line 293)
        x0_243098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 42), 'x0', False)
        # Processing the call keyword arguments (line 293)
        kwargs_243099 = {}
        # Getting the type of 'self' (line 293)
        self_243096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 19), 'self', False)
        # Obtaining the member 'jac_vector_vector' of a type (line 293)
        jac_vector_vector_243097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 19), self_243096, 'jac_vector_vector')
        # Calling jac_vector_vector(args, kwargs) (line 293)
        jac_vector_vector_call_result_243100 = invoke(stypy.reporting.localization.Localization(__file__, 293, 19), jac_vector_vector_243097, *[x0_243098], **kwargs_243099)
        
        # Assigning a type to the variable 'jac_true' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'jac_true', jac_vector_vector_call_result_243100)
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to approx_derivative(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'self' (line 295)
        self_243102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 37), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 295)
        fun_vector_vector_243103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 37), self_243102, 'fun_vector_vector')
        # Getting the type of 'x0' (line 295)
        x0_243104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 61), 'x0', False)
        # Processing the call keyword arguments (line 295)
        kwargs_243105 = {}
        # Getting the type of 'approx_derivative' (line 295)
        approx_derivative_243101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 19), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 295)
        approx_derivative_call_result_243106 = invoke(stypy.reporting.localization.Localization(__file__, 295, 19), approx_derivative_243101, *[fun_vector_vector_243103, x0_243104], **kwargs_243105)
        
        # Assigning a type to the variable 'jac_diff' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'jac_diff', approx_derivative_call_result_243106)
        
        # Call to assert_allclose(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'jac_diff' (line 296)
        jac_diff_243108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 24), 'jac_diff', False)
        # Getting the type of 'jac_true' (line 296)
        jac_true_243109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 34), 'jac_true', False)
        # Processing the call keyword arguments (line 296)
        float_243110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 49), 'float')
        keyword_243111 = float_243110
        kwargs_243112 = {'rtol': keyword_243111}
        # Getting the type of 'assert_allclose' (line 296)
        assert_allclose_243107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 296)
        assert_allclose_call_result_243113 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), assert_allclose_243107, *[jac_diff_243108, jac_true_243109], **kwargs_243112)
        
        
        # Assigning a Call to a Name (line 298):
        
        # Assigning a Call to a Name (line 298):
        
        # Call to approx_derivative(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'self' (line 298)
        self_243115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 37), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 298)
        fun_vector_vector_243116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 37), self_243115, 'fun_vector_vector')
        # Getting the type of 'x0' (line 298)
        x0_243117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 61), 'x0', False)
        # Processing the call keyword arguments (line 298)
        
        # Obtaining an instance of the builtin type 'tuple' (line 299)
        tuple_243118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 299)
        # Adding element type (line 299)
        # Getting the type of 'lb' (line 299)
        lb_243119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 45), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 45), tuple_243118, lb_243119)
        # Adding element type (line 299)
        # Getting the type of 'np' (line 299)
        np_243120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 49), 'np', False)
        # Obtaining the member 'inf' of a type (line 299)
        inf_243121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 49), np_243120, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 45), tuple_243118, inf_243121)
        
        keyword_243122 = tuple_243118
        kwargs_243123 = {'bounds': keyword_243122}
        # Getting the type of 'approx_derivative' (line 298)
        approx_derivative_243114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 19), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 298)
        approx_derivative_call_result_243124 = invoke(stypy.reporting.localization.Localization(__file__, 298, 19), approx_derivative_243114, *[fun_vector_vector_243116, x0_243117], **kwargs_243123)
        
        # Assigning a type to the variable 'jac_diff' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'jac_diff', approx_derivative_call_result_243124)
        
        # Call to assert_allclose(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'jac_diff' (line 300)
        jac_diff_243126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 24), 'jac_diff', False)
        # Getting the type of 'jac_true' (line 300)
        jac_true_243127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 34), 'jac_true', False)
        # Processing the call keyword arguments (line 300)
        float_243128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 49), 'float')
        keyword_243129 = float_243128
        kwargs_243130 = {'rtol': keyword_243129}
        # Getting the type of 'assert_allclose' (line 300)
        assert_allclose_243125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 300)
        assert_allclose_call_result_243131 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), assert_allclose_243125, *[jac_diff_243126, jac_true_243127], **kwargs_243130)
        
        
        # Assigning a Call to a Name (line 302):
        
        # Assigning a Call to a Name (line 302):
        
        # Call to approx_derivative(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'self' (line 302)
        self_243133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 37), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 302)
        fun_vector_vector_243134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 37), self_243133, 'fun_vector_vector')
        # Getting the type of 'x0' (line 302)
        x0_243135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 61), 'x0', False)
        # Processing the call keyword arguments (line 302)
        
        # Obtaining an instance of the builtin type 'tuple' (line 303)
        tuple_243136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 303)
        # Adding element type (line 303)
        
        # Getting the type of 'np' (line 303)
        np_243137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 46), 'np', False)
        # Obtaining the member 'inf' of a type (line 303)
        inf_243138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 46), np_243137, 'inf')
        # Applying the 'usub' unary operator (line 303)
        result___neg___243139 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 45), 'usub', inf_243138)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 45), tuple_243136, result___neg___243139)
        # Adding element type (line 303)
        # Getting the type of 'ub' (line 303)
        ub_243140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 54), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 45), tuple_243136, ub_243140)
        
        keyword_243141 = tuple_243136
        kwargs_243142 = {'bounds': keyword_243141}
        # Getting the type of 'approx_derivative' (line 302)
        approx_derivative_243132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 19), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 302)
        approx_derivative_call_result_243143 = invoke(stypy.reporting.localization.Localization(__file__, 302, 19), approx_derivative_243132, *[fun_vector_vector_243134, x0_243135], **kwargs_243142)
        
        # Assigning a type to the variable 'jac_diff' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'jac_diff', approx_derivative_call_result_243143)
        
        # Call to assert_allclose(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'jac_diff' (line 304)
        jac_diff_243145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 24), 'jac_diff', False)
        # Getting the type of 'jac_true' (line 304)
        jac_true_243146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 34), 'jac_true', False)
        # Processing the call keyword arguments (line 304)
        float_243147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 49), 'float')
        keyword_243148 = float_243147
        kwargs_243149 = {'rtol': keyword_243148}
        # Getting the type of 'assert_allclose' (line 304)
        assert_allclose_243144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 304)
        assert_allclose_call_result_243150 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), assert_allclose_243144, *[jac_diff_243145, jac_true_243146], **kwargs_243149)
        
        
        # Assigning a Call to a Name (line 306):
        
        # Assigning a Call to a Name (line 306):
        
        # Call to approx_derivative(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'self' (line 306)
        self_243152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 37), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 306)
        fun_vector_vector_243153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 37), self_243152, 'fun_vector_vector')
        # Getting the type of 'x0' (line 306)
        x0_243154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 61), 'x0', False)
        # Processing the call keyword arguments (line 306)
        
        # Obtaining an instance of the builtin type 'tuple' (line 307)
        tuple_243155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 307)
        # Adding element type (line 307)
        # Getting the type of 'lb' (line 307)
        lb_243156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 45), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 45), tuple_243155, lb_243156)
        # Adding element type (line 307)
        # Getting the type of 'ub' (line 307)
        ub_243157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 49), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 45), tuple_243155, ub_243157)
        
        keyword_243158 = tuple_243155
        kwargs_243159 = {'bounds': keyword_243158}
        # Getting the type of 'approx_derivative' (line 306)
        approx_derivative_243151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 306)
        approx_derivative_call_result_243160 = invoke(stypy.reporting.localization.Localization(__file__, 306, 19), approx_derivative_243151, *[fun_vector_vector_243153, x0_243154], **kwargs_243159)
        
        # Assigning a type to the variable 'jac_diff' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'jac_diff', approx_derivative_call_result_243160)
        
        # Call to assert_allclose(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'jac_diff' (line 308)
        jac_diff_243162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 24), 'jac_diff', False)
        # Getting the type of 'jac_true' (line 308)
        jac_true_243163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 34), 'jac_true', False)
        # Processing the call keyword arguments (line 308)
        float_243164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 49), 'float')
        keyword_243165 = float_243164
        kwargs_243166 = {'rtol': keyword_243165}
        # Getting the type of 'assert_allclose' (line 308)
        assert_allclose_243161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 308)
        assert_allclose_call_result_243167 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), assert_allclose_243161, *[jac_diff_243162, jac_true_243163], **kwargs_243166)
        
        
        # ################# End of 'test_with_bounds_3_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_with_bounds_3_point' in the type store
        # Getting the type of 'stypy_return_type' (line 288)
        stypy_return_type_243168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243168)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_with_bounds_3_point'
        return stypy_return_type_243168


    @norecursion
    def test_tight_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tight_bounds'
        module_type_store = module_type_store.open_function_context('test_tight_bounds', 310, 4, False)
        # Assigning a type to the variable 'self' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.test_tight_bounds.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.test_tight_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.test_tight_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.test_tight_bounds.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.test_tight_bounds')
        TestApproxDerivativesDense.test_tight_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativesDense.test_tight_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.test_tight_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.test_tight_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.test_tight_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.test_tight_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.test_tight_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.test_tight_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tight_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tight_bounds(...)' code ##################

        
        # Assigning a Call to a Name (line 311):
        
        # Assigning a Call to a Name (line 311):
        
        # Call to array(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_243171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        # Adding element type (line 311)
        float_243172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 22), list_243171, float_243172)
        # Adding element type (line 311)
        float_243173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 22), list_243171, float_243173)
        
        # Processing the call keyword arguments (line 311)
        kwargs_243174 = {}
        # Getting the type of 'np' (line 311)
        np_243169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 311)
        array_243170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 13), np_243169, 'array')
        # Calling array(args, kwargs) (line 311)
        array_call_result_243175 = invoke(stypy.reporting.localization.Localization(__file__, 311, 13), array_243170, *[list_243171], **kwargs_243174)
        
        # Assigning a type to the variable 'x0' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'x0', array_call_result_243175)
        
        # Assigning a BinOp to a Name (line 312):
        
        # Assigning a BinOp to a Name (line 312):
        # Getting the type of 'x0' (line 312)
        x0_243176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 13), 'x0')
        float_243177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 18), 'float')
        # Applying the binary operator '-' (line 312)
        result_sub_243178 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 13), '-', x0_243176, float_243177)
        
        # Assigning a type to the variable 'lb' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'lb', result_sub_243178)
        
        # Assigning a BinOp to a Name (line 313):
        
        # Assigning a BinOp to a Name (line 313):
        # Getting the type of 'x0' (line 313)
        x0_243179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'x0')
        float_243180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 18), 'float')
        # Applying the binary operator '+' (line 313)
        result_add_243181 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 13), '+', x0_243179, float_243180)
        
        # Assigning a type to the variable 'ub' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'ub', result_add_243181)
        
        # Assigning a Call to a Name (line 314):
        
        # Assigning a Call to a Name (line 314):
        
        # Call to jac_vector_vector(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'x0' (line 314)
        x0_243184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 42), 'x0', False)
        # Processing the call keyword arguments (line 314)
        kwargs_243185 = {}
        # Getting the type of 'self' (line 314)
        self_243182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 19), 'self', False)
        # Obtaining the member 'jac_vector_vector' of a type (line 314)
        jac_vector_vector_243183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 19), self_243182, 'jac_vector_vector')
        # Calling jac_vector_vector(args, kwargs) (line 314)
        jac_vector_vector_call_result_243186 = invoke(stypy.reporting.localization.Localization(__file__, 314, 19), jac_vector_vector_243183, *[x0_243184], **kwargs_243185)
        
        # Assigning a type to the variable 'jac_true' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'jac_true', jac_vector_vector_call_result_243186)
        
        # Assigning a Call to a Name (line 315):
        
        # Assigning a Call to a Name (line 315):
        
        # Call to approx_derivative(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'self' (line 316)
        self_243188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 316)
        fun_vector_vector_243189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 12), self_243188, 'fun_vector_vector')
        # Getting the type of 'x0' (line 316)
        x0_243190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 36), 'x0', False)
        # Processing the call keyword arguments (line 315)
        str_243191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 47), 'str', '2-point')
        keyword_243192 = str_243191
        
        # Obtaining an instance of the builtin type 'tuple' (line 316)
        tuple_243193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 66), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 316)
        # Adding element type (line 316)
        # Getting the type of 'lb' (line 316)
        lb_243194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 66), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 66), tuple_243193, lb_243194)
        # Adding element type (line 316)
        # Getting the type of 'ub' (line 316)
        ub_243195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 70), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 66), tuple_243193, ub_243195)
        
        keyword_243196 = tuple_243193
        kwargs_243197 = {'method': keyword_243192, 'bounds': keyword_243196}
        # Getting the type of 'approx_derivative' (line 315)
        approx_derivative_243187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 19), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 315)
        approx_derivative_call_result_243198 = invoke(stypy.reporting.localization.Localization(__file__, 315, 19), approx_derivative_243187, *[fun_vector_vector_243189, x0_243190], **kwargs_243197)
        
        # Assigning a type to the variable 'jac_diff' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'jac_diff', approx_derivative_call_result_243198)
        
        # Call to assert_allclose(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'jac_diff' (line 317)
        jac_diff_243200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 24), 'jac_diff', False)
        # Getting the type of 'jac_true' (line 317)
        jac_true_243201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 34), 'jac_true', False)
        # Processing the call keyword arguments (line 317)
        float_243202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 49), 'float')
        keyword_243203 = float_243202
        kwargs_243204 = {'rtol': keyword_243203}
        # Getting the type of 'assert_allclose' (line 317)
        assert_allclose_243199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 317)
        assert_allclose_call_result_243205 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), assert_allclose_243199, *[jac_diff_243200, jac_true_243201], **kwargs_243204)
        
        
        # Assigning a Call to a Name (line 318):
        
        # Assigning a Call to a Name (line 318):
        
        # Call to approx_derivative(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'self' (line 319)
        self_243207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 319)
        fun_vector_vector_243208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 12), self_243207, 'fun_vector_vector')
        # Getting the type of 'x0' (line 319)
        x0_243209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 36), 'x0', False)
        # Processing the call keyword arguments (line 318)
        str_243210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 47), 'str', '2-point')
        keyword_243211 = str_243210
        float_243212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 21), 'float')
        keyword_243213 = float_243212
        
        # Obtaining an instance of the builtin type 'tuple' (line 320)
        tuple_243214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 320)
        # Adding element type (line 320)
        # Getting the type of 'lb' (line 320)
        lb_243215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 35), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 35), tuple_243214, lb_243215)
        # Adding element type (line 320)
        # Getting the type of 'ub' (line 320)
        ub_243216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 39), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 35), tuple_243214, ub_243216)
        
        keyword_243217 = tuple_243214
        kwargs_243218 = {'method': keyword_243211, 'rel_step': keyword_243213, 'bounds': keyword_243217}
        # Getting the type of 'approx_derivative' (line 318)
        approx_derivative_243206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 19), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 318)
        approx_derivative_call_result_243219 = invoke(stypy.reporting.localization.Localization(__file__, 318, 19), approx_derivative_243206, *[fun_vector_vector_243208, x0_243209], **kwargs_243218)
        
        # Assigning a type to the variable 'jac_diff' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'jac_diff', approx_derivative_call_result_243219)
        
        # Call to assert_allclose(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'jac_diff' (line 321)
        jac_diff_243221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 24), 'jac_diff', False)
        # Getting the type of 'jac_true' (line 321)
        jac_true_243222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 34), 'jac_true', False)
        # Processing the call keyword arguments (line 321)
        float_243223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 49), 'float')
        keyword_243224 = float_243223
        kwargs_243225 = {'rtol': keyword_243224}
        # Getting the type of 'assert_allclose' (line 321)
        assert_allclose_243220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 321)
        assert_allclose_call_result_243226 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), assert_allclose_243220, *[jac_diff_243221, jac_true_243222], **kwargs_243225)
        
        
        # Assigning a Call to a Name (line 323):
        
        # Assigning a Call to a Name (line 323):
        
        # Call to approx_derivative(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'self' (line 324)
        self_243228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 324)
        fun_vector_vector_243229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), self_243228, 'fun_vector_vector')
        # Getting the type of 'x0' (line 324)
        x0_243230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 36), 'x0', False)
        # Processing the call keyword arguments (line 323)
        
        # Obtaining an instance of the builtin type 'tuple' (line 324)
        tuple_243231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 324)
        # Adding element type (line 324)
        # Getting the type of 'lb' (line 324)
        lb_243232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 48), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 48), tuple_243231, lb_243232)
        # Adding element type (line 324)
        # Getting the type of 'ub' (line 324)
        ub_243233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 52), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 48), tuple_243231, ub_243233)
        
        keyword_243234 = tuple_243231
        kwargs_243235 = {'bounds': keyword_243234}
        # Getting the type of 'approx_derivative' (line 323)
        approx_derivative_243227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 323)
        approx_derivative_call_result_243236 = invoke(stypy.reporting.localization.Localization(__file__, 323, 19), approx_derivative_243227, *[fun_vector_vector_243229, x0_243230], **kwargs_243235)
        
        # Assigning a type to the variable 'jac_diff' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'jac_diff', approx_derivative_call_result_243236)
        
        # Call to assert_allclose(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'jac_diff' (line 325)
        jac_diff_243238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 24), 'jac_diff', False)
        # Getting the type of 'jac_true' (line 325)
        jac_true_243239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 34), 'jac_true', False)
        # Processing the call keyword arguments (line 325)
        float_243240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 49), 'float')
        keyword_243241 = float_243240
        kwargs_243242 = {'rtol': keyword_243241}
        # Getting the type of 'assert_allclose' (line 325)
        assert_allclose_243237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 325)
        assert_allclose_call_result_243243 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), assert_allclose_243237, *[jac_diff_243238, jac_true_243239], **kwargs_243242)
        
        
        # Assigning a Call to a Name (line 326):
        
        # Assigning a Call to a Name (line 326):
        
        # Call to approx_derivative(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'self' (line 327)
        self_243245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 327)
        fun_vector_vector_243246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 12), self_243245, 'fun_vector_vector')
        # Getting the type of 'x0' (line 327)
        x0_243247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 36), 'x0', False)
        # Processing the call keyword arguments (line 326)
        float_243248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 49), 'float')
        keyword_243249 = float_243248
        
        # Obtaining an instance of the builtin type 'tuple' (line 327)
        tuple_243250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 327)
        # Adding element type (line 327)
        # Getting the type of 'lb' (line 327)
        lb_243251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 63), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 63), tuple_243250, lb_243251)
        # Adding element type (line 327)
        # Getting the type of 'ub' (line 327)
        ub_243252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 67), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 63), tuple_243250, ub_243252)
        
        keyword_243253 = tuple_243250
        kwargs_243254 = {'bounds': keyword_243253, 'rel_step': keyword_243249}
        # Getting the type of 'approx_derivative' (line 326)
        approx_derivative_243244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 326)
        approx_derivative_call_result_243255 = invoke(stypy.reporting.localization.Localization(__file__, 326, 19), approx_derivative_243244, *[fun_vector_vector_243246, x0_243247], **kwargs_243254)
        
        # Assigning a type to the variable 'jac_diff' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'jac_diff', approx_derivative_call_result_243255)
        
        # Call to assert_allclose(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'jac_true' (line 328)
        jac_true_243257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 24), 'jac_true', False)
        # Getting the type of 'jac_diff' (line 328)
        jac_diff_243258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 34), 'jac_diff', False)
        # Processing the call keyword arguments (line 328)
        float_243259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 49), 'float')
        keyword_243260 = float_243259
        kwargs_243261 = {'rtol': keyword_243260}
        # Getting the type of 'assert_allclose' (line 328)
        assert_allclose_243256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 328)
        assert_allclose_call_result_243262 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), assert_allclose_243256, *[jac_true_243257, jac_diff_243258], **kwargs_243261)
        
        
        # ################# End of 'test_tight_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tight_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 310)
        stypy_return_type_243263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243263)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tight_bounds'
        return stypy_return_type_243263


    @norecursion
    def test_bound_switches(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bound_switches'
        module_type_store = module_type_store.open_function_context('test_bound_switches', 330, 4, False)
        # Assigning a type to the variable 'self' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.test_bound_switches.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.test_bound_switches.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.test_bound_switches.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.test_bound_switches.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.test_bound_switches')
        TestApproxDerivativesDense.test_bound_switches.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativesDense.test_bound_switches.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.test_bound_switches.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.test_bound_switches.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.test_bound_switches.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.test_bound_switches.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.test_bound_switches.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.test_bound_switches', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bound_switches', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bound_switches(...)' code ##################

        
        # Assigning a Num to a Name (line 331):
        
        # Assigning a Num to a Name (line 331):
        float_243264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 13), 'float')
        # Assigning a type to the variable 'lb' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'lb', float_243264)
        
        # Assigning a Num to a Name (line 332):
        
        # Assigning a Num to a Name (line 332):
        float_243265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 13), 'float')
        # Assigning a type to the variable 'ub' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'ub', float_243265)
        
        # Assigning a Num to a Name (line 333):
        
        # Assigning a Num to a Name (line 333):
        float_243266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 13), 'float')
        # Assigning a type to the variable 'x0' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'x0', float_243266)
        
        # Assigning a Call to a Name (line 334):
        
        # Assigning a Call to a Name (line 334):
        
        # Call to jac_with_nan(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'x0' (line 334)
        x0_243269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 37), 'x0', False)
        # Processing the call keyword arguments (line 334)
        kwargs_243270 = {}
        # Getting the type of 'self' (line 334)
        self_243267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'self', False)
        # Obtaining the member 'jac_with_nan' of a type (line 334)
        jac_with_nan_243268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 19), self_243267, 'jac_with_nan')
        # Calling jac_with_nan(args, kwargs) (line 334)
        jac_with_nan_call_result_243271 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), jac_with_nan_243268, *[x0_243269], **kwargs_243270)
        
        # Assigning a type to the variable 'jac_true' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'jac_true', jac_with_nan_call_result_243271)
        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to approx_derivative(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'self' (line 336)
        self_243273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'self', False)
        # Obtaining the member 'fun_with_nan' of a type (line 336)
        fun_with_nan_243274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), self_243273, 'fun_with_nan')
        # Getting the type of 'x0' (line 336)
        x0_243275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 31), 'x0', False)
        # Processing the call keyword arguments (line 335)
        str_243276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 42), 'str', '2-point')
        keyword_243277 = str_243276
        float_243278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 62), 'float')
        keyword_243279 = float_243278
        
        # Obtaining an instance of the builtin type 'tuple' (line 337)
        tuple_243280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 337)
        # Adding element type (line 337)
        # Getting the type of 'lb' (line 337)
        lb_243281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 20), tuple_243280, lb_243281)
        # Adding element type (line 337)
        # Getting the type of 'ub' (line 337)
        ub_243282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 24), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 20), tuple_243280, ub_243282)
        
        keyword_243283 = tuple_243280
        kwargs_243284 = {'method': keyword_243277, 'rel_step': keyword_243279, 'bounds': keyword_243283}
        # Getting the type of 'approx_derivative' (line 335)
        approx_derivative_243272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 335)
        approx_derivative_call_result_243285 = invoke(stypy.reporting.localization.Localization(__file__, 335, 21), approx_derivative_243272, *[fun_with_nan_243274, x0_243275], **kwargs_243284)
        
        # Assigning a type to the variable 'jac_diff_2' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'jac_diff_2', approx_derivative_call_result_243285)
        
        # Assigning a Call to a Name (line 338):
        
        # Assigning a Call to a Name (line 338):
        
        # Call to approx_derivative(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'self' (line 339)
        self_243287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'self', False)
        # Obtaining the member 'fun_with_nan' of a type (line 339)
        fun_with_nan_243288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 12), self_243287, 'fun_with_nan')
        # Getting the type of 'x0' (line 339)
        x0_243289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 'x0', False)
        # Processing the call keyword arguments (line 338)
        float_243290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 44), 'float')
        keyword_243291 = float_243290
        
        # Obtaining an instance of the builtin type 'tuple' (line 339)
        tuple_243292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 339)
        # Adding element type (line 339)
        # Getting the type of 'lb' (line 339)
        lb_243293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 58), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 58), tuple_243292, lb_243293)
        # Adding element type (line 339)
        # Getting the type of 'ub' (line 339)
        ub_243294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 62), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 58), tuple_243292, ub_243294)
        
        keyword_243295 = tuple_243292
        kwargs_243296 = {'bounds': keyword_243295, 'rel_step': keyword_243291}
        # Getting the type of 'approx_derivative' (line 338)
        approx_derivative_243286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 338)
        approx_derivative_call_result_243297 = invoke(stypy.reporting.localization.Localization(__file__, 338, 21), approx_derivative_243286, *[fun_with_nan_243288, x0_243289], **kwargs_243296)
        
        # Assigning a type to the variable 'jac_diff_3' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'jac_diff_3', approx_derivative_call_result_243297)
        
        # Call to assert_allclose(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'jac_diff_2' (line 340)
        jac_diff_2_243299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 24), 'jac_diff_2', False)
        # Getting the type of 'jac_true' (line 340)
        jac_true_243300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 340)
        float_243301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 51), 'float')
        keyword_243302 = float_243301
        kwargs_243303 = {'rtol': keyword_243302}
        # Getting the type of 'assert_allclose' (line 340)
        assert_allclose_243298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 340)
        assert_allclose_call_result_243304 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), assert_allclose_243298, *[jac_diff_2_243299, jac_true_243300], **kwargs_243303)
        
        
        # Call to assert_allclose(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'jac_diff_3' (line 341)
        jac_diff_3_243306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 24), 'jac_diff_3', False)
        # Getting the type of 'jac_true' (line 341)
        jac_true_243307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 341)
        float_243308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 51), 'float')
        keyword_243309 = float_243308
        kwargs_243310 = {'rtol': keyword_243309}
        # Getting the type of 'assert_allclose' (line 341)
        assert_allclose_243305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 341)
        assert_allclose_call_result_243311 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), assert_allclose_243305, *[jac_diff_3_243306, jac_true_243307], **kwargs_243310)
        
        
        # Assigning a Num to a Name (line 343):
        
        # Assigning a Num to a Name (line 343):
        float_243312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 13), 'float')
        # Assigning a type to the variable 'x0' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'x0', float_243312)
        
        # Assigning a Call to a Name (line 344):
        
        # Assigning a Call to a Name (line 344):
        
        # Call to jac_with_nan(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'x0' (line 344)
        x0_243315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 37), 'x0', False)
        # Processing the call keyword arguments (line 344)
        kwargs_243316 = {}
        # Getting the type of 'self' (line 344)
        self_243313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 19), 'self', False)
        # Obtaining the member 'jac_with_nan' of a type (line 344)
        jac_with_nan_243314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 19), self_243313, 'jac_with_nan')
        # Calling jac_with_nan(args, kwargs) (line 344)
        jac_with_nan_call_result_243317 = invoke(stypy.reporting.localization.Localization(__file__, 344, 19), jac_with_nan_243314, *[x0_243315], **kwargs_243316)
        
        # Assigning a type to the variable 'jac_true' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'jac_true', jac_with_nan_call_result_243317)
        
        # Assigning a Call to a Name (line 345):
        
        # Assigning a Call to a Name (line 345):
        
        # Call to approx_derivative(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'self' (line 346)
        self_243319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'self', False)
        # Obtaining the member 'fun_with_nan' of a type (line 346)
        fun_with_nan_243320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 12), self_243319, 'fun_with_nan')
        # Getting the type of 'x0' (line 346)
        x0_243321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 31), 'x0', False)
        # Processing the call keyword arguments (line 345)
        str_243322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 42), 'str', '2-point')
        keyword_243323 = str_243322
        float_243324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 62), 'float')
        keyword_243325 = float_243324
        
        # Obtaining an instance of the builtin type 'tuple' (line 347)
        tuple_243326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 347)
        # Adding element type (line 347)
        # Getting the type of 'lb' (line 347)
        lb_243327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 20), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 20), tuple_243326, lb_243327)
        # Adding element type (line 347)
        # Getting the type of 'ub' (line 347)
        ub_243328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 24), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 20), tuple_243326, ub_243328)
        
        keyword_243329 = tuple_243326
        kwargs_243330 = {'method': keyword_243323, 'rel_step': keyword_243325, 'bounds': keyword_243329}
        # Getting the type of 'approx_derivative' (line 345)
        approx_derivative_243318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 345)
        approx_derivative_call_result_243331 = invoke(stypy.reporting.localization.Localization(__file__, 345, 21), approx_derivative_243318, *[fun_with_nan_243320, x0_243321], **kwargs_243330)
        
        # Assigning a type to the variable 'jac_diff_2' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'jac_diff_2', approx_derivative_call_result_243331)
        
        # Assigning a Call to a Name (line 348):
        
        # Assigning a Call to a Name (line 348):
        
        # Call to approx_derivative(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'self' (line 349)
        self_243333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'self', False)
        # Obtaining the member 'fun_with_nan' of a type (line 349)
        fun_with_nan_243334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 12), self_243333, 'fun_with_nan')
        # Getting the type of 'x0' (line 349)
        x0_243335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 31), 'x0', False)
        # Processing the call keyword arguments (line 348)
        float_243336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 44), 'float')
        keyword_243337 = float_243336
        
        # Obtaining an instance of the builtin type 'tuple' (line 349)
        tuple_243338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 349)
        # Adding element type (line 349)
        # Getting the type of 'lb' (line 349)
        lb_243339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 58), 'lb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 58), tuple_243338, lb_243339)
        # Adding element type (line 349)
        # Getting the type of 'ub' (line 349)
        ub_243340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 62), 'ub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 58), tuple_243338, ub_243340)
        
        keyword_243341 = tuple_243338
        kwargs_243342 = {'bounds': keyword_243341, 'rel_step': keyword_243337}
        # Getting the type of 'approx_derivative' (line 348)
        approx_derivative_243332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 348)
        approx_derivative_call_result_243343 = invoke(stypy.reporting.localization.Localization(__file__, 348, 21), approx_derivative_243332, *[fun_with_nan_243334, x0_243335], **kwargs_243342)
        
        # Assigning a type to the variable 'jac_diff_3' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'jac_diff_3', approx_derivative_call_result_243343)
        
        # Call to assert_allclose(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'jac_diff_2' (line 350)
        jac_diff_2_243345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 24), 'jac_diff_2', False)
        # Getting the type of 'jac_true' (line 350)
        jac_true_243346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 350)
        float_243347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 51), 'float')
        keyword_243348 = float_243347
        kwargs_243349 = {'rtol': keyword_243348}
        # Getting the type of 'assert_allclose' (line 350)
        assert_allclose_243344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 350)
        assert_allclose_call_result_243350 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), assert_allclose_243344, *[jac_diff_2_243345, jac_true_243346], **kwargs_243349)
        
        
        # Call to assert_allclose(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'jac_diff_3' (line 351)
        jac_diff_3_243352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'jac_diff_3', False)
        # Getting the type of 'jac_true' (line 351)
        jac_true_243353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 351)
        float_243354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 51), 'float')
        keyword_243355 = float_243354
        kwargs_243356 = {'rtol': keyword_243355}
        # Getting the type of 'assert_allclose' (line 351)
        assert_allclose_243351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 351)
        assert_allclose_call_result_243357 = invoke(stypy.reporting.localization.Localization(__file__, 351, 8), assert_allclose_243351, *[jac_diff_3_243352, jac_true_243353], **kwargs_243356)
        
        
        # ################# End of 'test_bound_switches(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bound_switches' in the type store
        # Getting the type of 'stypy_return_type' (line 330)
        stypy_return_type_243358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243358)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bound_switches'
        return stypy_return_type_243358


    @norecursion
    def test_non_numpy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_non_numpy'
        module_type_store = module_type_store.open_function_context('test_non_numpy', 353, 4, False)
        # Assigning a type to the variable 'self' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.test_non_numpy.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.test_non_numpy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.test_non_numpy.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.test_non_numpy.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.test_non_numpy')
        TestApproxDerivativesDense.test_non_numpy.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativesDense.test_non_numpy.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.test_non_numpy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.test_non_numpy.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.test_non_numpy.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.test_non_numpy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.test_non_numpy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.test_non_numpy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_non_numpy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_non_numpy(...)' code ##################

        
        # Assigning a Num to a Name (line 354):
        
        # Assigning a Num to a Name (line 354):
        float_243359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 13), 'float')
        # Assigning a type to the variable 'x0' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'x0', float_243359)
        
        # Assigning a Call to a Name (line 355):
        
        # Assigning a Call to a Name (line 355):
        
        # Call to jac_non_numpy(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'x0' (line 355)
        x0_243362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 38), 'x0', False)
        # Processing the call keyword arguments (line 355)
        kwargs_243363 = {}
        # Getting the type of 'self' (line 355)
        self_243360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 19), 'self', False)
        # Obtaining the member 'jac_non_numpy' of a type (line 355)
        jac_non_numpy_243361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 19), self_243360, 'jac_non_numpy')
        # Calling jac_non_numpy(args, kwargs) (line 355)
        jac_non_numpy_call_result_243364 = invoke(stypy.reporting.localization.Localization(__file__, 355, 19), jac_non_numpy_243361, *[x0_243362], **kwargs_243363)
        
        # Assigning a type to the variable 'jac_true' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'jac_true', jac_non_numpy_call_result_243364)
        
        # Assigning a Call to a Name (line 356):
        
        # Assigning a Call to a Name (line 356):
        
        # Call to approx_derivative(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'self' (line 356)
        self_243366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 39), 'self', False)
        # Obtaining the member 'jac_non_numpy' of a type (line 356)
        jac_non_numpy_243367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 39), self_243366, 'jac_non_numpy')
        # Getting the type of 'x0' (line 356)
        x0_243368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 59), 'x0', False)
        # Processing the call keyword arguments (line 356)
        str_243369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 46), 'str', '2-point')
        keyword_243370 = str_243369
        kwargs_243371 = {'method': keyword_243370}
        # Getting the type of 'approx_derivative' (line 356)
        approx_derivative_243365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 356)
        approx_derivative_call_result_243372 = invoke(stypy.reporting.localization.Localization(__file__, 356, 21), approx_derivative_243365, *[jac_non_numpy_243367, x0_243368], **kwargs_243371)
        
        # Assigning a type to the variable 'jac_diff_2' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'jac_diff_2', approx_derivative_call_result_243372)
        
        # Assigning a Call to a Name (line 358):
        
        # Assigning a Call to a Name (line 358):
        
        # Call to approx_derivative(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'self' (line 358)
        self_243374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 39), 'self', False)
        # Obtaining the member 'jac_non_numpy' of a type (line 358)
        jac_non_numpy_243375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 39), self_243374, 'jac_non_numpy')
        # Getting the type of 'x0' (line 358)
        x0_243376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 59), 'x0', False)
        # Processing the call keyword arguments (line 358)
        kwargs_243377 = {}
        # Getting the type of 'approx_derivative' (line 358)
        approx_derivative_243373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 21), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 358)
        approx_derivative_call_result_243378 = invoke(stypy.reporting.localization.Localization(__file__, 358, 21), approx_derivative_243373, *[jac_non_numpy_243375, x0_243376], **kwargs_243377)
        
        # Assigning a type to the variable 'jac_diff_3' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'jac_diff_3', approx_derivative_call_result_243378)
        
        # Call to assert_allclose(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'jac_diff_2' (line 359)
        jac_diff_2_243380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 24), 'jac_diff_2', False)
        # Getting the type of 'jac_true' (line 359)
        jac_true_243381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 359)
        float_243382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 51), 'float')
        keyword_243383 = float_243382
        kwargs_243384 = {'rtol': keyword_243383}
        # Getting the type of 'assert_allclose' (line 359)
        assert_allclose_243379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 359)
        assert_allclose_call_result_243385 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), assert_allclose_243379, *[jac_diff_2_243380, jac_true_243381], **kwargs_243384)
        
        
        # Call to assert_allclose(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'jac_diff_3' (line 360)
        jac_diff_3_243387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 24), 'jac_diff_3', False)
        # Getting the type of 'jac_true' (line 360)
        jac_true_243388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 36), 'jac_true', False)
        # Processing the call keyword arguments (line 360)
        float_243389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 51), 'float')
        keyword_243390 = float_243389
        kwargs_243391 = {'rtol': keyword_243390}
        # Getting the type of 'assert_allclose' (line 360)
        assert_allclose_243386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 360)
        assert_allclose_call_result_243392 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), assert_allclose_243386, *[jac_diff_3_243387, jac_true_243388], **kwargs_243391)
        
        
        # Call to assert_raises(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'TypeError' (line 363)
        TypeError_243394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 22), 'TypeError', False)
        # Getting the type of 'approx_derivative' (line 363)
        approx_derivative_243395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 33), 'approx_derivative', False)
        # Getting the type of 'self' (line 363)
        self_243396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 52), 'self', False)
        # Obtaining the member 'jac_non_numpy' of a type (line 363)
        jac_non_numpy_243397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 52), self_243396, 'jac_non_numpy')
        # Getting the type of 'x0' (line 363)
        x0_243398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 72), 'x0', False)
        # Processing the call keyword arguments (line 363)
        
        # Call to dict(...): (line 364)
        # Processing the call keyword arguments (line 364)
        str_243400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 53), 'str', 'cs')
        keyword_243401 = str_243400
        kwargs_243402 = {'method': keyword_243401}
        # Getting the type of 'dict' (line 364)
        dict_243399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 41), 'dict', False)
        # Calling dict(args, kwargs) (line 364)
        dict_call_result_243403 = invoke(stypy.reporting.localization.Localization(__file__, 364, 41), dict_243399, *[], **kwargs_243402)
        
        kwargs_243404 = {'dict_call_result_243403': dict_call_result_243403}
        # Getting the type of 'assert_raises' (line 363)
        assert_raises_243393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 363)
        assert_raises_call_result_243405 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), assert_raises_243393, *[TypeError_243394, approx_derivative_243395, jac_non_numpy_243397, x0_243398], **kwargs_243404)
        
        
        # ################# End of 'test_non_numpy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_non_numpy' in the type store
        # Getting the type of 'stypy_return_type' (line 353)
        stypy_return_type_243406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243406)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_non_numpy'
        return stypy_return_type_243406


    @norecursion
    def test_check_derivative(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_check_derivative'
        module_type_store = module_type_store.open_function_context('test_check_derivative', 366, 4, False)
        # Assigning a type to the variable 'self' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativesDense.test_check_derivative.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativesDense.test_check_derivative.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativesDense.test_check_derivative.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativesDense.test_check_derivative.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativesDense.test_check_derivative')
        TestApproxDerivativesDense.test_check_derivative.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativesDense.test_check_derivative.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativesDense.test_check_derivative.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativesDense.test_check_derivative.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativesDense.test_check_derivative.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativesDense.test_check_derivative.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativesDense.test_check_derivative.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.test_check_derivative', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_check_derivative', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_check_derivative(...)' code ##################

        
        # Assigning a Call to a Name (line 367):
        
        # Assigning a Call to a Name (line 367):
        
        # Call to array(...): (line 367)
        # Processing the call arguments (line 367)
        
        # Obtaining an instance of the builtin type 'list' (line 367)
        list_243409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 367)
        # Adding element type (line 367)
        float_243410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 22), list_243409, float_243410)
        # Adding element type (line 367)
        int_243411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 22), list_243409, int_243411)
        
        # Processing the call keyword arguments (line 367)
        kwargs_243412 = {}
        # Getting the type of 'np' (line 367)
        np_243407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 367)
        array_243408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 13), np_243407, 'array')
        # Calling array(args, kwargs) (line 367)
        array_call_result_243413 = invoke(stypy.reporting.localization.Localization(__file__, 367, 13), array_243408, *[list_243409], **kwargs_243412)
        
        # Assigning a type to the variable 'x0' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'x0', array_call_result_243413)
        
        # Assigning a Call to a Name (line 368):
        
        # Assigning a Call to a Name (line 368):
        
        # Call to check_derivative(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'self' (line 368)
        self_243415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 36), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 368)
        fun_vector_vector_243416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 36), self_243415, 'fun_vector_vector')
        # Getting the type of 'self' (line 369)
        self_243417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 36), 'self', False)
        # Obtaining the member 'jac_vector_vector' of a type (line 369)
        jac_vector_vector_243418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 36), self_243417, 'jac_vector_vector')
        # Getting the type of 'x0' (line 369)
        x0_243419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 60), 'x0', False)
        # Processing the call keyword arguments (line 368)
        kwargs_243420 = {}
        # Getting the type of 'check_derivative' (line 368)
        check_derivative_243414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 19), 'check_derivative', False)
        # Calling check_derivative(args, kwargs) (line 368)
        check_derivative_call_result_243421 = invoke(stypy.reporting.localization.Localization(__file__, 368, 19), check_derivative_243414, *[fun_vector_vector_243416, jac_vector_vector_243418, x0_243419], **kwargs_243420)
        
        # Assigning a type to the variable 'accuracy' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'accuracy', check_derivative_call_result_243421)
        
        # Call to assert_(...): (line 370)
        # Processing the call arguments (line 370)
        
        # Getting the type of 'accuracy' (line 370)
        accuracy_243423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'accuracy', False)
        float_243424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 27), 'float')
        # Applying the binary operator '<' (line 370)
        result_lt_243425 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 16), '<', accuracy_243423, float_243424)
        
        # Processing the call keyword arguments (line 370)
        kwargs_243426 = {}
        # Getting the type of 'assert_' (line 370)
        assert__243422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 370)
        assert__call_result_243427 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), assert__243422, *[result_lt_243425], **kwargs_243426)
        
        
        # Assigning a Call to a Name (line 371):
        
        # Assigning a Call to a Name (line 371):
        
        # Call to check_derivative(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'self' (line 371)
        self_243429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 36), 'self', False)
        # Obtaining the member 'fun_vector_vector' of a type (line 371)
        fun_vector_vector_243430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 36), self_243429, 'fun_vector_vector')
        # Getting the type of 'self' (line 372)
        self_243431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 36), 'self', False)
        # Obtaining the member 'jac_vector_vector' of a type (line 372)
        jac_vector_vector_243432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 36), self_243431, 'jac_vector_vector')
        # Getting the type of 'x0' (line 372)
        x0_243433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 60), 'x0', False)
        # Processing the call keyword arguments (line 371)
        kwargs_243434 = {}
        # Getting the type of 'check_derivative' (line 371)
        check_derivative_243428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 19), 'check_derivative', False)
        # Calling check_derivative(args, kwargs) (line 371)
        check_derivative_call_result_243435 = invoke(stypy.reporting.localization.Localization(__file__, 371, 19), check_derivative_243428, *[fun_vector_vector_243430, jac_vector_vector_243432, x0_243433], **kwargs_243434)
        
        # Assigning a type to the variable 'accuracy' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'accuracy', check_derivative_call_result_243435)
        
        # Call to assert_(...): (line 373)
        # Processing the call arguments (line 373)
        
        # Getting the type of 'accuracy' (line 373)
        accuracy_243437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'accuracy', False)
        float_243438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 27), 'float')
        # Applying the binary operator '<' (line 373)
        result_lt_243439 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 16), '<', accuracy_243437, float_243438)
        
        # Processing the call keyword arguments (line 373)
        kwargs_243440 = {}
        # Getting the type of 'assert_' (line 373)
        assert__243436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 373)
        assert__call_result_243441 = invoke(stypy.reporting.localization.Localization(__file__, 373, 8), assert__243436, *[result_lt_243439], **kwargs_243440)
        
        
        # Assigning a Call to a Name (line 375):
        
        # Assigning a Call to a Name (line 375):
        
        # Call to array(...): (line 375)
        # Processing the call arguments (line 375)
        
        # Obtaining an instance of the builtin type 'list' (line 375)
        list_243444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 375)
        # Adding element type (line 375)
        float_243445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 22), list_243444, float_243445)
        # Adding element type (line 375)
        float_243446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 22), list_243444, float_243446)
        
        # Processing the call keyword arguments (line 375)
        kwargs_243447 = {}
        # Getting the type of 'np' (line 375)
        np_243442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 375)
        array_243443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 13), np_243442, 'array')
        # Calling array(args, kwargs) (line 375)
        array_call_result_243448 = invoke(stypy.reporting.localization.Localization(__file__, 375, 13), array_243443, *[list_243444], **kwargs_243447)
        
        # Assigning a type to the variable 'x0' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'x0', array_call_result_243448)
        
        # Assigning a Call to a Name (line 376):
        
        # Assigning a Call to a Name (line 376):
        
        # Call to check_derivative(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'self' (line 376)
        self_243450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 36), 'self', False)
        # Obtaining the member 'fun_zero_jacobian' of a type (line 376)
        fun_zero_jacobian_243451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 36), self_243450, 'fun_zero_jacobian')
        # Getting the type of 'self' (line 377)
        self_243452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 36), 'self', False)
        # Obtaining the member 'jac_zero_jacobian' of a type (line 377)
        jac_zero_jacobian_243453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 36), self_243452, 'jac_zero_jacobian')
        # Getting the type of 'x0' (line 377)
        x0_243454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 60), 'x0', False)
        # Processing the call keyword arguments (line 376)
        kwargs_243455 = {}
        # Getting the type of 'check_derivative' (line 376)
        check_derivative_243449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 'check_derivative', False)
        # Calling check_derivative(args, kwargs) (line 376)
        check_derivative_call_result_243456 = invoke(stypy.reporting.localization.Localization(__file__, 376, 19), check_derivative_243449, *[fun_zero_jacobian_243451, jac_zero_jacobian_243453, x0_243454], **kwargs_243455)
        
        # Assigning a type to the variable 'accuracy' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'accuracy', check_derivative_call_result_243456)
        
        # Call to assert_(...): (line 378)
        # Processing the call arguments (line 378)
        
        # Getting the type of 'accuracy' (line 378)
        accuracy_243458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 16), 'accuracy', False)
        int_243459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 28), 'int')
        # Applying the binary operator '==' (line 378)
        result_eq_243460 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 16), '==', accuracy_243458, int_243459)
        
        # Processing the call keyword arguments (line 378)
        kwargs_243461 = {}
        # Getting the type of 'assert_' (line 378)
        assert__243457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 378)
        assert__call_result_243462 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), assert__243457, *[result_eq_243460], **kwargs_243461)
        
        
        # Assigning a Call to a Name (line 379):
        
        # Assigning a Call to a Name (line 379):
        
        # Call to check_derivative(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'self' (line 379)
        self_243464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 36), 'self', False)
        # Obtaining the member 'fun_zero_jacobian' of a type (line 379)
        fun_zero_jacobian_243465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 36), self_243464, 'fun_zero_jacobian')
        # Getting the type of 'self' (line 380)
        self_243466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 36), 'self', False)
        # Obtaining the member 'jac_zero_jacobian' of a type (line 380)
        jac_zero_jacobian_243467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 36), self_243466, 'jac_zero_jacobian')
        # Getting the type of 'x0' (line 380)
        x0_243468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 60), 'x0', False)
        # Processing the call keyword arguments (line 379)
        kwargs_243469 = {}
        # Getting the type of 'check_derivative' (line 379)
        check_derivative_243463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 19), 'check_derivative', False)
        # Calling check_derivative(args, kwargs) (line 379)
        check_derivative_call_result_243470 = invoke(stypy.reporting.localization.Localization(__file__, 379, 19), check_derivative_243463, *[fun_zero_jacobian_243465, jac_zero_jacobian_243467, x0_243468], **kwargs_243469)
        
        # Assigning a type to the variable 'accuracy' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'accuracy', check_derivative_call_result_243470)
        
        # Call to assert_(...): (line 381)
        # Processing the call arguments (line 381)
        
        # Getting the type of 'accuracy' (line 381)
        accuracy_243472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'accuracy', False)
        int_243473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 28), 'int')
        # Applying the binary operator '==' (line 381)
        result_eq_243474 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 16), '==', accuracy_243472, int_243473)
        
        # Processing the call keyword arguments (line 381)
        kwargs_243475 = {}
        # Getting the type of 'assert_' (line 381)
        assert__243471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 381)
        assert__call_result_243476 = invoke(stypy.reporting.localization.Localization(__file__, 381, 8), assert__243471, *[result_eq_243474], **kwargs_243475)
        
        
        # ################# End of 'test_check_derivative(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_check_derivative' in the type store
        # Getting the type of 'stypy_return_type' (line 366)
        stypy_return_type_243477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243477)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_check_derivative'
        return stypy_return_type_243477


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 119, 0, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativesDense.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestApproxDerivativesDense' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'TestApproxDerivativesDense', TestApproxDerivativesDense)
# Declaration of the 'TestApproxDerivativeSparse' class

class TestApproxDerivativeSparse(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 386, 4, False)
        # Assigning a type to the variable 'self' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativeSparse.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativeSparse.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativeSparse.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativeSparse.setup_method.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativeSparse.setup_method')
        TestApproxDerivativeSparse.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativeSparse.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativeSparse.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativeSparse.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativeSparse.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativeSparse.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativeSparse.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativeSparse.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Call to seed(...): (line 387)
        # Processing the call arguments (line 387)
        int_243481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 23), 'int')
        # Processing the call keyword arguments (line 387)
        kwargs_243482 = {}
        # Getting the type of 'np' (line 387)
        np_243478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 387)
        random_243479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), np_243478, 'random')
        # Obtaining the member 'seed' of a type (line 387)
        seed_243480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), random_243479, 'seed')
        # Calling seed(args, kwargs) (line 387)
        seed_call_result_243483 = invoke(stypy.reporting.localization.Localization(__file__, 387, 8), seed_243480, *[int_243481], **kwargs_243482)
        
        
        # Assigning a Num to a Attribute (line 388):
        
        # Assigning a Num to a Attribute (line 388):
        int_243484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 17), 'int')
        # Getting the type of 'self' (line 388)
        self_243485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'self')
        # Setting the type of the member 'n' of a type (line 388)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), self_243485, 'n', int_243484)
        
        # Assigning a BinOp to a Attribute (line 389):
        
        # Assigning a BinOp to a Attribute (line 389):
        float_243486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 18), 'float')
        int_243487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 26), 'int')
        
        # Call to arange(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'self' (line 389)
        self_243490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 40), 'self', False)
        # Obtaining the member 'n' of a type (line 389)
        n_243491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 40), self_243490, 'n')
        # Processing the call keyword arguments (line 389)
        kwargs_243492 = {}
        # Getting the type of 'np' (line 389)
        np_243488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 30), 'np', False)
        # Obtaining the member 'arange' of a type (line 389)
        arange_243489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 30), np_243488, 'arange')
        # Calling arange(args, kwargs) (line 389)
        arange_call_result_243493 = invoke(stypy.reporting.localization.Localization(__file__, 389, 30), arange_243489, *[n_243491], **kwargs_243492)
        
        # Applying the binary operator '+' (line 389)
        result_add_243494 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 26), '+', int_243487, arange_call_result_243493)
        
        # Applying the binary operator '*' (line 389)
        result_mul_243495 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 18), '*', float_243486, result_add_243494)
        
        # Getting the type of 'self' (line 389)
        self_243496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'self')
        # Setting the type of the member 'lb' of a type (line 389)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), self_243496, 'lb', result_mul_243495)
        
        # Assigning a BinOp to a Attribute (line 390):
        
        # Assigning a BinOp to a Attribute (line 390):
        float_243497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 18), 'float')
        int_243498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 25), 'int')
        
        # Call to arange(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'self' (line 390)
        self_243501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 39), 'self', False)
        # Obtaining the member 'n' of a type (line 390)
        n_243502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 39), self_243501, 'n')
        # Processing the call keyword arguments (line 390)
        kwargs_243503 = {}
        # Getting the type of 'np' (line 390)
        np_243499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 29), 'np', False)
        # Obtaining the member 'arange' of a type (line 390)
        arange_243500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 29), np_243499, 'arange')
        # Calling arange(args, kwargs) (line 390)
        arange_call_result_243504 = invoke(stypy.reporting.localization.Localization(__file__, 390, 29), arange_243500, *[n_243502], **kwargs_243503)
        
        # Applying the binary operator '+' (line 390)
        result_add_243505 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 25), '+', int_243498, arange_call_result_243504)
        
        # Applying the binary operator '*' (line 390)
        result_mul_243506 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 18), '*', float_243497, result_add_243505)
        
        # Getting the type of 'self' (line 390)
        self_243507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'self')
        # Setting the type of the member 'ub' of a type (line 390)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), self_243507, 'ub', result_mul_243506)
        
        # Assigning a Call to a Attribute (line 391):
        
        # Assigning a Call to a Attribute (line 391):
        
        # Call to empty(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'self' (line 391)
        self_243510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 27), 'self', False)
        # Obtaining the member 'n' of a type (line 391)
        n_243511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 27), self_243510, 'n')
        # Processing the call keyword arguments (line 391)
        kwargs_243512 = {}
        # Getting the type of 'np' (line 391)
        np_243508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 18), 'np', False)
        # Obtaining the member 'empty' of a type (line 391)
        empty_243509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 18), np_243508, 'empty')
        # Calling empty(args, kwargs) (line 391)
        empty_call_result_243513 = invoke(stypy.reporting.localization.Localization(__file__, 391, 18), empty_243509, *[n_243511], **kwargs_243512)
        
        # Getting the type of 'self' (line 391)
        self_243514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'self')
        # Setting the type of the member 'x0' of a type (line 391)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), self_243514, 'x0', empty_call_result_243513)
        
        # Assigning a BinOp to a Subscript (line 392):
        
        # Assigning a BinOp to a Subscript (line 392):
        int_243515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 24), 'int')
        float_243516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 28), 'float')
        # Applying the binary operator '-' (line 392)
        result_sub_243517 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 24), '-', int_243515, float_243516)
        
        
        # Obtaining the type of the subscript
        int_243518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 46), 'int')
        slice_243519 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 392, 36), None, None, int_243518)
        # Getting the type of 'self' (line 392)
        self_243520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 36), 'self')
        # Obtaining the member 'lb' of a type (line 392)
        lb_243521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 36), self_243520, 'lb')
        # Obtaining the member '__getitem__' of a type (line 392)
        getitem___243522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 36), lb_243521, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 392)
        subscript_call_result_243523 = invoke(stypy.reporting.localization.Localization(__file__, 392, 36), getitem___243522, slice_243519)
        
        # Applying the binary operator '*' (line 392)
        result_mul_243524 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 23), '*', result_sub_243517, subscript_call_result_243523)
        
        # Getting the type of 'self' (line 392)
        self_243525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'self')
        # Obtaining the member 'x0' of a type (line 392)
        x0_243526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), self_243525, 'x0')
        int_243527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 18), 'int')
        slice_243528 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 392, 8), None, None, int_243527)
        # Storing an element on a container (line 392)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 8), x0_243526, (slice_243528, result_mul_243524))
        
        # Assigning a BinOp to a Subscript (line 393):
        
        # Assigning a BinOp to a Subscript (line 393):
        int_243529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 25), 'int')
        float_243530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 29), 'float')
        # Applying the binary operator '-' (line 393)
        result_sub_243531 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 25), '-', int_243529, float_243530)
        
        
        # Obtaining the type of the subscript
        int_243532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 45), 'int')
        int_243533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 48), 'int')
        slice_243534 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 393, 37), int_243532, None, int_243533)
        # Getting the type of 'self' (line 393)
        self_243535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 37), 'self')
        # Obtaining the member 'ub' of a type (line 393)
        ub_243536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 37), self_243535, 'ub')
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___243537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 37), ub_243536, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_243538 = invoke(stypy.reporting.localization.Localization(__file__, 393, 37), getitem___243537, slice_243534)
        
        # Applying the binary operator '*' (line 393)
        result_mul_243539 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 24), '*', result_sub_243531, subscript_call_result_243538)
        
        # Getting the type of 'self' (line 393)
        self_243540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'self')
        # Obtaining the member 'x0' of a type (line 393)
        x0_243541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), self_243540, 'x0')
        int_243542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 16), 'int')
        int_243543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 19), 'int')
        slice_243544 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 393, 8), int_243542, None, int_243543)
        # Storing an element on a container (line 393)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 8), x0_243541, (slice_243544, result_mul_243539))
        
        # Assigning a Call to a Attribute (line 395):
        
        # Assigning a Call to a Attribute (line 395):
        
        # Call to jac(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'self' (line 395)
        self_243547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 31), 'self', False)
        # Obtaining the member 'x0' of a type (line 395)
        x0_243548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 31), self_243547, 'x0')
        # Processing the call keyword arguments (line 395)
        kwargs_243549 = {}
        # Getting the type of 'self' (line 395)
        self_243545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 22), 'self', False)
        # Obtaining the member 'jac' of a type (line 395)
        jac_243546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 22), self_243545, 'jac')
        # Calling jac(args, kwargs) (line 395)
        jac_call_result_243550 = invoke(stypy.reporting.localization.Localization(__file__, 395, 22), jac_243546, *[x0_243548], **kwargs_243549)
        
        # Getting the type of 'self' (line 395)
        self_243551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'self')
        # Setting the type of the member 'J_true' of a type (line 395)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), self_243551, 'J_true', jac_call_result_243550)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 386)
        stypy_return_type_243552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_243552


    @norecursion
    def fun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun'
        module_type_store = module_type_store.open_function_context('fun', 397, 4, False)
        # Assigning a type to the variable 'self' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativeSparse.fun.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativeSparse.fun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativeSparse.fun.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativeSparse.fun.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativeSparse.fun')
        TestApproxDerivativeSparse.fun.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativeSparse.fun.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativeSparse.fun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativeSparse.fun.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativeSparse.fun.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativeSparse.fun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativeSparse.fun.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativeSparse.fun', ['x'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a BinOp to a Name (line 398):
        
        # Assigning a BinOp to a Name (line 398):
        
        # Obtaining the type of the subscript
        int_243553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 14), 'int')
        slice_243554 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 398, 12), int_243553, None, None)
        # Getting the type of 'x' (line 398)
        x_243555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'x')
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___243556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), x_243555, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 398)
        subscript_call_result_243557 = invoke(stypy.reporting.localization.Localization(__file__, 398, 12), getitem___243556, slice_243554)
        
        int_243558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 19), 'int')
        # Applying the binary operator '**' (line 398)
        result_pow_243559 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 12), '**', subscript_call_result_243557, int_243558)
        
        
        # Obtaining the type of the subscript
        int_243560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 26), 'int')
        slice_243561 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 398, 23), None, int_243560, None)
        # Getting the type of 'x' (line 398)
        x_243562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 23), 'x')
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___243563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 23), x_243562, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 398)
        subscript_call_result_243564 = invoke(stypy.reporting.localization.Localization(__file__, 398, 23), getitem___243563, slice_243561)
        
        int_243565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 31), 'int')
        # Applying the binary operator '**' (line 398)
        result_pow_243566 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 23), '**', subscript_call_result_243564, int_243565)
        
        # Applying the binary operator '-' (line 398)
        result_sub_243567 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 12), '-', result_pow_243559, result_pow_243566)
        
        # Assigning a type to the variable 'e' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'e', result_sub_243567)
        
        # Call to hstack(...): (line 399)
        # Processing the call arguments (line 399)
        
        # Obtaining an instance of the builtin type 'tuple' (line 399)
        tuple_243570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 399)
        # Adding element type (line 399)
        int_243571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 26), tuple_243570, int_243571)
        # Adding element type (line 399)
        int_243572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 29), 'int')
        # Getting the type of 'e' (line 399)
        e_243573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 33), 'e', False)
        # Applying the binary operator '*' (line 399)
        result_mul_243574 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 29), '*', int_243572, e_243573)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 26), tuple_243570, result_mul_243574)
        
        # Processing the call keyword arguments (line 399)
        kwargs_243575 = {}
        # Getting the type of 'np' (line 399)
        np_243568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 15), 'np', False)
        # Obtaining the member 'hstack' of a type (line 399)
        hstack_243569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 15), np_243568, 'hstack')
        # Calling hstack(args, kwargs) (line 399)
        hstack_call_result_243576 = invoke(stypy.reporting.localization.Localization(__file__, 399, 15), hstack_243569, *[tuple_243570], **kwargs_243575)
        
        
        # Call to hstack(...): (line 399)
        # Processing the call arguments (line 399)
        
        # Obtaining an instance of the builtin type 'tuple' (line 399)
        tuple_243579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 399)
        # Adding element type (line 399)
        int_243580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 50), 'int')
        # Getting the type of 'e' (line 399)
        e_243581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 54), 'e', False)
        # Applying the binary operator '*' (line 399)
        result_mul_243582 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 50), '*', int_243580, e_243581)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 50), tuple_243579, result_mul_243582)
        # Adding element type (line 399)
        int_243583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 50), tuple_243579, int_243583)
        
        # Processing the call keyword arguments (line 399)
        kwargs_243584 = {}
        # Getting the type of 'np' (line 399)
        np_243577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 39), 'np', False)
        # Obtaining the member 'hstack' of a type (line 399)
        hstack_243578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 39), np_243577, 'hstack')
        # Calling hstack(args, kwargs) (line 399)
        hstack_call_result_243585 = invoke(stypy.reporting.localization.Localization(__file__, 399, 39), hstack_243578, *[tuple_243579], **kwargs_243584)
        
        # Applying the binary operator '+' (line 399)
        result_add_243586 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 15), '+', hstack_call_result_243576, hstack_call_result_243585)
        
        # Assigning a type to the variable 'stypy_return_type' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'stypy_return_type', result_add_243586)
        
        # ################# End of 'fun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun' in the type store
        # Getting the type of 'stypy_return_type' (line 397)
        stypy_return_type_243587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243587)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun'
        return stypy_return_type_243587


    @norecursion
    def jac(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac'
        module_type_store = module_type_store.open_function_context('jac', 401, 4, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativeSparse.jac.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativeSparse.jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativeSparse.jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativeSparse.jac.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativeSparse.jac')
        TestApproxDerivativeSparse.jac.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestApproxDerivativeSparse.jac.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativeSparse.jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativeSparse.jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativeSparse.jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativeSparse.jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativeSparse.jac.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativeSparse.jac', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac(...)' code ##################

        
        # Assigning a Attribute to a Name (line 402):
        
        # Assigning a Attribute to a Name (line 402):
        # Getting the type of 'x' (line 402)
        x_243588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'x')
        # Obtaining the member 'size' of a type (line 402)
        size_243589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 12), x_243588, 'size')
        # Assigning a type to the variable 'n' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'n', size_243589)
        
        # Assigning a Call to a Name (line 403):
        
        # Assigning a Call to a Name (line 403):
        
        # Call to zeros(...): (line 403)
        # Processing the call arguments (line 403)
        
        # Obtaining an instance of the builtin type 'tuple' (line 403)
        tuple_243592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 403)
        # Adding element type (line 403)
        # Getting the type of 'n' (line 403)
        n_243593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 22), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 22), tuple_243592, n_243593)
        # Adding element type (line 403)
        # Getting the type of 'n' (line 403)
        n_243594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 25), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 22), tuple_243592, n_243594)
        
        # Processing the call keyword arguments (line 403)
        kwargs_243595 = {}
        # Getting the type of 'np' (line 403)
        np_243590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 403)
        zeros_243591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 12), np_243590, 'zeros')
        # Calling zeros(args, kwargs) (line 403)
        zeros_call_result_243596 = invoke(stypy.reporting.localization.Localization(__file__, 403, 12), zeros_243591, *[tuple_243592], **kwargs_243595)
        
        # Assigning a type to the variable 'J' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'J', zeros_call_result_243596)
        
        # Assigning a BinOp to a Subscript (line 404):
        
        # Assigning a BinOp to a Subscript (line 404):
        int_243597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 18), 'int')
        
        # Obtaining the type of the subscript
        int_243598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 25), 'int')
        # Getting the type of 'x' (line 404)
        x_243599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 23), 'x')
        # Obtaining the member '__getitem__' of a type (line 404)
        getitem___243600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 23), x_243599, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 404)
        subscript_call_result_243601 = invoke(stypy.reporting.localization.Localization(__file__, 404, 23), getitem___243600, int_243598)
        
        # Applying the binary operator '*' (line 404)
        result_mul_243602 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 18), '*', int_243597, subscript_call_result_243601)
        
        # Getting the type of 'J' (line 404)
        J_243603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'J')
        
        # Obtaining an instance of the builtin type 'tuple' (line 404)
        tuple_243604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 404)
        # Adding element type (line 404)
        int_243605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 10), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 10), tuple_243604, int_243605)
        # Adding element type (line 404)
        int_243606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 10), tuple_243604, int_243606)
        
        # Storing an element on a container (line 404)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 8), J_243603, (tuple_243604, result_mul_243602))
        
        # Assigning a BinOp to a Subscript (line 405):
        
        # Assigning a BinOp to a Subscript (line 405):
        int_243607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 18), 'int')
        
        # Obtaining the type of the subscript
        int_243608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 24), 'int')
        # Getting the type of 'x' (line 405)
        x_243609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 22), 'x')
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___243610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 22), x_243609, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 405)
        subscript_call_result_243611 = invoke(stypy.reporting.localization.Localization(__file__, 405, 22), getitem___243610, int_243608)
        
        int_243612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 28), 'int')
        # Applying the binary operator '**' (line 405)
        result_pow_243613 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 22), '**', subscript_call_result_243611, int_243612)
        
        # Applying the binary operator '*' (line 405)
        result_mul_243614 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 18), '*', int_243607, result_pow_243613)
        
        # Getting the type of 'J' (line 405)
        J_243615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'J')
        
        # Obtaining an instance of the builtin type 'tuple' (line 405)
        tuple_243616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 405)
        # Adding element type (line 405)
        int_243617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 10), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 10), tuple_243616, int_243617)
        # Adding element type (line 405)
        int_243618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 10), tuple_243616, int_243618)
        
        # Storing an element on a container (line 405)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 8), J_243615, (tuple_243616, result_mul_243614))
        
        
        # Call to range(...): (line 406)
        # Processing the call arguments (line 406)
        int_243620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 23), 'int')
        # Getting the type of 'n' (line 406)
        n_243621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 26), 'n', False)
        int_243622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 30), 'int')
        # Applying the binary operator '-' (line 406)
        result_sub_243623 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 26), '-', n_243621, int_243622)
        
        # Processing the call keyword arguments (line 406)
        kwargs_243624 = {}
        # Getting the type of 'range' (line 406)
        range_243619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 17), 'range', False)
        # Calling range(args, kwargs) (line 406)
        range_call_result_243625 = invoke(stypy.reporting.localization.Localization(__file__, 406, 17), range_243619, *[int_243620, result_sub_243623], **kwargs_243624)
        
        # Testing the type of a for loop iterable (line 406)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 406, 8), range_call_result_243625)
        # Getting the type of the for loop variable (line 406)
        for_loop_var_243626 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 406, 8), range_call_result_243625)
        # Assigning a type to the variable 'i' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'i', for_loop_var_243626)
        # SSA begins for a for statement (line 406)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Subscript (line 407):
        
        # Assigning a BinOp to a Subscript (line 407):
        int_243627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 26), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 407)
        i_243628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 33), 'i')
        int_243629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 35), 'int')
        # Applying the binary operator '-' (line 407)
        result_sub_243630 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 33), '-', i_243628, int_243629)
        
        # Getting the type of 'x' (line 407)
        x_243631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 31), 'x')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___243632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 31), x_243631, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_243633 = invoke(stypy.reporting.localization.Localization(__file__, 407, 31), getitem___243632, result_sub_243630)
        
        # Applying the binary operator '*' (line 407)
        result_mul_243634 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 26), '*', int_243627, subscript_call_result_243633)
        
        # Getting the type of 'J' (line 407)
        J_243635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'J')
        
        # Obtaining an instance of the builtin type 'tuple' (line 407)
        tuple_243636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 407)
        # Adding element type (line 407)
        # Getting the type of 'i' (line 407)
        i_243637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 14), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 14), tuple_243636, i_243637)
        # Adding element type (line 407)
        # Getting the type of 'i' (line 407)
        i_243638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 17), 'i')
        int_243639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 21), 'int')
        # Applying the binary operator '-' (line 407)
        result_sub_243640 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 17), '-', i_243638, int_243639)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 14), tuple_243636, result_sub_243640)
        
        # Storing an element on a container (line 407)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), J_243635, (tuple_243636, result_mul_243634))
        
        # Assigning a BinOp to a Subscript (line 408):
        
        # Assigning a BinOp to a Subscript (line 408):
        int_243641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 22), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 408)
        i_243642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 28), 'i')
        # Getting the type of 'x' (line 408)
        x_243643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 26), 'x')
        # Obtaining the member '__getitem__' of a type (line 408)
        getitem___243644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 26), x_243643, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 408)
        subscript_call_result_243645 = invoke(stypy.reporting.localization.Localization(__file__, 408, 26), getitem___243644, i_243642)
        
        int_243646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 32), 'int')
        # Applying the binary operator '**' (line 408)
        result_pow_243647 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 26), '**', subscript_call_result_243645, int_243646)
        
        # Applying the binary operator '*' (line 408)
        result_mul_243648 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 22), '*', int_243641, result_pow_243647)
        
        int_243649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 36), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 408)
        i_243650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 42), 'i')
        # Getting the type of 'x' (line 408)
        x_243651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 40), 'x')
        # Obtaining the member '__getitem__' of a type (line 408)
        getitem___243652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 40), x_243651, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 408)
        subscript_call_result_243653 = invoke(stypy.reporting.localization.Localization(__file__, 408, 40), getitem___243652, i_243650)
        
        # Applying the binary operator '*' (line 408)
        result_mul_243654 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 36), '*', int_243649, subscript_call_result_243653)
        
        # Applying the binary operator '-' (line 408)
        result_sub_243655 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 22), '-', result_mul_243648, result_mul_243654)
        
        # Getting the type of 'J' (line 408)
        J_243656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'J')
        
        # Obtaining an instance of the builtin type 'tuple' (line 408)
        tuple_243657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 408)
        # Adding element type (line 408)
        # Getting the type of 'i' (line 408)
        i_243658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 14), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 14), tuple_243657, i_243658)
        # Adding element type (line 408)
        # Getting the type of 'i' (line 408)
        i_243659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 17), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 14), tuple_243657, i_243659)
        
        # Storing an element on a container (line 408)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 12), J_243656, (tuple_243657, result_sub_243655))
        
        # Assigning a BinOp to a Subscript (line 409):
        
        # Assigning a BinOp to a Subscript (line 409):
        int_243660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 26), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 409)
        i_243661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 32), 'i')
        int_243662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 34), 'int')
        # Applying the binary operator '+' (line 409)
        result_add_243663 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 32), '+', i_243661, int_243662)
        
        # Getting the type of 'x' (line 409)
        x_243664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 30), 'x')
        # Obtaining the member '__getitem__' of a type (line 409)
        getitem___243665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 30), x_243664, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 409)
        subscript_call_result_243666 = invoke(stypy.reporting.localization.Localization(__file__, 409, 30), getitem___243665, result_add_243663)
        
        int_243667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 38), 'int')
        # Applying the binary operator '**' (line 409)
        result_pow_243668 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 30), '**', subscript_call_result_243666, int_243667)
        
        # Applying the binary operator '*' (line 409)
        result_mul_243669 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 26), '*', int_243660, result_pow_243668)
        
        # Getting the type of 'J' (line 409)
        J_243670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'J')
        
        # Obtaining an instance of the builtin type 'tuple' (line 409)
        tuple_243671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 409)
        # Adding element type (line 409)
        # Getting the type of 'i' (line 409)
        i_243672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 14), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 14), tuple_243671, i_243672)
        # Adding element type (line 409)
        # Getting the type of 'i' (line 409)
        i_243673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 17), 'i')
        int_243674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 21), 'int')
        # Applying the binary operator '+' (line 409)
        result_add_243675 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 17), '+', i_243673, int_243674)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 14), tuple_243671, result_add_243675)
        
        # Storing an element on a container (line 409)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 12), J_243670, (tuple_243671, result_mul_243669))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Subscript (line 410):
        
        # Assigning a BinOp to a Subscript (line 410):
        int_243676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 20), 'int')
        
        # Obtaining the type of the subscript
        int_243677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 26), 'int')
        # Getting the type of 'x' (line 410)
        x_243678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 24), 'x')
        # Obtaining the member '__getitem__' of a type (line 410)
        getitem___243679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 24), x_243678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 410)
        subscript_call_result_243680 = invoke(stypy.reporting.localization.Localization(__file__, 410, 24), getitem___243679, int_243677)
        
        int_243681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 31), 'int')
        # Applying the binary operator '**' (line 410)
        result_pow_243682 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 24), '**', subscript_call_result_243680, int_243681)
        
        # Applying the binary operator '*' (line 410)
        result_mul_243683 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 20), '*', int_243676, result_pow_243682)
        
        # Getting the type of 'J' (line 410)
        J_243684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'J')
        
        # Obtaining an instance of the builtin type 'tuple' (line 410)
        tuple_243685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 410)
        # Adding element type (line 410)
        int_243686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 10), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 10), tuple_243685, int_243686)
        # Adding element type (line 410)
        int_243687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 10), tuple_243685, int_243687)
        
        # Storing an element on a container (line 410)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 8), J_243684, (tuple_243685, result_mul_243683))
        
        # Assigning a BinOp to a Subscript (line 411):
        
        # Assigning a BinOp to a Subscript (line 411):
        int_243688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 20), 'int')
        
        # Obtaining the type of the subscript
        int_243689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 27), 'int')
        # Getting the type of 'x' (line 411)
        x_243690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 25), 'x')
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___243691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 25), x_243690, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_243692 = invoke(stypy.reporting.localization.Localization(__file__, 411, 25), getitem___243691, int_243689)
        
        # Applying the binary operator '*' (line 411)
        result_mul_243693 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 20), '*', int_243688, subscript_call_result_243692)
        
        # Getting the type of 'J' (line 411)
        J_243694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'J')
        
        # Obtaining an instance of the builtin type 'tuple' (line 411)
        tuple_243695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 411)
        # Adding element type (line 411)
        int_243696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 10), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 10), tuple_243695, int_243696)
        # Adding element type (line 411)
        int_243697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 10), tuple_243695, int_243697)
        
        # Storing an element on a container (line 411)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 8), J_243694, (tuple_243695, result_mul_243693))
        # Getting the type of 'J' (line 413)
        J_243698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 15), 'J')
        # Assigning a type to the variable 'stypy_return_type' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'stypy_return_type', J_243698)
        
        # ################# End of 'jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac' in the type store
        # Getting the type of 'stypy_return_type' (line 401)
        stypy_return_type_243699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243699)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac'
        return stypy_return_type_243699


    @norecursion
    def structure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'structure'
        module_type_store = module_type_store.open_function_context('structure', 415, 4, False)
        # Assigning a type to the variable 'self' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativeSparse.structure.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativeSparse.structure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativeSparse.structure.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativeSparse.structure.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativeSparse.structure')
        TestApproxDerivativeSparse.structure.__dict__.__setitem__('stypy_param_names_list', ['n'])
        TestApproxDerivativeSparse.structure.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativeSparse.structure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativeSparse.structure.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativeSparse.structure.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativeSparse.structure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativeSparse.structure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativeSparse.structure', ['n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'structure', localization, ['n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'structure(...)' code ##################

        
        # Assigning a Call to a Name (line 416):
        
        # Assigning a Call to a Name (line 416):
        
        # Call to zeros(...): (line 416)
        # Processing the call arguments (line 416)
        
        # Obtaining an instance of the builtin type 'tuple' (line 416)
        tuple_243702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 416)
        # Adding element type (line 416)
        # Getting the type of 'n' (line 416)
        n_243703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 22), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 22), tuple_243702, n_243703)
        # Adding element type (line 416)
        # Getting the type of 'n' (line 416)
        n_243704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 25), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 22), tuple_243702, n_243704)
        
        # Processing the call keyword arguments (line 416)
        # Getting the type of 'int' (line 416)
        int_243705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 35), 'int', False)
        keyword_243706 = int_243705
        kwargs_243707 = {'dtype': keyword_243706}
        # Getting the type of 'np' (line 416)
        np_243700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 416)
        zeros_243701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 12), np_243700, 'zeros')
        # Calling zeros(args, kwargs) (line 416)
        zeros_call_result_243708 = invoke(stypy.reporting.localization.Localization(__file__, 416, 12), zeros_243701, *[tuple_243702], **kwargs_243707)
        
        # Assigning a type to the variable 'A' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'A', zeros_call_result_243708)
        
        # Assigning a Num to a Subscript (line 417):
        
        # Assigning a Num to a Subscript (line 417):
        int_243709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 18), 'int')
        # Getting the type of 'A' (line 417)
        A_243710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 417)
        tuple_243711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 417)
        # Adding element type (line 417)
        int_243712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 10), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 10), tuple_243711, int_243712)
        # Adding element type (line 417)
        int_243713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 10), tuple_243711, int_243713)
        
        # Storing an element on a container (line 417)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 8), A_243710, (tuple_243711, int_243709))
        
        # Assigning a Num to a Subscript (line 418):
        
        # Assigning a Num to a Subscript (line 418):
        int_243714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 18), 'int')
        # Getting the type of 'A' (line 418)
        A_243715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 418)
        tuple_243716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 418)
        # Adding element type (line 418)
        int_243717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 10), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 10), tuple_243716, int_243717)
        # Adding element type (line 418)
        int_243718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 10), tuple_243716, int_243718)
        
        # Storing an element on a container (line 418)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 8), A_243715, (tuple_243716, int_243714))
        
        
        # Call to range(...): (line 419)
        # Processing the call arguments (line 419)
        int_243720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 23), 'int')
        # Getting the type of 'n' (line 419)
        n_243721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 26), 'n', False)
        int_243722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 30), 'int')
        # Applying the binary operator '-' (line 419)
        result_sub_243723 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 26), '-', n_243721, int_243722)
        
        # Processing the call keyword arguments (line 419)
        kwargs_243724 = {}
        # Getting the type of 'range' (line 419)
        range_243719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 17), 'range', False)
        # Calling range(args, kwargs) (line 419)
        range_call_result_243725 = invoke(stypy.reporting.localization.Localization(__file__, 419, 17), range_243719, *[int_243720, result_sub_243723], **kwargs_243724)
        
        # Testing the type of a for loop iterable (line 419)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 419, 8), range_call_result_243725)
        # Getting the type of the for loop variable (line 419)
        for_loop_var_243726 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 419, 8), range_call_result_243725)
        # Assigning a type to the variable 'i' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'i', for_loop_var_243726)
        # SSA begins for a for statement (line 419)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 420):
        
        # Assigning a Num to a Subscript (line 420):
        int_243727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 33), 'int')
        # Getting the type of 'A' (line 420)
        A_243728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'A')
        # Getting the type of 'i' (line 420)
        i_243729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 14), 'i')
        # Getting the type of 'i' (line 420)
        i_243730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 17), 'i')
        int_243731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 21), 'int')
        # Applying the binary operator '-' (line 420)
        result_sub_243732 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 17), '-', i_243730, int_243731)
        
        # Getting the type of 'i' (line 420)
        i_243733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 24), 'i')
        int_243734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 28), 'int')
        # Applying the binary operator '+' (line 420)
        result_add_243735 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 24), '+', i_243733, int_243734)
        
        slice_243736 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 420, 12), result_sub_243732, result_add_243735, None)
        # Storing an element on a container (line 420)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 12), A_243728, ((i_243729, slice_243736), int_243727))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Subscript (line 421):
        
        # Assigning a Num to a Subscript (line 421):
        int_243737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 20), 'int')
        # Getting the type of 'A' (line 421)
        A_243738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 421)
        tuple_243739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 421)
        # Adding element type (line 421)
        int_243740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 10), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 10), tuple_243739, int_243740)
        # Adding element type (line 421)
        int_243741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 10), tuple_243739, int_243741)
        
        # Storing an element on a container (line 421)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 8), A_243738, (tuple_243739, int_243737))
        
        # Assigning a Num to a Subscript (line 422):
        
        # Assigning a Num to a Subscript (line 422):
        int_243742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 20), 'int')
        # Getting the type of 'A' (line 422)
        A_243743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 422)
        tuple_243744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 422)
        # Adding element type (line 422)
        int_243745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 10), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 10), tuple_243744, int_243745)
        # Adding element type (line 422)
        int_243746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 10), tuple_243744, int_243746)
        
        # Storing an element on a container (line 422)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 8), A_243743, (tuple_243744, int_243742))
        # Getting the type of 'A' (line 424)
        A_243747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 15), 'A')
        # Assigning a type to the variable 'stypy_return_type' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'stypy_return_type', A_243747)
        
        # ################# End of 'structure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'structure' in the type store
        # Getting the type of 'stypy_return_type' (line 415)
        stypy_return_type_243748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243748)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'structure'
        return stypy_return_type_243748


    @norecursion
    def test_all(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_all'
        module_type_store = module_type_store.open_function_context('test_all', 426, 4, False)
        # Assigning a type to the variable 'self' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativeSparse.test_all.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativeSparse.test_all.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativeSparse.test_all.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativeSparse.test_all.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativeSparse.test_all')
        TestApproxDerivativeSparse.test_all.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativeSparse.test_all.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativeSparse.test_all.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativeSparse.test_all.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativeSparse.test_all.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativeSparse.test_all.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativeSparse.test_all.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativeSparse.test_all', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_all', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_all(...)' code ##################

        
        # Assigning a Call to a Name (line 427):
        
        # Assigning a Call to a Name (line 427):
        
        # Call to structure(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'self' (line 427)
        self_243751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 27), 'self', False)
        # Obtaining the member 'n' of a type (line 427)
        n_243752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 27), self_243751, 'n')
        # Processing the call keyword arguments (line 427)
        kwargs_243753 = {}
        # Getting the type of 'self' (line 427)
        self_243749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'self', False)
        # Obtaining the member 'structure' of a type (line 427)
        structure_243750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 12), self_243749, 'structure')
        # Calling structure(args, kwargs) (line 427)
        structure_call_result_243754 = invoke(stypy.reporting.localization.Localization(__file__, 427, 12), structure_243750, *[n_243752], **kwargs_243753)
        
        # Assigning a type to the variable 'A' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'A', structure_call_result_243754)
        
        # Assigning a Call to a Name (line 428):
        
        # Assigning a Call to a Name (line 428):
        
        # Call to arange(...): (line 428)
        # Processing the call arguments (line 428)
        # Getting the type of 'self' (line 428)
        self_243757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 26), 'self', False)
        # Obtaining the member 'n' of a type (line 428)
        n_243758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 26), self_243757, 'n')
        # Processing the call keyword arguments (line 428)
        kwargs_243759 = {}
        # Getting the type of 'np' (line 428)
        np_243755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 16), 'np', False)
        # Obtaining the member 'arange' of a type (line 428)
        arange_243756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 16), np_243755, 'arange')
        # Calling arange(args, kwargs) (line 428)
        arange_call_result_243760 = invoke(stypy.reporting.localization.Localization(__file__, 428, 16), arange_243756, *[n_243758], **kwargs_243759)
        
        # Assigning a type to the variable 'order' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'order', arange_call_result_243760)
        
        # Assigning a Call to a Name (line 429):
        
        # Assigning a Call to a Name (line 429):
        
        # Call to group_columns(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'A' (line 429)
        A_243762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 33), 'A', False)
        # Getting the type of 'order' (line 429)
        order_243763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 36), 'order', False)
        # Processing the call keyword arguments (line 429)
        kwargs_243764 = {}
        # Getting the type of 'group_columns' (line 429)
        group_columns_243761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 19), 'group_columns', False)
        # Calling group_columns(args, kwargs) (line 429)
        group_columns_call_result_243765 = invoke(stypy.reporting.localization.Localization(__file__, 429, 19), group_columns_243761, *[A_243762, order_243763], **kwargs_243764)
        
        # Assigning a type to the variable 'groups_1' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'groups_1', group_columns_call_result_243765)
        
        # Call to shuffle(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'order' (line 430)
        order_243769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 26), 'order', False)
        # Processing the call keyword arguments (line 430)
        kwargs_243770 = {}
        # Getting the type of 'np' (line 430)
        np_243766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 430)
        random_243767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), np_243766, 'random')
        # Obtaining the member 'shuffle' of a type (line 430)
        shuffle_243768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), random_243767, 'shuffle')
        # Calling shuffle(args, kwargs) (line 430)
        shuffle_call_result_243771 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), shuffle_243768, *[order_243769], **kwargs_243770)
        
        
        # Assigning a Call to a Name (line 431):
        
        # Assigning a Call to a Name (line 431):
        
        # Call to group_columns(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'A' (line 431)
        A_243773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 33), 'A', False)
        # Getting the type of 'order' (line 431)
        order_243774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 36), 'order', False)
        # Processing the call keyword arguments (line 431)
        kwargs_243775 = {}
        # Getting the type of 'group_columns' (line 431)
        group_columns_243772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 19), 'group_columns', False)
        # Calling group_columns(args, kwargs) (line 431)
        group_columns_call_result_243776 = invoke(stypy.reporting.localization.Localization(__file__, 431, 19), group_columns_243772, *[A_243773, order_243774], **kwargs_243775)
        
        # Assigning a type to the variable 'groups_2' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'groups_2', group_columns_call_result_243776)
        
        
        # Call to product(...): (line 433)
        # Processing the call arguments (line 433)
        
        # Obtaining an instance of the builtin type 'list' (line 434)
        list_243778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 434)
        # Adding element type (line 434)
        str_243779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 17), 'str', '2-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 16), list_243778, str_243779)
        # Adding element type (line 434)
        str_243780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 28), 'str', '3-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 16), list_243778, str_243780)
        # Adding element type (line 434)
        str_243781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 39), 'str', 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 16), list_243778, str_243781)
        
        
        # Obtaining an instance of the builtin type 'list' (line 434)
        list_243782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 434)
        # Adding element type (line 434)
        # Getting the type of 'groups_1' (line 434)
        groups_1_243783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 47), 'groups_1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 46), list_243782, groups_1_243783)
        # Adding element type (line 434)
        # Getting the type of 'groups_2' (line 434)
        groups_2_243784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 57), 'groups_2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 46), list_243782, groups_2_243784)
        
        
        # Obtaining an instance of the builtin type 'list' (line 435)
        list_243785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 435)
        # Adding element type (line 435)
        
        # Getting the type of 'np' (line 435)
        np_243786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 18), 'np', False)
        # Obtaining the member 'inf' of a type (line 435)
        inf_243787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 18), np_243786, 'inf')
        # Applying the 'usub' unary operator (line 435)
        result___neg___243788 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 17), 'usub', inf_243787)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 16), list_243785, result___neg___243788)
        # Adding element type (line 435)
        # Getting the type of 'self' (line 435)
        self_243789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 26), 'self', False)
        # Obtaining the member 'lb' of a type (line 435)
        lb_243790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 26), self_243789, 'lb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 16), list_243785, lb_243790)
        
        
        # Obtaining an instance of the builtin type 'list' (line 435)
        list_243791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 435)
        # Adding element type (line 435)
        # Getting the type of 'np' (line 435)
        np_243792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 37), 'np', False)
        # Obtaining the member 'inf' of a type (line 435)
        inf_243793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 37), np_243792, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 36), list_243791, inf_243793)
        # Adding element type (line 435)
        # Getting the type of 'self' (line 435)
        self_243794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 45), 'self', False)
        # Obtaining the member 'ub' of a type (line 435)
        ub_243795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 45), self_243794, 'ub')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 36), list_243791, ub_243795)
        
        # Processing the call keyword arguments (line 433)
        kwargs_243796 = {}
        # Getting the type of 'product' (line 433)
        product_243777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 36), 'product', False)
        # Calling product(args, kwargs) (line 433)
        product_call_result_243797 = invoke(stypy.reporting.localization.Localization(__file__, 433, 36), product_243777, *[list_243778, list_243782, list_243785, list_243791], **kwargs_243796)
        
        # Testing the type of a for loop iterable (line 433)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 433, 8), product_call_result_243797)
        # Getting the type of the for loop variable (line 433)
        for_loop_var_243798 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 433, 8), product_call_result_243797)
        # Assigning a type to the variable 'method' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'method', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 8), for_loop_var_243798))
        # Assigning a type to the variable 'groups' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'groups', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 8), for_loop_var_243798))
        # Assigning a type to the variable 'l' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'l', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 8), for_loop_var_243798))
        # Assigning a type to the variable 'u' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'u', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 8), for_loop_var_243798))
        # SSA begins for a for statement (line 433)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 436):
        
        # Assigning a Call to a Name (line 436):
        
        # Call to approx_derivative(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'self' (line 436)
        self_243800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 34), 'self', False)
        # Obtaining the member 'fun' of a type (line 436)
        fun_243801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 34), self_243800, 'fun')
        # Getting the type of 'self' (line 436)
        self_243802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 44), 'self', False)
        # Obtaining the member 'x0' of a type (line 436)
        x0_243803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 44), self_243802, 'x0')
        # Processing the call keyword arguments (line 436)
        # Getting the type of 'method' (line 436)
        method_243804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 60), 'method', False)
        keyword_243805 = method_243804
        
        # Obtaining an instance of the builtin type 'tuple' (line 437)
        tuple_243806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 437)
        # Adding element type (line 437)
        # Getting the type of 'l' (line 437)
        l_243807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 42), 'l', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 42), tuple_243806, l_243807)
        # Adding element type (line 437)
        # Getting the type of 'u' (line 437)
        u_243808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 45), 'u', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 42), tuple_243806, u_243808)
        
        keyword_243809 = tuple_243806
        
        # Obtaining an instance of the builtin type 'tuple' (line 437)
        tuple_243810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 437)
        # Adding element type (line 437)
        # Getting the type of 'A' (line 437)
        A_243811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 59), 'A', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 59), tuple_243810, A_243811)
        # Adding element type (line 437)
        # Getting the type of 'groups' (line 437)
        groups_243812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 62), 'groups', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 59), tuple_243810, groups_243812)
        
        keyword_243813 = tuple_243810
        kwargs_243814 = {'sparsity': keyword_243813, 'method': keyword_243805, 'bounds': keyword_243809}
        # Getting the type of 'approx_derivative' (line 436)
        approx_derivative_243799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 436)
        approx_derivative_call_result_243815 = invoke(stypy.reporting.localization.Localization(__file__, 436, 16), approx_derivative_243799, *[fun_243801, x0_243803], **kwargs_243814)
        
        # Assigning a type to the variable 'J' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'J', approx_derivative_call_result_243815)
        
        # Call to assert_(...): (line 438)
        # Processing the call arguments (line 438)
        
        # Call to isinstance(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'J' (line 438)
        J_243818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 31), 'J', False)
        # Getting the type of 'csr_matrix' (line 438)
        csr_matrix_243819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 34), 'csr_matrix', False)
        # Processing the call keyword arguments (line 438)
        kwargs_243820 = {}
        # Getting the type of 'isinstance' (line 438)
        isinstance_243817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 438)
        isinstance_call_result_243821 = invoke(stypy.reporting.localization.Localization(__file__, 438, 20), isinstance_243817, *[J_243818, csr_matrix_243819], **kwargs_243820)
        
        # Processing the call keyword arguments (line 438)
        kwargs_243822 = {}
        # Getting the type of 'assert_' (line 438)
        assert__243816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 438)
        assert__call_result_243823 = invoke(stypy.reporting.localization.Localization(__file__, 438, 12), assert__243816, *[isinstance_call_result_243821], **kwargs_243822)
        
        
        # Call to assert_allclose(...): (line 439)
        # Processing the call arguments (line 439)
        
        # Call to toarray(...): (line 439)
        # Processing the call keyword arguments (line 439)
        kwargs_243827 = {}
        # Getting the type of 'J' (line 439)
        J_243825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 28), 'J', False)
        # Obtaining the member 'toarray' of a type (line 439)
        toarray_243826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 28), J_243825, 'toarray')
        # Calling toarray(args, kwargs) (line 439)
        toarray_call_result_243828 = invoke(stypy.reporting.localization.Localization(__file__, 439, 28), toarray_243826, *[], **kwargs_243827)
        
        # Getting the type of 'self' (line 439)
        self_243829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 41), 'self', False)
        # Obtaining the member 'J_true' of a type (line 439)
        J_true_243830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 41), self_243829, 'J_true')
        # Processing the call keyword arguments (line 439)
        float_243831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 59), 'float')
        keyword_243832 = float_243831
        kwargs_243833 = {'rtol': keyword_243832}
        # Getting the type of 'assert_allclose' (line 439)
        assert_allclose_243824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 439)
        assert_allclose_call_result_243834 = invoke(stypy.reporting.localization.Localization(__file__, 439, 12), assert_allclose_243824, *[toarray_call_result_243828, J_true_243830], **kwargs_243833)
        
        
        # Assigning a BinOp to a Name (line 441):
        
        # Assigning a BinOp to a Name (line 441):
        float_243835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 23), 'float')
        
        # Call to ones_like(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'self' (line 441)
        self_243838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 43), 'self', False)
        # Obtaining the member 'x0' of a type (line 441)
        x0_243839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 43), self_243838, 'x0')
        # Processing the call keyword arguments (line 441)
        kwargs_243840 = {}
        # Getting the type of 'np' (line 441)
        np_243836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 30), 'np', False)
        # Obtaining the member 'ones_like' of a type (line 441)
        ones_like_243837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 30), np_243836, 'ones_like')
        # Calling ones_like(args, kwargs) (line 441)
        ones_like_call_result_243841 = invoke(stypy.reporting.localization.Localization(__file__, 441, 30), ones_like_243837, *[x0_243839], **kwargs_243840)
        
        # Applying the binary operator '*' (line 441)
        result_mul_243842 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 23), '*', float_243835, ones_like_call_result_243841)
        
        # Assigning a type to the variable 'rel_step' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'rel_step', result_mul_243842)
        
        # Getting the type of 'rel_step' (line 442)
        rel_step_243843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'rel_step')
        
        # Obtaining the type of the subscript
        int_243844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 23), 'int')
        slice_243845 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 442, 12), None, None, int_243844)
        # Getting the type of 'rel_step' (line 442)
        rel_step_243846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'rel_step')
        # Obtaining the member '__getitem__' of a type (line 442)
        getitem___243847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 12), rel_step_243846, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 442)
        subscript_call_result_243848 = invoke(stypy.reporting.localization.Localization(__file__, 442, 12), getitem___243847, slice_243845)
        
        int_243849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 29), 'int')
        # Applying the binary operator '*=' (line 442)
        result_imul_243850 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 12), '*=', subscript_call_result_243848, int_243849)
        # Getting the type of 'rel_step' (line 442)
        rel_step_243851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'rel_step')
        int_243852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 23), 'int')
        slice_243853 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 442, 12), None, None, int_243852)
        # Storing an element on a container (line 442)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 12), rel_step_243851, (slice_243853, result_imul_243850))
        
        
        # Assigning a Call to a Name (line 443):
        
        # Assigning a Call to a Name (line 443):
        
        # Call to approx_derivative(...): (line 443)
        # Processing the call arguments (line 443)
        # Getting the type of 'self' (line 443)
        self_243855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 34), 'self', False)
        # Obtaining the member 'fun' of a type (line 443)
        fun_243856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 34), self_243855, 'fun')
        # Getting the type of 'self' (line 443)
        self_243857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 44), 'self', False)
        # Obtaining the member 'x0' of a type (line 443)
        x0_243858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 44), self_243857, 'x0')
        # Processing the call keyword arguments (line 443)
        # Getting the type of 'method' (line 443)
        method_243859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 60), 'method', False)
        keyword_243860 = method_243859
        # Getting the type of 'rel_step' (line 444)
        rel_step_243861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 43), 'rel_step', False)
        keyword_243862 = rel_step_243861
        
        # Obtaining an instance of the builtin type 'tuple' (line 444)
        tuple_243863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 444)
        # Adding element type (line 444)
        # Getting the type of 'A' (line 444)
        A_243864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 63), 'A', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 63), tuple_243863, A_243864)
        # Adding element type (line 444)
        # Getting the type of 'groups' (line 444)
        groups_243865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 66), 'groups', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 63), tuple_243863, groups_243865)
        
        keyword_243866 = tuple_243863
        kwargs_243867 = {'sparsity': keyword_243866, 'method': keyword_243860, 'rel_step': keyword_243862}
        # Getting the type of 'approx_derivative' (line 443)
        approx_derivative_243854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 16), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 443)
        approx_derivative_call_result_243868 = invoke(stypy.reporting.localization.Localization(__file__, 443, 16), approx_derivative_243854, *[fun_243856, x0_243858], **kwargs_243867)
        
        # Assigning a type to the variable 'J' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 12), 'J', approx_derivative_call_result_243868)
        
        # Call to assert_allclose(...): (line 445)
        # Processing the call arguments (line 445)
        
        # Call to toarray(...): (line 445)
        # Processing the call keyword arguments (line 445)
        kwargs_243872 = {}
        # Getting the type of 'J' (line 445)
        J_243870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 28), 'J', False)
        # Obtaining the member 'toarray' of a type (line 445)
        toarray_243871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 28), J_243870, 'toarray')
        # Calling toarray(args, kwargs) (line 445)
        toarray_call_result_243873 = invoke(stypy.reporting.localization.Localization(__file__, 445, 28), toarray_243871, *[], **kwargs_243872)
        
        # Getting the type of 'self' (line 445)
        self_243874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 41), 'self', False)
        # Obtaining the member 'J_true' of a type (line 445)
        J_true_243875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 41), self_243874, 'J_true')
        # Processing the call keyword arguments (line 445)
        float_243876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 59), 'float')
        keyword_243877 = float_243876
        kwargs_243878 = {'rtol': keyword_243877}
        # Getting the type of 'assert_allclose' (line 445)
        assert_allclose_243869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 445)
        assert_allclose_call_result_243879 = invoke(stypy.reporting.localization.Localization(__file__, 445, 12), assert_allclose_243869, *[toarray_call_result_243873, J_true_243875], **kwargs_243878)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_all(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_all' in the type store
        # Getting the type of 'stypy_return_type' (line 426)
        stypy_return_type_243880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243880)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_all'
        return stypy_return_type_243880


    @norecursion
    def test_no_precomputed_groups(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_no_precomputed_groups'
        module_type_store = module_type_store.open_function_context('test_no_precomputed_groups', 447, 4, False)
        # Assigning a type to the variable 'self' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativeSparse.test_no_precomputed_groups.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativeSparse.test_no_precomputed_groups.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativeSparse.test_no_precomputed_groups.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativeSparse.test_no_precomputed_groups.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativeSparse.test_no_precomputed_groups')
        TestApproxDerivativeSparse.test_no_precomputed_groups.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativeSparse.test_no_precomputed_groups.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativeSparse.test_no_precomputed_groups.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativeSparse.test_no_precomputed_groups.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativeSparse.test_no_precomputed_groups.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativeSparse.test_no_precomputed_groups.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativeSparse.test_no_precomputed_groups.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativeSparse.test_no_precomputed_groups', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_no_precomputed_groups', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_no_precomputed_groups(...)' code ##################

        
        # Assigning a Call to a Name (line 448):
        
        # Assigning a Call to a Name (line 448):
        
        # Call to structure(...): (line 448)
        # Processing the call arguments (line 448)
        # Getting the type of 'self' (line 448)
        self_243883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 27), 'self', False)
        # Obtaining the member 'n' of a type (line 448)
        n_243884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 27), self_243883, 'n')
        # Processing the call keyword arguments (line 448)
        kwargs_243885 = {}
        # Getting the type of 'self' (line 448)
        self_243881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'self', False)
        # Obtaining the member 'structure' of a type (line 448)
        structure_243882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 12), self_243881, 'structure')
        # Calling structure(args, kwargs) (line 448)
        structure_call_result_243886 = invoke(stypy.reporting.localization.Localization(__file__, 448, 12), structure_243882, *[n_243884], **kwargs_243885)
        
        # Assigning a type to the variable 'A' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'A', structure_call_result_243886)
        
        # Assigning a Call to a Name (line 449):
        
        # Assigning a Call to a Name (line 449):
        
        # Call to approx_derivative(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'self' (line 449)
        self_243888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 30), 'self', False)
        # Obtaining the member 'fun' of a type (line 449)
        fun_243889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 30), self_243888, 'fun')
        # Getting the type of 'self' (line 449)
        self_243890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 40), 'self', False)
        # Obtaining the member 'x0' of a type (line 449)
        x0_243891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 40), self_243890, 'x0')
        # Processing the call keyword arguments (line 449)
        # Getting the type of 'A' (line 449)
        A_243892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 58), 'A', False)
        keyword_243893 = A_243892
        kwargs_243894 = {'sparsity': keyword_243893}
        # Getting the type of 'approx_derivative' (line 449)
        approx_derivative_243887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 449)
        approx_derivative_call_result_243895 = invoke(stypy.reporting.localization.Localization(__file__, 449, 12), approx_derivative_243887, *[fun_243889, x0_243891], **kwargs_243894)
        
        # Assigning a type to the variable 'J' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'J', approx_derivative_call_result_243895)
        
        # Call to assert_allclose(...): (line 450)
        # Processing the call arguments (line 450)
        
        # Call to toarray(...): (line 450)
        # Processing the call keyword arguments (line 450)
        kwargs_243899 = {}
        # Getting the type of 'J' (line 450)
        J_243897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 24), 'J', False)
        # Obtaining the member 'toarray' of a type (line 450)
        toarray_243898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 24), J_243897, 'toarray')
        # Calling toarray(args, kwargs) (line 450)
        toarray_call_result_243900 = invoke(stypy.reporting.localization.Localization(__file__, 450, 24), toarray_243898, *[], **kwargs_243899)
        
        # Getting the type of 'self' (line 450)
        self_243901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 37), 'self', False)
        # Obtaining the member 'J_true' of a type (line 450)
        J_true_243902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 37), self_243901, 'J_true')
        # Processing the call keyword arguments (line 450)
        float_243903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 55), 'float')
        keyword_243904 = float_243903
        kwargs_243905 = {'rtol': keyword_243904}
        # Getting the type of 'assert_allclose' (line 450)
        assert_allclose_243896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 450)
        assert_allclose_call_result_243906 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), assert_allclose_243896, *[toarray_call_result_243900, J_true_243902], **kwargs_243905)
        
        
        # ################# End of 'test_no_precomputed_groups(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_no_precomputed_groups' in the type store
        # Getting the type of 'stypy_return_type' (line 447)
        stypy_return_type_243907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243907)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_no_precomputed_groups'
        return stypy_return_type_243907


    @norecursion
    def test_equivalence(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_equivalence'
        module_type_store = module_type_store.open_function_context('test_equivalence', 452, 4, False)
        # Assigning a type to the variable 'self' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativeSparse.test_equivalence.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativeSparse.test_equivalence.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativeSparse.test_equivalence.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativeSparse.test_equivalence.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativeSparse.test_equivalence')
        TestApproxDerivativeSparse.test_equivalence.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativeSparse.test_equivalence.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativeSparse.test_equivalence.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativeSparse.test_equivalence.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativeSparse.test_equivalence.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativeSparse.test_equivalence.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativeSparse.test_equivalence.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativeSparse.test_equivalence', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 453):
        
        # Assigning a Call to a Name (line 453):
        
        # Call to ones(...): (line 453)
        # Processing the call arguments (line 453)
        
        # Obtaining an instance of the builtin type 'tuple' (line 453)
        tuple_243910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 453)
        # Adding element type (line 453)
        # Getting the type of 'self' (line 453)
        self_243911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 29), 'self', False)
        # Obtaining the member 'n' of a type (line 453)
        n_243912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 29), self_243911, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 29), tuple_243910, n_243912)
        # Adding element type (line 453)
        # Getting the type of 'self' (line 453)
        self_243913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 37), 'self', False)
        # Obtaining the member 'n' of a type (line 453)
        n_243914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 37), self_243913, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 29), tuple_243910, n_243914)
        
        # Processing the call keyword arguments (line 453)
        # Getting the type of 'int' (line 453)
        int_243915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 52), 'int', False)
        keyword_243916 = int_243915
        kwargs_243917 = {'dtype': keyword_243916}
        # Getting the type of 'np' (line 453)
        np_243908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 20), 'np', False)
        # Obtaining the member 'ones' of a type (line 453)
        ones_243909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 20), np_243908, 'ones')
        # Calling ones(args, kwargs) (line 453)
        ones_call_result_243918 = invoke(stypy.reporting.localization.Localization(__file__, 453, 20), ones_243909, *[tuple_243910], **kwargs_243917)
        
        # Assigning a type to the variable 'structure' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'structure', ones_call_result_243918)
        
        # Assigning a Call to a Name (line 454):
        
        # Assigning a Call to a Name (line 454):
        
        # Call to arange(...): (line 454)
        # Processing the call arguments (line 454)
        # Getting the type of 'self' (line 454)
        self_243921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 27), 'self', False)
        # Obtaining the member 'n' of a type (line 454)
        n_243922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 27), self_243921, 'n')
        # Processing the call keyword arguments (line 454)
        kwargs_243923 = {}
        # Getting the type of 'np' (line 454)
        np_243919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 454)
        arange_243920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 17), np_243919, 'arange')
        # Calling arange(args, kwargs) (line 454)
        arange_call_result_243924 = invoke(stypy.reporting.localization.Localization(__file__, 454, 17), arange_243920, *[n_243922], **kwargs_243923)
        
        # Assigning a type to the variable 'groups' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'groups', arange_call_result_243924)
        
        
        # Obtaining an instance of the builtin type 'list' (line 455)
        list_243925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 455)
        # Adding element type (line 455)
        str_243926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 23), 'str', '2-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 22), list_243925, str_243926)
        # Adding element type (line 455)
        str_243927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 34), 'str', '3-point')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 22), list_243925, str_243927)
        # Adding element type (line 455)
        str_243928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 45), 'str', 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 22), list_243925, str_243928)
        
        # Testing the type of a for loop iterable (line 455)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 455, 8), list_243925)
        # Getting the type of the for loop variable (line 455)
        for_loop_var_243929 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 455, 8), list_243925)
        # Assigning a type to the variable 'method' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'method', for_loop_var_243929)
        # SSA begins for a for statement (line 455)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 456):
        
        # Assigning a Call to a Name (line 456):
        
        # Call to approx_derivative(...): (line 456)
        # Processing the call arguments (line 456)
        # Getting the type of 'self' (line 456)
        self_243931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 40), 'self', False)
        # Obtaining the member 'fun' of a type (line 456)
        fun_243932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 40), self_243931, 'fun')
        # Getting the type of 'self' (line 456)
        self_243933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 50), 'self', False)
        # Obtaining the member 'x0' of a type (line 456)
        x0_243934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 50), self_243933, 'x0')
        # Processing the call keyword arguments (line 456)
        # Getting the type of 'method' (line 456)
        method_243935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 66), 'method', False)
        keyword_243936 = method_243935
        kwargs_243937 = {'method': keyword_243936}
        # Getting the type of 'approx_derivative' (line 456)
        approx_derivative_243930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 22), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 456)
        approx_derivative_call_result_243938 = invoke(stypy.reporting.localization.Localization(__file__, 456, 22), approx_derivative_243930, *[fun_243932, x0_243934], **kwargs_243937)
        
        # Assigning a type to the variable 'J_dense' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'J_dense', approx_derivative_call_result_243938)
        
        # Assigning a Call to a Name (line 457):
        
        # Assigning a Call to a Name (line 457):
        
        # Call to approx_derivative(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'self' (line 458)
        self_243940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 16), 'self', False)
        # Obtaining the member 'fun' of a type (line 458)
        fun_243941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 16), self_243940, 'fun')
        # Getting the type of 'self' (line 458)
        self_243942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 26), 'self', False)
        # Obtaining the member 'x0' of a type (line 458)
        x0_243943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 26), self_243942, 'x0')
        # Processing the call keyword arguments (line 457)
        
        # Obtaining an instance of the builtin type 'tuple' (line 458)
        tuple_243944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 458)
        # Adding element type (line 458)
        # Getting the type of 'structure' (line 458)
        structure_243945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 45), 'structure', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 45), tuple_243944, structure_243945)
        # Adding element type (line 458)
        # Getting the type of 'groups' (line 458)
        groups_243946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 56), 'groups', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 45), tuple_243944, groups_243946)
        
        keyword_243947 = tuple_243944
        # Getting the type of 'method' (line 458)
        method_243948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 72), 'method', False)
        keyword_243949 = method_243948
        kwargs_243950 = {'sparsity': keyword_243947, 'method': keyword_243949}
        # Getting the type of 'approx_derivative' (line 457)
        approx_derivative_243939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 23), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 457)
        approx_derivative_call_result_243951 = invoke(stypy.reporting.localization.Localization(__file__, 457, 23), approx_derivative_243939, *[fun_243941, x0_243943], **kwargs_243950)
        
        # Assigning a type to the variable 'J_sparse' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'J_sparse', approx_derivative_call_result_243951)
        
        # Call to assert_equal(...): (line 459)
        # Processing the call arguments (line 459)
        # Getting the type of 'J_dense' (line 459)
        J_dense_243953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 25), 'J_dense', False)
        
        # Call to toarray(...): (line 459)
        # Processing the call keyword arguments (line 459)
        kwargs_243956 = {}
        # Getting the type of 'J_sparse' (line 459)
        J_sparse_243954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 34), 'J_sparse', False)
        # Obtaining the member 'toarray' of a type (line 459)
        toarray_243955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 34), J_sparse_243954, 'toarray')
        # Calling toarray(args, kwargs) (line 459)
        toarray_call_result_243957 = invoke(stypy.reporting.localization.Localization(__file__, 459, 34), toarray_243955, *[], **kwargs_243956)
        
        # Processing the call keyword arguments (line 459)
        kwargs_243958 = {}
        # Getting the type of 'assert_equal' (line 459)
        assert_equal_243952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 459)
        assert_equal_call_result_243959 = invoke(stypy.reporting.localization.Localization(__file__, 459, 12), assert_equal_243952, *[J_dense_243953, toarray_call_result_243957], **kwargs_243958)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_equivalence(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_equivalence' in the type store
        # Getting the type of 'stypy_return_type' (line 452)
        stypy_return_type_243960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243960)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_equivalence'
        return stypy_return_type_243960


    @norecursion
    def test_check_derivative(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_check_derivative'
        module_type_store = module_type_store.open_function_context('test_check_derivative', 461, 4, False)
        # Assigning a type to the variable 'self' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestApproxDerivativeSparse.test_check_derivative.__dict__.__setitem__('stypy_localization', localization)
        TestApproxDerivativeSparse.test_check_derivative.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestApproxDerivativeSparse.test_check_derivative.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestApproxDerivativeSparse.test_check_derivative.__dict__.__setitem__('stypy_function_name', 'TestApproxDerivativeSparse.test_check_derivative')
        TestApproxDerivativeSparse.test_check_derivative.__dict__.__setitem__('stypy_param_names_list', [])
        TestApproxDerivativeSparse.test_check_derivative.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestApproxDerivativeSparse.test_check_derivative.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestApproxDerivativeSparse.test_check_derivative.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestApproxDerivativeSparse.test_check_derivative.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestApproxDerivativeSparse.test_check_derivative.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestApproxDerivativeSparse.test_check_derivative.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativeSparse.test_check_derivative', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_check_derivative', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_check_derivative(...)' code ##################


        @norecursion
        def jac(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'jac'
            module_type_store = module_type_store.open_function_context('jac', 462, 8, False)
            
            # Passed parameters checking function
            jac.stypy_localization = localization
            jac.stypy_type_of_self = None
            jac.stypy_type_store = module_type_store
            jac.stypy_function_name = 'jac'
            jac.stypy_param_names_list = ['x']
            jac.stypy_varargs_param_name = None
            jac.stypy_kwargs_param_name = None
            jac.stypy_call_defaults = defaults
            jac.stypy_call_varargs = varargs
            jac.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'jac', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'jac', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'jac(...)' code ##################

            
            # Call to csr_matrix(...): (line 463)
            # Processing the call arguments (line 463)
            
            # Call to jac(...): (line 463)
            # Processing the call arguments (line 463)
            # Getting the type of 'x' (line 463)
            x_243964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 39), 'x', False)
            # Processing the call keyword arguments (line 463)
            kwargs_243965 = {}
            # Getting the type of 'self' (line 463)
            self_243962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 30), 'self', False)
            # Obtaining the member 'jac' of a type (line 463)
            jac_243963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 30), self_243962, 'jac')
            # Calling jac(args, kwargs) (line 463)
            jac_call_result_243966 = invoke(stypy.reporting.localization.Localization(__file__, 463, 30), jac_243963, *[x_243964], **kwargs_243965)
            
            # Processing the call keyword arguments (line 463)
            kwargs_243967 = {}
            # Getting the type of 'csr_matrix' (line 463)
            csr_matrix_243961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 19), 'csr_matrix', False)
            # Calling csr_matrix(args, kwargs) (line 463)
            csr_matrix_call_result_243968 = invoke(stypy.reporting.localization.Localization(__file__, 463, 19), csr_matrix_243961, *[jac_call_result_243966], **kwargs_243967)
            
            # Assigning a type to the variable 'stypy_return_type' (line 463)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'stypy_return_type', csr_matrix_call_result_243968)
            
            # ################# End of 'jac(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'jac' in the type store
            # Getting the type of 'stypy_return_type' (line 462)
            stypy_return_type_243969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_243969)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'jac'
            return stypy_return_type_243969

        # Assigning a type to the variable 'jac' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'jac', jac)
        
        # Assigning a Call to a Name (line 465):
        
        # Assigning a Call to a Name (line 465):
        
        # Call to check_derivative(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'self' (line 465)
        self_243971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 36), 'self', False)
        # Obtaining the member 'fun' of a type (line 465)
        fun_243972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 36), self_243971, 'fun')
        # Getting the type of 'jac' (line 465)
        jac_243973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 46), 'jac', False)
        # Getting the type of 'self' (line 465)
        self_243974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 51), 'self', False)
        # Obtaining the member 'x0' of a type (line 465)
        x0_243975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 51), self_243974, 'x0')
        # Processing the call keyword arguments (line 465)
        
        # Obtaining an instance of the builtin type 'tuple' (line 466)
        tuple_243976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 466)
        # Adding element type (line 466)
        # Getting the type of 'self' (line 466)
        self_243977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 44), 'self', False)
        # Obtaining the member 'lb' of a type (line 466)
        lb_243978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 44), self_243977, 'lb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 44), tuple_243976, lb_243978)
        # Adding element type (line 466)
        # Getting the type of 'self' (line 466)
        self_243979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 53), 'self', False)
        # Obtaining the member 'ub' of a type (line 466)
        ub_243980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 53), self_243979, 'ub')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 44), tuple_243976, ub_243980)
        
        keyword_243981 = tuple_243976
        kwargs_243982 = {'bounds': keyword_243981}
        # Getting the type of 'check_derivative' (line 465)
        check_derivative_243970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 19), 'check_derivative', False)
        # Calling check_derivative(args, kwargs) (line 465)
        check_derivative_call_result_243983 = invoke(stypy.reporting.localization.Localization(__file__, 465, 19), check_derivative_243970, *[fun_243972, jac_243973, x0_243975], **kwargs_243982)
        
        # Assigning a type to the variable 'accuracy' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'accuracy', check_derivative_call_result_243983)
        
        # Call to assert_(...): (line 467)
        # Processing the call arguments (line 467)
        
        # Getting the type of 'accuracy' (line 467)
        accuracy_243985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'accuracy', False)
        float_243986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 27), 'float')
        # Applying the binary operator '<' (line 467)
        result_lt_243987 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 16), '<', accuracy_243985, float_243986)
        
        # Processing the call keyword arguments (line 467)
        kwargs_243988 = {}
        # Getting the type of 'assert_' (line 467)
        assert__243984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 467)
        assert__call_result_243989 = invoke(stypy.reporting.localization.Localization(__file__, 467, 8), assert__243984, *[result_lt_243987], **kwargs_243988)
        
        
        # Assigning a Call to a Name (line 469):
        
        # Assigning a Call to a Name (line 469):
        
        # Call to check_derivative(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 'self' (line 469)
        self_243991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 36), 'self', False)
        # Obtaining the member 'fun' of a type (line 469)
        fun_243992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 36), self_243991, 'fun')
        # Getting the type of 'jac' (line 469)
        jac_243993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 46), 'jac', False)
        # Getting the type of 'self' (line 469)
        self_243994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 51), 'self', False)
        # Obtaining the member 'x0' of a type (line 469)
        x0_243995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 51), self_243994, 'x0')
        # Processing the call keyword arguments (line 469)
        
        # Obtaining an instance of the builtin type 'tuple' (line 470)
        tuple_243996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 470)
        # Adding element type (line 470)
        # Getting the type of 'self' (line 470)
        self_243997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 44), 'self', False)
        # Obtaining the member 'lb' of a type (line 470)
        lb_243998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 44), self_243997, 'lb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 44), tuple_243996, lb_243998)
        # Adding element type (line 470)
        # Getting the type of 'self' (line 470)
        self_243999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 53), 'self', False)
        # Obtaining the member 'ub' of a type (line 470)
        ub_244000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 53), self_243999, 'ub')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 44), tuple_243996, ub_244000)
        
        keyword_244001 = tuple_243996
        kwargs_244002 = {'bounds': keyword_244001}
        # Getting the type of 'check_derivative' (line 469)
        check_derivative_243990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 19), 'check_derivative', False)
        # Calling check_derivative(args, kwargs) (line 469)
        check_derivative_call_result_244003 = invoke(stypy.reporting.localization.Localization(__file__, 469, 19), check_derivative_243990, *[fun_243992, jac_243993, x0_243995], **kwargs_244002)
        
        # Assigning a type to the variable 'accuracy' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'accuracy', check_derivative_call_result_244003)
        
        # Call to assert_(...): (line 471)
        # Processing the call arguments (line 471)
        
        # Getting the type of 'accuracy' (line 471)
        accuracy_244005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'accuracy', False)
        float_244006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 27), 'float')
        # Applying the binary operator '<' (line 471)
        result_lt_244007 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 16), '<', accuracy_244005, float_244006)
        
        # Processing the call keyword arguments (line 471)
        kwargs_244008 = {}
        # Getting the type of 'assert_' (line 471)
        assert__244004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 471)
        assert__call_result_244009 = invoke(stypy.reporting.localization.Localization(__file__, 471, 8), assert__244004, *[result_lt_244007], **kwargs_244008)
        
        
        # ################# End of 'test_check_derivative(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_check_derivative' in the type store
        # Getting the type of 'stypy_return_type' (line 461)
        stypy_return_type_244010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_244010)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_check_derivative'
        return stypy_return_type_244010


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 384, 0, False)
        # Assigning a type to the variable 'self' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestApproxDerivativeSparse.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestApproxDerivativeSparse' (line 384)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 0), 'TestApproxDerivativeSparse', TestApproxDerivativeSparse)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
