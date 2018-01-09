
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #
2: # Created by: Pearu Peterson, March 2002
3: #
4: ''' Test functions for scipy.linalg.matfuncs module
5: 
6: '''
7: from __future__ import division, print_function, absolute_import
8: 
9: import math
10: 
11: import numpy as np
12: from numpy import array, eye, exp, random
13: from numpy.linalg import matrix_power
14: from numpy.testing import (
15:         assert_allclose, assert_, assert_array_almost_equal, assert_equal,
16:         assert_array_almost_equal_nulp)
17: from scipy._lib._numpy_compat import suppress_warnings
18: 
19: from scipy.sparse import csc_matrix, SparseEfficiencyWarning
20: from scipy.sparse.construct import eye as speye
21: from scipy.sparse.linalg.matfuncs import (expm, _expm,
22:         ProductOperator, MatrixPowerOperator,
23:         _onenorm_matrix_power_nnm)
24: from scipy.linalg import logm
25: from scipy.special import factorial
26: import scipy.sparse
27: import scipy.sparse.linalg
28: 
29: 
30: def _burkardt_13_power(n, p):
31:     '''
32:     A helper function for testing matrix functions.
33: 
34:     Parameters
35:     ----------
36:     n : integer greater than 1
37:         Order of the square matrix to be returned.
38:     p : non-negative integer
39:         Power of the matrix.
40: 
41:     Returns
42:     -------
43:     out : ndarray representing a square matrix
44:         A Forsythe matrix of order n, raised to the power p.
45: 
46:     '''
47:     # Input validation.
48:     if n != int(n) or n < 2:
49:         raise ValueError('n must be an integer greater than 1')
50:     n = int(n)
51:     if p != int(p) or p < 0:
52:         raise ValueError('p must be a non-negative integer')
53:     p = int(p)
54: 
55:     # Construct the matrix explicitly.
56:     a, b = divmod(p, n)
57:     large = np.power(10.0, -n*a)
58:     small = large * np.power(10.0, -n)
59:     return np.diag([large]*(n-b), b) + np.diag([small]*b, b-n)
60: 
61: 
62: def test_onenorm_matrix_power_nnm():
63:     np.random.seed(1234)
64:     for n in range(1, 5):
65:         for p in range(5):
66:             M = np.random.random((n, n))
67:             Mp = np.linalg.matrix_power(M, p)
68:             observed = _onenorm_matrix_power_nnm(M, p)
69:             expected = np.linalg.norm(Mp, 1)
70:             assert_allclose(observed, expected)
71: 
72: 
73: class TestExpM(object):
74:     def test_zero_ndarray(self):
75:         a = array([[0.,0],[0,0]])
76:         assert_array_almost_equal(expm(a),[[1,0],[0,1]])
77: 
78:     def test_zero_sparse(self):
79:         a = csc_matrix([[0.,0],[0,0]])
80:         assert_array_almost_equal(expm(a).toarray(),[[1,0],[0,1]])
81: 
82:     def test_zero_matrix(self):
83:         a = np.matrix([[0.,0],[0,0]])
84:         assert_array_almost_equal(expm(a),[[1,0],[0,1]])
85: 
86:     def test_misc_types(self):
87:         A = expm(np.array([[1]]))
88:         assert_allclose(expm(((1,),)), A)
89:         assert_allclose(expm([[1]]), A)
90:         assert_allclose(expm(np.matrix([[1]])), A)
91:         assert_allclose(expm(np.array([[1]])), A)
92:         assert_allclose(expm(csc_matrix([[1]])).A, A)
93:         B = expm(np.array([[1j]]))
94:         assert_allclose(expm(((1j,),)), B)
95:         assert_allclose(expm([[1j]]), B)
96:         assert_allclose(expm(np.matrix([[1j]])), B)
97:         assert_allclose(expm(csc_matrix([[1j]])).A, B)
98: 
99:     def test_bidiagonal_sparse(self):
100:         A = csc_matrix([
101:             [1, 3, 0],
102:             [0, 1, 5],
103:             [0, 0, 2]], dtype=float)
104:         e1 = math.exp(1)
105:         e2 = math.exp(2)
106:         expected = np.array([
107:             [e1, 3*e1, 15*(e2 - 2*e1)],
108:             [0, e1, 5*(e2 - e1)],
109:             [0, 0, e2]], dtype=float)
110:         observed = expm(A).toarray()
111:         assert_array_almost_equal(observed, expected)
112: 
113:     def test_padecases_dtype_float(self):
114:         for dtype in [np.float32, np.float64]:
115:             for scale in [1e-2, 1e-1, 5e-1, 1, 10]:
116:                 A = scale * eye(3, dtype=dtype)
117:                 observed = expm(A)
118:                 expected = exp(scale) * eye(3, dtype=dtype)
119:                 assert_array_almost_equal_nulp(observed, expected, nulp=100)
120: 
121:     def test_padecases_dtype_complex(self):
122:         for dtype in [np.complex64, np.complex128]:
123:             for scale in [1e-2, 1e-1, 5e-1, 1, 10]:
124:                 A = scale * eye(3, dtype=dtype)
125:                 observed = expm(A)
126:                 expected = exp(scale) * eye(3, dtype=dtype)
127:                 assert_array_almost_equal_nulp(observed, expected, nulp=100)
128: 
129:     def test_padecases_dtype_sparse_float(self):
130:         # float32 and complex64 lead to errors in spsolve/UMFpack
131:         dtype = np.float64
132:         for scale in [1e-2, 1e-1, 5e-1, 1, 10]:
133:             a = scale * speye(3, 3, dtype=dtype, format='csc')
134:             e = exp(scale) * eye(3, dtype=dtype)
135:             with suppress_warnings() as sup:
136:                 sup.filter(SparseEfficiencyWarning,
137:                            "Changing the sparsity structure of a csc_matrix is expensive.")
138:                 exact_onenorm = _expm(a, use_exact_onenorm=True).toarray()
139:                 inexact_onenorm = _expm(a, use_exact_onenorm=False).toarray()
140:             assert_array_almost_equal_nulp(exact_onenorm, e, nulp=100)
141:             assert_array_almost_equal_nulp(inexact_onenorm, e, nulp=100)
142: 
143:     def test_padecases_dtype_sparse_complex(self):
144:         # float32 and complex64 lead to errors in spsolve/UMFpack
145:         dtype = np.complex128
146:         for scale in [1e-2, 1e-1, 5e-1, 1, 10]:
147:             a = scale * speye(3, 3, dtype=dtype, format='csc')
148:             e = exp(scale) * eye(3, dtype=dtype)
149:             with suppress_warnings() as sup:
150:                 sup.filter(SparseEfficiencyWarning,
151:                            "Changing the sparsity structure of a csc_matrix is expensive.")
152:                 assert_array_almost_equal_nulp(expm(a).toarray(), e, nulp=100)
153: 
154:     def test_logm_consistency(self):
155:         random.seed(1234)
156:         for dtype in [np.float64, np.complex128]:
157:             for n in range(1, 10):
158:                 for scale in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]:
159:                     # make logm(A) be of a given scale
160:                     A = (eye(n) + random.rand(n, n) * scale).astype(dtype)
161:                     if np.iscomplexobj(A):
162:                         A = A + 1j * random.rand(n, n) * scale
163:                     assert_array_almost_equal(expm(logm(A)), A)
164: 
165:     def test_integer_matrix(self):
166:         Q = np.array([
167:             [-3, 1, 1, 1],
168:             [1, -3, 1, 1],
169:             [1, 1, -3, 1],
170:             [1, 1, 1, -3]])
171:         assert_allclose(expm(Q), expm(1.0 * Q))
172: 
173:     def test_triangularity_perturbation(self):
174:         # Experiment (1) of
175:         # Awad H. Al-Mohy and Nicholas J. Higham (2012)
176:         # Improved Inverse Scaling and Squaring Algorithms
177:         # for the Matrix Logarithm.
178:         A = np.array([
179:             [3.2346e-1, 3e4, 3e4, 3e4],
180:             [0, 3.0089e-1, 3e4, 3e4],
181:             [0, 0, 3.221e-1, 3e4],
182:             [0, 0, 0, 3.0744e-1]],
183:             dtype=float)
184:         A_logm = np.array([
185:             [-1.12867982029050462e+00, 9.61418377142025565e+04,
186:              -4.52485573953179264e+09, 2.92496941103871812e+14],
187:             [0.00000000000000000e+00, -1.20101052953082288e+00,
188:              9.63469687211303099e+04, -4.68104828911105442e+09],
189:             [0.00000000000000000e+00, 0.00000000000000000e+00,
190:              -1.13289322264498393e+00, 9.53249183094775653e+04],
191:             [0.00000000000000000e+00, 0.00000000000000000e+00,
192:              0.00000000000000000e+00, -1.17947533272554850e+00]],
193:             dtype=float)
194:         assert_allclose(expm(A_logm), A, rtol=1e-4)
195: 
196:         # Perturb the upper triangular matrix by tiny amounts,
197:         # so that it becomes technically not upper triangular.
198:         random.seed(1234)
199:         tiny = 1e-17
200:         A_logm_perturbed = A_logm.copy()
201:         A_logm_perturbed[1, 0] = tiny
202:         with suppress_warnings() as sup:
203:             sup.filter(RuntimeWarning,
204:                        "scipy.linalg.solve\nIll-conditioned.*")
205:             A_expm_logm_perturbed = expm(A_logm_perturbed)
206:         rtol = 1e-4
207:         atol = 100 * tiny
208:         assert_(not np.allclose(A_expm_logm_perturbed, A, rtol=rtol, atol=atol))
209: 
210:     def test_burkardt_1(self):
211:         # This matrix is diagonal.
212:         # The calculation of the matrix exponential is simple.
213:         #
214:         # This is the first of a series of matrix exponential tests
215:         # collected by John Burkardt from the following sources.
216:         #
217:         # Alan Laub,
218:         # Review of "Linear System Theory" by Joao Hespanha,
219:         # SIAM Review,
220:         # Volume 52, Number 4, December 2010, pages 779--781.
221:         #
222:         # Cleve Moler and Charles Van Loan,
223:         # Nineteen Dubious Ways to Compute the Exponential of a Matrix,
224:         # Twenty-Five Years Later,
225:         # SIAM Review,
226:         # Volume 45, Number 1, March 2003, pages 3--49.
227:         #
228:         # Cleve Moler,
229:         # Cleve's Corner: A Balancing Act for the Matrix Exponential,
230:         # 23 July 2012.
231:         #
232:         # Robert Ward,
233:         # Numerical computation of the matrix exponential
234:         # with accuracy estimate,
235:         # SIAM Journal on Numerical Analysis,
236:         # Volume 14, Number 4, September 1977, pages 600--610.
237:         exp1 = np.exp(1)
238:         exp2 = np.exp(2)
239:         A = np.array([
240:             [1, 0],
241:             [0, 2],
242:             ], dtype=float)
243:         desired = np.array([
244:             [exp1, 0],
245:             [0, exp2],
246:             ], dtype=float)
247:         actual = expm(A)
248:         assert_allclose(actual, desired)
249: 
250:     def test_burkardt_2(self):
251:         # This matrix is symmetric.
252:         # The calculation of the matrix exponential is straightforward.
253:         A = np.array([
254:             [1, 3],
255:             [3, 2],
256:             ], dtype=float)
257:         desired = np.array([
258:             [39.322809708033859, 46.166301438885753],
259:             [46.166301438885768, 54.711576854329110],
260:             ], dtype=float)
261:         actual = expm(A)
262:         assert_allclose(actual, desired)
263: 
264:     def test_burkardt_3(self):
265:         # This example is due to Laub.
266:         # This matrix is ill-suited for the Taylor series approach.
267:         # As powers of A are computed, the entries blow up too quickly.
268:         exp1 = np.exp(1)
269:         exp39 = np.exp(39)
270:         A = np.array([
271:             [0, 1],
272:             [-39, -40],
273:             ], dtype=float)
274:         desired = np.array([
275:             [
276:                 39/(38*exp1) - 1/(38*exp39),
277:                 -np.expm1(-38) / (38*exp1)],
278:             [
279:                 39*np.expm1(-38) / (38*exp1),
280:                 -1/(38*exp1) + 39/(38*exp39)],
281:             ], dtype=float)
282:         actual = expm(A)
283:         assert_allclose(actual, desired)
284: 
285:     def test_burkardt_4(self):
286:         # This example is due to Moler and Van Loan.
287:         # The example will cause problems for the series summation approach,
288:         # as well as for diagonal Pade approximations.
289:         A = np.array([
290:             [-49, 24],
291:             [-64, 31],
292:             ], dtype=float)
293:         U = np.array([[3, 1], [4, 2]], dtype=float)
294:         V = np.array([[1, -1/2], [-2, 3/2]], dtype=float)
295:         w = np.array([-17, -1], dtype=float)
296:         desired = np.dot(U * np.exp(w), V)
297:         actual = expm(A)
298:         assert_allclose(actual, desired)
299: 
300:     def test_burkardt_5(self):
301:         # This example is due to Moler and Van Loan.
302:         # This matrix is strictly upper triangular
303:         # All powers of A are zero beyond some (low) limit.
304:         # This example will cause problems for Pade approximations.
305:         A = np.array([
306:             [0, 6, 0, 0],
307:             [0, 0, 6, 0],
308:             [0, 0, 0, 6],
309:             [0, 0, 0, 0],
310:             ], dtype=float)
311:         desired = np.array([
312:             [1, 6, 18, 36],
313:             [0, 1, 6, 18],
314:             [0, 0, 1, 6],
315:             [0, 0, 0, 1],
316:             ], dtype=float)
317:         actual = expm(A)
318:         assert_allclose(actual, desired)
319: 
320:     def test_burkardt_6(self):
321:         # This example is due to Moler and Van Loan.
322:         # This matrix does not have a complete set of eigenvectors.
323:         # That means the eigenvector approach will fail.
324:         exp1 = np.exp(1)
325:         A = np.array([
326:             [1, 1],
327:             [0, 1],
328:             ], dtype=float)
329:         desired = np.array([
330:             [exp1, exp1],
331:             [0, exp1],
332:             ], dtype=float)
333:         actual = expm(A)
334:         assert_allclose(actual, desired)
335: 
336:     def test_burkardt_7(self):
337:         # This example is due to Moler and Van Loan.
338:         # This matrix is very close to example 5.
339:         # Mathematically, it has a complete set of eigenvectors.
340:         # Numerically, however, the calculation will be suspect.
341:         exp1 = np.exp(1)
342:         eps = np.spacing(1)
343:         A = np.array([
344:             [1 + eps, 1],
345:             [0, 1 - eps],
346:             ], dtype=float)
347:         desired = np.array([
348:             [exp1, exp1],
349:             [0, exp1],
350:             ], dtype=float)
351:         actual = expm(A)
352:         assert_allclose(actual, desired)
353: 
354:     def test_burkardt_8(self):
355:         # This matrix was an example in Wikipedia.
356:         exp4 = np.exp(4)
357:         exp16 = np.exp(16)
358:         A = np.array([
359:             [21, 17, 6],
360:             [-5, -1, -6],
361:             [4, 4, 16],
362:             ], dtype=float)
363:         desired = np.array([
364:             [13*exp16 - exp4, 13*exp16 - 5*exp4, 2*exp16 - 2*exp4],
365:             [-9*exp16 + exp4, -9*exp16 + 5*exp4, -2*exp16 + 2*exp4],
366:             [16*exp16, 16*exp16, 4*exp16],
367:             ], dtype=float) * 0.25
368:         actual = expm(A)
369:         assert_allclose(actual, desired)
370: 
371:     def test_burkardt_9(self):
372:         # This matrix is due to the NAG Library.
373:         # It is an example for function F01ECF.
374:         A = np.array([
375:             [1, 2, 2, 2],
376:             [3, 1, 1, 2],
377:             [3, 2, 1, 2],
378:             [3, 3, 3, 1],
379:             ], dtype=float)
380:         desired = np.array([
381:             [740.7038, 610.8500, 542.2743, 549.1753],
382:             [731.2510, 603.5524, 535.0884, 542.2743],
383:             [823.7630, 679.4257, 603.5524, 610.8500],
384:             [998.4355, 823.7630, 731.2510, 740.7038],
385:             ], dtype=float)
386:         actual = expm(A)
387:         assert_allclose(actual, desired)
388: 
389:     def test_burkardt_10(self):
390:         # This is Ward's example #1.
391:         # It is defective and nonderogatory.
392:         A = np.array([
393:             [4, 2, 0],
394:             [1, 4, 1],
395:             [1, 1, 4],
396:             ], dtype=float)
397:         assert_allclose(sorted(scipy.linalg.eigvals(A)), (3, 3, 6))
398:         desired = np.array([
399:             [147.8666224463699, 183.7651386463682, 71.79703239999647],
400:             [127.7810855231823, 183.7651386463682, 91.88256932318415],
401:             [127.7810855231824, 163.6796017231806, 111.9681062463718],
402:             ], dtype=float)
403:         actual = expm(A)
404:         assert_allclose(actual, desired)
405: 
406:     def test_burkardt_11(self):
407:         # This is Ward's example #2.
408:         # It is a symmetric matrix.
409:         A = np.array([
410:             [29.87942128909879, 0.7815750847907159, -2.289519314033932],
411:             [0.7815750847907159, 25.72656945571064, 8.680737820540137],
412:             [-2.289519314033932, 8.680737820540137, 34.39400925519054],
413:             ], dtype=float)
414:         assert_allclose(scipy.linalg.eigvalsh(A), (20, 30, 40))
415:         desired = np.array([
416:              [
417:                  5.496313853692378E+15,
418:                  -1.823188097200898E+16,
419:                  -3.047577080858001E+16],
420:              [
421:                 -1.823188097200899E+16,
422:                 6.060522870222108E+16,
423:                 1.012918429302482E+17],
424:              [
425:                 -3.047577080858001E+16,
426:                 1.012918429302482E+17,
427:                 1.692944112408493E+17],
428:             ], dtype=float)
429:         actual = expm(A)
430:         assert_allclose(actual, desired)
431: 
432:     def test_burkardt_12(self):
433:         # This is Ward's example #3.
434:         # Ward's algorithm has difficulty estimating the accuracy
435:         # of its results.
436:         A = np.array([
437:             [-131, 19, 18],
438:             [-390, 56, 54],
439:             [-387, 57, 52],
440:             ], dtype=float)
441:         assert_allclose(sorted(scipy.linalg.eigvals(A)), (-20, -2, -1))
442:         desired = np.array([
443:             [-1.509644158793135, 0.3678794391096522, 0.1353352811751005],
444:             [-5.632570799891469, 1.471517758499875, 0.4060058435250609],
445:             [-4.934938326088363, 1.103638317328798, 0.5413411267617766],
446:             ], dtype=float)
447:         actual = expm(A)
448:         assert_allclose(actual, desired)
449: 
450:     def test_burkardt_13(self):
451:         # This is Ward's example #4.
452:         # This is a version of the Forsythe matrix.
453:         # The eigenvector problem is badly conditioned.
454:         # Ward's algorithm has difficulty esimating the accuracy
455:         # of its results for this problem.
456:         #
457:         # Check the construction of one instance of this family of matrices.
458:         A4_actual = _burkardt_13_power(4, 1)
459:         A4_desired = [[0, 1, 0, 0],
460:                       [0, 0, 1, 0],
461:                       [0, 0, 0, 1],
462:                       [1e-4, 0, 0, 0]]
463:         assert_allclose(A4_actual, A4_desired)
464:         # Check the expm for a few instances.
465:         for n in (2, 3, 4, 10):
466:             # Approximate expm using Taylor series.
467:             # This works well for this matrix family
468:             # because each matrix in the summation,
469:             # even before dividing by the factorial,
470:             # is entrywise positive with max entry 10**(-floor(p/n)*n).
471:             k = max(1, int(np.ceil(16/n)))
472:             desired = np.zeros((n, n), dtype=float)
473:             for p in range(n*k):
474:                 Ap = _burkardt_13_power(n, p)
475:                 assert_equal(np.min(Ap), 0)
476:                 assert_allclose(np.max(Ap), np.power(10, -np.floor(p/n)*n))
477:                 desired += Ap / factorial(p)
478:             actual = expm(_burkardt_13_power(n, 1))
479:             assert_allclose(actual, desired)
480: 
481:     def test_burkardt_14(self):
482:         # This is Moler's example.
483:         # This badly scaled matrix caused problems for MATLAB's expm().
484:         A = np.array([
485:             [0, 1e-8, 0],
486:             [-(2e10 + 4e8/6.), -3, 2e10],
487:             [200./3., 0, -200./3.],
488:             ], dtype=float)
489:         desired = np.array([
490:             [0.446849468283175, 1.54044157383952e-09, 0.462811453558774],
491:             [-5743067.77947947, -0.0152830038686819, -4526542.71278401],
492:             [0.447722977849494, 1.54270484519591e-09, 0.463480648837651],
493:             ], dtype=float)
494:         actual = expm(A)
495:         assert_allclose(actual, desired)
496: 
497: 
498: class TestOperators(object):
499: 
500:     def test_product_operator(self):
501:         random.seed(1234)
502:         n = 5
503:         k = 2
504:         nsamples = 10
505:         for i in range(nsamples):
506:             A = np.random.randn(n, n)
507:             B = np.random.randn(n, n)
508:             C = np.random.randn(n, n)
509:             D = np.random.randn(n, k)
510:             op = ProductOperator(A, B, C)
511:             assert_allclose(op.matmat(D), A.dot(B).dot(C).dot(D))
512:             assert_allclose(op.T.matmat(D), (A.dot(B).dot(C)).T.dot(D))
513: 
514:     def test_matrix_power_operator(self):
515:         random.seed(1234)
516:         n = 5
517:         k = 2
518:         p = 3
519:         nsamples = 10
520:         for i in range(nsamples):
521:             A = np.random.randn(n, n)
522:             B = np.random.randn(n, k)
523:             op = MatrixPowerOperator(A, p)
524:             assert_allclose(op.matmat(B), matrix_power(A, p).dot(B))
525:             assert_allclose(op.T.matmat(B), matrix_power(A, p).T.dot(B))
526: 
527: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_425957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', ' Test functions for scipy.linalg.matfuncs module\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import math' statement (line 9)
import math

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_425958 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_425958) is not StypyTypeError):

    if (import_425958 != 'pyd_module'):
        __import__(import_425958)
        sys_modules_425959 = sys.modules[import_425958]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_425959.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_425958)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy import array, eye, exp, random' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_425960 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy')

if (type(import_425960) is not StypyTypeError):

    if (import_425960 != 'pyd_module'):
        __import__(import_425960)
        sys_modules_425961 = sys.modules[import_425960]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', sys_modules_425961.module_type_store, module_type_store, ['array', 'eye', 'exp', 'random'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_425961, sys_modules_425961.module_type_store, module_type_store)
    else:
        from numpy import array, eye, exp, random

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', None, module_type_store, ['array', 'eye', 'exp', 'random'], [array, eye, exp, random])

else:
    # Assigning a type to the variable 'numpy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', import_425960)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.linalg import matrix_power' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_425962 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.linalg')

if (type(import_425962) is not StypyTypeError):

    if (import_425962 != 'pyd_module'):
        __import__(import_425962)
        sys_modules_425963 = sys.modules[import_425962]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.linalg', sys_modules_425963.module_type_store, module_type_store, ['matrix_power'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_425963, sys_modules_425963.module_type_store, module_type_store)
    else:
        from numpy.linalg import matrix_power

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.linalg', None, module_type_store, ['matrix_power'], [matrix_power])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.linalg', import_425962)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.testing import assert_allclose, assert_, assert_array_almost_equal, assert_equal, assert_array_almost_equal_nulp' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_425964 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing')

if (type(import_425964) is not StypyTypeError):

    if (import_425964 != 'pyd_module'):
        __import__(import_425964)
        sys_modules_425965 = sys.modules[import_425964]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', sys_modules_425965.module_type_store, module_type_store, ['assert_allclose', 'assert_', 'assert_array_almost_equal', 'assert_equal', 'assert_array_almost_equal_nulp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_425965, sys_modules_425965.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_, assert_array_almost_equal, assert_equal, assert_array_almost_equal_nulp

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_', 'assert_array_almost_equal', 'assert_equal', 'assert_array_almost_equal_nulp'], [assert_allclose, assert_, assert_array_almost_equal, assert_equal, assert_array_almost_equal_nulp])

else:
    # Assigning a type to the variable 'numpy.testing' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', import_425964)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_425966 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy._lib._numpy_compat')

if (type(import_425966) is not StypyTypeError):

    if (import_425966 != 'pyd_module'):
        __import__(import_425966)
        sys_modules_425967 = sys.modules[import_425966]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy._lib._numpy_compat', sys_modules_425967.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_425967, sys_modules_425967.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy._lib._numpy_compat', import_425966)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy.sparse import csc_matrix, SparseEfficiencyWarning' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_425968 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.sparse')

if (type(import_425968) is not StypyTypeError):

    if (import_425968 != 'pyd_module'):
        __import__(import_425968)
        sys_modules_425969 = sys.modules[import_425968]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.sparse', sys_modules_425969.module_type_store, module_type_store, ['csc_matrix', 'SparseEfficiencyWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_425969, sys_modules_425969.module_type_store, module_type_store)
    else:
        from scipy.sparse import csc_matrix, SparseEfficiencyWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.sparse', None, module_type_store, ['csc_matrix', 'SparseEfficiencyWarning'], [csc_matrix, SparseEfficiencyWarning])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.sparse', import_425968)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy.sparse.construct import speye' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_425970 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.construct')

if (type(import_425970) is not StypyTypeError):

    if (import_425970 != 'pyd_module'):
        __import__(import_425970)
        sys_modules_425971 = sys.modules[import_425970]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.construct', sys_modules_425971.module_type_store, module_type_store, ['eye'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_425971, sys_modules_425971.module_type_store, module_type_store)
    else:
        from scipy.sparse.construct import eye as speye

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.construct', None, module_type_store, ['eye'], [speye])

else:
    # Assigning a type to the variable 'scipy.sparse.construct' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.construct', import_425970)

# Adding an alias
module_type_store.add_alias('speye', 'eye')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from scipy.sparse.linalg.matfuncs import expm, _expm, ProductOperator, MatrixPowerOperator, _onenorm_matrix_power_nnm' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_425972 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.sparse.linalg.matfuncs')

if (type(import_425972) is not StypyTypeError):

    if (import_425972 != 'pyd_module'):
        __import__(import_425972)
        sys_modules_425973 = sys.modules[import_425972]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.sparse.linalg.matfuncs', sys_modules_425973.module_type_store, module_type_store, ['expm', '_expm', 'ProductOperator', 'MatrixPowerOperator', '_onenorm_matrix_power_nnm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_425973, sys_modules_425973.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.matfuncs import expm, _expm, ProductOperator, MatrixPowerOperator, _onenorm_matrix_power_nnm

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.sparse.linalg.matfuncs', None, module_type_store, ['expm', '_expm', 'ProductOperator', 'MatrixPowerOperator', '_onenorm_matrix_power_nnm'], [expm, _expm, ProductOperator, MatrixPowerOperator, _onenorm_matrix_power_nnm])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.matfuncs' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.sparse.linalg.matfuncs', import_425972)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from scipy.linalg import logm' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_425974 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.linalg')

if (type(import_425974) is not StypyTypeError):

    if (import_425974 != 'pyd_module'):
        __import__(import_425974)
        sys_modules_425975 = sys.modules[import_425974]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.linalg', sys_modules_425975.module_type_store, module_type_store, ['logm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_425975, sys_modules_425975.module_type_store, module_type_store)
    else:
        from scipy.linalg import logm

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.linalg', None, module_type_store, ['logm'], [logm])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.linalg', import_425974)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from scipy.special import factorial' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_425976 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'scipy.special')

if (type(import_425976) is not StypyTypeError):

    if (import_425976 != 'pyd_module'):
        __import__(import_425976)
        sys_modules_425977 = sys.modules[import_425976]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'scipy.special', sys_modules_425977.module_type_store, module_type_store, ['factorial'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_425977, sys_modules_425977.module_type_store, module_type_store)
    else:
        from scipy.special import factorial

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'scipy.special', None, module_type_store, ['factorial'], [factorial])

else:
    # Assigning a type to the variable 'scipy.special' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'scipy.special', import_425976)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'import scipy.sparse' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_425978 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse')

if (type(import_425978) is not StypyTypeError):

    if (import_425978 != 'pyd_module'):
        __import__(import_425978)
        sys_modules_425979 = sys.modules[import_425978]
        import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse', sys_modules_425979.module_type_store, module_type_store)
    else:
        import scipy.sparse

        import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse', scipy.sparse, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse', import_425978)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'import scipy.sparse.linalg' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_425980 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.sparse.linalg')

if (type(import_425980) is not StypyTypeError):

    if (import_425980 != 'pyd_module'):
        __import__(import_425980)
        sys_modules_425981 = sys.modules[import_425980]
        import_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.sparse.linalg', sys_modules_425981.module_type_store, module_type_store)
    else:
        import scipy.sparse.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.sparse.linalg', scipy.sparse.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.sparse.linalg', import_425980)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')


@norecursion
def _burkardt_13_power(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_burkardt_13_power'
    module_type_store = module_type_store.open_function_context('_burkardt_13_power', 30, 0, False)
    
    # Passed parameters checking function
    _burkardt_13_power.stypy_localization = localization
    _burkardt_13_power.stypy_type_of_self = None
    _burkardt_13_power.stypy_type_store = module_type_store
    _burkardt_13_power.stypy_function_name = '_burkardt_13_power'
    _burkardt_13_power.stypy_param_names_list = ['n', 'p']
    _burkardt_13_power.stypy_varargs_param_name = None
    _burkardt_13_power.stypy_kwargs_param_name = None
    _burkardt_13_power.stypy_call_defaults = defaults
    _burkardt_13_power.stypy_call_varargs = varargs
    _burkardt_13_power.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_burkardt_13_power', ['n', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_burkardt_13_power', localization, ['n', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_burkardt_13_power(...)' code ##################

    str_425982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, (-1)), 'str', '\n    A helper function for testing matrix functions.\n\n    Parameters\n    ----------\n    n : integer greater than 1\n        Order of the square matrix to be returned.\n    p : non-negative integer\n        Power of the matrix.\n\n    Returns\n    -------\n    out : ndarray representing a square matrix\n        A Forsythe matrix of order n, raised to the power p.\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'n' (line 48)
    n_425983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 7), 'n')
    
    # Call to int(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'n' (line 48)
    n_425985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'n', False)
    # Processing the call keyword arguments (line 48)
    kwargs_425986 = {}
    # Getting the type of 'int' (line 48)
    int_425984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'int', False)
    # Calling int(args, kwargs) (line 48)
    int_call_result_425987 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), int_425984, *[n_425985], **kwargs_425986)
    
    # Applying the binary operator '!=' (line 48)
    result_ne_425988 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 7), '!=', n_425983, int_call_result_425987)
    
    
    # Getting the type of 'n' (line 48)
    n_425989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'n')
    int_425990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 26), 'int')
    # Applying the binary operator '<' (line 48)
    result_lt_425991 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 22), '<', n_425989, int_425990)
    
    # Applying the binary operator 'or' (line 48)
    result_or_keyword_425992 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 7), 'or', result_ne_425988, result_lt_425991)
    
    # Testing the type of an if condition (line 48)
    if_condition_425993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 4), result_or_keyword_425992)
    # Assigning a type to the variable 'if_condition_425993' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'if_condition_425993', if_condition_425993)
    # SSA begins for if statement (line 48)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 49)
    # Processing the call arguments (line 49)
    str_425995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 25), 'str', 'n must be an integer greater than 1')
    # Processing the call keyword arguments (line 49)
    kwargs_425996 = {}
    # Getting the type of 'ValueError' (line 49)
    ValueError_425994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 49)
    ValueError_call_result_425997 = invoke(stypy.reporting.localization.Localization(__file__, 49, 14), ValueError_425994, *[str_425995], **kwargs_425996)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 49, 8), ValueError_call_result_425997, 'raise parameter', BaseException)
    # SSA join for if statement (line 48)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 50):
    
    # Assigning a Call to a Name (line 50):
    
    # Call to int(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'n' (line 50)
    n_425999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'n', False)
    # Processing the call keyword arguments (line 50)
    kwargs_426000 = {}
    # Getting the type of 'int' (line 50)
    int_425998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'int', False)
    # Calling int(args, kwargs) (line 50)
    int_call_result_426001 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), int_425998, *[n_425999], **kwargs_426000)
    
    # Assigning a type to the variable 'n' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'n', int_call_result_426001)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'p' (line 51)
    p_426002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 7), 'p')
    
    # Call to int(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'p' (line 51)
    p_426004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'p', False)
    # Processing the call keyword arguments (line 51)
    kwargs_426005 = {}
    # Getting the type of 'int' (line 51)
    int_426003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'int', False)
    # Calling int(args, kwargs) (line 51)
    int_call_result_426006 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), int_426003, *[p_426004], **kwargs_426005)
    
    # Applying the binary operator '!=' (line 51)
    result_ne_426007 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 7), '!=', p_426002, int_call_result_426006)
    
    
    # Getting the type of 'p' (line 51)
    p_426008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'p')
    int_426009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 26), 'int')
    # Applying the binary operator '<' (line 51)
    result_lt_426010 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 22), '<', p_426008, int_426009)
    
    # Applying the binary operator 'or' (line 51)
    result_or_keyword_426011 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 7), 'or', result_ne_426007, result_lt_426010)
    
    # Testing the type of an if condition (line 51)
    if_condition_426012 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 4), result_or_keyword_426011)
    # Assigning a type to the variable 'if_condition_426012' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'if_condition_426012', if_condition_426012)
    # SSA begins for if statement (line 51)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 52)
    # Processing the call arguments (line 52)
    str_426014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 25), 'str', 'p must be a non-negative integer')
    # Processing the call keyword arguments (line 52)
    kwargs_426015 = {}
    # Getting the type of 'ValueError' (line 52)
    ValueError_426013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 52)
    ValueError_call_result_426016 = invoke(stypy.reporting.localization.Localization(__file__, 52, 14), ValueError_426013, *[str_426014], **kwargs_426015)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 52, 8), ValueError_call_result_426016, 'raise parameter', BaseException)
    # SSA join for if statement (line 51)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 53):
    
    # Assigning a Call to a Name (line 53):
    
    # Call to int(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'p' (line 53)
    p_426018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'p', False)
    # Processing the call keyword arguments (line 53)
    kwargs_426019 = {}
    # Getting the type of 'int' (line 53)
    int_426017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'int', False)
    # Calling int(args, kwargs) (line 53)
    int_call_result_426020 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), int_426017, *[p_426018], **kwargs_426019)
    
    # Assigning a type to the variable 'p' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'p', int_call_result_426020)
    
    # Assigning a Call to a Tuple (line 56):
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    int_426021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'int')
    
    # Call to divmod(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'p' (line 56)
    p_426023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'p', False)
    # Getting the type of 'n' (line 56)
    n_426024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'n', False)
    # Processing the call keyword arguments (line 56)
    kwargs_426025 = {}
    # Getting the type of 'divmod' (line 56)
    divmod_426022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'divmod', False)
    # Calling divmod(args, kwargs) (line 56)
    divmod_call_result_426026 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), divmod_426022, *[p_426023, n_426024], **kwargs_426025)
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___426027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), divmod_call_result_426026, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_426028 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), getitem___426027, int_426021)
    
    # Assigning a type to the variable 'tuple_var_assignment_425955' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_425955', subscript_call_result_426028)
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    int_426029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'int')
    
    # Call to divmod(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'p' (line 56)
    p_426031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'p', False)
    # Getting the type of 'n' (line 56)
    n_426032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'n', False)
    # Processing the call keyword arguments (line 56)
    kwargs_426033 = {}
    # Getting the type of 'divmod' (line 56)
    divmod_426030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'divmod', False)
    # Calling divmod(args, kwargs) (line 56)
    divmod_call_result_426034 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), divmod_426030, *[p_426031, n_426032], **kwargs_426033)
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___426035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), divmod_call_result_426034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_426036 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), getitem___426035, int_426029)
    
    # Assigning a type to the variable 'tuple_var_assignment_425956' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_425956', subscript_call_result_426036)
    
    # Assigning a Name to a Name (line 56):
    # Getting the type of 'tuple_var_assignment_425955' (line 56)
    tuple_var_assignment_425955_426037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_425955')
    # Assigning a type to the variable 'a' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'a', tuple_var_assignment_425955_426037)
    
    # Assigning a Name to a Name (line 56):
    # Getting the type of 'tuple_var_assignment_425956' (line 56)
    tuple_var_assignment_425956_426038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_425956')
    # Assigning a type to the variable 'b' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 7), 'b', tuple_var_assignment_425956_426038)
    
    # Assigning a Call to a Name (line 57):
    
    # Assigning a Call to a Name (line 57):
    
    # Call to power(...): (line 57)
    # Processing the call arguments (line 57)
    float_426041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 21), 'float')
    
    # Getting the type of 'n' (line 57)
    n_426042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'n', False)
    # Applying the 'usub' unary operator (line 57)
    result___neg___426043 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 27), 'usub', n_426042)
    
    # Getting the type of 'a' (line 57)
    a_426044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'a', False)
    # Applying the binary operator '*' (line 57)
    result_mul_426045 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 27), '*', result___neg___426043, a_426044)
    
    # Processing the call keyword arguments (line 57)
    kwargs_426046 = {}
    # Getting the type of 'np' (line 57)
    np_426039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'np', False)
    # Obtaining the member 'power' of a type (line 57)
    power_426040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), np_426039, 'power')
    # Calling power(args, kwargs) (line 57)
    power_call_result_426047 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), power_426040, *[float_426041, result_mul_426045], **kwargs_426046)
    
    # Assigning a type to the variable 'large' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'large', power_call_result_426047)
    
    # Assigning a BinOp to a Name (line 58):
    
    # Assigning a BinOp to a Name (line 58):
    # Getting the type of 'large' (line 58)
    large_426048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'large')
    
    # Call to power(...): (line 58)
    # Processing the call arguments (line 58)
    float_426051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 29), 'float')
    
    # Getting the type of 'n' (line 58)
    n_426052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 36), 'n', False)
    # Applying the 'usub' unary operator (line 58)
    result___neg___426053 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 35), 'usub', n_426052)
    
    # Processing the call keyword arguments (line 58)
    kwargs_426054 = {}
    # Getting the type of 'np' (line 58)
    np_426049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'np', False)
    # Obtaining the member 'power' of a type (line 58)
    power_426050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 20), np_426049, 'power')
    # Calling power(args, kwargs) (line 58)
    power_call_result_426055 = invoke(stypy.reporting.localization.Localization(__file__, 58, 20), power_426050, *[float_426051, result___neg___426053], **kwargs_426054)
    
    # Applying the binary operator '*' (line 58)
    result_mul_426056 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 12), '*', large_426048, power_call_result_426055)
    
    # Assigning a type to the variable 'small' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'small', result_mul_426056)
    
    # Call to diag(...): (line 59)
    # Processing the call arguments (line 59)
    
    # Obtaining an instance of the builtin type 'list' (line 59)
    list_426059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 59)
    # Adding element type (line 59)
    # Getting the type of 'large' (line 59)
    large_426060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'large', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), list_426059, large_426060)
    
    # Getting the type of 'n' (line 59)
    n_426061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 28), 'n', False)
    # Getting the type of 'b' (line 59)
    b_426062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 30), 'b', False)
    # Applying the binary operator '-' (line 59)
    result_sub_426063 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 28), '-', n_426061, b_426062)
    
    # Applying the binary operator '*' (line 59)
    result_mul_426064 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 19), '*', list_426059, result_sub_426063)
    
    # Getting the type of 'b' (line 59)
    b_426065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 34), 'b', False)
    # Processing the call keyword arguments (line 59)
    kwargs_426066 = {}
    # Getting the type of 'np' (line 59)
    np_426057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'np', False)
    # Obtaining the member 'diag' of a type (line 59)
    diag_426058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 11), np_426057, 'diag')
    # Calling diag(args, kwargs) (line 59)
    diag_call_result_426067 = invoke(stypy.reporting.localization.Localization(__file__, 59, 11), diag_426058, *[result_mul_426064, b_426065], **kwargs_426066)
    
    
    # Call to diag(...): (line 59)
    # Processing the call arguments (line 59)
    
    # Obtaining an instance of the builtin type 'list' (line 59)
    list_426070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 47), 'list')
    # Adding type elements to the builtin type 'list' instance (line 59)
    # Adding element type (line 59)
    # Getting the type of 'small' (line 59)
    small_426071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 48), 'small', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 47), list_426070, small_426071)
    
    # Getting the type of 'b' (line 59)
    b_426072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 55), 'b', False)
    # Applying the binary operator '*' (line 59)
    result_mul_426073 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 47), '*', list_426070, b_426072)
    
    # Getting the type of 'b' (line 59)
    b_426074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 58), 'b', False)
    # Getting the type of 'n' (line 59)
    n_426075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 60), 'n', False)
    # Applying the binary operator '-' (line 59)
    result_sub_426076 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 58), '-', b_426074, n_426075)
    
    # Processing the call keyword arguments (line 59)
    kwargs_426077 = {}
    # Getting the type of 'np' (line 59)
    np_426068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 39), 'np', False)
    # Obtaining the member 'diag' of a type (line 59)
    diag_426069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 39), np_426068, 'diag')
    # Calling diag(args, kwargs) (line 59)
    diag_call_result_426078 = invoke(stypy.reporting.localization.Localization(__file__, 59, 39), diag_426069, *[result_mul_426073, result_sub_426076], **kwargs_426077)
    
    # Applying the binary operator '+' (line 59)
    result_add_426079 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 11), '+', diag_call_result_426067, diag_call_result_426078)
    
    # Assigning a type to the variable 'stypy_return_type' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type', result_add_426079)
    
    # ################# End of '_burkardt_13_power(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_burkardt_13_power' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_426080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_426080)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_burkardt_13_power'
    return stypy_return_type_426080

# Assigning a type to the variable '_burkardt_13_power' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), '_burkardt_13_power', _burkardt_13_power)

@norecursion
def test_onenorm_matrix_power_nnm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_onenorm_matrix_power_nnm'
    module_type_store = module_type_store.open_function_context('test_onenorm_matrix_power_nnm', 62, 0, False)
    
    # Passed parameters checking function
    test_onenorm_matrix_power_nnm.stypy_localization = localization
    test_onenorm_matrix_power_nnm.stypy_type_of_self = None
    test_onenorm_matrix_power_nnm.stypy_type_store = module_type_store
    test_onenorm_matrix_power_nnm.stypy_function_name = 'test_onenorm_matrix_power_nnm'
    test_onenorm_matrix_power_nnm.stypy_param_names_list = []
    test_onenorm_matrix_power_nnm.stypy_varargs_param_name = None
    test_onenorm_matrix_power_nnm.stypy_kwargs_param_name = None
    test_onenorm_matrix_power_nnm.stypy_call_defaults = defaults
    test_onenorm_matrix_power_nnm.stypy_call_varargs = varargs
    test_onenorm_matrix_power_nnm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_onenorm_matrix_power_nnm', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_onenorm_matrix_power_nnm', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_onenorm_matrix_power_nnm(...)' code ##################

    
    # Call to seed(...): (line 63)
    # Processing the call arguments (line 63)
    int_426084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 19), 'int')
    # Processing the call keyword arguments (line 63)
    kwargs_426085 = {}
    # Getting the type of 'np' (line 63)
    np_426081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 63)
    random_426082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 4), np_426081, 'random')
    # Obtaining the member 'seed' of a type (line 63)
    seed_426083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 4), random_426082, 'seed')
    # Calling seed(args, kwargs) (line 63)
    seed_call_result_426086 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), seed_426083, *[int_426084], **kwargs_426085)
    
    
    
    # Call to range(...): (line 64)
    # Processing the call arguments (line 64)
    int_426088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 19), 'int')
    int_426089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 22), 'int')
    # Processing the call keyword arguments (line 64)
    kwargs_426090 = {}
    # Getting the type of 'range' (line 64)
    range_426087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'range', False)
    # Calling range(args, kwargs) (line 64)
    range_call_result_426091 = invoke(stypy.reporting.localization.Localization(__file__, 64, 13), range_426087, *[int_426088, int_426089], **kwargs_426090)
    
    # Testing the type of a for loop iterable (line 64)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 64, 4), range_call_result_426091)
    # Getting the type of the for loop variable (line 64)
    for_loop_var_426092 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 64, 4), range_call_result_426091)
    # Assigning a type to the variable 'n' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'n', for_loop_var_426092)
    # SSA begins for a for statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 65)
    # Processing the call arguments (line 65)
    int_426094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 23), 'int')
    # Processing the call keyword arguments (line 65)
    kwargs_426095 = {}
    # Getting the type of 'range' (line 65)
    range_426093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 17), 'range', False)
    # Calling range(args, kwargs) (line 65)
    range_call_result_426096 = invoke(stypy.reporting.localization.Localization(__file__, 65, 17), range_426093, *[int_426094], **kwargs_426095)
    
    # Testing the type of a for loop iterable (line 65)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 65, 8), range_call_result_426096)
    # Getting the type of the for loop variable (line 65)
    for_loop_var_426097 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 65, 8), range_call_result_426096)
    # Assigning a type to the variable 'p' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'p', for_loop_var_426097)
    # SSA begins for a for statement (line 65)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 66):
    
    # Assigning a Call to a Name (line 66):
    
    # Call to random(...): (line 66)
    # Processing the call arguments (line 66)
    
    # Obtaining an instance of the builtin type 'tuple' (line 66)
    tuple_426101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 66)
    # Adding element type (line 66)
    # Getting the type of 'n' (line 66)
    n_426102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 34), tuple_426101, n_426102)
    # Adding element type (line 66)
    # Getting the type of 'n' (line 66)
    n_426103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 37), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 34), tuple_426101, n_426103)
    
    # Processing the call keyword arguments (line 66)
    kwargs_426104 = {}
    # Getting the type of 'np' (line 66)
    np_426098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'np', False)
    # Obtaining the member 'random' of a type (line 66)
    random_426099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 16), np_426098, 'random')
    # Obtaining the member 'random' of a type (line 66)
    random_426100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 16), random_426099, 'random')
    # Calling random(args, kwargs) (line 66)
    random_call_result_426105 = invoke(stypy.reporting.localization.Localization(__file__, 66, 16), random_426100, *[tuple_426101], **kwargs_426104)
    
    # Assigning a type to the variable 'M' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'M', random_call_result_426105)
    
    # Assigning a Call to a Name (line 67):
    
    # Assigning a Call to a Name (line 67):
    
    # Call to matrix_power(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'M' (line 67)
    M_426109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 40), 'M', False)
    # Getting the type of 'p' (line 67)
    p_426110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 43), 'p', False)
    # Processing the call keyword arguments (line 67)
    kwargs_426111 = {}
    # Getting the type of 'np' (line 67)
    np_426106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'np', False)
    # Obtaining the member 'linalg' of a type (line 67)
    linalg_426107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 17), np_426106, 'linalg')
    # Obtaining the member 'matrix_power' of a type (line 67)
    matrix_power_426108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 17), linalg_426107, 'matrix_power')
    # Calling matrix_power(args, kwargs) (line 67)
    matrix_power_call_result_426112 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), matrix_power_426108, *[M_426109, p_426110], **kwargs_426111)
    
    # Assigning a type to the variable 'Mp' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'Mp', matrix_power_call_result_426112)
    
    # Assigning a Call to a Name (line 68):
    
    # Assigning a Call to a Name (line 68):
    
    # Call to _onenorm_matrix_power_nnm(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'M' (line 68)
    M_426114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 49), 'M', False)
    # Getting the type of 'p' (line 68)
    p_426115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 52), 'p', False)
    # Processing the call keyword arguments (line 68)
    kwargs_426116 = {}
    # Getting the type of '_onenorm_matrix_power_nnm' (line 68)
    _onenorm_matrix_power_nnm_426113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), '_onenorm_matrix_power_nnm', False)
    # Calling _onenorm_matrix_power_nnm(args, kwargs) (line 68)
    _onenorm_matrix_power_nnm_call_result_426117 = invoke(stypy.reporting.localization.Localization(__file__, 68, 23), _onenorm_matrix_power_nnm_426113, *[M_426114, p_426115], **kwargs_426116)
    
    # Assigning a type to the variable 'observed' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'observed', _onenorm_matrix_power_nnm_call_result_426117)
    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to norm(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'Mp' (line 69)
    Mp_426121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 38), 'Mp', False)
    int_426122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 42), 'int')
    # Processing the call keyword arguments (line 69)
    kwargs_426123 = {}
    # Getting the type of 'np' (line 69)
    np_426118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'np', False)
    # Obtaining the member 'linalg' of a type (line 69)
    linalg_426119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 23), np_426118, 'linalg')
    # Obtaining the member 'norm' of a type (line 69)
    norm_426120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 23), linalg_426119, 'norm')
    # Calling norm(args, kwargs) (line 69)
    norm_call_result_426124 = invoke(stypy.reporting.localization.Localization(__file__, 69, 23), norm_426120, *[Mp_426121, int_426122], **kwargs_426123)
    
    # Assigning a type to the variable 'expected' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'expected', norm_call_result_426124)
    
    # Call to assert_allclose(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'observed' (line 70)
    observed_426126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'observed', False)
    # Getting the type of 'expected' (line 70)
    expected_426127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'expected', False)
    # Processing the call keyword arguments (line 70)
    kwargs_426128 = {}
    # Getting the type of 'assert_allclose' (line 70)
    assert_allclose_426125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 70)
    assert_allclose_call_result_426129 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), assert_allclose_426125, *[observed_426126, expected_426127], **kwargs_426128)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_onenorm_matrix_power_nnm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_onenorm_matrix_power_nnm' in the type store
    # Getting the type of 'stypy_return_type' (line 62)
    stypy_return_type_426130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_426130)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_onenorm_matrix_power_nnm'
    return stypy_return_type_426130

# Assigning a type to the variable 'test_onenorm_matrix_power_nnm' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'test_onenorm_matrix_power_nnm', test_onenorm_matrix_power_nnm)
# Declaration of the 'TestExpM' class

class TestExpM(object, ):

    @norecursion
    def test_zero_ndarray(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_zero_ndarray'
        module_type_store = module_type_store.open_function_context('test_zero_ndarray', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_zero_ndarray.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_zero_ndarray.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_zero_ndarray.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_zero_ndarray.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_zero_ndarray')
        TestExpM.test_zero_ndarray.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_zero_ndarray.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_zero_ndarray.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_zero_ndarray.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_zero_ndarray.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_zero_ndarray.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_zero_ndarray.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_zero_ndarray', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_zero_ndarray', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_zero_ndarray(...)' code ##################

        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to array(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_426132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_426133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        float_426134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 19), list_426133, float_426134)
        # Adding element type (line 75)
        int_426135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 19), list_426133, int_426135)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 18), list_426132, list_426133)
        # Adding element type (line 75)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_426136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        int_426137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 26), list_426136, int_426137)
        # Adding element type (line 75)
        int_426138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 26), list_426136, int_426138)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 18), list_426132, list_426136)
        
        # Processing the call keyword arguments (line 75)
        kwargs_426139 = {}
        # Getting the type of 'array' (line 75)
        array_426131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'array', False)
        # Calling array(args, kwargs) (line 75)
        array_call_result_426140 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), array_426131, *[list_426132], **kwargs_426139)
        
        # Assigning a type to the variable 'a' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'a', array_call_result_426140)
        
        # Call to assert_array_almost_equal(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Call to expm(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'a' (line 76)
        a_426143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 39), 'a', False)
        # Processing the call keyword arguments (line 76)
        kwargs_426144 = {}
        # Getting the type of 'expm' (line 76)
        expm_426142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 34), 'expm', False)
        # Calling expm(args, kwargs) (line 76)
        expm_call_result_426145 = invoke(stypy.reporting.localization.Localization(__file__, 76, 34), expm_426142, *[a_426143], **kwargs_426144)
        
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_426146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_426147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        int_426148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 43), list_426147, int_426148)
        # Adding element type (line 76)
        int_426149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 43), list_426147, int_426149)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 42), list_426146, list_426147)
        # Adding element type (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_426150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        int_426151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 49), list_426150, int_426151)
        # Adding element type (line 76)
        int_426152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 49), list_426150, int_426152)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 42), list_426146, list_426150)
        
        # Processing the call keyword arguments (line 76)
        kwargs_426153 = {}
        # Getting the type of 'assert_array_almost_equal' (line 76)
        assert_array_almost_equal_426141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 76)
        assert_array_almost_equal_call_result_426154 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), assert_array_almost_equal_426141, *[expm_call_result_426145, list_426146], **kwargs_426153)
        
        
        # ################# End of 'test_zero_ndarray(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_zero_ndarray' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_426155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426155)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_zero_ndarray'
        return stypy_return_type_426155


    @norecursion
    def test_zero_sparse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_zero_sparse'
        module_type_store = module_type_store.open_function_context('test_zero_sparse', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_zero_sparse.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_zero_sparse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_zero_sparse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_zero_sparse.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_zero_sparse')
        TestExpM.test_zero_sparse.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_zero_sparse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_zero_sparse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_zero_sparse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_zero_sparse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_zero_sparse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_zero_sparse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_zero_sparse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_zero_sparse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_zero_sparse(...)' code ##################

        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to csc_matrix(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_426157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_426158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        float_426159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 24), list_426158, float_426159)
        # Adding element type (line 79)
        int_426160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 24), list_426158, int_426160)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 23), list_426157, list_426158)
        # Adding element type (line 79)
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_426161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        int_426162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 31), list_426161, int_426162)
        # Adding element type (line 79)
        int_426163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 31), list_426161, int_426163)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 23), list_426157, list_426161)
        
        # Processing the call keyword arguments (line 79)
        kwargs_426164 = {}
        # Getting the type of 'csc_matrix' (line 79)
        csc_matrix_426156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 79)
        csc_matrix_call_result_426165 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), csc_matrix_426156, *[list_426157], **kwargs_426164)
        
        # Assigning a type to the variable 'a' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'a', csc_matrix_call_result_426165)
        
        # Call to assert_array_almost_equal(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Call to toarray(...): (line 80)
        # Processing the call keyword arguments (line 80)
        kwargs_426172 = {}
        
        # Call to expm(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'a' (line 80)
        a_426168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 39), 'a', False)
        # Processing the call keyword arguments (line 80)
        kwargs_426169 = {}
        # Getting the type of 'expm' (line 80)
        expm_426167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 'expm', False)
        # Calling expm(args, kwargs) (line 80)
        expm_call_result_426170 = invoke(stypy.reporting.localization.Localization(__file__, 80, 34), expm_426167, *[a_426168], **kwargs_426169)
        
        # Obtaining the member 'toarray' of a type (line 80)
        toarray_426171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 34), expm_call_result_426170, 'toarray')
        # Calling toarray(args, kwargs) (line 80)
        toarray_call_result_426173 = invoke(stypy.reporting.localization.Localization(__file__, 80, 34), toarray_426171, *[], **kwargs_426172)
        
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_426174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_426175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        int_426176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 53), list_426175, int_426176)
        # Adding element type (line 80)
        int_426177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 53), list_426175, int_426177)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 52), list_426174, list_426175)
        # Adding element type (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_426178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 59), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        int_426179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 59), list_426178, int_426179)
        # Adding element type (line 80)
        int_426180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 59), list_426178, int_426180)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 52), list_426174, list_426178)
        
        # Processing the call keyword arguments (line 80)
        kwargs_426181 = {}
        # Getting the type of 'assert_array_almost_equal' (line 80)
        assert_array_almost_equal_426166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 80)
        assert_array_almost_equal_call_result_426182 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assert_array_almost_equal_426166, *[toarray_call_result_426173, list_426174], **kwargs_426181)
        
        
        # ################# End of 'test_zero_sparse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_zero_sparse' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_426183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426183)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_zero_sparse'
        return stypy_return_type_426183


    @norecursion
    def test_zero_matrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_zero_matrix'
        module_type_store = module_type_store.open_function_context('test_zero_matrix', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_zero_matrix.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_zero_matrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_zero_matrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_zero_matrix.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_zero_matrix')
        TestExpM.test_zero_matrix.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_zero_matrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_zero_matrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_zero_matrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_zero_matrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_zero_matrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_zero_matrix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_zero_matrix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_zero_matrix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_zero_matrix(...)' code ##################

        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to matrix(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_426186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_426187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        float_426188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 23), list_426187, float_426188)
        # Adding element type (line 83)
        int_426189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 23), list_426187, int_426189)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 22), list_426186, list_426187)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_426190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        int_426191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 30), list_426190, int_426191)
        # Adding element type (line 83)
        int_426192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 30), list_426190, int_426192)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 22), list_426186, list_426190)
        
        # Processing the call keyword arguments (line 83)
        kwargs_426193 = {}
        # Getting the type of 'np' (line 83)
        np_426184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'np', False)
        # Obtaining the member 'matrix' of a type (line 83)
        matrix_426185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), np_426184, 'matrix')
        # Calling matrix(args, kwargs) (line 83)
        matrix_call_result_426194 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), matrix_426185, *[list_426186], **kwargs_426193)
        
        # Assigning a type to the variable 'a' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'a', matrix_call_result_426194)
        
        # Call to assert_array_almost_equal(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Call to expm(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'a' (line 84)
        a_426197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 39), 'a', False)
        # Processing the call keyword arguments (line 84)
        kwargs_426198 = {}
        # Getting the type of 'expm' (line 84)
        expm_426196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 34), 'expm', False)
        # Calling expm(args, kwargs) (line 84)
        expm_call_result_426199 = invoke(stypy.reporting.localization.Localization(__file__, 84, 34), expm_426196, *[a_426197], **kwargs_426198)
        
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_426200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_426201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        int_426202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 43), list_426201, int_426202)
        # Adding element type (line 84)
        int_426203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 43), list_426201, int_426203)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 42), list_426200, list_426201)
        # Adding element type (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_426204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        int_426205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 49), list_426204, int_426205)
        # Adding element type (line 84)
        int_426206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 49), list_426204, int_426206)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 42), list_426200, list_426204)
        
        # Processing the call keyword arguments (line 84)
        kwargs_426207 = {}
        # Getting the type of 'assert_array_almost_equal' (line 84)
        assert_array_almost_equal_426195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 84)
        assert_array_almost_equal_call_result_426208 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), assert_array_almost_equal_426195, *[expm_call_result_426199, list_426200], **kwargs_426207)
        
        
        # ################# End of 'test_zero_matrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_zero_matrix' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_426209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426209)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_zero_matrix'
        return stypy_return_type_426209


    @norecursion
    def test_misc_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_misc_types'
        module_type_store = module_type_store.open_function_context('test_misc_types', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_misc_types.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_misc_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_misc_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_misc_types.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_misc_types')
        TestExpM.test_misc_types.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_misc_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_misc_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_misc_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_misc_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_misc_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_misc_types.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_misc_types', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_misc_types', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_misc_types(...)' code ##################

        
        # Assigning a Call to a Name (line 87):
        
        # Assigning a Call to a Name (line 87):
        
        # Call to expm(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Call to array(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_426213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_426214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        int_426215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 27), list_426214, int_426215)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 26), list_426213, list_426214)
        
        # Processing the call keyword arguments (line 87)
        kwargs_426216 = {}
        # Getting the type of 'np' (line 87)
        np_426211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 87)
        array_426212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 17), np_426211, 'array')
        # Calling array(args, kwargs) (line 87)
        array_call_result_426217 = invoke(stypy.reporting.localization.Localization(__file__, 87, 17), array_426212, *[list_426213], **kwargs_426216)
        
        # Processing the call keyword arguments (line 87)
        kwargs_426218 = {}
        # Getting the type of 'expm' (line 87)
        expm_426210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'expm', False)
        # Calling expm(args, kwargs) (line 87)
        expm_call_result_426219 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), expm_426210, *[array_call_result_426217], **kwargs_426218)
        
        # Assigning a type to the variable 'A' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'A', expm_call_result_426219)
        
        # Call to assert_allclose(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to expm(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_426222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_426223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        int_426224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 31), tuple_426223, int_426224)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 30), tuple_426222, tuple_426223)
        
        # Processing the call keyword arguments (line 88)
        kwargs_426225 = {}
        # Getting the type of 'expm' (line 88)
        expm_426221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'expm', False)
        # Calling expm(args, kwargs) (line 88)
        expm_call_result_426226 = invoke(stypy.reporting.localization.Localization(__file__, 88, 24), expm_426221, *[tuple_426222], **kwargs_426225)
        
        # Getting the type of 'A' (line 88)
        A_426227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 39), 'A', False)
        # Processing the call keyword arguments (line 88)
        kwargs_426228 = {}
        # Getting the type of 'assert_allclose' (line 88)
        assert_allclose_426220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 88)
        assert_allclose_call_result_426229 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), assert_allclose_426220, *[expm_call_result_426226, A_426227], **kwargs_426228)
        
        
        # Call to assert_allclose(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to expm(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_426232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_426233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        int_426234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 30), list_426233, int_426234)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 29), list_426232, list_426233)
        
        # Processing the call keyword arguments (line 89)
        kwargs_426235 = {}
        # Getting the type of 'expm' (line 89)
        expm_426231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'expm', False)
        # Calling expm(args, kwargs) (line 89)
        expm_call_result_426236 = invoke(stypy.reporting.localization.Localization(__file__, 89, 24), expm_426231, *[list_426232], **kwargs_426235)
        
        # Getting the type of 'A' (line 89)
        A_426237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 37), 'A', False)
        # Processing the call keyword arguments (line 89)
        kwargs_426238 = {}
        # Getting the type of 'assert_allclose' (line 89)
        assert_allclose_426230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 89)
        assert_allclose_call_result_426239 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assert_allclose_426230, *[expm_call_result_426236, A_426237], **kwargs_426238)
        
        
        # Call to assert_allclose(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to expm(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to matrix(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_426244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_426245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        int_426246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 40), list_426245, int_426246)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 39), list_426244, list_426245)
        
        # Processing the call keyword arguments (line 90)
        kwargs_426247 = {}
        # Getting the type of 'np' (line 90)
        np_426242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 29), 'np', False)
        # Obtaining the member 'matrix' of a type (line 90)
        matrix_426243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 29), np_426242, 'matrix')
        # Calling matrix(args, kwargs) (line 90)
        matrix_call_result_426248 = invoke(stypy.reporting.localization.Localization(__file__, 90, 29), matrix_426243, *[list_426244], **kwargs_426247)
        
        # Processing the call keyword arguments (line 90)
        kwargs_426249 = {}
        # Getting the type of 'expm' (line 90)
        expm_426241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'expm', False)
        # Calling expm(args, kwargs) (line 90)
        expm_call_result_426250 = invoke(stypy.reporting.localization.Localization(__file__, 90, 24), expm_426241, *[matrix_call_result_426248], **kwargs_426249)
        
        # Getting the type of 'A' (line 90)
        A_426251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 48), 'A', False)
        # Processing the call keyword arguments (line 90)
        kwargs_426252 = {}
        # Getting the type of 'assert_allclose' (line 90)
        assert_allclose_426240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 90)
        assert_allclose_call_result_426253 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), assert_allclose_426240, *[expm_call_result_426250, A_426251], **kwargs_426252)
        
        
        # Call to assert_allclose(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to expm(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to array(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_426258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_426259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        int_426260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 39), list_426259, int_426260)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 38), list_426258, list_426259)
        
        # Processing the call keyword arguments (line 91)
        kwargs_426261 = {}
        # Getting the type of 'np' (line 91)
        np_426256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'np', False)
        # Obtaining the member 'array' of a type (line 91)
        array_426257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 29), np_426256, 'array')
        # Calling array(args, kwargs) (line 91)
        array_call_result_426262 = invoke(stypy.reporting.localization.Localization(__file__, 91, 29), array_426257, *[list_426258], **kwargs_426261)
        
        # Processing the call keyword arguments (line 91)
        kwargs_426263 = {}
        # Getting the type of 'expm' (line 91)
        expm_426255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'expm', False)
        # Calling expm(args, kwargs) (line 91)
        expm_call_result_426264 = invoke(stypy.reporting.localization.Localization(__file__, 91, 24), expm_426255, *[array_call_result_426262], **kwargs_426263)
        
        # Getting the type of 'A' (line 91)
        A_426265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 47), 'A', False)
        # Processing the call keyword arguments (line 91)
        kwargs_426266 = {}
        # Getting the type of 'assert_allclose' (line 91)
        assert_allclose_426254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 91)
        assert_allclose_call_result_426267 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), assert_allclose_426254, *[expm_call_result_426264, A_426265], **kwargs_426266)
        
        
        # Call to assert_allclose(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to expm(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to csc_matrix(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_426271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_426272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        int_426273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 41), list_426272, int_426273)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 40), list_426271, list_426272)
        
        # Processing the call keyword arguments (line 92)
        kwargs_426274 = {}
        # Getting the type of 'csc_matrix' (line 92)
        csc_matrix_426270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 29), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 92)
        csc_matrix_call_result_426275 = invoke(stypy.reporting.localization.Localization(__file__, 92, 29), csc_matrix_426270, *[list_426271], **kwargs_426274)
        
        # Processing the call keyword arguments (line 92)
        kwargs_426276 = {}
        # Getting the type of 'expm' (line 92)
        expm_426269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'expm', False)
        # Calling expm(args, kwargs) (line 92)
        expm_call_result_426277 = invoke(stypy.reporting.localization.Localization(__file__, 92, 24), expm_426269, *[csc_matrix_call_result_426275], **kwargs_426276)
        
        # Obtaining the member 'A' of a type (line 92)
        A_426278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), expm_call_result_426277, 'A')
        # Getting the type of 'A' (line 92)
        A_426279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 51), 'A', False)
        # Processing the call keyword arguments (line 92)
        kwargs_426280 = {}
        # Getting the type of 'assert_allclose' (line 92)
        assert_allclose_426268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 92)
        assert_allclose_call_result_426281 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), assert_allclose_426268, *[A_426278, A_426279], **kwargs_426280)
        
        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Call to expm(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Call to array(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_426285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_426286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        complex_426287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 28), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 27), list_426286, complex_426287)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 26), list_426285, list_426286)
        
        # Processing the call keyword arguments (line 93)
        kwargs_426288 = {}
        # Getting the type of 'np' (line 93)
        np_426283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 93)
        array_426284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 17), np_426283, 'array')
        # Calling array(args, kwargs) (line 93)
        array_call_result_426289 = invoke(stypy.reporting.localization.Localization(__file__, 93, 17), array_426284, *[list_426285], **kwargs_426288)
        
        # Processing the call keyword arguments (line 93)
        kwargs_426290 = {}
        # Getting the type of 'expm' (line 93)
        expm_426282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'expm', False)
        # Calling expm(args, kwargs) (line 93)
        expm_call_result_426291 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), expm_426282, *[array_call_result_426289], **kwargs_426290)
        
        # Assigning a type to the variable 'B' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'B', expm_call_result_426291)
        
        # Call to assert_allclose(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to expm(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Obtaining an instance of the builtin type 'tuple' (line 94)
        tuple_426294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 94)
        # Adding element type (line 94)
        
        # Obtaining an instance of the builtin type 'tuple' (line 94)
        tuple_426295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 94)
        # Adding element type (line 94)
        complex_426296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 31), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 31), tuple_426295, complex_426296)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 30), tuple_426294, tuple_426295)
        
        # Processing the call keyword arguments (line 94)
        kwargs_426297 = {}
        # Getting the type of 'expm' (line 94)
        expm_426293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'expm', False)
        # Calling expm(args, kwargs) (line 94)
        expm_call_result_426298 = invoke(stypy.reporting.localization.Localization(__file__, 94, 24), expm_426293, *[tuple_426294], **kwargs_426297)
        
        # Getting the type of 'B' (line 94)
        B_426299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 40), 'B', False)
        # Processing the call keyword arguments (line 94)
        kwargs_426300 = {}
        # Getting the type of 'assert_allclose' (line 94)
        assert_allclose_426292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 94)
        assert_allclose_call_result_426301 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), assert_allclose_426292, *[expm_call_result_426298, B_426299], **kwargs_426300)
        
        
        # Call to assert_allclose(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Call to expm(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_426304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_426305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        complex_426306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 31), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 30), list_426305, complex_426306)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 29), list_426304, list_426305)
        
        # Processing the call keyword arguments (line 95)
        kwargs_426307 = {}
        # Getting the type of 'expm' (line 95)
        expm_426303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'expm', False)
        # Calling expm(args, kwargs) (line 95)
        expm_call_result_426308 = invoke(stypy.reporting.localization.Localization(__file__, 95, 24), expm_426303, *[list_426304], **kwargs_426307)
        
        # Getting the type of 'B' (line 95)
        B_426309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 38), 'B', False)
        # Processing the call keyword arguments (line 95)
        kwargs_426310 = {}
        # Getting the type of 'assert_allclose' (line 95)
        assert_allclose_426302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 95)
        assert_allclose_call_result_426311 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), assert_allclose_426302, *[expm_call_result_426308, B_426309], **kwargs_426310)
        
        
        # Call to assert_allclose(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to expm(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to matrix(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_426316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_426317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        complex_426318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 41), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 40), list_426317, complex_426318)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 39), list_426316, list_426317)
        
        # Processing the call keyword arguments (line 96)
        kwargs_426319 = {}
        # Getting the type of 'np' (line 96)
        np_426314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 29), 'np', False)
        # Obtaining the member 'matrix' of a type (line 96)
        matrix_426315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 29), np_426314, 'matrix')
        # Calling matrix(args, kwargs) (line 96)
        matrix_call_result_426320 = invoke(stypy.reporting.localization.Localization(__file__, 96, 29), matrix_426315, *[list_426316], **kwargs_426319)
        
        # Processing the call keyword arguments (line 96)
        kwargs_426321 = {}
        # Getting the type of 'expm' (line 96)
        expm_426313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 24), 'expm', False)
        # Calling expm(args, kwargs) (line 96)
        expm_call_result_426322 = invoke(stypy.reporting.localization.Localization(__file__, 96, 24), expm_426313, *[matrix_call_result_426320], **kwargs_426321)
        
        # Getting the type of 'B' (line 96)
        B_426323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 49), 'B', False)
        # Processing the call keyword arguments (line 96)
        kwargs_426324 = {}
        # Getting the type of 'assert_allclose' (line 96)
        assert_allclose_426312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 96)
        assert_allclose_call_result_426325 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), assert_allclose_426312, *[expm_call_result_426322, B_426323], **kwargs_426324)
        
        
        # Call to assert_allclose(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to expm(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to csc_matrix(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_426329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_426330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        complex_426331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 42), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 41), list_426330, complex_426331)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 40), list_426329, list_426330)
        
        # Processing the call keyword arguments (line 97)
        kwargs_426332 = {}
        # Getting the type of 'csc_matrix' (line 97)
        csc_matrix_426328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 29), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 97)
        csc_matrix_call_result_426333 = invoke(stypy.reporting.localization.Localization(__file__, 97, 29), csc_matrix_426328, *[list_426329], **kwargs_426332)
        
        # Processing the call keyword arguments (line 97)
        kwargs_426334 = {}
        # Getting the type of 'expm' (line 97)
        expm_426327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'expm', False)
        # Calling expm(args, kwargs) (line 97)
        expm_call_result_426335 = invoke(stypy.reporting.localization.Localization(__file__, 97, 24), expm_426327, *[csc_matrix_call_result_426333], **kwargs_426334)
        
        # Obtaining the member 'A' of a type (line 97)
        A_426336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 24), expm_call_result_426335, 'A')
        # Getting the type of 'B' (line 97)
        B_426337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 52), 'B', False)
        # Processing the call keyword arguments (line 97)
        kwargs_426338 = {}
        # Getting the type of 'assert_allclose' (line 97)
        assert_allclose_426326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 97)
        assert_allclose_call_result_426339 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assert_allclose_426326, *[A_426336, B_426337], **kwargs_426338)
        
        
        # ################# End of 'test_misc_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_misc_types' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_426340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426340)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_misc_types'
        return stypy_return_type_426340


    @norecursion
    def test_bidiagonal_sparse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bidiagonal_sparse'
        module_type_store = module_type_store.open_function_context('test_bidiagonal_sparse', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_bidiagonal_sparse.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_bidiagonal_sparse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_bidiagonal_sparse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_bidiagonal_sparse.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_bidiagonal_sparse')
        TestExpM.test_bidiagonal_sparse.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_bidiagonal_sparse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_bidiagonal_sparse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_bidiagonal_sparse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_bidiagonal_sparse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_bidiagonal_sparse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_bidiagonal_sparse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_bidiagonal_sparse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bidiagonal_sparse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bidiagonal_sparse(...)' code ##################

        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to csc_matrix(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_426342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        # Adding element type (line 100)
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_426343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        int_426344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 12), list_426343, int_426344)
        # Adding element type (line 101)
        int_426345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 12), list_426343, int_426345)
        # Adding element type (line 101)
        int_426346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 12), list_426343, int_426346)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 23), list_426342, list_426343)
        # Adding element type (line 100)
        
        # Obtaining an instance of the builtin type 'list' (line 102)
        list_426347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 102)
        # Adding element type (line 102)
        int_426348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 12), list_426347, int_426348)
        # Adding element type (line 102)
        int_426349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 12), list_426347, int_426349)
        # Adding element type (line 102)
        int_426350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 12), list_426347, int_426350)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 23), list_426342, list_426347)
        # Adding element type (line 100)
        
        # Obtaining an instance of the builtin type 'list' (line 103)
        list_426351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 103)
        # Adding element type (line 103)
        int_426352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 12), list_426351, int_426352)
        # Adding element type (line 103)
        int_426353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 12), list_426351, int_426353)
        # Adding element type (line 103)
        int_426354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 12), list_426351, int_426354)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 23), list_426342, list_426351)
        
        # Processing the call keyword arguments (line 100)
        # Getting the type of 'float' (line 103)
        float_426355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'float', False)
        keyword_426356 = float_426355
        kwargs_426357 = {'dtype': keyword_426356}
        # Getting the type of 'csc_matrix' (line 100)
        csc_matrix_426341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 100)
        csc_matrix_call_result_426358 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), csc_matrix_426341, *[list_426342], **kwargs_426357)
        
        # Assigning a type to the variable 'A' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'A', csc_matrix_call_result_426358)
        
        # Assigning a Call to a Name (line 104):
        
        # Assigning a Call to a Name (line 104):
        
        # Call to exp(...): (line 104)
        # Processing the call arguments (line 104)
        int_426361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 22), 'int')
        # Processing the call keyword arguments (line 104)
        kwargs_426362 = {}
        # Getting the type of 'math' (line 104)
        math_426359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'math', False)
        # Obtaining the member 'exp' of a type (line 104)
        exp_426360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 13), math_426359, 'exp')
        # Calling exp(args, kwargs) (line 104)
        exp_call_result_426363 = invoke(stypy.reporting.localization.Localization(__file__, 104, 13), exp_426360, *[int_426361], **kwargs_426362)
        
        # Assigning a type to the variable 'e1' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'e1', exp_call_result_426363)
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to exp(...): (line 105)
        # Processing the call arguments (line 105)
        int_426366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 22), 'int')
        # Processing the call keyword arguments (line 105)
        kwargs_426367 = {}
        # Getting the type of 'math' (line 105)
        math_426364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'math', False)
        # Obtaining the member 'exp' of a type (line 105)
        exp_426365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 13), math_426364, 'exp')
        # Calling exp(args, kwargs) (line 105)
        exp_call_result_426368 = invoke(stypy.reporting.localization.Localization(__file__, 105, 13), exp_426365, *[int_426366], **kwargs_426367)
        
        # Assigning a type to the variable 'e2' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'e2', exp_call_result_426368)
        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to array(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Obtaining an instance of the builtin type 'list' (line 106)
        list_426371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 106)
        # Adding element type (line 106)
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_426372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        # Getting the type of 'e1' (line 107)
        e1_426373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'e1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_426372, e1_426373)
        # Adding element type (line 107)
        int_426374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 17), 'int')
        # Getting the type of 'e1' (line 107)
        e1_426375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'e1', False)
        # Applying the binary operator '*' (line 107)
        result_mul_426376 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 17), '*', int_426374, e1_426375)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_426372, result_mul_426376)
        # Adding element type (line 107)
        int_426377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 23), 'int')
        # Getting the type of 'e2' (line 107)
        e2_426378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'e2', False)
        int_426379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 32), 'int')
        # Getting the type of 'e1' (line 107)
        e1_426380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 34), 'e1', False)
        # Applying the binary operator '*' (line 107)
        result_mul_426381 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 32), '*', int_426379, e1_426380)
        
        # Applying the binary operator '-' (line 107)
        result_sub_426382 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 27), '-', e2_426378, result_mul_426381)
        
        # Applying the binary operator '*' (line 107)
        result_mul_426383 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 23), '*', int_426377, result_sub_426382)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_426372, result_mul_426383)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 28), list_426371, list_426372)
        # Adding element type (line 106)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_426384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        int_426385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 12), list_426384, int_426385)
        # Adding element type (line 108)
        # Getting the type of 'e1' (line 108)
        e1_426386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'e1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 12), list_426384, e1_426386)
        # Adding element type (line 108)
        int_426387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 20), 'int')
        # Getting the type of 'e2' (line 108)
        e2_426388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'e2', False)
        # Getting the type of 'e1' (line 108)
        e1_426389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 28), 'e1', False)
        # Applying the binary operator '-' (line 108)
        result_sub_426390 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 23), '-', e2_426388, e1_426389)
        
        # Applying the binary operator '*' (line 108)
        result_mul_426391 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 20), '*', int_426387, result_sub_426390)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 12), list_426384, result_mul_426391)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 28), list_426371, list_426384)
        # Adding element type (line 106)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_426392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        int_426393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 12), list_426392, int_426393)
        # Adding element type (line 109)
        int_426394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 12), list_426392, int_426394)
        # Adding element type (line 109)
        # Getting the type of 'e2' (line 109)
        e2_426395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'e2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 12), list_426392, e2_426395)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 28), list_426371, list_426392)
        
        # Processing the call keyword arguments (line 106)
        # Getting the type of 'float' (line 109)
        float_426396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 31), 'float', False)
        keyword_426397 = float_426396
        kwargs_426398 = {'dtype': keyword_426397}
        # Getting the type of 'np' (line 106)
        np_426369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 106)
        array_426370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 19), np_426369, 'array')
        # Calling array(args, kwargs) (line 106)
        array_call_result_426399 = invoke(stypy.reporting.localization.Localization(__file__, 106, 19), array_426370, *[list_426371], **kwargs_426398)
        
        # Assigning a type to the variable 'expected' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'expected', array_call_result_426399)
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to toarray(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_426405 = {}
        
        # Call to expm(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'A' (line 110)
        A_426401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 24), 'A', False)
        # Processing the call keyword arguments (line 110)
        kwargs_426402 = {}
        # Getting the type of 'expm' (line 110)
        expm_426400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'expm', False)
        # Calling expm(args, kwargs) (line 110)
        expm_call_result_426403 = invoke(stypy.reporting.localization.Localization(__file__, 110, 19), expm_426400, *[A_426401], **kwargs_426402)
        
        # Obtaining the member 'toarray' of a type (line 110)
        toarray_426404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 19), expm_call_result_426403, 'toarray')
        # Calling toarray(args, kwargs) (line 110)
        toarray_call_result_426406 = invoke(stypy.reporting.localization.Localization(__file__, 110, 19), toarray_426404, *[], **kwargs_426405)
        
        # Assigning a type to the variable 'observed' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'observed', toarray_call_result_426406)
        
        # Call to assert_array_almost_equal(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'observed' (line 111)
        observed_426408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 34), 'observed', False)
        # Getting the type of 'expected' (line 111)
        expected_426409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 44), 'expected', False)
        # Processing the call keyword arguments (line 111)
        kwargs_426410 = {}
        # Getting the type of 'assert_array_almost_equal' (line 111)
        assert_array_almost_equal_426407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 111)
        assert_array_almost_equal_call_result_426411 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), assert_array_almost_equal_426407, *[observed_426408, expected_426409], **kwargs_426410)
        
        
        # ################# End of 'test_bidiagonal_sparse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bidiagonal_sparse' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_426412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426412)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bidiagonal_sparse'
        return stypy_return_type_426412


    @norecursion
    def test_padecases_dtype_float(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_padecases_dtype_float'
        module_type_store = module_type_store.open_function_context('test_padecases_dtype_float', 113, 4, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_padecases_dtype_float.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_padecases_dtype_float.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_padecases_dtype_float.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_padecases_dtype_float.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_padecases_dtype_float')
        TestExpM.test_padecases_dtype_float.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_padecases_dtype_float.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_padecases_dtype_float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_padecases_dtype_float.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_padecases_dtype_float.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_padecases_dtype_float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_padecases_dtype_float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_padecases_dtype_float', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_padecases_dtype_float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_padecases_dtype_float(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_426413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        # Adding element type (line 114)
        # Getting the type of 'np' (line 114)
        np_426414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'np')
        # Obtaining the member 'float32' of a type (line 114)
        float32_426415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 22), np_426414, 'float32')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 21), list_426413, float32_426415)
        # Adding element type (line 114)
        # Getting the type of 'np' (line 114)
        np_426416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 34), 'np')
        # Obtaining the member 'float64' of a type (line 114)
        float64_426417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 34), np_426416, 'float64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 21), list_426413, float64_426417)
        
        # Testing the type of a for loop iterable (line 114)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 114, 8), list_426413)
        # Getting the type of the for loop variable (line 114)
        for_loop_var_426418 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 114, 8), list_426413)
        # Assigning a type to the variable 'dtype' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'dtype', for_loop_var_426418)
        # SSA begins for a for statement (line 114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_426419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        # Adding element type (line 115)
        float_426420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 25), list_426419, float_426420)
        # Adding element type (line 115)
        float_426421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 25), list_426419, float_426421)
        # Adding element type (line 115)
        float_426422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 25), list_426419, float_426422)
        # Adding element type (line 115)
        int_426423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 25), list_426419, int_426423)
        # Adding element type (line 115)
        int_426424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 25), list_426419, int_426424)
        
        # Testing the type of a for loop iterable (line 115)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 12), list_426419)
        # Getting the type of the for loop variable (line 115)
        for_loop_var_426425 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 12), list_426419)
        # Assigning a type to the variable 'scale' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'scale', for_loop_var_426425)
        # SSA begins for a for statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 116):
        
        # Assigning a BinOp to a Name (line 116):
        # Getting the type of 'scale' (line 116)
        scale_426426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'scale')
        
        # Call to eye(...): (line 116)
        # Processing the call arguments (line 116)
        int_426428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 32), 'int')
        # Processing the call keyword arguments (line 116)
        # Getting the type of 'dtype' (line 116)
        dtype_426429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 41), 'dtype', False)
        keyword_426430 = dtype_426429
        kwargs_426431 = {'dtype': keyword_426430}
        # Getting the type of 'eye' (line 116)
        eye_426427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 28), 'eye', False)
        # Calling eye(args, kwargs) (line 116)
        eye_call_result_426432 = invoke(stypy.reporting.localization.Localization(__file__, 116, 28), eye_426427, *[int_426428], **kwargs_426431)
        
        # Applying the binary operator '*' (line 116)
        result_mul_426433 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 20), '*', scale_426426, eye_call_result_426432)
        
        # Assigning a type to the variable 'A' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'A', result_mul_426433)
        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to expm(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'A' (line 117)
        A_426435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'A', False)
        # Processing the call keyword arguments (line 117)
        kwargs_426436 = {}
        # Getting the type of 'expm' (line 117)
        expm_426434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'expm', False)
        # Calling expm(args, kwargs) (line 117)
        expm_call_result_426437 = invoke(stypy.reporting.localization.Localization(__file__, 117, 27), expm_426434, *[A_426435], **kwargs_426436)
        
        # Assigning a type to the variable 'observed' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'observed', expm_call_result_426437)
        
        # Assigning a BinOp to a Name (line 118):
        
        # Assigning a BinOp to a Name (line 118):
        
        # Call to exp(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'scale' (line 118)
        scale_426439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'scale', False)
        # Processing the call keyword arguments (line 118)
        kwargs_426440 = {}
        # Getting the type of 'exp' (line 118)
        exp_426438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'exp', False)
        # Calling exp(args, kwargs) (line 118)
        exp_call_result_426441 = invoke(stypy.reporting.localization.Localization(__file__, 118, 27), exp_426438, *[scale_426439], **kwargs_426440)
        
        
        # Call to eye(...): (line 118)
        # Processing the call arguments (line 118)
        int_426443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 44), 'int')
        # Processing the call keyword arguments (line 118)
        # Getting the type of 'dtype' (line 118)
        dtype_426444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 53), 'dtype', False)
        keyword_426445 = dtype_426444
        kwargs_426446 = {'dtype': keyword_426445}
        # Getting the type of 'eye' (line 118)
        eye_426442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 40), 'eye', False)
        # Calling eye(args, kwargs) (line 118)
        eye_call_result_426447 = invoke(stypy.reporting.localization.Localization(__file__, 118, 40), eye_426442, *[int_426443], **kwargs_426446)
        
        # Applying the binary operator '*' (line 118)
        result_mul_426448 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 27), '*', exp_call_result_426441, eye_call_result_426447)
        
        # Assigning a type to the variable 'expected' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'expected', result_mul_426448)
        
        # Call to assert_array_almost_equal_nulp(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'observed' (line 119)
        observed_426450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 47), 'observed', False)
        # Getting the type of 'expected' (line 119)
        expected_426451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 57), 'expected', False)
        # Processing the call keyword arguments (line 119)
        int_426452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 72), 'int')
        keyword_426453 = int_426452
        kwargs_426454 = {'nulp': keyword_426453}
        # Getting the type of 'assert_array_almost_equal_nulp' (line 119)
        assert_array_almost_equal_nulp_426449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'assert_array_almost_equal_nulp', False)
        # Calling assert_array_almost_equal_nulp(args, kwargs) (line 119)
        assert_array_almost_equal_nulp_call_result_426455 = invoke(stypy.reporting.localization.Localization(__file__, 119, 16), assert_array_almost_equal_nulp_426449, *[observed_426450, expected_426451], **kwargs_426454)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_padecases_dtype_float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_padecases_dtype_float' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_426456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426456)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_padecases_dtype_float'
        return stypy_return_type_426456


    @norecursion
    def test_padecases_dtype_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_padecases_dtype_complex'
        module_type_store = module_type_store.open_function_context('test_padecases_dtype_complex', 121, 4, False)
        # Assigning a type to the variable 'self' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_padecases_dtype_complex.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_padecases_dtype_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_padecases_dtype_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_padecases_dtype_complex.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_padecases_dtype_complex')
        TestExpM.test_padecases_dtype_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_padecases_dtype_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_padecases_dtype_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_padecases_dtype_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_padecases_dtype_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_padecases_dtype_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_padecases_dtype_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_padecases_dtype_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_padecases_dtype_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_padecases_dtype_complex(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_426457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        # Adding element type (line 122)
        # Getting the type of 'np' (line 122)
        np_426458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 22), 'np')
        # Obtaining the member 'complex64' of a type (line 122)
        complex64_426459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 22), np_426458, 'complex64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_426457, complex64_426459)
        # Adding element type (line 122)
        # Getting the type of 'np' (line 122)
        np_426460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 36), 'np')
        # Obtaining the member 'complex128' of a type (line 122)
        complex128_426461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 36), np_426460, 'complex128')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_426457, complex128_426461)
        
        # Testing the type of a for loop iterable (line 122)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 122, 8), list_426457)
        # Getting the type of the for loop variable (line 122)
        for_loop_var_426462 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 122, 8), list_426457)
        # Assigning a type to the variable 'dtype' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'dtype', for_loop_var_426462)
        # SSA begins for a for statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_426463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        float_426464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 25), list_426463, float_426464)
        # Adding element type (line 123)
        float_426465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 25), list_426463, float_426465)
        # Adding element type (line 123)
        float_426466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 25), list_426463, float_426466)
        # Adding element type (line 123)
        int_426467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 25), list_426463, int_426467)
        # Adding element type (line 123)
        int_426468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 25), list_426463, int_426468)
        
        # Testing the type of a for loop iterable (line 123)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 123, 12), list_426463)
        # Getting the type of the for loop variable (line 123)
        for_loop_var_426469 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 123, 12), list_426463)
        # Assigning a type to the variable 'scale' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'scale', for_loop_var_426469)
        # SSA begins for a for statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 124):
        
        # Assigning a BinOp to a Name (line 124):
        # Getting the type of 'scale' (line 124)
        scale_426470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'scale')
        
        # Call to eye(...): (line 124)
        # Processing the call arguments (line 124)
        int_426472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 32), 'int')
        # Processing the call keyword arguments (line 124)
        # Getting the type of 'dtype' (line 124)
        dtype_426473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 41), 'dtype', False)
        keyword_426474 = dtype_426473
        kwargs_426475 = {'dtype': keyword_426474}
        # Getting the type of 'eye' (line 124)
        eye_426471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'eye', False)
        # Calling eye(args, kwargs) (line 124)
        eye_call_result_426476 = invoke(stypy.reporting.localization.Localization(__file__, 124, 28), eye_426471, *[int_426472], **kwargs_426475)
        
        # Applying the binary operator '*' (line 124)
        result_mul_426477 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 20), '*', scale_426470, eye_call_result_426476)
        
        # Assigning a type to the variable 'A' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'A', result_mul_426477)
        
        # Assigning a Call to a Name (line 125):
        
        # Assigning a Call to a Name (line 125):
        
        # Call to expm(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'A' (line 125)
        A_426479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 32), 'A', False)
        # Processing the call keyword arguments (line 125)
        kwargs_426480 = {}
        # Getting the type of 'expm' (line 125)
        expm_426478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'expm', False)
        # Calling expm(args, kwargs) (line 125)
        expm_call_result_426481 = invoke(stypy.reporting.localization.Localization(__file__, 125, 27), expm_426478, *[A_426479], **kwargs_426480)
        
        # Assigning a type to the variable 'observed' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'observed', expm_call_result_426481)
        
        # Assigning a BinOp to a Name (line 126):
        
        # Assigning a BinOp to a Name (line 126):
        
        # Call to exp(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'scale' (line 126)
        scale_426483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 31), 'scale', False)
        # Processing the call keyword arguments (line 126)
        kwargs_426484 = {}
        # Getting the type of 'exp' (line 126)
        exp_426482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 27), 'exp', False)
        # Calling exp(args, kwargs) (line 126)
        exp_call_result_426485 = invoke(stypy.reporting.localization.Localization(__file__, 126, 27), exp_426482, *[scale_426483], **kwargs_426484)
        
        
        # Call to eye(...): (line 126)
        # Processing the call arguments (line 126)
        int_426487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 44), 'int')
        # Processing the call keyword arguments (line 126)
        # Getting the type of 'dtype' (line 126)
        dtype_426488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 53), 'dtype', False)
        keyword_426489 = dtype_426488
        kwargs_426490 = {'dtype': keyword_426489}
        # Getting the type of 'eye' (line 126)
        eye_426486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 40), 'eye', False)
        # Calling eye(args, kwargs) (line 126)
        eye_call_result_426491 = invoke(stypy.reporting.localization.Localization(__file__, 126, 40), eye_426486, *[int_426487], **kwargs_426490)
        
        # Applying the binary operator '*' (line 126)
        result_mul_426492 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 27), '*', exp_call_result_426485, eye_call_result_426491)
        
        # Assigning a type to the variable 'expected' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'expected', result_mul_426492)
        
        # Call to assert_array_almost_equal_nulp(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'observed' (line 127)
        observed_426494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 47), 'observed', False)
        # Getting the type of 'expected' (line 127)
        expected_426495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 57), 'expected', False)
        # Processing the call keyword arguments (line 127)
        int_426496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 72), 'int')
        keyword_426497 = int_426496
        kwargs_426498 = {'nulp': keyword_426497}
        # Getting the type of 'assert_array_almost_equal_nulp' (line 127)
        assert_array_almost_equal_nulp_426493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'assert_array_almost_equal_nulp', False)
        # Calling assert_array_almost_equal_nulp(args, kwargs) (line 127)
        assert_array_almost_equal_nulp_call_result_426499 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), assert_array_almost_equal_nulp_426493, *[observed_426494, expected_426495], **kwargs_426498)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_padecases_dtype_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_padecases_dtype_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_426500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426500)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_padecases_dtype_complex'
        return stypy_return_type_426500


    @norecursion
    def test_padecases_dtype_sparse_float(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_padecases_dtype_sparse_float'
        module_type_store = module_type_store.open_function_context('test_padecases_dtype_sparse_float', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_padecases_dtype_sparse_float.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_padecases_dtype_sparse_float.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_padecases_dtype_sparse_float.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_padecases_dtype_sparse_float.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_padecases_dtype_sparse_float')
        TestExpM.test_padecases_dtype_sparse_float.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_padecases_dtype_sparse_float.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_padecases_dtype_sparse_float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_padecases_dtype_sparse_float.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_padecases_dtype_sparse_float.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_padecases_dtype_sparse_float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_padecases_dtype_sparse_float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_padecases_dtype_sparse_float', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_padecases_dtype_sparse_float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_padecases_dtype_sparse_float(...)' code ##################

        
        # Assigning a Attribute to a Name (line 131):
        
        # Assigning a Attribute to a Name (line 131):
        # Getting the type of 'np' (line 131)
        np_426501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'np')
        # Obtaining the member 'float64' of a type (line 131)
        float64_426502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), np_426501, 'float64')
        # Assigning a type to the variable 'dtype' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'dtype', float64_426502)
        
        
        # Obtaining an instance of the builtin type 'list' (line 132)
        list_426503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 132)
        # Adding element type (line 132)
        float_426504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_426503, float_426504)
        # Adding element type (line 132)
        float_426505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_426503, float_426505)
        # Adding element type (line 132)
        float_426506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_426503, float_426506)
        # Adding element type (line 132)
        int_426507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_426503, int_426507)
        # Adding element type (line 132)
        int_426508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_426503, int_426508)
        
        # Testing the type of a for loop iterable (line 132)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 132, 8), list_426503)
        # Getting the type of the for loop variable (line 132)
        for_loop_var_426509 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 132, 8), list_426503)
        # Assigning a type to the variable 'scale' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'scale', for_loop_var_426509)
        # SSA begins for a for statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 133):
        
        # Assigning a BinOp to a Name (line 133):
        # Getting the type of 'scale' (line 133)
        scale_426510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'scale')
        
        # Call to speye(...): (line 133)
        # Processing the call arguments (line 133)
        int_426512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 30), 'int')
        int_426513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 33), 'int')
        # Processing the call keyword arguments (line 133)
        # Getting the type of 'dtype' (line 133)
        dtype_426514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 42), 'dtype', False)
        keyword_426515 = dtype_426514
        str_426516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 56), 'str', 'csc')
        keyword_426517 = str_426516
        kwargs_426518 = {'dtype': keyword_426515, 'format': keyword_426517}
        # Getting the type of 'speye' (line 133)
        speye_426511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'speye', False)
        # Calling speye(args, kwargs) (line 133)
        speye_call_result_426519 = invoke(stypy.reporting.localization.Localization(__file__, 133, 24), speye_426511, *[int_426512, int_426513], **kwargs_426518)
        
        # Applying the binary operator '*' (line 133)
        result_mul_426520 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 16), '*', scale_426510, speye_call_result_426519)
        
        # Assigning a type to the variable 'a' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'a', result_mul_426520)
        
        # Assigning a BinOp to a Name (line 134):
        
        # Assigning a BinOp to a Name (line 134):
        
        # Call to exp(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'scale' (line 134)
        scale_426522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 'scale', False)
        # Processing the call keyword arguments (line 134)
        kwargs_426523 = {}
        # Getting the type of 'exp' (line 134)
        exp_426521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'exp', False)
        # Calling exp(args, kwargs) (line 134)
        exp_call_result_426524 = invoke(stypy.reporting.localization.Localization(__file__, 134, 16), exp_426521, *[scale_426522], **kwargs_426523)
        
        
        # Call to eye(...): (line 134)
        # Processing the call arguments (line 134)
        int_426526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 33), 'int')
        # Processing the call keyword arguments (line 134)
        # Getting the type of 'dtype' (line 134)
        dtype_426527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 42), 'dtype', False)
        keyword_426528 = dtype_426527
        kwargs_426529 = {'dtype': keyword_426528}
        # Getting the type of 'eye' (line 134)
        eye_426525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 29), 'eye', False)
        # Calling eye(args, kwargs) (line 134)
        eye_call_result_426530 = invoke(stypy.reporting.localization.Localization(__file__, 134, 29), eye_426525, *[int_426526], **kwargs_426529)
        
        # Applying the binary operator '*' (line 134)
        result_mul_426531 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 16), '*', exp_call_result_426524, eye_call_result_426530)
        
        # Assigning a type to the variable 'e' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'e', result_mul_426531)
        
        # Call to suppress_warnings(...): (line 135)
        # Processing the call keyword arguments (line 135)
        kwargs_426533 = {}
        # Getting the type of 'suppress_warnings' (line 135)
        suppress_warnings_426532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 135)
        suppress_warnings_call_result_426534 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), suppress_warnings_426532, *[], **kwargs_426533)
        
        with_426535 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 135, 17), suppress_warnings_call_result_426534, 'with parameter', '__enter__', '__exit__')

        if with_426535:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 135)
            enter___426536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 17), suppress_warnings_call_result_426534, '__enter__')
            with_enter_426537 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), enter___426536)
            # Assigning a type to the variable 'sup' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'sup', with_enter_426537)
            
            # Call to filter(...): (line 136)
            # Processing the call arguments (line 136)
            # Getting the type of 'SparseEfficiencyWarning' (line 136)
            SparseEfficiencyWarning_426540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'SparseEfficiencyWarning', False)
            str_426541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 27), 'str', 'Changing the sparsity structure of a csc_matrix is expensive.')
            # Processing the call keyword arguments (line 136)
            kwargs_426542 = {}
            # Getting the type of 'sup' (line 136)
            sup_426538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'sup', False)
            # Obtaining the member 'filter' of a type (line 136)
            filter_426539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 16), sup_426538, 'filter')
            # Calling filter(args, kwargs) (line 136)
            filter_call_result_426543 = invoke(stypy.reporting.localization.Localization(__file__, 136, 16), filter_426539, *[SparseEfficiencyWarning_426540, str_426541], **kwargs_426542)
            
            
            # Assigning a Call to a Name (line 138):
            
            # Assigning a Call to a Name (line 138):
            
            # Call to toarray(...): (line 138)
            # Processing the call keyword arguments (line 138)
            kwargs_426551 = {}
            
            # Call to _expm(...): (line 138)
            # Processing the call arguments (line 138)
            # Getting the type of 'a' (line 138)
            a_426545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 38), 'a', False)
            # Processing the call keyword arguments (line 138)
            # Getting the type of 'True' (line 138)
            True_426546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 59), 'True', False)
            keyword_426547 = True_426546
            kwargs_426548 = {'use_exact_onenorm': keyword_426547}
            # Getting the type of '_expm' (line 138)
            _expm_426544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 32), '_expm', False)
            # Calling _expm(args, kwargs) (line 138)
            _expm_call_result_426549 = invoke(stypy.reporting.localization.Localization(__file__, 138, 32), _expm_426544, *[a_426545], **kwargs_426548)
            
            # Obtaining the member 'toarray' of a type (line 138)
            toarray_426550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 32), _expm_call_result_426549, 'toarray')
            # Calling toarray(args, kwargs) (line 138)
            toarray_call_result_426552 = invoke(stypy.reporting.localization.Localization(__file__, 138, 32), toarray_426550, *[], **kwargs_426551)
            
            # Assigning a type to the variable 'exact_onenorm' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'exact_onenorm', toarray_call_result_426552)
            
            # Assigning a Call to a Name (line 139):
            
            # Assigning a Call to a Name (line 139):
            
            # Call to toarray(...): (line 139)
            # Processing the call keyword arguments (line 139)
            kwargs_426560 = {}
            
            # Call to _expm(...): (line 139)
            # Processing the call arguments (line 139)
            # Getting the type of 'a' (line 139)
            a_426554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 40), 'a', False)
            # Processing the call keyword arguments (line 139)
            # Getting the type of 'False' (line 139)
            False_426555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 61), 'False', False)
            keyword_426556 = False_426555
            kwargs_426557 = {'use_exact_onenorm': keyword_426556}
            # Getting the type of '_expm' (line 139)
            _expm_426553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 34), '_expm', False)
            # Calling _expm(args, kwargs) (line 139)
            _expm_call_result_426558 = invoke(stypy.reporting.localization.Localization(__file__, 139, 34), _expm_426553, *[a_426554], **kwargs_426557)
            
            # Obtaining the member 'toarray' of a type (line 139)
            toarray_426559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 34), _expm_call_result_426558, 'toarray')
            # Calling toarray(args, kwargs) (line 139)
            toarray_call_result_426561 = invoke(stypy.reporting.localization.Localization(__file__, 139, 34), toarray_426559, *[], **kwargs_426560)
            
            # Assigning a type to the variable 'inexact_onenorm' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'inexact_onenorm', toarray_call_result_426561)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 135)
            exit___426562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 17), suppress_warnings_call_result_426534, '__exit__')
            with_exit_426563 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), exit___426562, None, None, None)

        
        # Call to assert_array_almost_equal_nulp(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'exact_onenorm' (line 140)
        exact_onenorm_426565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 43), 'exact_onenorm', False)
        # Getting the type of 'e' (line 140)
        e_426566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 58), 'e', False)
        # Processing the call keyword arguments (line 140)
        int_426567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 66), 'int')
        keyword_426568 = int_426567
        kwargs_426569 = {'nulp': keyword_426568}
        # Getting the type of 'assert_array_almost_equal_nulp' (line 140)
        assert_array_almost_equal_nulp_426564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'assert_array_almost_equal_nulp', False)
        # Calling assert_array_almost_equal_nulp(args, kwargs) (line 140)
        assert_array_almost_equal_nulp_call_result_426570 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), assert_array_almost_equal_nulp_426564, *[exact_onenorm_426565, e_426566], **kwargs_426569)
        
        
        # Call to assert_array_almost_equal_nulp(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'inexact_onenorm' (line 141)
        inexact_onenorm_426572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 43), 'inexact_onenorm', False)
        # Getting the type of 'e' (line 141)
        e_426573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 60), 'e', False)
        # Processing the call keyword arguments (line 141)
        int_426574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 68), 'int')
        keyword_426575 = int_426574
        kwargs_426576 = {'nulp': keyword_426575}
        # Getting the type of 'assert_array_almost_equal_nulp' (line 141)
        assert_array_almost_equal_nulp_426571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'assert_array_almost_equal_nulp', False)
        # Calling assert_array_almost_equal_nulp(args, kwargs) (line 141)
        assert_array_almost_equal_nulp_call_result_426577 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), assert_array_almost_equal_nulp_426571, *[inexact_onenorm_426572, e_426573], **kwargs_426576)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_padecases_dtype_sparse_float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_padecases_dtype_sparse_float' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_426578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426578)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_padecases_dtype_sparse_float'
        return stypy_return_type_426578


    @norecursion
    def test_padecases_dtype_sparse_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_padecases_dtype_sparse_complex'
        module_type_store = module_type_store.open_function_context('test_padecases_dtype_sparse_complex', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_padecases_dtype_sparse_complex.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_padecases_dtype_sparse_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_padecases_dtype_sparse_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_padecases_dtype_sparse_complex.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_padecases_dtype_sparse_complex')
        TestExpM.test_padecases_dtype_sparse_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_padecases_dtype_sparse_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_padecases_dtype_sparse_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_padecases_dtype_sparse_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_padecases_dtype_sparse_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_padecases_dtype_sparse_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_padecases_dtype_sparse_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_padecases_dtype_sparse_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_padecases_dtype_sparse_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_padecases_dtype_sparse_complex(...)' code ##################

        
        # Assigning a Attribute to a Name (line 145):
        
        # Assigning a Attribute to a Name (line 145):
        # Getting the type of 'np' (line 145)
        np_426579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'np')
        # Obtaining the member 'complex128' of a type (line 145)
        complex128_426580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 16), np_426579, 'complex128')
        # Assigning a type to the variable 'dtype' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'dtype', complex128_426580)
        
        
        # Obtaining an instance of the builtin type 'list' (line 146)
        list_426581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 146)
        # Adding element type (line 146)
        float_426582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 21), list_426581, float_426582)
        # Adding element type (line 146)
        float_426583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 21), list_426581, float_426583)
        # Adding element type (line 146)
        float_426584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 21), list_426581, float_426584)
        # Adding element type (line 146)
        int_426585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 21), list_426581, int_426585)
        # Adding element type (line 146)
        int_426586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 21), list_426581, int_426586)
        
        # Testing the type of a for loop iterable (line 146)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 8), list_426581)
        # Getting the type of the for loop variable (line 146)
        for_loop_var_426587 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 8), list_426581)
        # Assigning a type to the variable 'scale' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'scale', for_loop_var_426587)
        # SSA begins for a for statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 147):
        
        # Assigning a BinOp to a Name (line 147):
        # Getting the type of 'scale' (line 147)
        scale_426588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'scale')
        
        # Call to speye(...): (line 147)
        # Processing the call arguments (line 147)
        int_426590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 30), 'int')
        int_426591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 33), 'int')
        # Processing the call keyword arguments (line 147)
        # Getting the type of 'dtype' (line 147)
        dtype_426592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 42), 'dtype', False)
        keyword_426593 = dtype_426592
        str_426594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 56), 'str', 'csc')
        keyword_426595 = str_426594
        kwargs_426596 = {'dtype': keyword_426593, 'format': keyword_426595}
        # Getting the type of 'speye' (line 147)
        speye_426589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 24), 'speye', False)
        # Calling speye(args, kwargs) (line 147)
        speye_call_result_426597 = invoke(stypy.reporting.localization.Localization(__file__, 147, 24), speye_426589, *[int_426590, int_426591], **kwargs_426596)
        
        # Applying the binary operator '*' (line 147)
        result_mul_426598 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 16), '*', scale_426588, speye_call_result_426597)
        
        # Assigning a type to the variable 'a' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'a', result_mul_426598)
        
        # Assigning a BinOp to a Name (line 148):
        
        # Assigning a BinOp to a Name (line 148):
        
        # Call to exp(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'scale' (line 148)
        scale_426600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'scale', False)
        # Processing the call keyword arguments (line 148)
        kwargs_426601 = {}
        # Getting the type of 'exp' (line 148)
        exp_426599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'exp', False)
        # Calling exp(args, kwargs) (line 148)
        exp_call_result_426602 = invoke(stypy.reporting.localization.Localization(__file__, 148, 16), exp_426599, *[scale_426600], **kwargs_426601)
        
        
        # Call to eye(...): (line 148)
        # Processing the call arguments (line 148)
        int_426604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 33), 'int')
        # Processing the call keyword arguments (line 148)
        # Getting the type of 'dtype' (line 148)
        dtype_426605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 42), 'dtype', False)
        keyword_426606 = dtype_426605
        kwargs_426607 = {'dtype': keyword_426606}
        # Getting the type of 'eye' (line 148)
        eye_426603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 29), 'eye', False)
        # Calling eye(args, kwargs) (line 148)
        eye_call_result_426608 = invoke(stypy.reporting.localization.Localization(__file__, 148, 29), eye_426603, *[int_426604], **kwargs_426607)
        
        # Applying the binary operator '*' (line 148)
        result_mul_426609 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 16), '*', exp_call_result_426602, eye_call_result_426608)
        
        # Assigning a type to the variable 'e' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'e', result_mul_426609)
        
        # Call to suppress_warnings(...): (line 149)
        # Processing the call keyword arguments (line 149)
        kwargs_426611 = {}
        # Getting the type of 'suppress_warnings' (line 149)
        suppress_warnings_426610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 149)
        suppress_warnings_call_result_426612 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), suppress_warnings_426610, *[], **kwargs_426611)
        
        with_426613 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 149, 17), suppress_warnings_call_result_426612, 'with parameter', '__enter__', '__exit__')

        if with_426613:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 149)
            enter___426614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 17), suppress_warnings_call_result_426612, '__enter__')
            with_enter_426615 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), enter___426614)
            # Assigning a type to the variable 'sup' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'sup', with_enter_426615)
            
            # Call to filter(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'SparseEfficiencyWarning' (line 150)
            SparseEfficiencyWarning_426618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 'SparseEfficiencyWarning', False)
            str_426619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 27), 'str', 'Changing the sparsity structure of a csc_matrix is expensive.')
            # Processing the call keyword arguments (line 150)
            kwargs_426620 = {}
            # Getting the type of 'sup' (line 150)
            sup_426616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'sup', False)
            # Obtaining the member 'filter' of a type (line 150)
            filter_426617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), sup_426616, 'filter')
            # Calling filter(args, kwargs) (line 150)
            filter_call_result_426621 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), filter_426617, *[SparseEfficiencyWarning_426618, str_426619], **kwargs_426620)
            
            
            # Call to assert_array_almost_equal_nulp(...): (line 152)
            # Processing the call arguments (line 152)
            
            # Call to toarray(...): (line 152)
            # Processing the call keyword arguments (line 152)
            kwargs_426628 = {}
            
            # Call to expm(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'a' (line 152)
            a_426624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 52), 'a', False)
            # Processing the call keyword arguments (line 152)
            kwargs_426625 = {}
            # Getting the type of 'expm' (line 152)
            expm_426623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 47), 'expm', False)
            # Calling expm(args, kwargs) (line 152)
            expm_call_result_426626 = invoke(stypy.reporting.localization.Localization(__file__, 152, 47), expm_426623, *[a_426624], **kwargs_426625)
            
            # Obtaining the member 'toarray' of a type (line 152)
            toarray_426627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 47), expm_call_result_426626, 'toarray')
            # Calling toarray(args, kwargs) (line 152)
            toarray_call_result_426629 = invoke(stypy.reporting.localization.Localization(__file__, 152, 47), toarray_426627, *[], **kwargs_426628)
            
            # Getting the type of 'e' (line 152)
            e_426630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 66), 'e', False)
            # Processing the call keyword arguments (line 152)
            int_426631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 74), 'int')
            keyword_426632 = int_426631
            kwargs_426633 = {'nulp': keyword_426632}
            # Getting the type of 'assert_array_almost_equal_nulp' (line 152)
            assert_array_almost_equal_nulp_426622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'assert_array_almost_equal_nulp', False)
            # Calling assert_array_almost_equal_nulp(args, kwargs) (line 152)
            assert_array_almost_equal_nulp_call_result_426634 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), assert_array_almost_equal_nulp_426622, *[toarray_call_result_426629, e_426630], **kwargs_426633)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 149)
            exit___426635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 17), suppress_warnings_call_result_426612, '__exit__')
            with_exit_426636 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), exit___426635, None, None, None)

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_padecases_dtype_sparse_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_padecases_dtype_sparse_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_426637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426637)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_padecases_dtype_sparse_complex'
        return stypy_return_type_426637


    @norecursion
    def test_logm_consistency(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_logm_consistency'
        module_type_store = module_type_store.open_function_context('test_logm_consistency', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_logm_consistency.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_logm_consistency.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_logm_consistency.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_logm_consistency.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_logm_consistency')
        TestExpM.test_logm_consistency.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_logm_consistency.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_logm_consistency.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_logm_consistency.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_logm_consistency.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_logm_consistency.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_logm_consistency.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_logm_consistency', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_logm_consistency', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_logm_consistency(...)' code ##################

        
        # Call to seed(...): (line 155)
        # Processing the call arguments (line 155)
        int_426640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 20), 'int')
        # Processing the call keyword arguments (line 155)
        kwargs_426641 = {}
        # Getting the type of 'random' (line 155)
        random_426638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'random', False)
        # Obtaining the member 'seed' of a type (line 155)
        seed_426639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), random_426638, 'seed')
        # Calling seed(args, kwargs) (line 155)
        seed_call_result_426642 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), seed_426639, *[int_426640], **kwargs_426641)
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_426643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        # Adding element type (line 156)
        # Getting the type of 'np' (line 156)
        np_426644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 22), 'np')
        # Obtaining the member 'float64' of a type (line 156)
        float64_426645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 22), np_426644, 'float64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 21), list_426643, float64_426645)
        # Adding element type (line 156)
        # Getting the type of 'np' (line 156)
        np_426646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 34), 'np')
        # Obtaining the member 'complex128' of a type (line 156)
        complex128_426647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 34), np_426646, 'complex128')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 21), list_426643, complex128_426647)
        
        # Testing the type of a for loop iterable (line 156)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 156, 8), list_426643)
        # Getting the type of the for loop variable (line 156)
        for_loop_var_426648 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 156, 8), list_426643)
        # Assigning a type to the variable 'dtype' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'dtype', for_loop_var_426648)
        # SSA begins for a for statement (line 156)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 157)
        # Processing the call arguments (line 157)
        int_426650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 27), 'int')
        int_426651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 30), 'int')
        # Processing the call keyword arguments (line 157)
        kwargs_426652 = {}
        # Getting the type of 'range' (line 157)
        range_426649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'range', False)
        # Calling range(args, kwargs) (line 157)
        range_call_result_426653 = invoke(stypy.reporting.localization.Localization(__file__, 157, 21), range_426649, *[int_426650, int_426651], **kwargs_426652)
        
        # Testing the type of a for loop iterable (line 157)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 12), range_call_result_426653)
        # Getting the type of the for loop variable (line 157)
        for_loop_var_426654 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 12), range_call_result_426653)
        # Assigning a type to the variable 'n' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'n', for_loop_var_426654)
        # SSA begins for a for statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 158)
        list_426655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 158)
        # Adding element type (line 158)
        float_426656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 29), list_426655, float_426656)
        # Adding element type (line 158)
        float_426657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 29), list_426655, float_426657)
        # Adding element type (line 158)
        float_426658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 29), list_426655, float_426658)
        # Adding element type (line 158)
        float_426659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 29), list_426655, float_426659)
        # Adding element type (line 158)
        int_426660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 29), list_426655, int_426660)
        # Adding element type (line 158)
        float_426661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 29), list_426655, float_426661)
        # Adding element type (line 158)
        float_426662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 29), list_426655, float_426662)
        
        # Testing the type of a for loop iterable (line 158)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 158, 16), list_426655)
        # Getting the type of the for loop variable (line 158)
        for_loop_var_426663 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 158, 16), list_426655)
        # Assigning a type to the variable 'scale' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'scale', for_loop_var_426663)
        # SSA begins for a for statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to astype(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'dtype' (line 160)
        dtype_426678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 68), 'dtype', False)
        # Processing the call keyword arguments (line 160)
        kwargs_426679 = {}
        
        # Call to eye(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'n' (line 160)
        n_426665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 29), 'n', False)
        # Processing the call keyword arguments (line 160)
        kwargs_426666 = {}
        # Getting the type of 'eye' (line 160)
        eye_426664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 25), 'eye', False)
        # Calling eye(args, kwargs) (line 160)
        eye_call_result_426667 = invoke(stypy.reporting.localization.Localization(__file__, 160, 25), eye_426664, *[n_426665], **kwargs_426666)
        
        
        # Call to rand(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'n' (line 160)
        n_426670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 46), 'n', False)
        # Getting the type of 'n' (line 160)
        n_426671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 49), 'n', False)
        # Processing the call keyword arguments (line 160)
        kwargs_426672 = {}
        # Getting the type of 'random' (line 160)
        random_426668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 34), 'random', False)
        # Obtaining the member 'rand' of a type (line 160)
        rand_426669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 34), random_426668, 'rand')
        # Calling rand(args, kwargs) (line 160)
        rand_call_result_426673 = invoke(stypy.reporting.localization.Localization(__file__, 160, 34), rand_426669, *[n_426670, n_426671], **kwargs_426672)
        
        # Getting the type of 'scale' (line 160)
        scale_426674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 54), 'scale', False)
        # Applying the binary operator '*' (line 160)
        result_mul_426675 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 34), '*', rand_call_result_426673, scale_426674)
        
        # Applying the binary operator '+' (line 160)
        result_add_426676 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 25), '+', eye_call_result_426667, result_mul_426675)
        
        # Obtaining the member 'astype' of a type (line 160)
        astype_426677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 25), result_add_426676, 'astype')
        # Calling astype(args, kwargs) (line 160)
        astype_call_result_426680 = invoke(stypy.reporting.localization.Localization(__file__, 160, 25), astype_426677, *[dtype_426678], **kwargs_426679)
        
        # Assigning a type to the variable 'A' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'A', astype_call_result_426680)
        
        
        # Call to iscomplexobj(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'A' (line 161)
        A_426683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 39), 'A', False)
        # Processing the call keyword arguments (line 161)
        kwargs_426684 = {}
        # Getting the type of 'np' (line 161)
        np_426681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'np', False)
        # Obtaining the member 'iscomplexobj' of a type (line 161)
        iscomplexobj_426682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 23), np_426681, 'iscomplexobj')
        # Calling iscomplexobj(args, kwargs) (line 161)
        iscomplexobj_call_result_426685 = invoke(stypy.reporting.localization.Localization(__file__, 161, 23), iscomplexobj_426682, *[A_426683], **kwargs_426684)
        
        # Testing the type of an if condition (line 161)
        if_condition_426686 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 20), iscomplexobj_call_result_426685)
        # Assigning a type to the variable 'if_condition_426686' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'if_condition_426686', if_condition_426686)
        # SSA begins for if statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 162):
        
        # Assigning a BinOp to a Name (line 162):
        # Getting the type of 'A' (line 162)
        A_426687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 28), 'A')
        complex_426688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 32), 'complex')
        
        # Call to rand(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'n' (line 162)
        n_426691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 49), 'n', False)
        # Getting the type of 'n' (line 162)
        n_426692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 52), 'n', False)
        # Processing the call keyword arguments (line 162)
        kwargs_426693 = {}
        # Getting the type of 'random' (line 162)
        random_426689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 37), 'random', False)
        # Obtaining the member 'rand' of a type (line 162)
        rand_426690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 37), random_426689, 'rand')
        # Calling rand(args, kwargs) (line 162)
        rand_call_result_426694 = invoke(stypy.reporting.localization.Localization(__file__, 162, 37), rand_426690, *[n_426691, n_426692], **kwargs_426693)
        
        # Applying the binary operator '*' (line 162)
        result_mul_426695 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 32), '*', complex_426688, rand_call_result_426694)
        
        # Getting the type of 'scale' (line 162)
        scale_426696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 57), 'scale')
        # Applying the binary operator '*' (line 162)
        result_mul_426697 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 55), '*', result_mul_426695, scale_426696)
        
        # Applying the binary operator '+' (line 162)
        result_add_426698 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 28), '+', A_426687, result_mul_426697)
        
        # Assigning a type to the variable 'A' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 24), 'A', result_add_426698)
        # SSA join for if statement (line 161)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_array_almost_equal(...): (line 163)
        # Processing the call arguments (line 163)
        
        # Call to expm(...): (line 163)
        # Processing the call arguments (line 163)
        
        # Call to logm(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'A' (line 163)
        A_426702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 56), 'A', False)
        # Processing the call keyword arguments (line 163)
        kwargs_426703 = {}
        # Getting the type of 'logm' (line 163)
        logm_426701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 51), 'logm', False)
        # Calling logm(args, kwargs) (line 163)
        logm_call_result_426704 = invoke(stypy.reporting.localization.Localization(__file__, 163, 51), logm_426701, *[A_426702], **kwargs_426703)
        
        # Processing the call keyword arguments (line 163)
        kwargs_426705 = {}
        # Getting the type of 'expm' (line 163)
        expm_426700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 46), 'expm', False)
        # Calling expm(args, kwargs) (line 163)
        expm_call_result_426706 = invoke(stypy.reporting.localization.Localization(__file__, 163, 46), expm_426700, *[logm_call_result_426704], **kwargs_426705)
        
        # Getting the type of 'A' (line 163)
        A_426707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 61), 'A', False)
        # Processing the call keyword arguments (line 163)
        kwargs_426708 = {}
        # Getting the type of 'assert_array_almost_equal' (line 163)
        assert_array_almost_equal_426699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 163)
        assert_array_almost_equal_call_result_426709 = invoke(stypy.reporting.localization.Localization(__file__, 163, 20), assert_array_almost_equal_426699, *[expm_call_result_426706, A_426707], **kwargs_426708)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_logm_consistency(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_logm_consistency' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_426710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426710)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_logm_consistency'
        return stypy_return_type_426710


    @norecursion
    def test_integer_matrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_integer_matrix'
        module_type_store = module_type_store.open_function_context('test_integer_matrix', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_integer_matrix.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_integer_matrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_integer_matrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_integer_matrix.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_integer_matrix')
        TestExpM.test_integer_matrix.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_integer_matrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_integer_matrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_integer_matrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_integer_matrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_integer_matrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_integer_matrix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_integer_matrix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_integer_matrix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_integer_matrix(...)' code ##################

        
        # Assigning a Call to a Name (line 166):
        
        # Assigning a Call to a Name (line 166):
        
        # Call to array(...): (line 166)
        # Processing the call arguments (line 166)
        
        # Obtaining an instance of the builtin type 'list' (line 166)
        list_426713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 166)
        # Adding element type (line 166)
        
        # Obtaining an instance of the builtin type 'list' (line 167)
        list_426714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 167)
        # Adding element type (line 167)
        int_426715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 12), list_426714, int_426715)
        # Adding element type (line 167)
        int_426716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 12), list_426714, int_426716)
        # Adding element type (line 167)
        int_426717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 12), list_426714, int_426717)
        # Adding element type (line 167)
        int_426718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 12), list_426714, int_426718)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 21), list_426713, list_426714)
        # Adding element type (line 166)
        
        # Obtaining an instance of the builtin type 'list' (line 168)
        list_426719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 168)
        # Adding element type (line 168)
        int_426720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 12), list_426719, int_426720)
        # Adding element type (line 168)
        int_426721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 12), list_426719, int_426721)
        # Adding element type (line 168)
        int_426722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 12), list_426719, int_426722)
        # Adding element type (line 168)
        int_426723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 12), list_426719, int_426723)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 21), list_426713, list_426719)
        # Adding element type (line 166)
        
        # Obtaining an instance of the builtin type 'list' (line 169)
        list_426724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 169)
        # Adding element type (line 169)
        int_426725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 12), list_426724, int_426725)
        # Adding element type (line 169)
        int_426726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 12), list_426724, int_426726)
        # Adding element type (line 169)
        int_426727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 12), list_426724, int_426727)
        # Adding element type (line 169)
        int_426728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 12), list_426724, int_426728)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 21), list_426713, list_426724)
        # Adding element type (line 166)
        
        # Obtaining an instance of the builtin type 'list' (line 170)
        list_426729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 170)
        # Adding element type (line 170)
        int_426730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 12), list_426729, int_426730)
        # Adding element type (line 170)
        int_426731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 12), list_426729, int_426731)
        # Adding element type (line 170)
        int_426732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 12), list_426729, int_426732)
        # Adding element type (line 170)
        int_426733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 12), list_426729, int_426733)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 21), list_426713, list_426729)
        
        # Processing the call keyword arguments (line 166)
        kwargs_426734 = {}
        # Getting the type of 'np' (line 166)
        np_426711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 166)
        array_426712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), np_426711, 'array')
        # Calling array(args, kwargs) (line 166)
        array_call_result_426735 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), array_426712, *[list_426713], **kwargs_426734)
        
        # Assigning a type to the variable 'Q' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'Q', array_call_result_426735)
        
        # Call to assert_allclose(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Call to expm(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'Q' (line 171)
        Q_426738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), 'Q', False)
        # Processing the call keyword arguments (line 171)
        kwargs_426739 = {}
        # Getting the type of 'expm' (line 171)
        expm_426737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 24), 'expm', False)
        # Calling expm(args, kwargs) (line 171)
        expm_call_result_426740 = invoke(stypy.reporting.localization.Localization(__file__, 171, 24), expm_426737, *[Q_426738], **kwargs_426739)
        
        
        # Call to expm(...): (line 171)
        # Processing the call arguments (line 171)
        float_426742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 38), 'float')
        # Getting the type of 'Q' (line 171)
        Q_426743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 44), 'Q', False)
        # Applying the binary operator '*' (line 171)
        result_mul_426744 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 38), '*', float_426742, Q_426743)
        
        # Processing the call keyword arguments (line 171)
        kwargs_426745 = {}
        # Getting the type of 'expm' (line 171)
        expm_426741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 33), 'expm', False)
        # Calling expm(args, kwargs) (line 171)
        expm_call_result_426746 = invoke(stypy.reporting.localization.Localization(__file__, 171, 33), expm_426741, *[result_mul_426744], **kwargs_426745)
        
        # Processing the call keyword arguments (line 171)
        kwargs_426747 = {}
        # Getting the type of 'assert_allclose' (line 171)
        assert_allclose_426736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 171)
        assert_allclose_call_result_426748 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), assert_allclose_426736, *[expm_call_result_426740, expm_call_result_426746], **kwargs_426747)
        
        
        # ################# End of 'test_integer_matrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_integer_matrix' in the type store
        # Getting the type of 'stypy_return_type' (line 165)
        stypy_return_type_426749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426749)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_integer_matrix'
        return stypy_return_type_426749


    @norecursion
    def test_triangularity_perturbation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_triangularity_perturbation'
        module_type_store = module_type_store.open_function_context('test_triangularity_perturbation', 173, 4, False)
        # Assigning a type to the variable 'self' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_triangularity_perturbation.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_triangularity_perturbation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_triangularity_perturbation.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_triangularity_perturbation.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_triangularity_perturbation')
        TestExpM.test_triangularity_perturbation.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_triangularity_perturbation.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_triangularity_perturbation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_triangularity_perturbation.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_triangularity_perturbation.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_triangularity_perturbation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_triangularity_perturbation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_triangularity_perturbation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_triangularity_perturbation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_triangularity_perturbation(...)' code ##################

        
        # Assigning a Call to a Name (line 178):
        
        # Assigning a Call to a Name (line 178):
        
        # Call to array(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Obtaining an instance of the builtin type 'list' (line 178)
        list_426752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 178)
        # Adding element type (line 178)
        
        # Obtaining an instance of the builtin type 'list' (line 179)
        list_426753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 179)
        # Adding element type (line 179)
        float_426754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 12), list_426753, float_426754)
        # Adding element type (line 179)
        float_426755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 12), list_426753, float_426755)
        # Adding element type (line 179)
        float_426756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 12), list_426753, float_426756)
        # Adding element type (line 179)
        float_426757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 12), list_426753, float_426757)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 21), list_426752, list_426753)
        # Adding element type (line 178)
        
        # Obtaining an instance of the builtin type 'list' (line 180)
        list_426758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 180)
        # Adding element type (line 180)
        int_426759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 12), list_426758, int_426759)
        # Adding element type (line 180)
        float_426760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 12), list_426758, float_426760)
        # Adding element type (line 180)
        float_426761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 12), list_426758, float_426761)
        # Adding element type (line 180)
        float_426762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 12), list_426758, float_426762)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 21), list_426752, list_426758)
        # Adding element type (line 178)
        
        # Obtaining an instance of the builtin type 'list' (line 181)
        list_426763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 181)
        # Adding element type (line 181)
        int_426764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 12), list_426763, int_426764)
        # Adding element type (line 181)
        int_426765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 12), list_426763, int_426765)
        # Adding element type (line 181)
        float_426766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 12), list_426763, float_426766)
        # Adding element type (line 181)
        float_426767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 12), list_426763, float_426767)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 21), list_426752, list_426763)
        # Adding element type (line 178)
        
        # Obtaining an instance of the builtin type 'list' (line 182)
        list_426768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 182)
        # Adding element type (line 182)
        int_426769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 12), list_426768, int_426769)
        # Adding element type (line 182)
        int_426770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 12), list_426768, int_426770)
        # Adding element type (line 182)
        int_426771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 12), list_426768, int_426771)
        # Adding element type (line 182)
        float_426772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 12), list_426768, float_426772)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 21), list_426752, list_426768)
        
        # Processing the call keyword arguments (line 178)
        # Getting the type of 'float' (line 183)
        float_426773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 18), 'float', False)
        keyword_426774 = float_426773
        kwargs_426775 = {'dtype': keyword_426774}
        # Getting the type of 'np' (line 178)
        np_426750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 178)
        array_426751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), np_426750, 'array')
        # Calling array(args, kwargs) (line 178)
        array_call_result_426776 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), array_426751, *[list_426752], **kwargs_426775)
        
        # Assigning a type to the variable 'A' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'A', array_call_result_426776)
        
        # Assigning a Call to a Name (line 184):
        
        # Assigning a Call to a Name (line 184):
        
        # Call to array(...): (line 184)
        # Processing the call arguments (line 184)
        
        # Obtaining an instance of the builtin type 'list' (line 184)
        list_426779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 184)
        # Adding element type (line 184)
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_426780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        # Adding element type (line 185)
        float_426781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 12), list_426780, float_426781)
        # Adding element type (line 185)
        float_426782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 12), list_426780, float_426782)
        # Adding element type (line 185)
        float_426783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 12), list_426780, float_426783)
        # Adding element type (line 185)
        float_426784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 12), list_426780, float_426784)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 26), list_426779, list_426780)
        # Adding element type (line 184)
        
        # Obtaining an instance of the builtin type 'list' (line 187)
        list_426785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 187)
        # Adding element type (line 187)
        float_426786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 12), list_426785, float_426786)
        # Adding element type (line 187)
        float_426787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 12), list_426785, float_426787)
        # Adding element type (line 187)
        float_426788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 12), list_426785, float_426788)
        # Adding element type (line 187)
        float_426789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 12), list_426785, float_426789)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 26), list_426779, list_426785)
        # Adding element type (line 184)
        
        # Obtaining an instance of the builtin type 'list' (line 189)
        list_426790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 189)
        # Adding element type (line 189)
        float_426791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 12), list_426790, float_426791)
        # Adding element type (line 189)
        float_426792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 12), list_426790, float_426792)
        # Adding element type (line 189)
        float_426793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 12), list_426790, float_426793)
        # Adding element type (line 189)
        float_426794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 12), list_426790, float_426794)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 26), list_426779, list_426790)
        # Adding element type (line 184)
        
        # Obtaining an instance of the builtin type 'list' (line 191)
        list_426795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 191)
        # Adding element type (line 191)
        float_426796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), list_426795, float_426796)
        # Adding element type (line 191)
        float_426797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), list_426795, float_426797)
        # Adding element type (line 191)
        float_426798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), list_426795, float_426798)
        # Adding element type (line 191)
        float_426799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), list_426795, float_426799)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 26), list_426779, list_426795)
        
        # Processing the call keyword arguments (line 184)
        # Getting the type of 'float' (line 193)
        float_426800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 18), 'float', False)
        keyword_426801 = float_426800
        kwargs_426802 = {'dtype': keyword_426801}
        # Getting the type of 'np' (line 184)
        np_426777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 184)
        array_426778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 17), np_426777, 'array')
        # Calling array(args, kwargs) (line 184)
        array_call_result_426803 = invoke(stypy.reporting.localization.Localization(__file__, 184, 17), array_426778, *[list_426779], **kwargs_426802)
        
        # Assigning a type to the variable 'A_logm' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'A_logm', array_call_result_426803)
        
        # Call to assert_allclose(...): (line 194)
        # Processing the call arguments (line 194)
        
        # Call to expm(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'A_logm' (line 194)
        A_logm_426806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 29), 'A_logm', False)
        # Processing the call keyword arguments (line 194)
        kwargs_426807 = {}
        # Getting the type of 'expm' (line 194)
        expm_426805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 24), 'expm', False)
        # Calling expm(args, kwargs) (line 194)
        expm_call_result_426808 = invoke(stypy.reporting.localization.Localization(__file__, 194, 24), expm_426805, *[A_logm_426806], **kwargs_426807)
        
        # Getting the type of 'A' (line 194)
        A_426809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 38), 'A', False)
        # Processing the call keyword arguments (line 194)
        float_426810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 46), 'float')
        keyword_426811 = float_426810
        kwargs_426812 = {'rtol': keyword_426811}
        # Getting the type of 'assert_allclose' (line 194)
        assert_allclose_426804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 194)
        assert_allclose_call_result_426813 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), assert_allclose_426804, *[expm_call_result_426808, A_426809], **kwargs_426812)
        
        
        # Call to seed(...): (line 198)
        # Processing the call arguments (line 198)
        int_426816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 20), 'int')
        # Processing the call keyword arguments (line 198)
        kwargs_426817 = {}
        # Getting the type of 'random' (line 198)
        random_426814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'random', False)
        # Obtaining the member 'seed' of a type (line 198)
        seed_426815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), random_426814, 'seed')
        # Calling seed(args, kwargs) (line 198)
        seed_call_result_426818 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), seed_426815, *[int_426816], **kwargs_426817)
        
        
        # Assigning a Num to a Name (line 199):
        
        # Assigning a Num to a Name (line 199):
        float_426819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 15), 'float')
        # Assigning a type to the variable 'tiny' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'tiny', float_426819)
        
        # Assigning a Call to a Name (line 200):
        
        # Assigning a Call to a Name (line 200):
        
        # Call to copy(...): (line 200)
        # Processing the call keyword arguments (line 200)
        kwargs_426822 = {}
        # Getting the type of 'A_logm' (line 200)
        A_logm_426820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 27), 'A_logm', False)
        # Obtaining the member 'copy' of a type (line 200)
        copy_426821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 27), A_logm_426820, 'copy')
        # Calling copy(args, kwargs) (line 200)
        copy_call_result_426823 = invoke(stypy.reporting.localization.Localization(__file__, 200, 27), copy_426821, *[], **kwargs_426822)
        
        # Assigning a type to the variable 'A_logm_perturbed' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'A_logm_perturbed', copy_call_result_426823)
        
        # Assigning a Name to a Subscript (line 201):
        
        # Assigning a Name to a Subscript (line 201):
        # Getting the type of 'tiny' (line 201)
        tiny_426824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 33), 'tiny')
        # Getting the type of 'A_logm_perturbed' (line 201)
        A_logm_perturbed_426825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'A_logm_perturbed')
        
        # Obtaining an instance of the builtin type 'tuple' (line 201)
        tuple_426826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 201)
        # Adding element type (line 201)
        int_426827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 25), tuple_426826, int_426827)
        # Adding element type (line 201)
        int_426828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 25), tuple_426826, int_426828)
        
        # Storing an element on a container (line 201)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 8), A_logm_perturbed_426825, (tuple_426826, tiny_426824))
        
        # Call to suppress_warnings(...): (line 202)
        # Processing the call keyword arguments (line 202)
        kwargs_426830 = {}
        # Getting the type of 'suppress_warnings' (line 202)
        suppress_warnings_426829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 202)
        suppress_warnings_call_result_426831 = invoke(stypy.reporting.localization.Localization(__file__, 202, 13), suppress_warnings_426829, *[], **kwargs_426830)
        
        with_426832 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 202, 13), suppress_warnings_call_result_426831, 'with parameter', '__enter__', '__exit__')

        if with_426832:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 202)
            enter___426833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 13), suppress_warnings_call_result_426831, '__enter__')
            with_enter_426834 = invoke(stypy.reporting.localization.Localization(__file__, 202, 13), enter___426833)
            # Assigning a type to the variable 'sup' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 13), 'sup', with_enter_426834)
            
            # Call to filter(...): (line 203)
            # Processing the call arguments (line 203)
            # Getting the type of 'RuntimeWarning' (line 203)
            RuntimeWarning_426837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'RuntimeWarning', False)
            str_426838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 23), 'str', 'scipy.linalg.solve\nIll-conditioned.*')
            # Processing the call keyword arguments (line 203)
            kwargs_426839 = {}
            # Getting the type of 'sup' (line 203)
            sup_426835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 203)
            filter_426836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 12), sup_426835, 'filter')
            # Calling filter(args, kwargs) (line 203)
            filter_call_result_426840 = invoke(stypy.reporting.localization.Localization(__file__, 203, 12), filter_426836, *[RuntimeWarning_426837, str_426838], **kwargs_426839)
            
            
            # Assigning a Call to a Name (line 205):
            
            # Assigning a Call to a Name (line 205):
            
            # Call to expm(...): (line 205)
            # Processing the call arguments (line 205)
            # Getting the type of 'A_logm_perturbed' (line 205)
            A_logm_perturbed_426842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 41), 'A_logm_perturbed', False)
            # Processing the call keyword arguments (line 205)
            kwargs_426843 = {}
            # Getting the type of 'expm' (line 205)
            expm_426841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 36), 'expm', False)
            # Calling expm(args, kwargs) (line 205)
            expm_call_result_426844 = invoke(stypy.reporting.localization.Localization(__file__, 205, 36), expm_426841, *[A_logm_perturbed_426842], **kwargs_426843)
            
            # Assigning a type to the variable 'A_expm_logm_perturbed' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'A_expm_logm_perturbed', expm_call_result_426844)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 202)
            exit___426845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 13), suppress_warnings_call_result_426831, '__exit__')
            with_exit_426846 = invoke(stypy.reporting.localization.Localization(__file__, 202, 13), exit___426845, None, None, None)

        
        # Assigning a Num to a Name (line 206):
        
        # Assigning a Num to a Name (line 206):
        float_426847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 15), 'float')
        # Assigning a type to the variable 'rtol' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'rtol', float_426847)
        
        # Assigning a BinOp to a Name (line 207):
        
        # Assigning a BinOp to a Name (line 207):
        int_426848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 15), 'int')
        # Getting the type of 'tiny' (line 207)
        tiny_426849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 21), 'tiny')
        # Applying the binary operator '*' (line 207)
        result_mul_426850 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 15), '*', int_426848, tiny_426849)
        
        # Assigning a type to the variable 'atol' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'atol', result_mul_426850)
        
        # Call to assert_(...): (line 208)
        # Processing the call arguments (line 208)
        
        
        # Call to allclose(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'A_expm_logm_perturbed' (line 208)
        A_expm_logm_perturbed_426854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 32), 'A_expm_logm_perturbed', False)
        # Getting the type of 'A' (line 208)
        A_426855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 55), 'A', False)
        # Processing the call keyword arguments (line 208)
        # Getting the type of 'rtol' (line 208)
        rtol_426856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 63), 'rtol', False)
        keyword_426857 = rtol_426856
        # Getting the type of 'atol' (line 208)
        atol_426858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 74), 'atol', False)
        keyword_426859 = atol_426858
        kwargs_426860 = {'rtol': keyword_426857, 'atol': keyword_426859}
        # Getting the type of 'np' (line 208)
        np_426852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'np', False)
        # Obtaining the member 'allclose' of a type (line 208)
        allclose_426853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), np_426852, 'allclose')
        # Calling allclose(args, kwargs) (line 208)
        allclose_call_result_426861 = invoke(stypy.reporting.localization.Localization(__file__, 208, 20), allclose_426853, *[A_expm_logm_perturbed_426854, A_426855], **kwargs_426860)
        
        # Applying the 'not' unary operator (line 208)
        result_not__426862 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 16), 'not', allclose_call_result_426861)
        
        # Processing the call keyword arguments (line 208)
        kwargs_426863 = {}
        # Getting the type of 'assert_' (line 208)
        assert__426851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 208)
        assert__call_result_426864 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), assert__426851, *[result_not__426862], **kwargs_426863)
        
        
        # ################# End of 'test_triangularity_perturbation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_triangularity_perturbation' in the type store
        # Getting the type of 'stypy_return_type' (line 173)
        stypy_return_type_426865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426865)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_triangularity_perturbation'
        return stypy_return_type_426865


    @norecursion
    def test_burkardt_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_1'
        module_type_store = module_type_store.open_function_context('test_burkardt_1', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_1.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_1.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_1')
        TestExpM.test_burkardt_1.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_1(...)' code ##################

        
        # Assigning a Call to a Name (line 237):
        
        # Assigning a Call to a Name (line 237):
        
        # Call to exp(...): (line 237)
        # Processing the call arguments (line 237)
        int_426868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 22), 'int')
        # Processing the call keyword arguments (line 237)
        kwargs_426869 = {}
        # Getting the type of 'np' (line 237)
        np_426866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'np', False)
        # Obtaining the member 'exp' of a type (line 237)
        exp_426867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 15), np_426866, 'exp')
        # Calling exp(args, kwargs) (line 237)
        exp_call_result_426870 = invoke(stypy.reporting.localization.Localization(__file__, 237, 15), exp_426867, *[int_426868], **kwargs_426869)
        
        # Assigning a type to the variable 'exp1' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'exp1', exp_call_result_426870)
        
        # Assigning a Call to a Name (line 238):
        
        # Assigning a Call to a Name (line 238):
        
        # Call to exp(...): (line 238)
        # Processing the call arguments (line 238)
        int_426873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 22), 'int')
        # Processing the call keyword arguments (line 238)
        kwargs_426874 = {}
        # Getting the type of 'np' (line 238)
        np_426871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'np', False)
        # Obtaining the member 'exp' of a type (line 238)
        exp_426872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 15), np_426871, 'exp')
        # Calling exp(args, kwargs) (line 238)
        exp_call_result_426875 = invoke(stypy.reporting.localization.Localization(__file__, 238, 15), exp_426872, *[int_426873], **kwargs_426874)
        
        # Assigning a type to the variable 'exp2' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'exp2', exp_call_result_426875)
        
        # Assigning a Call to a Name (line 239):
        
        # Assigning a Call to a Name (line 239):
        
        # Call to array(...): (line 239)
        # Processing the call arguments (line 239)
        
        # Obtaining an instance of the builtin type 'list' (line 239)
        list_426878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 239)
        # Adding element type (line 239)
        
        # Obtaining an instance of the builtin type 'list' (line 240)
        list_426879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 240)
        # Adding element type (line 240)
        int_426880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 12), list_426879, int_426880)
        # Adding element type (line 240)
        int_426881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 12), list_426879, int_426881)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 21), list_426878, list_426879)
        # Adding element type (line 239)
        
        # Obtaining an instance of the builtin type 'list' (line 241)
        list_426882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 241)
        # Adding element type (line 241)
        int_426883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 12), list_426882, int_426883)
        # Adding element type (line 241)
        int_426884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 12), list_426882, int_426884)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 21), list_426878, list_426882)
        
        # Processing the call keyword arguments (line 239)
        # Getting the type of 'float' (line 242)
        float_426885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 21), 'float', False)
        keyword_426886 = float_426885
        kwargs_426887 = {'dtype': keyword_426886}
        # Getting the type of 'np' (line 239)
        np_426876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 239)
        array_426877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), np_426876, 'array')
        # Calling array(args, kwargs) (line 239)
        array_call_result_426888 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), array_426877, *[list_426878], **kwargs_426887)
        
        # Assigning a type to the variable 'A' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'A', array_call_result_426888)
        
        # Assigning a Call to a Name (line 243):
        
        # Assigning a Call to a Name (line 243):
        
        # Call to array(...): (line 243)
        # Processing the call arguments (line 243)
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_426891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        
        # Obtaining an instance of the builtin type 'list' (line 244)
        list_426892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 244)
        # Adding element type (line 244)
        # Getting the type of 'exp1' (line 244)
        exp1_426893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 13), 'exp1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 12), list_426892, exp1_426893)
        # Adding element type (line 244)
        int_426894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 12), list_426892, int_426894)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 27), list_426891, list_426892)
        # Adding element type (line 243)
        
        # Obtaining an instance of the builtin type 'list' (line 245)
        list_426895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 245)
        # Adding element type (line 245)
        int_426896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 12), list_426895, int_426896)
        # Adding element type (line 245)
        # Getting the type of 'exp2' (line 245)
        exp2_426897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'exp2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 12), list_426895, exp2_426897)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 27), list_426891, list_426895)
        
        # Processing the call keyword arguments (line 243)
        # Getting the type of 'float' (line 246)
        float_426898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 21), 'float', False)
        keyword_426899 = float_426898
        kwargs_426900 = {'dtype': keyword_426899}
        # Getting the type of 'np' (line 243)
        np_426889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 243)
        array_426890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 18), np_426889, 'array')
        # Calling array(args, kwargs) (line 243)
        array_call_result_426901 = invoke(stypy.reporting.localization.Localization(__file__, 243, 18), array_426890, *[list_426891], **kwargs_426900)
        
        # Assigning a type to the variable 'desired' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'desired', array_call_result_426901)
        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Call to expm(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'A' (line 247)
        A_426903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 22), 'A', False)
        # Processing the call keyword arguments (line 247)
        kwargs_426904 = {}
        # Getting the type of 'expm' (line 247)
        expm_426902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 17), 'expm', False)
        # Calling expm(args, kwargs) (line 247)
        expm_call_result_426905 = invoke(stypy.reporting.localization.Localization(__file__, 247, 17), expm_426902, *[A_426903], **kwargs_426904)
        
        # Assigning a type to the variable 'actual' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'actual', expm_call_result_426905)
        
        # Call to assert_allclose(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'actual' (line 248)
        actual_426907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'actual', False)
        # Getting the type of 'desired' (line 248)
        desired_426908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 32), 'desired', False)
        # Processing the call keyword arguments (line 248)
        kwargs_426909 = {}
        # Getting the type of 'assert_allclose' (line 248)
        assert_allclose_426906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 248)
        assert_allclose_call_result_426910 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), assert_allclose_426906, *[actual_426907, desired_426908], **kwargs_426909)
        
        
        # ################# End of 'test_burkardt_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_1' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_426911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426911)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_1'
        return stypy_return_type_426911


    @norecursion
    def test_burkardt_2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_2'
        module_type_store = module_type_store.open_function_context('test_burkardt_2', 250, 4, False)
        # Assigning a type to the variable 'self' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_2.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_2.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_2')
        TestExpM.test_burkardt_2.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_2(...)' code ##################

        
        # Assigning a Call to a Name (line 253):
        
        # Assigning a Call to a Name (line 253):
        
        # Call to array(...): (line 253)
        # Processing the call arguments (line 253)
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_426914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_426915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        int_426916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 12), list_426915, int_426916)
        # Adding element type (line 254)
        int_426917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 12), list_426915, int_426917)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 21), list_426914, list_426915)
        # Adding element type (line 253)
        
        # Obtaining an instance of the builtin type 'list' (line 255)
        list_426918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 255)
        # Adding element type (line 255)
        int_426919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 12), list_426918, int_426919)
        # Adding element type (line 255)
        int_426920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 12), list_426918, int_426920)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 21), list_426914, list_426918)
        
        # Processing the call keyword arguments (line 253)
        # Getting the type of 'float' (line 256)
        float_426921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 21), 'float', False)
        keyword_426922 = float_426921
        kwargs_426923 = {'dtype': keyword_426922}
        # Getting the type of 'np' (line 253)
        np_426912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 253)
        array_426913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 12), np_426912, 'array')
        # Calling array(args, kwargs) (line 253)
        array_call_result_426924 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), array_426913, *[list_426914], **kwargs_426923)
        
        # Assigning a type to the variable 'A' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'A', array_call_result_426924)
        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Call to array(...): (line 257)
        # Processing the call arguments (line 257)
        
        # Obtaining an instance of the builtin type 'list' (line 257)
        list_426927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 257)
        # Adding element type (line 257)
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_426928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        # Adding element type (line 258)
        float_426929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 12), list_426928, float_426929)
        # Adding element type (line 258)
        float_426930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 12), list_426928, float_426930)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 27), list_426927, list_426928)
        # Adding element type (line 257)
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_426931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        float_426932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 12), list_426931, float_426932)
        # Adding element type (line 259)
        float_426933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 12), list_426931, float_426933)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 27), list_426927, list_426931)
        
        # Processing the call keyword arguments (line 257)
        # Getting the type of 'float' (line 260)
        float_426934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 21), 'float', False)
        keyword_426935 = float_426934
        kwargs_426936 = {'dtype': keyword_426935}
        # Getting the type of 'np' (line 257)
        np_426925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 257)
        array_426926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 18), np_426925, 'array')
        # Calling array(args, kwargs) (line 257)
        array_call_result_426937 = invoke(stypy.reporting.localization.Localization(__file__, 257, 18), array_426926, *[list_426927], **kwargs_426936)
        
        # Assigning a type to the variable 'desired' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'desired', array_call_result_426937)
        
        # Assigning a Call to a Name (line 261):
        
        # Assigning a Call to a Name (line 261):
        
        # Call to expm(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'A' (line 261)
        A_426939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 22), 'A', False)
        # Processing the call keyword arguments (line 261)
        kwargs_426940 = {}
        # Getting the type of 'expm' (line 261)
        expm_426938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 17), 'expm', False)
        # Calling expm(args, kwargs) (line 261)
        expm_call_result_426941 = invoke(stypy.reporting.localization.Localization(__file__, 261, 17), expm_426938, *[A_426939], **kwargs_426940)
        
        # Assigning a type to the variable 'actual' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'actual', expm_call_result_426941)
        
        # Call to assert_allclose(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'actual' (line 262)
        actual_426943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'actual', False)
        # Getting the type of 'desired' (line 262)
        desired_426944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 32), 'desired', False)
        # Processing the call keyword arguments (line 262)
        kwargs_426945 = {}
        # Getting the type of 'assert_allclose' (line 262)
        assert_allclose_426942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 262)
        assert_allclose_call_result_426946 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), assert_allclose_426942, *[actual_426943, desired_426944], **kwargs_426945)
        
        
        # ################# End of 'test_burkardt_2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_2' in the type store
        # Getting the type of 'stypy_return_type' (line 250)
        stypy_return_type_426947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426947)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_2'
        return stypy_return_type_426947


    @norecursion
    def test_burkardt_3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_3'
        module_type_store = module_type_store.open_function_context('test_burkardt_3', 264, 4, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_3.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_3.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_3')
        TestExpM.test_burkardt_3.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_3(...)' code ##################

        
        # Assigning a Call to a Name (line 268):
        
        # Assigning a Call to a Name (line 268):
        
        # Call to exp(...): (line 268)
        # Processing the call arguments (line 268)
        int_426950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 22), 'int')
        # Processing the call keyword arguments (line 268)
        kwargs_426951 = {}
        # Getting the type of 'np' (line 268)
        np_426948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'np', False)
        # Obtaining the member 'exp' of a type (line 268)
        exp_426949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 15), np_426948, 'exp')
        # Calling exp(args, kwargs) (line 268)
        exp_call_result_426952 = invoke(stypy.reporting.localization.Localization(__file__, 268, 15), exp_426949, *[int_426950], **kwargs_426951)
        
        # Assigning a type to the variable 'exp1' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'exp1', exp_call_result_426952)
        
        # Assigning a Call to a Name (line 269):
        
        # Assigning a Call to a Name (line 269):
        
        # Call to exp(...): (line 269)
        # Processing the call arguments (line 269)
        int_426955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 23), 'int')
        # Processing the call keyword arguments (line 269)
        kwargs_426956 = {}
        # Getting the type of 'np' (line 269)
        np_426953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'np', False)
        # Obtaining the member 'exp' of a type (line 269)
        exp_426954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 16), np_426953, 'exp')
        # Calling exp(args, kwargs) (line 269)
        exp_call_result_426957 = invoke(stypy.reporting.localization.Localization(__file__, 269, 16), exp_426954, *[int_426955], **kwargs_426956)
        
        # Assigning a type to the variable 'exp39' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'exp39', exp_call_result_426957)
        
        # Assigning a Call to a Name (line 270):
        
        # Assigning a Call to a Name (line 270):
        
        # Call to array(...): (line 270)
        # Processing the call arguments (line 270)
        
        # Obtaining an instance of the builtin type 'list' (line 270)
        list_426960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 270)
        # Adding element type (line 270)
        
        # Obtaining an instance of the builtin type 'list' (line 271)
        list_426961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 271)
        # Adding element type (line 271)
        int_426962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 12), list_426961, int_426962)
        # Adding element type (line 271)
        int_426963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 12), list_426961, int_426963)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 21), list_426960, list_426961)
        # Adding element type (line 270)
        
        # Obtaining an instance of the builtin type 'list' (line 272)
        list_426964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 272)
        # Adding element type (line 272)
        int_426965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 12), list_426964, int_426965)
        # Adding element type (line 272)
        int_426966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 12), list_426964, int_426966)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 21), list_426960, list_426964)
        
        # Processing the call keyword arguments (line 270)
        # Getting the type of 'float' (line 273)
        float_426967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 21), 'float', False)
        keyword_426968 = float_426967
        kwargs_426969 = {'dtype': keyword_426968}
        # Getting the type of 'np' (line 270)
        np_426958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 270)
        array_426959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 12), np_426958, 'array')
        # Calling array(args, kwargs) (line 270)
        array_call_result_426970 = invoke(stypy.reporting.localization.Localization(__file__, 270, 12), array_426959, *[list_426960], **kwargs_426969)
        
        # Assigning a type to the variable 'A' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'A', array_call_result_426970)
        
        # Assigning a Call to a Name (line 274):
        
        # Assigning a Call to a Name (line 274):
        
        # Call to array(...): (line 274)
        # Processing the call arguments (line 274)
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_426973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_426974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        int_426975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 16), 'int')
        int_426976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 20), 'int')
        # Getting the type of 'exp1' (line 276)
        exp1_426977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 23), 'exp1', False)
        # Applying the binary operator '*' (line 276)
        result_mul_426978 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 20), '*', int_426976, exp1_426977)
        
        # Applying the binary operator 'div' (line 276)
        result_div_426979 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 16), 'div', int_426975, result_mul_426978)
        
        int_426980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 31), 'int')
        int_426981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 34), 'int')
        # Getting the type of 'exp39' (line 276)
        exp39_426982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 37), 'exp39', False)
        # Applying the binary operator '*' (line 276)
        result_mul_426983 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 34), '*', int_426981, exp39_426982)
        
        # Applying the binary operator 'div' (line 276)
        result_div_426984 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 31), 'div', int_426980, result_mul_426983)
        
        # Applying the binary operator '-' (line 276)
        result_sub_426985 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 16), '-', result_div_426979, result_div_426984)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 12), list_426974, result_sub_426985)
        # Adding element type (line 275)
        
        
        # Call to expm1(...): (line 277)
        # Processing the call arguments (line 277)
        int_426988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 26), 'int')
        # Processing the call keyword arguments (line 277)
        kwargs_426989 = {}
        # Getting the type of 'np' (line 277)
        np_426986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 17), 'np', False)
        # Obtaining the member 'expm1' of a type (line 277)
        expm1_426987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 17), np_426986, 'expm1')
        # Calling expm1(args, kwargs) (line 277)
        expm1_call_result_426990 = invoke(stypy.reporting.localization.Localization(__file__, 277, 17), expm1_426987, *[int_426988], **kwargs_426989)
        
        # Applying the 'usub' unary operator (line 277)
        result___neg___426991 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 16), 'usub', expm1_call_result_426990)
        
        int_426992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 34), 'int')
        # Getting the type of 'exp1' (line 277)
        exp1_426993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 37), 'exp1', False)
        # Applying the binary operator '*' (line 277)
        result_mul_426994 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 34), '*', int_426992, exp1_426993)
        
        # Applying the binary operator 'div' (line 277)
        result_div_426995 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 16), 'div', result___neg___426991, result_mul_426994)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 12), list_426974, result_div_426995)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 27), list_426973, list_426974)
        # Adding element type (line 274)
        
        # Obtaining an instance of the builtin type 'list' (line 278)
        list_426996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 278)
        # Adding element type (line 278)
        int_426997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 16), 'int')
        
        # Call to expm1(...): (line 279)
        # Processing the call arguments (line 279)
        int_427000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 28), 'int')
        # Processing the call keyword arguments (line 279)
        kwargs_427001 = {}
        # Getting the type of 'np' (line 279)
        np_426998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 19), 'np', False)
        # Obtaining the member 'expm1' of a type (line 279)
        expm1_426999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 19), np_426998, 'expm1')
        # Calling expm1(args, kwargs) (line 279)
        expm1_call_result_427002 = invoke(stypy.reporting.localization.Localization(__file__, 279, 19), expm1_426999, *[int_427000], **kwargs_427001)
        
        # Applying the binary operator '*' (line 279)
        result_mul_427003 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 16), '*', int_426997, expm1_call_result_427002)
        
        int_427004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 36), 'int')
        # Getting the type of 'exp1' (line 279)
        exp1_427005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 39), 'exp1', False)
        # Applying the binary operator '*' (line 279)
        result_mul_427006 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 36), '*', int_427004, exp1_427005)
        
        # Applying the binary operator 'div' (line 279)
        result_div_427007 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 33), 'div', result_mul_427003, result_mul_427006)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 12), list_426996, result_div_427007)
        # Adding element type (line 278)
        int_427008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 16), 'int')
        int_427009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 20), 'int')
        # Getting the type of 'exp1' (line 280)
        exp1_427010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 23), 'exp1', False)
        # Applying the binary operator '*' (line 280)
        result_mul_427011 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 20), '*', int_427009, exp1_427010)
        
        # Applying the binary operator 'div' (line 280)
        result_div_427012 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 16), 'div', int_427008, result_mul_427011)
        
        int_427013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 31), 'int')
        int_427014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 35), 'int')
        # Getting the type of 'exp39' (line 280)
        exp39_427015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 38), 'exp39', False)
        # Applying the binary operator '*' (line 280)
        result_mul_427016 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 35), '*', int_427014, exp39_427015)
        
        # Applying the binary operator 'div' (line 280)
        result_div_427017 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 31), 'div', int_427013, result_mul_427016)
        
        # Applying the binary operator '+' (line 280)
        result_add_427018 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 16), '+', result_div_427012, result_div_427017)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 12), list_426996, result_add_427018)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 27), list_426973, list_426996)
        
        # Processing the call keyword arguments (line 274)
        # Getting the type of 'float' (line 281)
        float_427019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 21), 'float', False)
        keyword_427020 = float_427019
        kwargs_427021 = {'dtype': keyword_427020}
        # Getting the type of 'np' (line 274)
        np_426971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 274)
        array_426972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 18), np_426971, 'array')
        # Calling array(args, kwargs) (line 274)
        array_call_result_427022 = invoke(stypy.reporting.localization.Localization(__file__, 274, 18), array_426972, *[list_426973], **kwargs_427021)
        
        # Assigning a type to the variable 'desired' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'desired', array_call_result_427022)
        
        # Assigning a Call to a Name (line 282):
        
        # Assigning a Call to a Name (line 282):
        
        # Call to expm(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'A' (line 282)
        A_427024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 22), 'A', False)
        # Processing the call keyword arguments (line 282)
        kwargs_427025 = {}
        # Getting the type of 'expm' (line 282)
        expm_427023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'expm', False)
        # Calling expm(args, kwargs) (line 282)
        expm_call_result_427026 = invoke(stypy.reporting.localization.Localization(__file__, 282, 17), expm_427023, *[A_427024], **kwargs_427025)
        
        # Assigning a type to the variable 'actual' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'actual', expm_call_result_427026)
        
        # Call to assert_allclose(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'actual' (line 283)
        actual_427028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 24), 'actual', False)
        # Getting the type of 'desired' (line 283)
        desired_427029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 32), 'desired', False)
        # Processing the call keyword arguments (line 283)
        kwargs_427030 = {}
        # Getting the type of 'assert_allclose' (line 283)
        assert_allclose_427027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 283)
        assert_allclose_call_result_427031 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), assert_allclose_427027, *[actual_427028, desired_427029], **kwargs_427030)
        
        
        # ################# End of 'test_burkardt_3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_3' in the type store
        # Getting the type of 'stypy_return_type' (line 264)
        stypy_return_type_427032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427032)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_3'
        return stypy_return_type_427032


    @norecursion
    def test_burkardt_4(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_4'
        module_type_store = module_type_store.open_function_context('test_burkardt_4', 285, 4, False)
        # Assigning a type to the variable 'self' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_4.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_4.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_4.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_4.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_4')
        TestExpM.test_burkardt_4.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_4.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_4.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_4.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_4.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_4.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_4.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_4', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_4', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_4(...)' code ##################

        
        # Assigning a Call to a Name (line 289):
        
        # Assigning a Call to a Name (line 289):
        
        # Call to array(...): (line 289)
        # Processing the call arguments (line 289)
        
        # Obtaining an instance of the builtin type 'list' (line 289)
        list_427035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 289)
        # Adding element type (line 289)
        
        # Obtaining an instance of the builtin type 'list' (line 290)
        list_427036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 290)
        # Adding element type (line 290)
        int_427037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 12), list_427036, int_427037)
        # Adding element type (line 290)
        int_427038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 12), list_427036, int_427038)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 21), list_427035, list_427036)
        # Adding element type (line 289)
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_427039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        # Adding element type (line 291)
        int_427040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 12), list_427039, int_427040)
        # Adding element type (line 291)
        int_427041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 12), list_427039, int_427041)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 21), list_427035, list_427039)
        
        # Processing the call keyword arguments (line 289)
        # Getting the type of 'float' (line 292)
        float_427042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 21), 'float', False)
        keyword_427043 = float_427042
        kwargs_427044 = {'dtype': keyword_427043}
        # Getting the type of 'np' (line 289)
        np_427033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 289)
        array_427034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 12), np_427033, 'array')
        # Calling array(args, kwargs) (line 289)
        array_call_result_427045 = invoke(stypy.reporting.localization.Localization(__file__, 289, 12), array_427034, *[list_427035], **kwargs_427044)
        
        # Assigning a type to the variable 'A' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'A', array_call_result_427045)
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to array(...): (line 293)
        # Processing the call arguments (line 293)
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_427048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_427049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        int_427050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 22), list_427049, int_427050)
        # Adding element type (line 293)
        int_427051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 22), list_427049, int_427051)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 21), list_427048, list_427049)
        # Adding element type (line 293)
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_427052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        int_427053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 30), list_427052, int_427053)
        # Adding element type (line 293)
        int_427054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 30), list_427052, int_427054)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 21), list_427048, list_427052)
        
        # Processing the call keyword arguments (line 293)
        # Getting the type of 'float' (line 293)
        float_427055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 45), 'float', False)
        keyword_427056 = float_427055
        kwargs_427057 = {'dtype': keyword_427056}
        # Getting the type of 'np' (line 293)
        np_427046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 293)
        array_427047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 12), np_427046, 'array')
        # Calling array(args, kwargs) (line 293)
        array_call_result_427058 = invoke(stypy.reporting.localization.Localization(__file__, 293, 12), array_427047, *[list_427048], **kwargs_427057)
        
        # Assigning a type to the variable 'U' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'U', array_call_result_427058)
        
        # Assigning a Call to a Name (line 294):
        
        # Assigning a Call to a Name (line 294):
        
        # Call to array(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Obtaining an instance of the builtin type 'list' (line 294)
        list_427061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 294)
        # Adding element type (line 294)
        
        # Obtaining an instance of the builtin type 'list' (line 294)
        list_427062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 294)
        # Adding element type (line 294)
        int_427063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_427062, int_427063)
        # Adding element type (line 294)
        int_427064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 26), 'int')
        int_427065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 29), 'int')
        # Applying the binary operator 'div' (line 294)
        result_div_427066 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 26), 'div', int_427064, int_427065)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_427062, result_div_427066)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 21), list_427061, list_427062)
        # Adding element type (line 294)
        
        # Obtaining an instance of the builtin type 'list' (line 294)
        list_427067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 294)
        # Adding element type (line 294)
        int_427068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 33), list_427067, int_427068)
        # Adding element type (line 294)
        int_427069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 38), 'int')
        int_427070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 40), 'int')
        # Applying the binary operator 'div' (line 294)
        result_div_427071 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 38), 'div', int_427069, int_427070)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 33), list_427067, result_div_427071)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 21), list_427061, list_427067)
        
        # Processing the call keyword arguments (line 294)
        # Getting the type of 'float' (line 294)
        float_427072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 51), 'float', False)
        keyword_427073 = float_427072
        kwargs_427074 = {'dtype': keyword_427073}
        # Getting the type of 'np' (line 294)
        np_427059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 294)
        array_427060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 12), np_427059, 'array')
        # Calling array(args, kwargs) (line 294)
        array_call_result_427075 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), array_427060, *[list_427061], **kwargs_427074)
        
        # Assigning a type to the variable 'V' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'V', array_call_result_427075)
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to array(...): (line 295)
        # Processing the call arguments (line 295)
        
        # Obtaining an instance of the builtin type 'list' (line 295)
        list_427078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 295)
        # Adding element type (line 295)
        int_427079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 21), list_427078, int_427079)
        # Adding element type (line 295)
        int_427080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 21), list_427078, int_427080)
        
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'float' (line 295)
        float_427081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 38), 'float', False)
        keyword_427082 = float_427081
        kwargs_427083 = {'dtype': keyword_427082}
        # Getting the type of 'np' (line 295)
        np_427076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 295)
        array_427077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 12), np_427076, 'array')
        # Calling array(args, kwargs) (line 295)
        array_call_result_427084 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), array_427077, *[list_427078], **kwargs_427083)
        
        # Assigning a type to the variable 'w' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'w', array_call_result_427084)
        
        # Assigning a Call to a Name (line 296):
        
        # Assigning a Call to a Name (line 296):
        
        # Call to dot(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'U' (line 296)
        U_427087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 25), 'U', False)
        
        # Call to exp(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'w' (line 296)
        w_427090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 36), 'w', False)
        # Processing the call keyword arguments (line 296)
        kwargs_427091 = {}
        # Getting the type of 'np' (line 296)
        np_427088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 29), 'np', False)
        # Obtaining the member 'exp' of a type (line 296)
        exp_427089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 29), np_427088, 'exp')
        # Calling exp(args, kwargs) (line 296)
        exp_call_result_427092 = invoke(stypy.reporting.localization.Localization(__file__, 296, 29), exp_427089, *[w_427090], **kwargs_427091)
        
        # Applying the binary operator '*' (line 296)
        result_mul_427093 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 25), '*', U_427087, exp_call_result_427092)
        
        # Getting the type of 'V' (line 296)
        V_427094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 40), 'V', False)
        # Processing the call keyword arguments (line 296)
        kwargs_427095 = {}
        # Getting the type of 'np' (line 296)
        np_427085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 18), 'np', False)
        # Obtaining the member 'dot' of a type (line 296)
        dot_427086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 18), np_427085, 'dot')
        # Calling dot(args, kwargs) (line 296)
        dot_call_result_427096 = invoke(stypy.reporting.localization.Localization(__file__, 296, 18), dot_427086, *[result_mul_427093, V_427094], **kwargs_427095)
        
        # Assigning a type to the variable 'desired' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'desired', dot_call_result_427096)
        
        # Assigning a Call to a Name (line 297):
        
        # Assigning a Call to a Name (line 297):
        
        # Call to expm(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'A' (line 297)
        A_427098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 22), 'A', False)
        # Processing the call keyword arguments (line 297)
        kwargs_427099 = {}
        # Getting the type of 'expm' (line 297)
        expm_427097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 17), 'expm', False)
        # Calling expm(args, kwargs) (line 297)
        expm_call_result_427100 = invoke(stypy.reporting.localization.Localization(__file__, 297, 17), expm_427097, *[A_427098], **kwargs_427099)
        
        # Assigning a type to the variable 'actual' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'actual', expm_call_result_427100)
        
        # Call to assert_allclose(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'actual' (line 298)
        actual_427102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 24), 'actual', False)
        # Getting the type of 'desired' (line 298)
        desired_427103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 32), 'desired', False)
        # Processing the call keyword arguments (line 298)
        kwargs_427104 = {}
        # Getting the type of 'assert_allclose' (line 298)
        assert_allclose_427101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 298)
        assert_allclose_call_result_427105 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), assert_allclose_427101, *[actual_427102, desired_427103], **kwargs_427104)
        
        
        # ################# End of 'test_burkardt_4(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_4' in the type store
        # Getting the type of 'stypy_return_type' (line 285)
        stypy_return_type_427106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427106)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_4'
        return stypy_return_type_427106


    @norecursion
    def test_burkardt_5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_5'
        module_type_store = module_type_store.open_function_context('test_burkardt_5', 300, 4, False)
        # Assigning a type to the variable 'self' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_5.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_5.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_5.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_5')
        TestExpM.test_burkardt_5.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_5.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_5.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_5.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_5.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_5', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_5', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_5(...)' code ##################

        
        # Assigning a Call to a Name (line 305):
        
        # Assigning a Call to a Name (line 305):
        
        # Call to array(...): (line 305)
        # Processing the call arguments (line 305)
        
        # Obtaining an instance of the builtin type 'list' (line 305)
        list_427109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 305)
        # Adding element type (line 305)
        
        # Obtaining an instance of the builtin type 'list' (line 306)
        list_427110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 306)
        # Adding element type (line 306)
        int_427111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 12), list_427110, int_427111)
        # Adding element type (line 306)
        int_427112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 12), list_427110, int_427112)
        # Adding element type (line 306)
        int_427113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 12), list_427110, int_427113)
        # Adding element type (line 306)
        int_427114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 12), list_427110, int_427114)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 21), list_427109, list_427110)
        # Adding element type (line 305)
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_427115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        int_427116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 12), list_427115, int_427116)
        # Adding element type (line 307)
        int_427117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 12), list_427115, int_427117)
        # Adding element type (line 307)
        int_427118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 12), list_427115, int_427118)
        # Adding element type (line 307)
        int_427119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 12), list_427115, int_427119)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 21), list_427109, list_427115)
        # Adding element type (line 305)
        
        # Obtaining an instance of the builtin type 'list' (line 308)
        list_427120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 308)
        # Adding element type (line 308)
        int_427121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 12), list_427120, int_427121)
        # Adding element type (line 308)
        int_427122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 12), list_427120, int_427122)
        # Adding element type (line 308)
        int_427123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 12), list_427120, int_427123)
        # Adding element type (line 308)
        int_427124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 12), list_427120, int_427124)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 21), list_427109, list_427120)
        # Adding element type (line 305)
        
        # Obtaining an instance of the builtin type 'list' (line 309)
        list_427125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 309)
        # Adding element type (line 309)
        int_427126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 12), list_427125, int_427126)
        # Adding element type (line 309)
        int_427127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 12), list_427125, int_427127)
        # Adding element type (line 309)
        int_427128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 12), list_427125, int_427128)
        # Adding element type (line 309)
        int_427129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 12), list_427125, int_427129)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 21), list_427109, list_427125)
        
        # Processing the call keyword arguments (line 305)
        # Getting the type of 'float' (line 310)
        float_427130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 21), 'float', False)
        keyword_427131 = float_427130
        kwargs_427132 = {'dtype': keyword_427131}
        # Getting the type of 'np' (line 305)
        np_427107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 305)
        array_427108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 12), np_427107, 'array')
        # Calling array(args, kwargs) (line 305)
        array_call_result_427133 = invoke(stypy.reporting.localization.Localization(__file__, 305, 12), array_427108, *[list_427109], **kwargs_427132)
        
        # Assigning a type to the variable 'A' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'A', array_call_result_427133)
        
        # Assigning a Call to a Name (line 311):
        
        # Assigning a Call to a Name (line 311):
        
        # Call to array(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_427136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        # Adding element type (line 311)
        
        # Obtaining an instance of the builtin type 'list' (line 312)
        list_427137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 312)
        # Adding element type (line 312)
        int_427138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 12), list_427137, int_427138)
        # Adding element type (line 312)
        int_427139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 12), list_427137, int_427139)
        # Adding element type (line 312)
        int_427140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 12), list_427137, int_427140)
        # Adding element type (line 312)
        int_427141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 12), list_427137, int_427141)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 27), list_427136, list_427137)
        # Adding element type (line 311)
        
        # Obtaining an instance of the builtin type 'list' (line 313)
        list_427142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 313)
        # Adding element type (line 313)
        int_427143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 12), list_427142, int_427143)
        # Adding element type (line 313)
        int_427144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 12), list_427142, int_427144)
        # Adding element type (line 313)
        int_427145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 12), list_427142, int_427145)
        # Adding element type (line 313)
        int_427146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 12), list_427142, int_427146)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 27), list_427136, list_427142)
        # Adding element type (line 311)
        
        # Obtaining an instance of the builtin type 'list' (line 314)
        list_427147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 314)
        # Adding element type (line 314)
        int_427148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 12), list_427147, int_427148)
        # Adding element type (line 314)
        int_427149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 12), list_427147, int_427149)
        # Adding element type (line 314)
        int_427150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 12), list_427147, int_427150)
        # Adding element type (line 314)
        int_427151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 12), list_427147, int_427151)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 27), list_427136, list_427147)
        # Adding element type (line 311)
        
        # Obtaining an instance of the builtin type 'list' (line 315)
        list_427152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 315)
        # Adding element type (line 315)
        int_427153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 12), list_427152, int_427153)
        # Adding element type (line 315)
        int_427154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 12), list_427152, int_427154)
        # Adding element type (line 315)
        int_427155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 12), list_427152, int_427155)
        # Adding element type (line 315)
        int_427156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 12), list_427152, int_427156)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 27), list_427136, list_427152)
        
        # Processing the call keyword arguments (line 311)
        # Getting the type of 'float' (line 316)
        float_427157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 21), 'float', False)
        keyword_427158 = float_427157
        kwargs_427159 = {'dtype': keyword_427158}
        # Getting the type of 'np' (line 311)
        np_427134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 311)
        array_427135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 18), np_427134, 'array')
        # Calling array(args, kwargs) (line 311)
        array_call_result_427160 = invoke(stypy.reporting.localization.Localization(__file__, 311, 18), array_427135, *[list_427136], **kwargs_427159)
        
        # Assigning a type to the variable 'desired' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'desired', array_call_result_427160)
        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to expm(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'A' (line 317)
        A_427162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 22), 'A', False)
        # Processing the call keyword arguments (line 317)
        kwargs_427163 = {}
        # Getting the type of 'expm' (line 317)
        expm_427161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 17), 'expm', False)
        # Calling expm(args, kwargs) (line 317)
        expm_call_result_427164 = invoke(stypy.reporting.localization.Localization(__file__, 317, 17), expm_427161, *[A_427162], **kwargs_427163)
        
        # Assigning a type to the variable 'actual' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'actual', expm_call_result_427164)
        
        # Call to assert_allclose(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'actual' (line 318)
        actual_427166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 24), 'actual', False)
        # Getting the type of 'desired' (line 318)
        desired_427167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 32), 'desired', False)
        # Processing the call keyword arguments (line 318)
        kwargs_427168 = {}
        # Getting the type of 'assert_allclose' (line 318)
        assert_allclose_427165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 318)
        assert_allclose_call_result_427169 = invoke(stypy.reporting.localization.Localization(__file__, 318, 8), assert_allclose_427165, *[actual_427166, desired_427167], **kwargs_427168)
        
        
        # ################# End of 'test_burkardt_5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_5' in the type store
        # Getting the type of 'stypy_return_type' (line 300)
        stypy_return_type_427170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427170)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_5'
        return stypy_return_type_427170


    @norecursion
    def test_burkardt_6(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_6'
        module_type_store = module_type_store.open_function_context('test_burkardt_6', 320, 4, False)
        # Assigning a type to the variable 'self' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_6.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_6.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_6.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_6.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_6')
        TestExpM.test_burkardt_6.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_6.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_6.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_6.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_6.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_6.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_6.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_6', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_6', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_6(...)' code ##################

        
        # Assigning a Call to a Name (line 324):
        
        # Assigning a Call to a Name (line 324):
        
        # Call to exp(...): (line 324)
        # Processing the call arguments (line 324)
        int_427173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 22), 'int')
        # Processing the call keyword arguments (line 324)
        kwargs_427174 = {}
        # Getting the type of 'np' (line 324)
        np_427171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 15), 'np', False)
        # Obtaining the member 'exp' of a type (line 324)
        exp_427172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 15), np_427171, 'exp')
        # Calling exp(args, kwargs) (line 324)
        exp_call_result_427175 = invoke(stypy.reporting.localization.Localization(__file__, 324, 15), exp_427172, *[int_427173], **kwargs_427174)
        
        # Assigning a type to the variable 'exp1' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'exp1', exp_call_result_427175)
        
        # Assigning a Call to a Name (line 325):
        
        # Assigning a Call to a Name (line 325):
        
        # Call to array(...): (line 325)
        # Processing the call arguments (line 325)
        
        # Obtaining an instance of the builtin type 'list' (line 325)
        list_427178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 325)
        # Adding element type (line 325)
        
        # Obtaining an instance of the builtin type 'list' (line 326)
        list_427179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 326)
        # Adding element type (line 326)
        int_427180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 12), list_427179, int_427180)
        # Adding element type (line 326)
        int_427181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 12), list_427179, int_427181)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 21), list_427178, list_427179)
        # Adding element type (line 325)
        
        # Obtaining an instance of the builtin type 'list' (line 327)
        list_427182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 327)
        # Adding element type (line 327)
        int_427183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 12), list_427182, int_427183)
        # Adding element type (line 327)
        int_427184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 12), list_427182, int_427184)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 21), list_427178, list_427182)
        
        # Processing the call keyword arguments (line 325)
        # Getting the type of 'float' (line 328)
        float_427185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 21), 'float', False)
        keyword_427186 = float_427185
        kwargs_427187 = {'dtype': keyword_427186}
        # Getting the type of 'np' (line 325)
        np_427176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 325)
        array_427177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), np_427176, 'array')
        # Calling array(args, kwargs) (line 325)
        array_call_result_427188 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), array_427177, *[list_427178], **kwargs_427187)
        
        # Assigning a type to the variable 'A' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'A', array_call_result_427188)
        
        # Assigning a Call to a Name (line 329):
        
        # Assigning a Call to a Name (line 329):
        
        # Call to array(...): (line 329)
        # Processing the call arguments (line 329)
        
        # Obtaining an instance of the builtin type 'list' (line 329)
        list_427191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 329)
        # Adding element type (line 329)
        
        # Obtaining an instance of the builtin type 'list' (line 330)
        list_427192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 330)
        # Adding element type (line 330)
        # Getting the type of 'exp1' (line 330)
        exp1_427193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 13), 'exp1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 12), list_427192, exp1_427193)
        # Adding element type (line 330)
        # Getting the type of 'exp1' (line 330)
        exp1_427194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'exp1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 12), list_427192, exp1_427194)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 27), list_427191, list_427192)
        # Adding element type (line 329)
        
        # Obtaining an instance of the builtin type 'list' (line 331)
        list_427195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 331)
        # Adding element type (line 331)
        int_427196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 12), list_427195, int_427196)
        # Adding element type (line 331)
        # Getting the type of 'exp1' (line 331)
        exp1_427197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'exp1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 12), list_427195, exp1_427197)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 27), list_427191, list_427195)
        
        # Processing the call keyword arguments (line 329)
        # Getting the type of 'float' (line 332)
        float_427198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 21), 'float', False)
        keyword_427199 = float_427198
        kwargs_427200 = {'dtype': keyword_427199}
        # Getting the type of 'np' (line 329)
        np_427189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 329)
        array_427190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 18), np_427189, 'array')
        # Calling array(args, kwargs) (line 329)
        array_call_result_427201 = invoke(stypy.reporting.localization.Localization(__file__, 329, 18), array_427190, *[list_427191], **kwargs_427200)
        
        # Assigning a type to the variable 'desired' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'desired', array_call_result_427201)
        
        # Assigning a Call to a Name (line 333):
        
        # Assigning a Call to a Name (line 333):
        
        # Call to expm(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'A' (line 333)
        A_427203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 22), 'A', False)
        # Processing the call keyword arguments (line 333)
        kwargs_427204 = {}
        # Getting the type of 'expm' (line 333)
        expm_427202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 17), 'expm', False)
        # Calling expm(args, kwargs) (line 333)
        expm_call_result_427205 = invoke(stypy.reporting.localization.Localization(__file__, 333, 17), expm_427202, *[A_427203], **kwargs_427204)
        
        # Assigning a type to the variable 'actual' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'actual', expm_call_result_427205)
        
        # Call to assert_allclose(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'actual' (line 334)
        actual_427207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 24), 'actual', False)
        # Getting the type of 'desired' (line 334)
        desired_427208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 32), 'desired', False)
        # Processing the call keyword arguments (line 334)
        kwargs_427209 = {}
        # Getting the type of 'assert_allclose' (line 334)
        assert_allclose_427206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 334)
        assert_allclose_call_result_427210 = invoke(stypy.reporting.localization.Localization(__file__, 334, 8), assert_allclose_427206, *[actual_427207, desired_427208], **kwargs_427209)
        
        
        # ################# End of 'test_burkardt_6(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_6' in the type store
        # Getting the type of 'stypy_return_type' (line 320)
        stypy_return_type_427211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427211)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_6'
        return stypy_return_type_427211


    @norecursion
    def test_burkardt_7(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_7'
        module_type_store = module_type_store.open_function_context('test_burkardt_7', 336, 4, False)
        # Assigning a type to the variable 'self' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_7.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_7.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_7.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_7.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_7')
        TestExpM.test_burkardt_7.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_7.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_7.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_7.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_7.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_7.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_7.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_7', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_7', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_7(...)' code ##################

        
        # Assigning a Call to a Name (line 341):
        
        # Assigning a Call to a Name (line 341):
        
        # Call to exp(...): (line 341)
        # Processing the call arguments (line 341)
        int_427214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 22), 'int')
        # Processing the call keyword arguments (line 341)
        kwargs_427215 = {}
        # Getting the type of 'np' (line 341)
        np_427212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'np', False)
        # Obtaining the member 'exp' of a type (line 341)
        exp_427213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), np_427212, 'exp')
        # Calling exp(args, kwargs) (line 341)
        exp_call_result_427216 = invoke(stypy.reporting.localization.Localization(__file__, 341, 15), exp_427213, *[int_427214], **kwargs_427215)
        
        # Assigning a type to the variable 'exp1' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'exp1', exp_call_result_427216)
        
        # Assigning a Call to a Name (line 342):
        
        # Assigning a Call to a Name (line 342):
        
        # Call to spacing(...): (line 342)
        # Processing the call arguments (line 342)
        int_427219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 25), 'int')
        # Processing the call keyword arguments (line 342)
        kwargs_427220 = {}
        # Getting the type of 'np' (line 342)
        np_427217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 14), 'np', False)
        # Obtaining the member 'spacing' of a type (line 342)
        spacing_427218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 14), np_427217, 'spacing')
        # Calling spacing(args, kwargs) (line 342)
        spacing_call_result_427221 = invoke(stypy.reporting.localization.Localization(__file__, 342, 14), spacing_427218, *[int_427219], **kwargs_427220)
        
        # Assigning a type to the variable 'eps' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'eps', spacing_call_result_427221)
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to array(...): (line 343)
        # Processing the call arguments (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_427224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 344)
        list_427225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 344)
        # Adding element type (line 344)
        int_427226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 13), 'int')
        # Getting the type of 'eps' (line 344)
        eps_427227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 17), 'eps', False)
        # Applying the binary operator '+' (line 344)
        result_add_427228 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 13), '+', int_427226, eps_427227)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 12), list_427225, result_add_427228)
        # Adding element type (line 344)
        int_427229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 12), list_427225, int_427229)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 21), list_427224, list_427225)
        # Adding element type (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 345)
        list_427230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 345)
        # Adding element type (line 345)
        int_427231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 12), list_427230, int_427231)
        # Adding element type (line 345)
        int_427232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 16), 'int')
        # Getting the type of 'eps' (line 345)
        eps_427233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 20), 'eps', False)
        # Applying the binary operator '-' (line 345)
        result_sub_427234 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 16), '-', int_427232, eps_427233)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 12), list_427230, result_sub_427234)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 21), list_427224, list_427230)
        
        # Processing the call keyword arguments (line 343)
        # Getting the type of 'float' (line 346)
        float_427235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 21), 'float', False)
        keyword_427236 = float_427235
        kwargs_427237 = {'dtype': keyword_427236}
        # Getting the type of 'np' (line 343)
        np_427222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 343)
        array_427223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 12), np_427222, 'array')
        # Calling array(args, kwargs) (line 343)
        array_call_result_427238 = invoke(stypy.reporting.localization.Localization(__file__, 343, 12), array_427223, *[list_427224], **kwargs_427237)
        
        # Assigning a type to the variable 'A' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'A', array_call_result_427238)
        
        # Assigning a Call to a Name (line 347):
        
        # Assigning a Call to a Name (line 347):
        
        # Call to array(...): (line 347)
        # Processing the call arguments (line 347)
        
        # Obtaining an instance of the builtin type 'list' (line 347)
        list_427241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 347)
        # Adding element type (line 347)
        
        # Obtaining an instance of the builtin type 'list' (line 348)
        list_427242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 348)
        # Adding element type (line 348)
        # Getting the type of 'exp1' (line 348)
        exp1_427243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 13), 'exp1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 12), list_427242, exp1_427243)
        # Adding element type (line 348)
        # Getting the type of 'exp1' (line 348)
        exp1_427244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 19), 'exp1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 12), list_427242, exp1_427244)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 27), list_427241, list_427242)
        # Adding element type (line 347)
        
        # Obtaining an instance of the builtin type 'list' (line 349)
        list_427245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 349)
        # Adding element type (line 349)
        int_427246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 12), list_427245, int_427246)
        # Adding element type (line 349)
        # Getting the type of 'exp1' (line 349)
        exp1_427247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 16), 'exp1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 12), list_427245, exp1_427247)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 27), list_427241, list_427245)
        
        # Processing the call keyword arguments (line 347)
        # Getting the type of 'float' (line 350)
        float_427248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 21), 'float', False)
        keyword_427249 = float_427248
        kwargs_427250 = {'dtype': keyword_427249}
        # Getting the type of 'np' (line 347)
        np_427239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 347)
        array_427240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 18), np_427239, 'array')
        # Calling array(args, kwargs) (line 347)
        array_call_result_427251 = invoke(stypy.reporting.localization.Localization(__file__, 347, 18), array_427240, *[list_427241], **kwargs_427250)
        
        # Assigning a type to the variable 'desired' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'desired', array_call_result_427251)
        
        # Assigning a Call to a Name (line 351):
        
        # Assigning a Call to a Name (line 351):
        
        # Call to expm(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'A' (line 351)
        A_427253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 22), 'A', False)
        # Processing the call keyword arguments (line 351)
        kwargs_427254 = {}
        # Getting the type of 'expm' (line 351)
        expm_427252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 17), 'expm', False)
        # Calling expm(args, kwargs) (line 351)
        expm_call_result_427255 = invoke(stypy.reporting.localization.Localization(__file__, 351, 17), expm_427252, *[A_427253], **kwargs_427254)
        
        # Assigning a type to the variable 'actual' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'actual', expm_call_result_427255)
        
        # Call to assert_allclose(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'actual' (line 352)
        actual_427257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 24), 'actual', False)
        # Getting the type of 'desired' (line 352)
        desired_427258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 32), 'desired', False)
        # Processing the call keyword arguments (line 352)
        kwargs_427259 = {}
        # Getting the type of 'assert_allclose' (line 352)
        assert_allclose_427256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 352)
        assert_allclose_call_result_427260 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), assert_allclose_427256, *[actual_427257, desired_427258], **kwargs_427259)
        
        
        # ################# End of 'test_burkardt_7(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_7' in the type store
        # Getting the type of 'stypy_return_type' (line 336)
        stypy_return_type_427261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_7'
        return stypy_return_type_427261


    @norecursion
    def test_burkardt_8(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_8'
        module_type_store = module_type_store.open_function_context('test_burkardt_8', 354, 4, False)
        # Assigning a type to the variable 'self' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_8.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_8.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_8.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_8.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_8')
        TestExpM.test_burkardt_8.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_8.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_8.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_8.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_8.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_8.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_8.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_8', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_8', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_8(...)' code ##################

        
        # Assigning a Call to a Name (line 356):
        
        # Assigning a Call to a Name (line 356):
        
        # Call to exp(...): (line 356)
        # Processing the call arguments (line 356)
        int_427264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 22), 'int')
        # Processing the call keyword arguments (line 356)
        kwargs_427265 = {}
        # Getting the type of 'np' (line 356)
        np_427262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 15), 'np', False)
        # Obtaining the member 'exp' of a type (line 356)
        exp_427263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 15), np_427262, 'exp')
        # Calling exp(args, kwargs) (line 356)
        exp_call_result_427266 = invoke(stypy.reporting.localization.Localization(__file__, 356, 15), exp_427263, *[int_427264], **kwargs_427265)
        
        # Assigning a type to the variable 'exp4' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'exp4', exp_call_result_427266)
        
        # Assigning a Call to a Name (line 357):
        
        # Assigning a Call to a Name (line 357):
        
        # Call to exp(...): (line 357)
        # Processing the call arguments (line 357)
        int_427269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 23), 'int')
        # Processing the call keyword arguments (line 357)
        kwargs_427270 = {}
        # Getting the type of 'np' (line 357)
        np_427267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 16), 'np', False)
        # Obtaining the member 'exp' of a type (line 357)
        exp_427268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 16), np_427267, 'exp')
        # Calling exp(args, kwargs) (line 357)
        exp_call_result_427271 = invoke(stypy.reporting.localization.Localization(__file__, 357, 16), exp_427268, *[int_427269], **kwargs_427270)
        
        # Assigning a type to the variable 'exp16' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'exp16', exp_call_result_427271)
        
        # Assigning a Call to a Name (line 358):
        
        # Assigning a Call to a Name (line 358):
        
        # Call to array(...): (line 358)
        # Processing the call arguments (line 358)
        
        # Obtaining an instance of the builtin type 'list' (line 358)
        list_427274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 358)
        # Adding element type (line 358)
        
        # Obtaining an instance of the builtin type 'list' (line 359)
        list_427275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 359)
        # Adding element type (line 359)
        int_427276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 12), list_427275, int_427276)
        # Adding element type (line 359)
        int_427277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 12), list_427275, int_427277)
        # Adding element type (line 359)
        int_427278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 12), list_427275, int_427278)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 21), list_427274, list_427275)
        # Adding element type (line 358)
        
        # Obtaining an instance of the builtin type 'list' (line 360)
        list_427279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 360)
        # Adding element type (line 360)
        int_427280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 12), list_427279, int_427280)
        # Adding element type (line 360)
        int_427281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 12), list_427279, int_427281)
        # Adding element type (line 360)
        int_427282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 12), list_427279, int_427282)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 21), list_427274, list_427279)
        # Adding element type (line 358)
        
        # Obtaining an instance of the builtin type 'list' (line 361)
        list_427283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 361)
        # Adding element type (line 361)
        int_427284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 12), list_427283, int_427284)
        # Adding element type (line 361)
        int_427285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 12), list_427283, int_427285)
        # Adding element type (line 361)
        int_427286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 12), list_427283, int_427286)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 21), list_427274, list_427283)
        
        # Processing the call keyword arguments (line 358)
        # Getting the type of 'float' (line 362)
        float_427287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 21), 'float', False)
        keyword_427288 = float_427287
        kwargs_427289 = {'dtype': keyword_427288}
        # Getting the type of 'np' (line 358)
        np_427272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 358)
        array_427273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), np_427272, 'array')
        # Calling array(args, kwargs) (line 358)
        array_call_result_427290 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), array_427273, *[list_427274], **kwargs_427289)
        
        # Assigning a type to the variable 'A' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'A', array_call_result_427290)
        
        # Assigning a BinOp to a Name (line 363):
        
        # Assigning a BinOp to a Name (line 363):
        
        # Call to array(...): (line 363)
        # Processing the call arguments (line 363)
        
        # Obtaining an instance of the builtin type 'list' (line 363)
        list_427293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 363)
        # Adding element type (line 363)
        
        # Obtaining an instance of the builtin type 'list' (line 364)
        list_427294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 364)
        # Adding element type (line 364)
        int_427295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 13), 'int')
        # Getting the type of 'exp16' (line 364)
        exp16_427296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'exp16', False)
        # Applying the binary operator '*' (line 364)
        result_mul_427297 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 13), '*', int_427295, exp16_427296)
        
        # Getting the type of 'exp4' (line 364)
        exp4_427298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 24), 'exp4', False)
        # Applying the binary operator '-' (line 364)
        result_sub_427299 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 13), '-', result_mul_427297, exp4_427298)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 12), list_427294, result_sub_427299)
        # Adding element type (line 364)
        int_427300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 30), 'int')
        # Getting the type of 'exp16' (line 364)
        exp16_427301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 33), 'exp16', False)
        # Applying the binary operator '*' (line 364)
        result_mul_427302 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 30), '*', int_427300, exp16_427301)
        
        int_427303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 41), 'int')
        # Getting the type of 'exp4' (line 364)
        exp4_427304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 43), 'exp4', False)
        # Applying the binary operator '*' (line 364)
        result_mul_427305 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 41), '*', int_427303, exp4_427304)
        
        # Applying the binary operator '-' (line 364)
        result_sub_427306 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 30), '-', result_mul_427302, result_mul_427305)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 12), list_427294, result_sub_427306)
        # Adding element type (line 364)
        int_427307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 49), 'int')
        # Getting the type of 'exp16' (line 364)
        exp16_427308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 51), 'exp16', False)
        # Applying the binary operator '*' (line 364)
        result_mul_427309 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 49), '*', int_427307, exp16_427308)
        
        int_427310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 59), 'int')
        # Getting the type of 'exp4' (line 364)
        exp4_427311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 61), 'exp4', False)
        # Applying the binary operator '*' (line 364)
        result_mul_427312 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 59), '*', int_427310, exp4_427311)
        
        # Applying the binary operator '-' (line 364)
        result_sub_427313 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 49), '-', result_mul_427309, result_mul_427312)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 12), list_427294, result_sub_427313)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 27), list_427293, list_427294)
        # Adding element type (line 363)
        
        # Obtaining an instance of the builtin type 'list' (line 365)
        list_427314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 365)
        # Adding element type (line 365)
        int_427315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 13), 'int')
        # Getting the type of 'exp16' (line 365)
        exp16_427316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 16), 'exp16', False)
        # Applying the binary operator '*' (line 365)
        result_mul_427317 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 13), '*', int_427315, exp16_427316)
        
        # Getting the type of 'exp4' (line 365)
        exp4_427318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 24), 'exp4', False)
        # Applying the binary operator '+' (line 365)
        result_add_427319 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 13), '+', result_mul_427317, exp4_427318)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 12), list_427314, result_add_427319)
        # Adding element type (line 365)
        int_427320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 30), 'int')
        # Getting the type of 'exp16' (line 365)
        exp16_427321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 33), 'exp16', False)
        # Applying the binary operator '*' (line 365)
        result_mul_427322 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 30), '*', int_427320, exp16_427321)
        
        int_427323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 41), 'int')
        # Getting the type of 'exp4' (line 365)
        exp4_427324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 43), 'exp4', False)
        # Applying the binary operator '*' (line 365)
        result_mul_427325 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 41), '*', int_427323, exp4_427324)
        
        # Applying the binary operator '+' (line 365)
        result_add_427326 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 30), '+', result_mul_427322, result_mul_427325)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 12), list_427314, result_add_427326)
        # Adding element type (line 365)
        int_427327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 49), 'int')
        # Getting the type of 'exp16' (line 365)
        exp16_427328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 52), 'exp16', False)
        # Applying the binary operator '*' (line 365)
        result_mul_427329 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 49), '*', int_427327, exp16_427328)
        
        int_427330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 60), 'int')
        # Getting the type of 'exp4' (line 365)
        exp4_427331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 62), 'exp4', False)
        # Applying the binary operator '*' (line 365)
        result_mul_427332 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 60), '*', int_427330, exp4_427331)
        
        # Applying the binary operator '+' (line 365)
        result_add_427333 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 49), '+', result_mul_427329, result_mul_427332)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 12), list_427314, result_add_427333)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 27), list_427293, list_427314)
        # Adding element type (line 363)
        
        # Obtaining an instance of the builtin type 'list' (line 366)
        list_427334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 366)
        # Adding element type (line 366)
        int_427335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 13), 'int')
        # Getting the type of 'exp16' (line 366)
        exp16_427336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 'exp16', False)
        # Applying the binary operator '*' (line 366)
        result_mul_427337 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 13), '*', int_427335, exp16_427336)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 12), list_427334, result_mul_427337)
        # Adding element type (line 366)
        int_427338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 23), 'int')
        # Getting the type of 'exp16' (line 366)
        exp16_427339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 26), 'exp16', False)
        # Applying the binary operator '*' (line 366)
        result_mul_427340 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 23), '*', int_427338, exp16_427339)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 12), list_427334, result_mul_427340)
        # Adding element type (line 366)
        int_427341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 33), 'int')
        # Getting the type of 'exp16' (line 366)
        exp16_427342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 35), 'exp16', False)
        # Applying the binary operator '*' (line 366)
        result_mul_427343 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 33), '*', int_427341, exp16_427342)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 12), list_427334, result_mul_427343)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 27), list_427293, list_427334)
        
        # Processing the call keyword arguments (line 363)
        # Getting the type of 'float' (line 367)
        float_427344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 21), 'float', False)
        keyword_427345 = float_427344
        kwargs_427346 = {'dtype': keyword_427345}
        # Getting the type of 'np' (line 363)
        np_427291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 363)
        array_427292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 18), np_427291, 'array')
        # Calling array(args, kwargs) (line 363)
        array_call_result_427347 = invoke(stypy.reporting.localization.Localization(__file__, 363, 18), array_427292, *[list_427293], **kwargs_427346)
        
        float_427348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 30), 'float')
        # Applying the binary operator '*' (line 363)
        result_mul_427349 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 18), '*', array_call_result_427347, float_427348)
        
        # Assigning a type to the variable 'desired' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'desired', result_mul_427349)
        
        # Assigning a Call to a Name (line 368):
        
        # Assigning a Call to a Name (line 368):
        
        # Call to expm(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'A' (line 368)
        A_427351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 22), 'A', False)
        # Processing the call keyword arguments (line 368)
        kwargs_427352 = {}
        # Getting the type of 'expm' (line 368)
        expm_427350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 17), 'expm', False)
        # Calling expm(args, kwargs) (line 368)
        expm_call_result_427353 = invoke(stypy.reporting.localization.Localization(__file__, 368, 17), expm_427350, *[A_427351], **kwargs_427352)
        
        # Assigning a type to the variable 'actual' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'actual', expm_call_result_427353)
        
        # Call to assert_allclose(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'actual' (line 369)
        actual_427355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 24), 'actual', False)
        # Getting the type of 'desired' (line 369)
        desired_427356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 32), 'desired', False)
        # Processing the call keyword arguments (line 369)
        kwargs_427357 = {}
        # Getting the type of 'assert_allclose' (line 369)
        assert_allclose_427354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 369)
        assert_allclose_call_result_427358 = invoke(stypy.reporting.localization.Localization(__file__, 369, 8), assert_allclose_427354, *[actual_427355, desired_427356], **kwargs_427357)
        
        
        # ################# End of 'test_burkardt_8(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_8' in the type store
        # Getting the type of 'stypy_return_type' (line 354)
        stypy_return_type_427359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427359)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_8'
        return stypy_return_type_427359


    @norecursion
    def test_burkardt_9(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_9'
        module_type_store = module_type_store.open_function_context('test_burkardt_9', 371, 4, False)
        # Assigning a type to the variable 'self' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_9.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_9.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_9.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_9.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_9')
        TestExpM.test_burkardt_9.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_9.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_9.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_9.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_9.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_9.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_9.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_9', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_9', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_9(...)' code ##################

        
        # Assigning a Call to a Name (line 374):
        
        # Assigning a Call to a Name (line 374):
        
        # Call to array(...): (line 374)
        # Processing the call arguments (line 374)
        
        # Obtaining an instance of the builtin type 'list' (line 374)
        list_427362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 374)
        # Adding element type (line 374)
        
        # Obtaining an instance of the builtin type 'list' (line 375)
        list_427363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 375)
        # Adding element type (line 375)
        int_427364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 12), list_427363, int_427364)
        # Adding element type (line 375)
        int_427365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 12), list_427363, int_427365)
        # Adding element type (line 375)
        int_427366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 12), list_427363, int_427366)
        # Adding element type (line 375)
        int_427367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 12), list_427363, int_427367)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 21), list_427362, list_427363)
        # Adding element type (line 374)
        
        # Obtaining an instance of the builtin type 'list' (line 376)
        list_427368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 376)
        # Adding element type (line 376)
        int_427369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 12), list_427368, int_427369)
        # Adding element type (line 376)
        int_427370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 12), list_427368, int_427370)
        # Adding element type (line 376)
        int_427371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 12), list_427368, int_427371)
        # Adding element type (line 376)
        int_427372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 12), list_427368, int_427372)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 21), list_427362, list_427368)
        # Adding element type (line 374)
        
        # Obtaining an instance of the builtin type 'list' (line 377)
        list_427373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 377)
        # Adding element type (line 377)
        int_427374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 12), list_427373, int_427374)
        # Adding element type (line 377)
        int_427375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 12), list_427373, int_427375)
        # Adding element type (line 377)
        int_427376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 12), list_427373, int_427376)
        # Adding element type (line 377)
        int_427377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 12), list_427373, int_427377)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 21), list_427362, list_427373)
        # Adding element type (line 374)
        
        # Obtaining an instance of the builtin type 'list' (line 378)
        list_427378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 378)
        # Adding element type (line 378)
        int_427379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 12), list_427378, int_427379)
        # Adding element type (line 378)
        int_427380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 12), list_427378, int_427380)
        # Adding element type (line 378)
        int_427381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 12), list_427378, int_427381)
        # Adding element type (line 378)
        int_427382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 12), list_427378, int_427382)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 21), list_427362, list_427378)
        
        # Processing the call keyword arguments (line 374)
        # Getting the type of 'float' (line 379)
        float_427383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 21), 'float', False)
        keyword_427384 = float_427383
        kwargs_427385 = {'dtype': keyword_427384}
        # Getting the type of 'np' (line 374)
        np_427360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 374)
        array_427361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), np_427360, 'array')
        # Calling array(args, kwargs) (line 374)
        array_call_result_427386 = invoke(stypy.reporting.localization.Localization(__file__, 374, 12), array_427361, *[list_427362], **kwargs_427385)
        
        # Assigning a type to the variable 'A' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'A', array_call_result_427386)
        
        # Assigning a Call to a Name (line 380):
        
        # Assigning a Call to a Name (line 380):
        
        # Call to array(...): (line 380)
        # Processing the call arguments (line 380)
        
        # Obtaining an instance of the builtin type 'list' (line 380)
        list_427389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 380)
        # Adding element type (line 380)
        
        # Obtaining an instance of the builtin type 'list' (line 381)
        list_427390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 381)
        # Adding element type (line 381)
        float_427391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 12), list_427390, float_427391)
        # Adding element type (line 381)
        float_427392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 12), list_427390, float_427392)
        # Adding element type (line 381)
        float_427393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 12), list_427390, float_427393)
        # Adding element type (line 381)
        float_427394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 12), list_427390, float_427394)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 27), list_427389, list_427390)
        # Adding element type (line 380)
        
        # Obtaining an instance of the builtin type 'list' (line 382)
        list_427395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 382)
        # Adding element type (line 382)
        float_427396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_427395, float_427396)
        # Adding element type (line 382)
        float_427397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_427395, float_427397)
        # Adding element type (line 382)
        float_427398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_427395, float_427398)
        # Adding element type (line 382)
        float_427399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_427395, float_427399)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 27), list_427389, list_427395)
        # Adding element type (line 380)
        
        # Obtaining an instance of the builtin type 'list' (line 383)
        list_427400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 383)
        # Adding element type (line 383)
        float_427401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 12), list_427400, float_427401)
        # Adding element type (line 383)
        float_427402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 12), list_427400, float_427402)
        # Adding element type (line 383)
        float_427403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 12), list_427400, float_427403)
        # Adding element type (line 383)
        float_427404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 12), list_427400, float_427404)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 27), list_427389, list_427400)
        # Adding element type (line 380)
        
        # Obtaining an instance of the builtin type 'list' (line 384)
        list_427405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 384)
        # Adding element type (line 384)
        float_427406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 12), list_427405, float_427406)
        # Adding element type (line 384)
        float_427407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 12), list_427405, float_427407)
        # Adding element type (line 384)
        float_427408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 12), list_427405, float_427408)
        # Adding element type (line 384)
        float_427409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 12), list_427405, float_427409)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 27), list_427389, list_427405)
        
        # Processing the call keyword arguments (line 380)
        # Getting the type of 'float' (line 385)
        float_427410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 21), 'float', False)
        keyword_427411 = float_427410
        kwargs_427412 = {'dtype': keyword_427411}
        # Getting the type of 'np' (line 380)
        np_427387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 380)
        array_427388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 18), np_427387, 'array')
        # Calling array(args, kwargs) (line 380)
        array_call_result_427413 = invoke(stypy.reporting.localization.Localization(__file__, 380, 18), array_427388, *[list_427389], **kwargs_427412)
        
        # Assigning a type to the variable 'desired' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'desired', array_call_result_427413)
        
        # Assigning a Call to a Name (line 386):
        
        # Assigning a Call to a Name (line 386):
        
        # Call to expm(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'A' (line 386)
        A_427415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 22), 'A', False)
        # Processing the call keyword arguments (line 386)
        kwargs_427416 = {}
        # Getting the type of 'expm' (line 386)
        expm_427414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 17), 'expm', False)
        # Calling expm(args, kwargs) (line 386)
        expm_call_result_427417 = invoke(stypy.reporting.localization.Localization(__file__, 386, 17), expm_427414, *[A_427415], **kwargs_427416)
        
        # Assigning a type to the variable 'actual' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'actual', expm_call_result_427417)
        
        # Call to assert_allclose(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'actual' (line 387)
        actual_427419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 24), 'actual', False)
        # Getting the type of 'desired' (line 387)
        desired_427420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 32), 'desired', False)
        # Processing the call keyword arguments (line 387)
        kwargs_427421 = {}
        # Getting the type of 'assert_allclose' (line 387)
        assert_allclose_427418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 387)
        assert_allclose_call_result_427422 = invoke(stypy.reporting.localization.Localization(__file__, 387, 8), assert_allclose_427418, *[actual_427419, desired_427420], **kwargs_427421)
        
        
        # ################# End of 'test_burkardt_9(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_9' in the type store
        # Getting the type of 'stypy_return_type' (line 371)
        stypy_return_type_427423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427423)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_9'
        return stypy_return_type_427423


    @norecursion
    def test_burkardt_10(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_10'
        module_type_store = module_type_store.open_function_context('test_burkardt_10', 389, 4, False)
        # Assigning a type to the variable 'self' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_10.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_10.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_10.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_10.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_10')
        TestExpM.test_burkardt_10.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_10.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_10.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_10.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_10.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_10.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_10.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_10', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_10', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_10(...)' code ##################

        
        # Assigning a Call to a Name (line 392):
        
        # Assigning a Call to a Name (line 392):
        
        # Call to array(...): (line 392)
        # Processing the call arguments (line 392)
        
        # Obtaining an instance of the builtin type 'list' (line 392)
        list_427426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 392)
        # Adding element type (line 392)
        
        # Obtaining an instance of the builtin type 'list' (line 393)
        list_427427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 393)
        # Adding element type (line 393)
        int_427428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 12), list_427427, int_427428)
        # Adding element type (line 393)
        int_427429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 12), list_427427, int_427429)
        # Adding element type (line 393)
        int_427430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 12), list_427427, int_427430)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 21), list_427426, list_427427)
        # Adding element type (line 392)
        
        # Obtaining an instance of the builtin type 'list' (line 394)
        list_427431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 394)
        # Adding element type (line 394)
        int_427432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 12), list_427431, int_427432)
        # Adding element type (line 394)
        int_427433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 12), list_427431, int_427433)
        # Adding element type (line 394)
        int_427434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 12), list_427431, int_427434)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 21), list_427426, list_427431)
        # Adding element type (line 392)
        
        # Obtaining an instance of the builtin type 'list' (line 395)
        list_427435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 395)
        # Adding element type (line 395)
        int_427436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_427435, int_427436)
        # Adding element type (line 395)
        int_427437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_427435, int_427437)
        # Adding element type (line 395)
        int_427438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_427435, int_427438)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 21), list_427426, list_427435)
        
        # Processing the call keyword arguments (line 392)
        # Getting the type of 'float' (line 396)
        float_427439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 21), 'float', False)
        keyword_427440 = float_427439
        kwargs_427441 = {'dtype': keyword_427440}
        # Getting the type of 'np' (line 392)
        np_427424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 392)
        array_427425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 12), np_427424, 'array')
        # Calling array(args, kwargs) (line 392)
        array_call_result_427442 = invoke(stypy.reporting.localization.Localization(__file__, 392, 12), array_427425, *[list_427426], **kwargs_427441)
        
        # Assigning a type to the variable 'A' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'A', array_call_result_427442)
        
        # Call to assert_allclose(...): (line 397)
        # Processing the call arguments (line 397)
        
        # Call to sorted(...): (line 397)
        # Processing the call arguments (line 397)
        
        # Call to eigvals(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'A' (line 397)
        A_427448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 52), 'A', False)
        # Processing the call keyword arguments (line 397)
        kwargs_427449 = {}
        # Getting the type of 'scipy' (line 397)
        scipy_427445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 31), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 397)
        linalg_427446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 31), scipy_427445, 'linalg')
        # Obtaining the member 'eigvals' of a type (line 397)
        eigvals_427447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 31), linalg_427446, 'eigvals')
        # Calling eigvals(args, kwargs) (line 397)
        eigvals_call_result_427450 = invoke(stypy.reporting.localization.Localization(__file__, 397, 31), eigvals_427447, *[A_427448], **kwargs_427449)
        
        # Processing the call keyword arguments (line 397)
        kwargs_427451 = {}
        # Getting the type of 'sorted' (line 397)
        sorted_427444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 24), 'sorted', False)
        # Calling sorted(args, kwargs) (line 397)
        sorted_call_result_427452 = invoke(stypy.reporting.localization.Localization(__file__, 397, 24), sorted_427444, *[eigvals_call_result_427450], **kwargs_427451)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 397)
        tuple_427453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 397)
        # Adding element type (line 397)
        int_427454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 58), tuple_427453, int_427454)
        # Adding element type (line 397)
        int_427455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 58), tuple_427453, int_427455)
        # Adding element type (line 397)
        int_427456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 58), tuple_427453, int_427456)
        
        # Processing the call keyword arguments (line 397)
        kwargs_427457 = {}
        # Getting the type of 'assert_allclose' (line 397)
        assert_allclose_427443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 397)
        assert_allclose_call_result_427458 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), assert_allclose_427443, *[sorted_call_result_427452, tuple_427453], **kwargs_427457)
        
        
        # Assigning a Call to a Name (line 398):
        
        # Assigning a Call to a Name (line 398):
        
        # Call to array(...): (line 398)
        # Processing the call arguments (line 398)
        
        # Obtaining an instance of the builtin type 'list' (line 398)
        list_427461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 398)
        # Adding element type (line 398)
        
        # Obtaining an instance of the builtin type 'list' (line 399)
        list_427462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 399)
        # Adding element type (line 399)
        float_427463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 12), list_427462, float_427463)
        # Adding element type (line 399)
        float_427464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 12), list_427462, float_427464)
        # Adding element type (line 399)
        float_427465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 12), list_427462, float_427465)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 27), list_427461, list_427462)
        # Adding element type (line 398)
        
        # Obtaining an instance of the builtin type 'list' (line 400)
        list_427466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 400)
        # Adding element type (line 400)
        float_427467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 12), list_427466, float_427467)
        # Adding element type (line 400)
        float_427468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 12), list_427466, float_427468)
        # Adding element type (line 400)
        float_427469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 12), list_427466, float_427469)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 27), list_427461, list_427466)
        # Adding element type (line 398)
        
        # Obtaining an instance of the builtin type 'list' (line 401)
        list_427470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 401)
        # Adding element type (line 401)
        float_427471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_427470, float_427471)
        # Adding element type (line 401)
        float_427472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_427470, float_427472)
        # Adding element type (line 401)
        float_427473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_427470, float_427473)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 27), list_427461, list_427470)
        
        # Processing the call keyword arguments (line 398)
        # Getting the type of 'float' (line 402)
        float_427474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 21), 'float', False)
        keyword_427475 = float_427474
        kwargs_427476 = {'dtype': keyword_427475}
        # Getting the type of 'np' (line 398)
        np_427459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 398)
        array_427460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 18), np_427459, 'array')
        # Calling array(args, kwargs) (line 398)
        array_call_result_427477 = invoke(stypy.reporting.localization.Localization(__file__, 398, 18), array_427460, *[list_427461], **kwargs_427476)
        
        # Assigning a type to the variable 'desired' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'desired', array_call_result_427477)
        
        # Assigning a Call to a Name (line 403):
        
        # Assigning a Call to a Name (line 403):
        
        # Call to expm(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'A' (line 403)
        A_427479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 22), 'A', False)
        # Processing the call keyword arguments (line 403)
        kwargs_427480 = {}
        # Getting the type of 'expm' (line 403)
        expm_427478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 17), 'expm', False)
        # Calling expm(args, kwargs) (line 403)
        expm_call_result_427481 = invoke(stypy.reporting.localization.Localization(__file__, 403, 17), expm_427478, *[A_427479], **kwargs_427480)
        
        # Assigning a type to the variable 'actual' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'actual', expm_call_result_427481)
        
        # Call to assert_allclose(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'actual' (line 404)
        actual_427483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 24), 'actual', False)
        # Getting the type of 'desired' (line 404)
        desired_427484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 32), 'desired', False)
        # Processing the call keyword arguments (line 404)
        kwargs_427485 = {}
        # Getting the type of 'assert_allclose' (line 404)
        assert_allclose_427482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 404)
        assert_allclose_call_result_427486 = invoke(stypy.reporting.localization.Localization(__file__, 404, 8), assert_allclose_427482, *[actual_427483, desired_427484], **kwargs_427485)
        
        
        # ################# End of 'test_burkardt_10(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_10' in the type store
        # Getting the type of 'stypy_return_type' (line 389)
        stypy_return_type_427487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427487)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_10'
        return stypy_return_type_427487


    @norecursion
    def test_burkardt_11(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_11'
        module_type_store = module_type_store.open_function_context('test_burkardt_11', 406, 4, False)
        # Assigning a type to the variable 'self' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_11.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_11.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_11.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_11.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_11')
        TestExpM.test_burkardt_11.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_11.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_11.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_11.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_11.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_11.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_11.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_11', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_11', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_11(...)' code ##################

        
        # Assigning a Call to a Name (line 409):
        
        # Assigning a Call to a Name (line 409):
        
        # Call to array(...): (line 409)
        # Processing the call arguments (line 409)
        
        # Obtaining an instance of the builtin type 'list' (line 409)
        list_427490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 409)
        # Adding element type (line 409)
        
        # Obtaining an instance of the builtin type 'list' (line 410)
        list_427491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 410)
        # Adding element type (line 410)
        float_427492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 12), list_427491, float_427492)
        # Adding element type (line 410)
        float_427493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 12), list_427491, float_427493)
        # Adding element type (line 410)
        float_427494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 12), list_427491, float_427494)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 21), list_427490, list_427491)
        # Adding element type (line 409)
        
        # Obtaining an instance of the builtin type 'list' (line 411)
        list_427495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 411)
        # Adding element type (line 411)
        float_427496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 12), list_427495, float_427496)
        # Adding element type (line 411)
        float_427497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 12), list_427495, float_427497)
        # Adding element type (line 411)
        float_427498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 12), list_427495, float_427498)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 21), list_427490, list_427495)
        # Adding element type (line 409)
        
        # Obtaining an instance of the builtin type 'list' (line 412)
        list_427499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 412)
        # Adding element type (line 412)
        float_427500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_427499, float_427500)
        # Adding element type (line 412)
        float_427501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_427499, float_427501)
        # Adding element type (line 412)
        float_427502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_427499, float_427502)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 21), list_427490, list_427499)
        
        # Processing the call keyword arguments (line 409)
        # Getting the type of 'float' (line 413)
        float_427503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 21), 'float', False)
        keyword_427504 = float_427503
        kwargs_427505 = {'dtype': keyword_427504}
        # Getting the type of 'np' (line 409)
        np_427488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 409)
        array_427489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 12), np_427488, 'array')
        # Calling array(args, kwargs) (line 409)
        array_call_result_427506 = invoke(stypy.reporting.localization.Localization(__file__, 409, 12), array_427489, *[list_427490], **kwargs_427505)
        
        # Assigning a type to the variable 'A' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'A', array_call_result_427506)
        
        # Call to assert_allclose(...): (line 414)
        # Processing the call arguments (line 414)
        
        # Call to eigvalsh(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'A' (line 414)
        A_427511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 46), 'A', False)
        # Processing the call keyword arguments (line 414)
        kwargs_427512 = {}
        # Getting the type of 'scipy' (line 414)
        scipy_427508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 24), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 414)
        linalg_427509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 24), scipy_427508, 'linalg')
        # Obtaining the member 'eigvalsh' of a type (line 414)
        eigvalsh_427510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 24), linalg_427509, 'eigvalsh')
        # Calling eigvalsh(args, kwargs) (line 414)
        eigvalsh_call_result_427513 = invoke(stypy.reporting.localization.Localization(__file__, 414, 24), eigvalsh_427510, *[A_427511], **kwargs_427512)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 414)
        tuple_427514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 414)
        # Adding element type (line 414)
        int_427515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 51), tuple_427514, int_427515)
        # Adding element type (line 414)
        int_427516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 51), tuple_427514, int_427516)
        # Adding element type (line 414)
        int_427517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 51), tuple_427514, int_427517)
        
        # Processing the call keyword arguments (line 414)
        kwargs_427518 = {}
        # Getting the type of 'assert_allclose' (line 414)
        assert_allclose_427507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 414)
        assert_allclose_call_result_427519 = invoke(stypy.reporting.localization.Localization(__file__, 414, 8), assert_allclose_427507, *[eigvalsh_call_result_427513, tuple_427514], **kwargs_427518)
        
        
        # Assigning a Call to a Name (line 415):
        
        # Assigning a Call to a Name (line 415):
        
        # Call to array(...): (line 415)
        # Processing the call arguments (line 415)
        
        # Obtaining an instance of the builtin type 'list' (line 415)
        list_427522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 415)
        # Adding element type (line 415)
        
        # Obtaining an instance of the builtin type 'list' (line 416)
        list_427523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 416)
        # Adding element type (line 416)
        float_427524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 13), list_427523, float_427524)
        # Adding element type (line 416)
        float_427525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 13), list_427523, float_427525)
        # Adding element type (line 416)
        float_427526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 13), list_427523, float_427526)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 27), list_427522, list_427523)
        # Adding element type (line 415)
        
        # Obtaining an instance of the builtin type 'list' (line 420)
        list_427527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 420)
        # Adding element type (line 420)
        float_427528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 13), list_427527, float_427528)
        # Adding element type (line 420)
        float_427529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 13), list_427527, float_427529)
        # Adding element type (line 420)
        float_427530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 13), list_427527, float_427530)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 27), list_427522, list_427527)
        # Adding element type (line 415)
        
        # Obtaining an instance of the builtin type 'list' (line 424)
        list_427531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 424)
        # Adding element type (line 424)
        float_427532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 13), list_427531, float_427532)
        # Adding element type (line 424)
        float_427533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 13), list_427531, float_427533)
        # Adding element type (line 424)
        float_427534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 13), list_427531, float_427534)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 27), list_427522, list_427531)
        
        # Processing the call keyword arguments (line 415)
        # Getting the type of 'float' (line 428)
        float_427535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 21), 'float', False)
        keyword_427536 = float_427535
        kwargs_427537 = {'dtype': keyword_427536}
        # Getting the type of 'np' (line 415)
        np_427520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 415)
        array_427521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 18), np_427520, 'array')
        # Calling array(args, kwargs) (line 415)
        array_call_result_427538 = invoke(stypy.reporting.localization.Localization(__file__, 415, 18), array_427521, *[list_427522], **kwargs_427537)
        
        # Assigning a type to the variable 'desired' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'desired', array_call_result_427538)
        
        # Assigning a Call to a Name (line 429):
        
        # Assigning a Call to a Name (line 429):
        
        # Call to expm(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'A' (line 429)
        A_427540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 22), 'A', False)
        # Processing the call keyword arguments (line 429)
        kwargs_427541 = {}
        # Getting the type of 'expm' (line 429)
        expm_427539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 17), 'expm', False)
        # Calling expm(args, kwargs) (line 429)
        expm_call_result_427542 = invoke(stypy.reporting.localization.Localization(__file__, 429, 17), expm_427539, *[A_427540], **kwargs_427541)
        
        # Assigning a type to the variable 'actual' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'actual', expm_call_result_427542)
        
        # Call to assert_allclose(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'actual' (line 430)
        actual_427544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 24), 'actual', False)
        # Getting the type of 'desired' (line 430)
        desired_427545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 32), 'desired', False)
        # Processing the call keyword arguments (line 430)
        kwargs_427546 = {}
        # Getting the type of 'assert_allclose' (line 430)
        assert_allclose_427543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 430)
        assert_allclose_call_result_427547 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), assert_allclose_427543, *[actual_427544, desired_427545], **kwargs_427546)
        
        
        # ################# End of 'test_burkardt_11(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_11' in the type store
        # Getting the type of 'stypy_return_type' (line 406)
        stypy_return_type_427548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427548)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_11'
        return stypy_return_type_427548


    @norecursion
    def test_burkardt_12(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_12'
        module_type_store = module_type_store.open_function_context('test_burkardt_12', 432, 4, False)
        # Assigning a type to the variable 'self' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_12.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_12.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_12.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_12.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_12')
        TestExpM.test_burkardt_12.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_12.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_12.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_12.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_12.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_12.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_12.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_12', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_12', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_12(...)' code ##################

        
        # Assigning a Call to a Name (line 436):
        
        # Assigning a Call to a Name (line 436):
        
        # Call to array(...): (line 436)
        # Processing the call arguments (line 436)
        
        # Obtaining an instance of the builtin type 'list' (line 436)
        list_427551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 436)
        # Adding element type (line 436)
        
        # Obtaining an instance of the builtin type 'list' (line 437)
        list_427552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 437)
        # Adding element type (line 437)
        int_427553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 12), list_427552, int_427553)
        # Adding element type (line 437)
        int_427554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 12), list_427552, int_427554)
        # Adding element type (line 437)
        int_427555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 12), list_427552, int_427555)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 21), list_427551, list_427552)
        # Adding element type (line 436)
        
        # Obtaining an instance of the builtin type 'list' (line 438)
        list_427556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 438)
        # Adding element type (line 438)
        int_427557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 12), list_427556, int_427557)
        # Adding element type (line 438)
        int_427558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 12), list_427556, int_427558)
        # Adding element type (line 438)
        int_427559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 12), list_427556, int_427559)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 21), list_427551, list_427556)
        # Adding element type (line 436)
        
        # Obtaining an instance of the builtin type 'list' (line 439)
        list_427560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 439)
        # Adding element type (line 439)
        int_427561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 12), list_427560, int_427561)
        # Adding element type (line 439)
        int_427562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 12), list_427560, int_427562)
        # Adding element type (line 439)
        int_427563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 12), list_427560, int_427563)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 21), list_427551, list_427560)
        
        # Processing the call keyword arguments (line 436)
        # Getting the type of 'float' (line 440)
        float_427564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 21), 'float', False)
        keyword_427565 = float_427564
        kwargs_427566 = {'dtype': keyword_427565}
        # Getting the type of 'np' (line 436)
        np_427549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 436)
        array_427550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 12), np_427549, 'array')
        # Calling array(args, kwargs) (line 436)
        array_call_result_427567 = invoke(stypy.reporting.localization.Localization(__file__, 436, 12), array_427550, *[list_427551], **kwargs_427566)
        
        # Assigning a type to the variable 'A' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'A', array_call_result_427567)
        
        # Call to assert_allclose(...): (line 441)
        # Processing the call arguments (line 441)
        
        # Call to sorted(...): (line 441)
        # Processing the call arguments (line 441)
        
        # Call to eigvals(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'A' (line 441)
        A_427573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 52), 'A', False)
        # Processing the call keyword arguments (line 441)
        kwargs_427574 = {}
        # Getting the type of 'scipy' (line 441)
        scipy_427570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 31), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 441)
        linalg_427571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 31), scipy_427570, 'linalg')
        # Obtaining the member 'eigvals' of a type (line 441)
        eigvals_427572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 31), linalg_427571, 'eigvals')
        # Calling eigvals(args, kwargs) (line 441)
        eigvals_call_result_427575 = invoke(stypy.reporting.localization.Localization(__file__, 441, 31), eigvals_427572, *[A_427573], **kwargs_427574)
        
        # Processing the call keyword arguments (line 441)
        kwargs_427576 = {}
        # Getting the type of 'sorted' (line 441)
        sorted_427569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 24), 'sorted', False)
        # Calling sorted(args, kwargs) (line 441)
        sorted_call_result_427577 = invoke(stypy.reporting.localization.Localization(__file__, 441, 24), sorted_427569, *[eigvals_call_result_427575], **kwargs_427576)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 441)
        tuple_427578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 441)
        # Adding element type (line 441)
        int_427579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 58), tuple_427578, int_427579)
        # Adding element type (line 441)
        int_427580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 58), tuple_427578, int_427580)
        # Adding element type (line 441)
        int_427581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 58), tuple_427578, int_427581)
        
        # Processing the call keyword arguments (line 441)
        kwargs_427582 = {}
        # Getting the type of 'assert_allclose' (line 441)
        assert_allclose_427568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 441)
        assert_allclose_call_result_427583 = invoke(stypy.reporting.localization.Localization(__file__, 441, 8), assert_allclose_427568, *[sorted_call_result_427577, tuple_427578], **kwargs_427582)
        
        
        # Assigning a Call to a Name (line 442):
        
        # Assigning a Call to a Name (line 442):
        
        # Call to array(...): (line 442)
        # Processing the call arguments (line 442)
        
        # Obtaining an instance of the builtin type 'list' (line 442)
        list_427586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 442)
        # Adding element type (line 442)
        
        # Obtaining an instance of the builtin type 'list' (line 443)
        list_427587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 443)
        # Adding element type (line 443)
        float_427588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 12), list_427587, float_427588)
        # Adding element type (line 443)
        float_427589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 12), list_427587, float_427589)
        # Adding element type (line 443)
        float_427590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 12), list_427587, float_427590)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 27), list_427586, list_427587)
        # Adding element type (line 442)
        
        # Obtaining an instance of the builtin type 'list' (line 444)
        list_427591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 444)
        # Adding element type (line 444)
        float_427592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 12), list_427591, float_427592)
        # Adding element type (line 444)
        float_427593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 12), list_427591, float_427593)
        # Adding element type (line 444)
        float_427594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 12), list_427591, float_427594)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 27), list_427586, list_427591)
        # Adding element type (line 442)
        
        # Obtaining an instance of the builtin type 'list' (line 445)
        list_427595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 445)
        # Adding element type (line 445)
        float_427596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 12), list_427595, float_427596)
        # Adding element type (line 445)
        float_427597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 12), list_427595, float_427597)
        # Adding element type (line 445)
        float_427598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 12), list_427595, float_427598)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 27), list_427586, list_427595)
        
        # Processing the call keyword arguments (line 442)
        # Getting the type of 'float' (line 446)
        float_427599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 21), 'float', False)
        keyword_427600 = float_427599
        kwargs_427601 = {'dtype': keyword_427600}
        # Getting the type of 'np' (line 442)
        np_427584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 442)
        array_427585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 18), np_427584, 'array')
        # Calling array(args, kwargs) (line 442)
        array_call_result_427602 = invoke(stypy.reporting.localization.Localization(__file__, 442, 18), array_427585, *[list_427586], **kwargs_427601)
        
        # Assigning a type to the variable 'desired' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'desired', array_call_result_427602)
        
        # Assigning a Call to a Name (line 447):
        
        # Assigning a Call to a Name (line 447):
        
        # Call to expm(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'A' (line 447)
        A_427604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 22), 'A', False)
        # Processing the call keyword arguments (line 447)
        kwargs_427605 = {}
        # Getting the type of 'expm' (line 447)
        expm_427603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 17), 'expm', False)
        # Calling expm(args, kwargs) (line 447)
        expm_call_result_427606 = invoke(stypy.reporting.localization.Localization(__file__, 447, 17), expm_427603, *[A_427604], **kwargs_427605)
        
        # Assigning a type to the variable 'actual' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'actual', expm_call_result_427606)
        
        # Call to assert_allclose(...): (line 448)
        # Processing the call arguments (line 448)
        # Getting the type of 'actual' (line 448)
        actual_427608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 24), 'actual', False)
        # Getting the type of 'desired' (line 448)
        desired_427609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 32), 'desired', False)
        # Processing the call keyword arguments (line 448)
        kwargs_427610 = {}
        # Getting the type of 'assert_allclose' (line 448)
        assert_allclose_427607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 448)
        assert_allclose_call_result_427611 = invoke(stypy.reporting.localization.Localization(__file__, 448, 8), assert_allclose_427607, *[actual_427608, desired_427609], **kwargs_427610)
        
        
        # ################# End of 'test_burkardt_12(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_12' in the type store
        # Getting the type of 'stypy_return_type' (line 432)
        stypy_return_type_427612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427612)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_12'
        return stypy_return_type_427612


    @norecursion
    def test_burkardt_13(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_13'
        module_type_store = module_type_store.open_function_context('test_burkardt_13', 450, 4, False)
        # Assigning a type to the variable 'self' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_13.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_13.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_13.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_13.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_13')
        TestExpM.test_burkardt_13.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_13.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_13.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_13.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_13.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_13.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_13.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_13', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_13', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_13(...)' code ##################

        
        # Assigning a Call to a Name (line 458):
        
        # Assigning a Call to a Name (line 458):
        
        # Call to _burkardt_13_power(...): (line 458)
        # Processing the call arguments (line 458)
        int_427614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 39), 'int')
        int_427615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 42), 'int')
        # Processing the call keyword arguments (line 458)
        kwargs_427616 = {}
        # Getting the type of '_burkardt_13_power' (line 458)
        _burkardt_13_power_427613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 20), '_burkardt_13_power', False)
        # Calling _burkardt_13_power(args, kwargs) (line 458)
        _burkardt_13_power_call_result_427617 = invoke(stypy.reporting.localization.Localization(__file__, 458, 20), _burkardt_13_power_427613, *[int_427614, int_427615], **kwargs_427616)
        
        # Assigning a type to the variable 'A4_actual' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'A4_actual', _burkardt_13_power_call_result_427617)
        
        # Assigning a List to a Name (line 459):
        
        # Assigning a List to a Name (line 459):
        
        # Obtaining an instance of the builtin type 'list' (line 459)
        list_427618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 459)
        # Adding element type (line 459)
        
        # Obtaining an instance of the builtin type 'list' (line 459)
        list_427619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 459)
        # Adding element type (line 459)
        int_427620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 22), list_427619, int_427620)
        # Adding element type (line 459)
        int_427621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 22), list_427619, int_427621)
        # Adding element type (line 459)
        int_427622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 22), list_427619, int_427622)
        # Adding element type (line 459)
        int_427623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 22), list_427619, int_427623)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 21), list_427618, list_427619)
        # Adding element type (line 459)
        
        # Obtaining an instance of the builtin type 'list' (line 460)
        list_427624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 460)
        # Adding element type (line 460)
        int_427625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 22), list_427624, int_427625)
        # Adding element type (line 460)
        int_427626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 22), list_427624, int_427626)
        # Adding element type (line 460)
        int_427627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 22), list_427624, int_427627)
        # Adding element type (line 460)
        int_427628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 22), list_427624, int_427628)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 21), list_427618, list_427624)
        # Adding element type (line 459)
        
        # Obtaining an instance of the builtin type 'list' (line 461)
        list_427629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 461)
        # Adding element type (line 461)
        int_427630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 22), list_427629, int_427630)
        # Adding element type (line 461)
        int_427631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 22), list_427629, int_427631)
        # Adding element type (line 461)
        int_427632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 22), list_427629, int_427632)
        # Adding element type (line 461)
        int_427633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 22), list_427629, int_427633)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 21), list_427618, list_427629)
        # Adding element type (line 459)
        
        # Obtaining an instance of the builtin type 'list' (line 462)
        list_427634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 462)
        # Adding element type (line 462)
        float_427635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 22), list_427634, float_427635)
        # Adding element type (line 462)
        int_427636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 22), list_427634, int_427636)
        # Adding element type (line 462)
        int_427637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 22), list_427634, int_427637)
        # Adding element type (line 462)
        int_427638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 22), list_427634, int_427638)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 21), list_427618, list_427634)
        
        # Assigning a type to the variable 'A4_desired' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'A4_desired', list_427618)
        
        # Call to assert_allclose(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'A4_actual' (line 463)
        A4_actual_427640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 24), 'A4_actual', False)
        # Getting the type of 'A4_desired' (line 463)
        A4_desired_427641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 35), 'A4_desired', False)
        # Processing the call keyword arguments (line 463)
        kwargs_427642 = {}
        # Getting the type of 'assert_allclose' (line 463)
        assert_allclose_427639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 463)
        assert_allclose_call_result_427643 = invoke(stypy.reporting.localization.Localization(__file__, 463, 8), assert_allclose_427639, *[A4_actual_427640, A4_desired_427641], **kwargs_427642)
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 465)
        tuple_427644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 465)
        # Adding element type (line 465)
        int_427645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 18), tuple_427644, int_427645)
        # Adding element type (line 465)
        int_427646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 18), tuple_427644, int_427646)
        # Adding element type (line 465)
        int_427647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 18), tuple_427644, int_427647)
        # Adding element type (line 465)
        int_427648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 18), tuple_427644, int_427648)
        
        # Testing the type of a for loop iterable (line 465)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 465, 8), tuple_427644)
        # Getting the type of the for loop variable (line 465)
        for_loop_var_427649 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 465, 8), tuple_427644)
        # Assigning a type to the variable 'n' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'n', for_loop_var_427649)
        # SSA begins for a for statement (line 465)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 471):
        
        # Assigning a Call to a Name (line 471):
        
        # Call to max(...): (line 471)
        # Processing the call arguments (line 471)
        int_427651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 20), 'int')
        
        # Call to int(...): (line 471)
        # Processing the call arguments (line 471)
        
        # Call to ceil(...): (line 471)
        # Processing the call arguments (line 471)
        int_427655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 35), 'int')
        # Getting the type of 'n' (line 471)
        n_427656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 38), 'n', False)
        # Applying the binary operator 'div' (line 471)
        result_div_427657 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 35), 'div', int_427655, n_427656)
        
        # Processing the call keyword arguments (line 471)
        kwargs_427658 = {}
        # Getting the type of 'np' (line 471)
        np_427653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 27), 'np', False)
        # Obtaining the member 'ceil' of a type (line 471)
        ceil_427654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 27), np_427653, 'ceil')
        # Calling ceil(args, kwargs) (line 471)
        ceil_call_result_427659 = invoke(stypy.reporting.localization.Localization(__file__, 471, 27), ceil_427654, *[result_div_427657], **kwargs_427658)
        
        # Processing the call keyword arguments (line 471)
        kwargs_427660 = {}
        # Getting the type of 'int' (line 471)
        int_427652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 23), 'int', False)
        # Calling int(args, kwargs) (line 471)
        int_call_result_427661 = invoke(stypy.reporting.localization.Localization(__file__, 471, 23), int_427652, *[ceil_call_result_427659], **kwargs_427660)
        
        # Processing the call keyword arguments (line 471)
        kwargs_427662 = {}
        # Getting the type of 'max' (line 471)
        max_427650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'max', False)
        # Calling max(args, kwargs) (line 471)
        max_call_result_427663 = invoke(stypy.reporting.localization.Localization(__file__, 471, 16), max_427650, *[int_427651, int_call_result_427661], **kwargs_427662)
        
        # Assigning a type to the variable 'k' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'k', max_call_result_427663)
        
        # Assigning a Call to a Name (line 472):
        
        # Assigning a Call to a Name (line 472):
        
        # Call to zeros(...): (line 472)
        # Processing the call arguments (line 472)
        
        # Obtaining an instance of the builtin type 'tuple' (line 472)
        tuple_427666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 472)
        # Adding element type (line 472)
        # Getting the type of 'n' (line 472)
        n_427667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 32), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 32), tuple_427666, n_427667)
        # Adding element type (line 472)
        # Getting the type of 'n' (line 472)
        n_427668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 35), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 32), tuple_427666, n_427668)
        
        # Processing the call keyword arguments (line 472)
        # Getting the type of 'float' (line 472)
        float_427669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 45), 'float', False)
        keyword_427670 = float_427669
        kwargs_427671 = {'dtype': keyword_427670}
        # Getting the type of 'np' (line 472)
        np_427664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 22), 'np', False)
        # Obtaining the member 'zeros' of a type (line 472)
        zeros_427665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 22), np_427664, 'zeros')
        # Calling zeros(args, kwargs) (line 472)
        zeros_call_result_427672 = invoke(stypy.reporting.localization.Localization(__file__, 472, 22), zeros_427665, *[tuple_427666], **kwargs_427671)
        
        # Assigning a type to the variable 'desired' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'desired', zeros_call_result_427672)
        
        
        # Call to range(...): (line 473)
        # Processing the call arguments (line 473)
        # Getting the type of 'n' (line 473)
        n_427674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 27), 'n', False)
        # Getting the type of 'k' (line 473)
        k_427675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 29), 'k', False)
        # Applying the binary operator '*' (line 473)
        result_mul_427676 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 27), '*', n_427674, k_427675)
        
        # Processing the call keyword arguments (line 473)
        kwargs_427677 = {}
        # Getting the type of 'range' (line 473)
        range_427673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 21), 'range', False)
        # Calling range(args, kwargs) (line 473)
        range_call_result_427678 = invoke(stypy.reporting.localization.Localization(__file__, 473, 21), range_427673, *[result_mul_427676], **kwargs_427677)
        
        # Testing the type of a for loop iterable (line 473)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 473, 12), range_call_result_427678)
        # Getting the type of the for loop variable (line 473)
        for_loop_var_427679 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 473, 12), range_call_result_427678)
        # Assigning a type to the variable 'p' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'p', for_loop_var_427679)
        # SSA begins for a for statement (line 473)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 474):
        
        # Assigning a Call to a Name (line 474):
        
        # Call to _burkardt_13_power(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 'n' (line 474)
        n_427681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 40), 'n', False)
        # Getting the type of 'p' (line 474)
        p_427682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 43), 'p', False)
        # Processing the call keyword arguments (line 474)
        kwargs_427683 = {}
        # Getting the type of '_burkardt_13_power' (line 474)
        _burkardt_13_power_427680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 21), '_burkardt_13_power', False)
        # Calling _burkardt_13_power(args, kwargs) (line 474)
        _burkardt_13_power_call_result_427684 = invoke(stypy.reporting.localization.Localization(__file__, 474, 21), _burkardt_13_power_427680, *[n_427681, p_427682], **kwargs_427683)
        
        # Assigning a type to the variable 'Ap' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'Ap', _burkardt_13_power_call_result_427684)
        
        # Call to assert_equal(...): (line 475)
        # Processing the call arguments (line 475)
        
        # Call to min(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'Ap' (line 475)
        Ap_427688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 36), 'Ap', False)
        # Processing the call keyword arguments (line 475)
        kwargs_427689 = {}
        # Getting the type of 'np' (line 475)
        np_427686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 29), 'np', False)
        # Obtaining the member 'min' of a type (line 475)
        min_427687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 29), np_427686, 'min')
        # Calling min(args, kwargs) (line 475)
        min_call_result_427690 = invoke(stypy.reporting.localization.Localization(__file__, 475, 29), min_427687, *[Ap_427688], **kwargs_427689)
        
        int_427691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 41), 'int')
        # Processing the call keyword arguments (line 475)
        kwargs_427692 = {}
        # Getting the type of 'assert_equal' (line 475)
        assert_equal_427685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 16), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 475)
        assert_equal_call_result_427693 = invoke(stypy.reporting.localization.Localization(__file__, 475, 16), assert_equal_427685, *[min_call_result_427690, int_427691], **kwargs_427692)
        
        
        # Call to assert_allclose(...): (line 476)
        # Processing the call arguments (line 476)
        
        # Call to max(...): (line 476)
        # Processing the call arguments (line 476)
        # Getting the type of 'Ap' (line 476)
        Ap_427697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 39), 'Ap', False)
        # Processing the call keyword arguments (line 476)
        kwargs_427698 = {}
        # Getting the type of 'np' (line 476)
        np_427695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 32), 'np', False)
        # Obtaining the member 'max' of a type (line 476)
        max_427696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 32), np_427695, 'max')
        # Calling max(args, kwargs) (line 476)
        max_call_result_427699 = invoke(stypy.reporting.localization.Localization(__file__, 476, 32), max_427696, *[Ap_427697], **kwargs_427698)
        
        
        # Call to power(...): (line 476)
        # Processing the call arguments (line 476)
        int_427702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 53), 'int')
        
        
        # Call to floor(...): (line 476)
        # Processing the call arguments (line 476)
        # Getting the type of 'p' (line 476)
        p_427705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 67), 'p', False)
        # Getting the type of 'n' (line 476)
        n_427706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 69), 'n', False)
        # Applying the binary operator 'div' (line 476)
        result_div_427707 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 67), 'div', p_427705, n_427706)
        
        # Processing the call keyword arguments (line 476)
        kwargs_427708 = {}
        # Getting the type of 'np' (line 476)
        np_427703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 58), 'np', False)
        # Obtaining the member 'floor' of a type (line 476)
        floor_427704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 58), np_427703, 'floor')
        # Calling floor(args, kwargs) (line 476)
        floor_call_result_427709 = invoke(stypy.reporting.localization.Localization(__file__, 476, 58), floor_427704, *[result_div_427707], **kwargs_427708)
        
        # Applying the 'usub' unary operator (line 476)
        result___neg___427710 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 57), 'usub', floor_call_result_427709)
        
        # Getting the type of 'n' (line 476)
        n_427711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 72), 'n', False)
        # Applying the binary operator '*' (line 476)
        result_mul_427712 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 57), '*', result___neg___427710, n_427711)
        
        # Processing the call keyword arguments (line 476)
        kwargs_427713 = {}
        # Getting the type of 'np' (line 476)
        np_427700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 44), 'np', False)
        # Obtaining the member 'power' of a type (line 476)
        power_427701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 44), np_427700, 'power')
        # Calling power(args, kwargs) (line 476)
        power_call_result_427714 = invoke(stypy.reporting.localization.Localization(__file__, 476, 44), power_427701, *[int_427702, result_mul_427712], **kwargs_427713)
        
        # Processing the call keyword arguments (line 476)
        kwargs_427715 = {}
        # Getting the type of 'assert_allclose' (line 476)
        assert_allclose_427694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 476)
        assert_allclose_call_result_427716 = invoke(stypy.reporting.localization.Localization(__file__, 476, 16), assert_allclose_427694, *[max_call_result_427699, power_call_result_427714], **kwargs_427715)
        
        
        # Getting the type of 'desired' (line 477)
        desired_427717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 16), 'desired')
        # Getting the type of 'Ap' (line 477)
        Ap_427718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 27), 'Ap')
        
        # Call to factorial(...): (line 477)
        # Processing the call arguments (line 477)
        # Getting the type of 'p' (line 477)
        p_427720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 42), 'p', False)
        # Processing the call keyword arguments (line 477)
        kwargs_427721 = {}
        # Getting the type of 'factorial' (line 477)
        factorial_427719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 32), 'factorial', False)
        # Calling factorial(args, kwargs) (line 477)
        factorial_call_result_427722 = invoke(stypy.reporting.localization.Localization(__file__, 477, 32), factorial_427719, *[p_427720], **kwargs_427721)
        
        # Applying the binary operator 'div' (line 477)
        result_div_427723 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 27), 'div', Ap_427718, factorial_call_result_427722)
        
        # Applying the binary operator '+=' (line 477)
        result_iadd_427724 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 16), '+=', desired_427717, result_div_427723)
        # Assigning a type to the variable 'desired' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 16), 'desired', result_iadd_427724)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 478):
        
        # Assigning a Call to a Name (line 478):
        
        # Call to expm(...): (line 478)
        # Processing the call arguments (line 478)
        
        # Call to _burkardt_13_power(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'n' (line 478)
        n_427727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 45), 'n', False)
        int_427728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 48), 'int')
        # Processing the call keyword arguments (line 478)
        kwargs_427729 = {}
        # Getting the type of '_burkardt_13_power' (line 478)
        _burkardt_13_power_427726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 26), '_burkardt_13_power', False)
        # Calling _burkardt_13_power(args, kwargs) (line 478)
        _burkardt_13_power_call_result_427730 = invoke(stypy.reporting.localization.Localization(__file__, 478, 26), _burkardt_13_power_427726, *[n_427727, int_427728], **kwargs_427729)
        
        # Processing the call keyword arguments (line 478)
        kwargs_427731 = {}
        # Getting the type of 'expm' (line 478)
        expm_427725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 21), 'expm', False)
        # Calling expm(args, kwargs) (line 478)
        expm_call_result_427732 = invoke(stypy.reporting.localization.Localization(__file__, 478, 21), expm_427725, *[_burkardt_13_power_call_result_427730], **kwargs_427731)
        
        # Assigning a type to the variable 'actual' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'actual', expm_call_result_427732)
        
        # Call to assert_allclose(...): (line 479)
        # Processing the call arguments (line 479)
        # Getting the type of 'actual' (line 479)
        actual_427734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 28), 'actual', False)
        # Getting the type of 'desired' (line 479)
        desired_427735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 36), 'desired', False)
        # Processing the call keyword arguments (line 479)
        kwargs_427736 = {}
        # Getting the type of 'assert_allclose' (line 479)
        assert_allclose_427733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 479)
        assert_allclose_call_result_427737 = invoke(stypy.reporting.localization.Localization(__file__, 479, 12), assert_allclose_427733, *[actual_427734, desired_427735], **kwargs_427736)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_burkardt_13(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_13' in the type store
        # Getting the type of 'stypy_return_type' (line 450)
        stypy_return_type_427738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_13'
        return stypy_return_type_427738


    @norecursion
    def test_burkardt_14(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_burkardt_14'
        module_type_store = module_type_store.open_function_context('test_burkardt_14', 481, 4, False)
        # Assigning a type to the variable 'self' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpM.test_burkardt_14.__dict__.__setitem__('stypy_localization', localization)
        TestExpM.test_burkardt_14.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpM.test_burkardt_14.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpM.test_burkardt_14.__dict__.__setitem__('stypy_function_name', 'TestExpM.test_burkardt_14')
        TestExpM.test_burkardt_14.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpM.test_burkardt_14.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpM.test_burkardt_14.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpM.test_burkardt_14.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpM.test_burkardt_14.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpM.test_burkardt_14.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpM.test_burkardt_14.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.test_burkardt_14', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_burkardt_14', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_burkardt_14(...)' code ##################

        
        # Assigning a Call to a Name (line 484):
        
        # Assigning a Call to a Name (line 484):
        
        # Call to array(...): (line 484)
        # Processing the call arguments (line 484)
        
        # Obtaining an instance of the builtin type 'list' (line 484)
        list_427741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 484)
        # Adding element type (line 484)
        
        # Obtaining an instance of the builtin type 'list' (line 485)
        list_427742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 485)
        # Adding element type (line 485)
        int_427743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 12), list_427742, int_427743)
        # Adding element type (line 485)
        float_427744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 12), list_427742, float_427744)
        # Adding element type (line 485)
        int_427745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 12), list_427742, int_427745)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 21), list_427741, list_427742)
        # Adding element type (line 484)
        
        # Obtaining an instance of the builtin type 'list' (line 486)
        list_427746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 486)
        # Adding element type (line 486)
        
        float_427747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 15), 'float')
        float_427748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 22), 'float')
        float_427749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 26), 'float')
        # Applying the binary operator 'div' (line 486)
        result_div_427750 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 22), 'div', float_427748, float_427749)
        
        # Applying the binary operator '+' (line 486)
        result_add_427751 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 15), '+', float_427747, result_div_427750)
        
        # Applying the 'usub' unary operator (line 486)
        result___neg___427752 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 13), 'usub', result_add_427751)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 12), list_427746, result___neg___427752)
        # Adding element type (line 486)
        int_427753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 12), list_427746, int_427753)
        # Adding element type (line 486)
        float_427754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 12), list_427746, float_427754)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 21), list_427741, list_427746)
        # Adding element type (line 484)
        
        # Obtaining an instance of the builtin type 'list' (line 487)
        list_427755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 487)
        # Adding element type (line 487)
        float_427756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 13), 'float')
        float_427757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 18), 'float')
        # Applying the binary operator 'div' (line 487)
        result_div_427758 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 13), 'div', float_427756, float_427757)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), list_427755, result_div_427758)
        # Adding element type (line 487)
        int_427759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), list_427755, int_427759)
        # Adding element type (line 487)
        float_427760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 25), 'float')
        float_427761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 31), 'float')
        # Applying the binary operator 'div' (line 487)
        result_div_427762 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 25), 'div', float_427760, float_427761)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), list_427755, result_div_427762)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 21), list_427741, list_427755)
        
        # Processing the call keyword arguments (line 484)
        # Getting the type of 'float' (line 488)
        float_427763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 21), 'float', False)
        keyword_427764 = float_427763
        kwargs_427765 = {'dtype': keyword_427764}
        # Getting the type of 'np' (line 484)
        np_427739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 484)
        array_427740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 12), np_427739, 'array')
        # Calling array(args, kwargs) (line 484)
        array_call_result_427766 = invoke(stypy.reporting.localization.Localization(__file__, 484, 12), array_427740, *[list_427741], **kwargs_427765)
        
        # Assigning a type to the variable 'A' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'A', array_call_result_427766)
        
        # Assigning a Call to a Name (line 489):
        
        # Assigning a Call to a Name (line 489):
        
        # Call to array(...): (line 489)
        # Processing the call arguments (line 489)
        
        # Obtaining an instance of the builtin type 'list' (line 489)
        list_427769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 489)
        # Adding element type (line 489)
        
        # Obtaining an instance of the builtin type 'list' (line 490)
        list_427770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 490)
        # Adding element type (line 490)
        float_427771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 12), list_427770, float_427771)
        # Adding element type (line 490)
        float_427772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 12), list_427770, float_427772)
        # Adding element type (line 490)
        float_427773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 12), list_427770, float_427773)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 27), list_427769, list_427770)
        # Adding element type (line 489)
        
        # Obtaining an instance of the builtin type 'list' (line 491)
        list_427774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 491)
        # Adding element type (line 491)
        float_427775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 12), list_427774, float_427775)
        # Adding element type (line 491)
        float_427776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 12), list_427774, float_427776)
        # Adding element type (line 491)
        float_427777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 12), list_427774, float_427777)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 27), list_427769, list_427774)
        # Adding element type (line 489)
        
        # Obtaining an instance of the builtin type 'list' (line 492)
        list_427778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 492)
        # Adding element type (line 492)
        float_427779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 492, 12), list_427778, float_427779)
        # Adding element type (line 492)
        float_427780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 492, 12), list_427778, float_427780)
        # Adding element type (line 492)
        float_427781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 492, 12), list_427778, float_427781)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 27), list_427769, list_427778)
        
        # Processing the call keyword arguments (line 489)
        # Getting the type of 'float' (line 493)
        float_427782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 21), 'float', False)
        keyword_427783 = float_427782
        kwargs_427784 = {'dtype': keyword_427783}
        # Getting the type of 'np' (line 489)
        np_427767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 489)
        array_427768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 18), np_427767, 'array')
        # Calling array(args, kwargs) (line 489)
        array_call_result_427785 = invoke(stypy.reporting.localization.Localization(__file__, 489, 18), array_427768, *[list_427769], **kwargs_427784)
        
        # Assigning a type to the variable 'desired' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'desired', array_call_result_427785)
        
        # Assigning a Call to a Name (line 494):
        
        # Assigning a Call to a Name (line 494):
        
        # Call to expm(...): (line 494)
        # Processing the call arguments (line 494)
        # Getting the type of 'A' (line 494)
        A_427787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 22), 'A', False)
        # Processing the call keyword arguments (line 494)
        kwargs_427788 = {}
        # Getting the type of 'expm' (line 494)
        expm_427786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 17), 'expm', False)
        # Calling expm(args, kwargs) (line 494)
        expm_call_result_427789 = invoke(stypy.reporting.localization.Localization(__file__, 494, 17), expm_427786, *[A_427787], **kwargs_427788)
        
        # Assigning a type to the variable 'actual' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'actual', expm_call_result_427789)
        
        # Call to assert_allclose(...): (line 495)
        # Processing the call arguments (line 495)
        # Getting the type of 'actual' (line 495)
        actual_427791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 24), 'actual', False)
        # Getting the type of 'desired' (line 495)
        desired_427792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 32), 'desired', False)
        # Processing the call keyword arguments (line 495)
        kwargs_427793 = {}
        # Getting the type of 'assert_allclose' (line 495)
        assert_allclose_427790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 495)
        assert_allclose_call_result_427794 = invoke(stypy.reporting.localization.Localization(__file__, 495, 8), assert_allclose_427790, *[actual_427791, desired_427792], **kwargs_427793)
        
        
        # ################# End of 'test_burkardt_14(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_burkardt_14' in the type store
        # Getting the type of 'stypy_return_type' (line 481)
        stypy_return_type_427795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427795)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_burkardt_14'
        return stypy_return_type_427795


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 73, 0, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpM.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestExpM' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'TestExpM', TestExpM)
# Declaration of the 'TestOperators' class

class TestOperators(object, ):

    @norecursion
    def test_product_operator(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_product_operator'
        module_type_store = module_type_store.open_function_context('test_product_operator', 500, 4, False)
        # Assigning a type to the variable 'self' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOperators.test_product_operator.__dict__.__setitem__('stypy_localization', localization)
        TestOperators.test_product_operator.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOperators.test_product_operator.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOperators.test_product_operator.__dict__.__setitem__('stypy_function_name', 'TestOperators.test_product_operator')
        TestOperators.test_product_operator.__dict__.__setitem__('stypy_param_names_list', [])
        TestOperators.test_product_operator.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOperators.test_product_operator.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOperators.test_product_operator.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOperators.test_product_operator.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOperators.test_product_operator.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOperators.test_product_operator.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOperators.test_product_operator', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_product_operator', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_product_operator(...)' code ##################

        
        # Call to seed(...): (line 501)
        # Processing the call arguments (line 501)
        int_427798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 20), 'int')
        # Processing the call keyword arguments (line 501)
        kwargs_427799 = {}
        # Getting the type of 'random' (line 501)
        random_427796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'random', False)
        # Obtaining the member 'seed' of a type (line 501)
        seed_427797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 8), random_427796, 'seed')
        # Calling seed(args, kwargs) (line 501)
        seed_call_result_427800 = invoke(stypy.reporting.localization.Localization(__file__, 501, 8), seed_427797, *[int_427798], **kwargs_427799)
        
        
        # Assigning a Num to a Name (line 502):
        
        # Assigning a Num to a Name (line 502):
        int_427801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 12), 'int')
        # Assigning a type to the variable 'n' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'n', int_427801)
        
        # Assigning a Num to a Name (line 503):
        
        # Assigning a Num to a Name (line 503):
        int_427802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 12), 'int')
        # Assigning a type to the variable 'k' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'k', int_427802)
        
        # Assigning a Num to a Name (line 504):
        
        # Assigning a Num to a Name (line 504):
        int_427803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 19), 'int')
        # Assigning a type to the variable 'nsamples' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'nsamples', int_427803)
        
        
        # Call to range(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'nsamples' (line 505)
        nsamples_427805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 23), 'nsamples', False)
        # Processing the call keyword arguments (line 505)
        kwargs_427806 = {}
        # Getting the type of 'range' (line 505)
        range_427804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 17), 'range', False)
        # Calling range(args, kwargs) (line 505)
        range_call_result_427807 = invoke(stypy.reporting.localization.Localization(__file__, 505, 17), range_427804, *[nsamples_427805], **kwargs_427806)
        
        # Testing the type of a for loop iterable (line 505)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 505, 8), range_call_result_427807)
        # Getting the type of the for loop variable (line 505)
        for_loop_var_427808 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 505, 8), range_call_result_427807)
        # Assigning a type to the variable 'i' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'i', for_loop_var_427808)
        # SSA begins for a for statement (line 505)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 506):
        
        # Assigning a Call to a Name (line 506):
        
        # Call to randn(...): (line 506)
        # Processing the call arguments (line 506)
        # Getting the type of 'n' (line 506)
        n_427812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 32), 'n', False)
        # Getting the type of 'n' (line 506)
        n_427813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 35), 'n', False)
        # Processing the call keyword arguments (line 506)
        kwargs_427814 = {}
        # Getting the type of 'np' (line 506)
        np_427809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 506)
        random_427810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 16), np_427809, 'random')
        # Obtaining the member 'randn' of a type (line 506)
        randn_427811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 16), random_427810, 'randn')
        # Calling randn(args, kwargs) (line 506)
        randn_call_result_427815 = invoke(stypy.reporting.localization.Localization(__file__, 506, 16), randn_427811, *[n_427812, n_427813], **kwargs_427814)
        
        # Assigning a type to the variable 'A' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 12), 'A', randn_call_result_427815)
        
        # Assigning a Call to a Name (line 507):
        
        # Assigning a Call to a Name (line 507):
        
        # Call to randn(...): (line 507)
        # Processing the call arguments (line 507)
        # Getting the type of 'n' (line 507)
        n_427819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 32), 'n', False)
        # Getting the type of 'n' (line 507)
        n_427820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 35), 'n', False)
        # Processing the call keyword arguments (line 507)
        kwargs_427821 = {}
        # Getting the type of 'np' (line 507)
        np_427816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 507)
        random_427817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 16), np_427816, 'random')
        # Obtaining the member 'randn' of a type (line 507)
        randn_427818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 16), random_427817, 'randn')
        # Calling randn(args, kwargs) (line 507)
        randn_call_result_427822 = invoke(stypy.reporting.localization.Localization(__file__, 507, 16), randn_427818, *[n_427819, n_427820], **kwargs_427821)
        
        # Assigning a type to the variable 'B' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'B', randn_call_result_427822)
        
        # Assigning a Call to a Name (line 508):
        
        # Assigning a Call to a Name (line 508):
        
        # Call to randn(...): (line 508)
        # Processing the call arguments (line 508)
        # Getting the type of 'n' (line 508)
        n_427826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 32), 'n', False)
        # Getting the type of 'n' (line 508)
        n_427827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 35), 'n', False)
        # Processing the call keyword arguments (line 508)
        kwargs_427828 = {}
        # Getting the type of 'np' (line 508)
        np_427823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 508)
        random_427824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 16), np_427823, 'random')
        # Obtaining the member 'randn' of a type (line 508)
        randn_427825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 16), random_427824, 'randn')
        # Calling randn(args, kwargs) (line 508)
        randn_call_result_427829 = invoke(stypy.reporting.localization.Localization(__file__, 508, 16), randn_427825, *[n_427826, n_427827], **kwargs_427828)
        
        # Assigning a type to the variable 'C' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'C', randn_call_result_427829)
        
        # Assigning a Call to a Name (line 509):
        
        # Assigning a Call to a Name (line 509):
        
        # Call to randn(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'n' (line 509)
        n_427833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 32), 'n', False)
        # Getting the type of 'k' (line 509)
        k_427834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 35), 'k', False)
        # Processing the call keyword arguments (line 509)
        kwargs_427835 = {}
        # Getting the type of 'np' (line 509)
        np_427830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 509)
        random_427831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 16), np_427830, 'random')
        # Obtaining the member 'randn' of a type (line 509)
        randn_427832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 16), random_427831, 'randn')
        # Calling randn(args, kwargs) (line 509)
        randn_call_result_427836 = invoke(stypy.reporting.localization.Localization(__file__, 509, 16), randn_427832, *[n_427833, k_427834], **kwargs_427835)
        
        # Assigning a type to the variable 'D' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'D', randn_call_result_427836)
        
        # Assigning a Call to a Name (line 510):
        
        # Assigning a Call to a Name (line 510):
        
        # Call to ProductOperator(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'A' (line 510)
        A_427838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 33), 'A', False)
        # Getting the type of 'B' (line 510)
        B_427839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 36), 'B', False)
        # Getting the type of 'C' (line 510)
        C_427840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 39), 'C', False)
        # Processing the call keyword arguments (line 510)
        kwargs_427841 = {}
        # Getting the type of 'ProductOperator' (line 510)
        ProductOperator_427837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 17), 'ProductOperator', False)
        # Calling ProductOperator(args, kwargs) (line 510)
        ProductOperator_call_result_427842 = invoke(stypy.reporting.localization.Localization(__file__, 510, 17), ProductOperator_427837, *[A_427838, B_427839, C_427840], **kwargs_427841)
        
        # Assigning a type to the variable 'op' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'op', ProductOperator_call_result_427842)
        
        # Call to assert_allclose(...): (line 511)
        # Processing the call arguments (line 511)
        
        # Call to matmat(...): (line 511)
        # Processing the call arguments (line 511)
        # Getting the type of 'D' (line 511)
        D_427846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 38), 'D', False)
        # Processing the call keyword arguments (line 511)
        kwargs_427847 = {}
        # Getting the type of 'op' (line 511)
        op_427844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 28), 'op', False)
        # Obtaining the member 'matmat' of a type (line 511)
        matmat_427845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 28), op_427844, 'matmat')
        # Calling matmat(args, kwargs) (line 511)
        matmat_call_result_427848 = invoke(stypy.reporting.localization.Localization(__file__, 511, 28), matmat_427845, *[D_427846], **kwargs_427847)
        
        
        # Call to dot(...): (line 511)
        # Processing the call arguments (line 511)
        # Getting the type of 'D' (line 511)
        D_427859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 62), 'D', False)
        # Processing the call keyword arguments (line 511)
        kwargs_427860 = {}
        
        # Call to dot(...): (line 511)
        # Processing the call arguments (line 511)
        # Getting the type of 'C' (line 511)
        C_427855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 55), 'C', False)
        # Processing the call keyword arguments (line 511)
        kwargs_427856 = {}
        
        # Call to dot(...): (line 511)
        # Processing the call arguments (line 511)
        # Getting the type of 'B' (line 511)
        B_427851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 48), 'B', False)
        # Processing the call keyword arguments (line 511)
        kwargs_427852 = {}
        # Getting the type of 'A' (line 511)
        A_427849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 42), 'A', False)
        # Obtaining the member 'dot' of a type (line 511)
        dot_427850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 42), A_427849, 'dot')
        # Calling dot(args, kwargs) (line 511)
        dot_call_result_427853 = invoke(stypy.reporting.localization.Localization(__file__, 511, 42), dot_427850, *[B_427851], **kwargs_427852)
        
        # Obtaining the member 'dot' of a type (line 511)
        dot_427854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 42), dot_call_result_427853, 'dot')
        # Calling dot(args, kwargs) (line 511)
        dot_call_result_427857 = invoke(stypy.reporting.localization.Localization(__file__, 511, 42), dot_427854, *[C_427855], **kwargs_427856)
        
        # Obtaining the member 'dot' of a type (line 511)
        dot_427858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 42), dot_call_result_427857, 'dot')
        # Calling dot(args, kwargs) (line 511)
        dot_call_result_427861 = invoke(stypy.reporting.localization.Localization(__file__, 511, 42), dot_427858, *[D_427859], **kwargs_427860)
        
        # Processing the call keyword arguments (line 511)
        kwargs_427862 = {}
        # Getting the type of 'assert_allclose' (line 511)
        assert_allclose_427843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 511)
        assert_allclose_call_result_427863 = invoke(stypy.reporting.localization.Localization(__file__, 511, 12), assert_allclose_427843, *[matmat_call_result_427848, dot_call_result_427861], **kwargs_427862)
        
        
        # Call to assert_allclose(...): (line 512)
        # Processing the call arguments (line 512)
        
        # Call to matmat(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'D' (line 512)
        D_427868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 40), 'D', False)
        # Processing the call keyword arguments (line 512)
        kwargs_427869 = {}
        # Getting the type of 'op' (line 512)
        op_427865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 28), 'op', False)
        # Obtaining the member 'T' of a type (line 512)
        T_427866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 28), op_427865, 'T')
        # Obtaining the member 'matmat' of a type (line 512)
        matmat_427867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 28), T_427866, 'matmat')
        # Calling matmat(args, kwargs) (line 512)
        matmat_call_result_427870 = invoke(stypy.reporting.localization.Localization(__file__, 512, 28), matmat_427867, *[D_427868], **kwargs_427869)
        
        
        # Call to dot(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'D' (line 512)
        D_427882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 68), 'D', False)
        # Processing the call keyword arguments (line 512)
        kwargs_427883 = {}
        
        # Call to dot(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'C' (line 512)
        C_427877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 58), 'C', False)
        # Processing the call keyword arguments (line 512)
        kwargs_427878 = {}
        
        # Call to dot(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'B' (line 512)
        B_427873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 51), 'B', False)
        # Processing the call keyword arguments (line 512)
        kwargs_427874 = {}
        # Getting the type of 'A' (line 512)
        A_427871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 45), 'A', False)
        # Obtaining the member 'dot' of a type (line 512)
        dot_427872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 45), A_427871, 'dot')
        # Calling dot(args, kwargs) (line 512)
        dot_call_result_427875 = invoke(stypy.reporting.localization.Localization(__file__, 512, 45), dot_427872, *[B_427873], **kwargs_427874)
        
        # Obtaining the member 'dot' of a type (line 512)
        dot_427876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 45), dot_call_result_427875, 'dot')
        # Calling dot(args, kwargs) (line 512)
        dot_call_result_427879 = invoke(stypy.reporting.localization.Localization(__file__, 512, 45), dot_427876, *[C_427877], **kwargs_427878)
        
        # Obtaining the member 'T' of a type (line 512)
        T_427880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 45), dot_call_result_427879, 'T')
        # Obtaining the member 'dot' of a type (line 512)
        dot_427881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 45), T_427880, 'dot')
        # Calling dot(args, kwargs) (line 512)
        dot_call_result_427884 = invoke(stypy.reporting.localization.Localization(__file__, 512, 45), dot_427881, *[D_427882], **kwargs_427883)
        
        # Processing the call keyword arguments (line 512)
        kwargs_427885 = {}
        # Getting the type of 'assert_allclose' (line 512)
        assert_allclose_427864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 512)
        assert_allclose_call_result_427886 = invoke(stypy.reporting.localization.Localization(__file__, 512, 12), assert_allclose_427864, *[matmat_call_result_427870, dot_call_result_427884], **kwargs_427885)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_product_operator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_product_operator' in the type store
        # Getting the type of 'stypy_return_type' (line 500)
        stypy_return_type_427887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427887)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_product_operator'
        return stypy_return_type_427887


    @norecursion
    def test_matrix_power_operator(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_matrix_power_operator'
        module_type_store = module_type_store.open_function_context('test_matrix_power_operator', 514, 4, False)
        # Assigning a type to the variable 'self' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOperators.test_matrix_power_operator.__dict__.__setitem__('stypy_localization', localization)
        TestOperators.test_matrix_power_operator.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOperators.test_matrix_power_operator.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOperators.test_matrix_power_operator.__dict__.__setitem__('stypy_function_name', 'TestOperators.test_matrix_power_operator')
        TestOperators.test_matrix_power_operator.__dict__.__setitem__('stypy_param_names_list', [])
        TestOperators.test_matrix_power_operator.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOperators.test_matrix_power_operator.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOperators.test_matrix_power_operator.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOperators.test_matrix_power_operator.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOperators.test_matrix_power_operator.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOperators.test_matrix_power_operator.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOperators.test_matrix_power_operator', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_matrix_power_operator', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_matrix_power_operator(...)' code ##################

        
        # Call to seed(...): (line 515)
        # Processing the call arguments (line 515)
        int_427890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 20), 'int')
        # Processing the call keyword arguments (line 515)
        kwargs_427891 = {}
        # Getting the type of 'random' (line 515)
        random_427888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'random', False)
        # Obtaining the member 'seed' of a type (line 515)
        seed_427889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 8), random_427888, 'seed')
        # Calling seed(args, kwargs) (line 515)
        seed_call_result_427892 = invoke(stypy.reporting.localization.Localization(__file__, 515, 8), seed_427889, *[int_427890], **kwargs_427891)
        
        
        # Assigning a Num to a Name (line 516):
        
        # Assigning a Num to a Name (line 516):
        int_427893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 12), 'int')
        # Assigning a type to the variable 'n' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'n', int_427893)
        
        # Assigning a Num to a Name (line 517):
        
        # Assigning a Num to a Name (line 517):
        int_427894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 12), 'int')
        # Assigning a type to the variable 'k' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'k', int_427894)
        
        # Assigning a Num to a Name (line 518):
        
        # Assigning a Num to a Name (line 518):
        int_427895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 12), 'int')
        # Assigning a type to the variable 'p' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'p', int_427895)
        
        # Assigning a Num to a Name (line 519):
        
        # Assigning a Num to a Name (line 519):
        int_427896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 19), 'int')
        # Assigning a type to the variable 'nsamples' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'nsamples', int_427896)
        
        
        # Call to range(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 'nsamples' (line 520)
        nsamples_427898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 23), 'nsamples', False)
        # Processing the call keyword arguments (line 520)
        kwargs_427899 = {}
        # Getting the type of 'range' (line 520)
        range_427897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 17), 'range', False)
        # Calling range(args, kwargs) (line 520)
        range_call_result_427900 = invoke(stypy.reporting.localization.Localization(__file__, 520, 17), range_427897, *[nsamples_427898], **kwargs_427899)
        
        # Testing the type of a for loop iterable (line 520)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 520, 8), range_call_result_427900)
        # Getting the type of the for loop variable (line 520)
        for_loop_var_427901 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 520, 8), range_call_result_427900)
        # Assigning a type to the variable 'i' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'i', for_loop_var_427901)
        # SSA begins for a for statement (line 520)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 521):
        
        # Assigning a Call to a Name (line 521):
        
        # Call to randn(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'n' (line 521)
        n_427905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 32), 'n', False)
        # Getting the type of 'n' (line 521)
        n_427906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 35), 'n', False)
        # Processing the call keyword arguments (line 521)
        kwargs_427907 = {}
        # Getting the type of 'np' (line 521)
        np_427902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 521)
        random_427903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 16), np_427902, 'random')
        # Obtaining the member 'randn' of a type (line 521)
        randn_427904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 16), random_427903, 'randn')
        # Calling randn(args, kwargs) (line 521)
        randn_call_result_427908 = invoke(stypy.reporting.localization.Localization(__file__, 521, 16), randn_427904, *[n_427905, n_427906], **kwargs_427907)
        
        # Assigning a type to the variable 'A' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'A', randn_call_result_427908)
        
        # Assigning a Call to a Name (line 522):
        
        # Assigning a Call to a Name (line 522):
        
        # Call to randn(...): (line 522)
        # Processing the call arguments (line 522)
        # Getting the type of 'n' (line 522)
        n_427912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 32), 'n', False)
        # Getting the type of 'k' (line 522)
        k_427913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 35), 'k', False)
        # Processing the call keyword arguments (line 522)
        kwargs_427914 = {}
        # Getting the type of 'np' (line 522)
        np_427909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 522)
        random_427910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 16), np_427909, 'random')
        # Obtaining the member 'randn' of a type (line 522)
        randn_427911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 16), random_427910, 'randn')
        # Calling randn(args, kwargs) (line 522)
        randn_call_result_427915 = invoke(stypy.reporting.localization.Localization(__file__, 522, 16), randn_427911, *[n_427912, k_427913], **kwargs_427914)
        
        # Assigning a type to the variable 'B' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'B', randn_call_result_427915)
        
        # Assigning a Call to a Name (line 523):
        
        # Assigning a Call to a Name (line 523):
        
        # Call to MatrixPowerOperator(...): (line 523)
        # Processing the call arguments (line 523)
        # Getting the type of 'A' (line 523)
        A_427917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 37), 'A', False)
        # Getting the type of 'p' (line 523)
        p_427918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 40), 'p', False)
        # Processing the call keyword arguments (line 523)
        kwargs_427919 = {}
        # Getting the type of 'MatrixPowerOperator' (line 523)
        MatrixPowerOperator_427916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 17), 'MatrixPowerOperator', False)
        # Calling MatrixPowerOperator(args, kwargs) (line 523)
        MatrixPowerOperator_call_result_427920 = invoke(stypy.reporting.localization.Localization(__file__, 523, 17), MatrixPowerOperator_427916, *[A_427917, p_427918], **kwargs_427919)
        
        # Assigning a type to the variable 'op' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'op', MatrixPowerOperator_call_result_427920)
        
        # Call to assert_allclose(...): (line 524)
        # Processing the call arguments (line 524)
        
        # Call to matmat(...): (line 524)
        # Processing the call arguments (line 524)
        # Getting the type of 'B' (line 524)
        B_427924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 38), 'B', False)
        # Processing the call keyword arguments (line 524)
        kwargs_427925 = {}
        # Getting the type of 'op' (line 524)
        op_427922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 28), 'op', False)
        # Obtaining the member 'matmat' of a type (line 524)
        matmat_427923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 28), op_427922, 'matmat')
        # Calling matmat(args, kwargs) (line 524)
        matmat_call_result_427926 = invoke(stypy.reporting.localization.Localization(__file__, 524, 28), matmat_427923, *[B_427924], **kwargs_427925)
        
        
        # Call to dot(...): (line 524)
        # Processing the call arguments (line 524)
        # Getting the type of 'B' (line 524)
        B_427933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 65), 'B', False)
        # Processing the call keyword arguments (line 524)
        kwargs_427934 = {}
        
        # Call to matrix_power(...): (line 524)
        # Processing the call arguments (line 524)
        # Getting the type of 'A' (line 524)
        A_427928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 55), 'A', False)
        # Getting the type of 'p' (line 524)
        p_427929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 58), 'p', False)
        # Processing the call keyword arguments (line 524)
        kwargs_427930 = {}
        # Getting the type of 'matrix_power' (line 524)
        matrix_power_427927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 42), 'matrix_power', False)
        # Calling matrix_power(args, kwargs) (line 524)
        matrix_power_call_result_427931 = invoke(stypy.reporting.localization.Localization(__file__, 524, 42), matrix_power_427927, *[A_427928, p_427929], **kwargs_427930)
        
        # Obtaining the member 'dot' of a type (line 524)
        dot_427932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 42), matrix_power_call_result_427931, 'dot')
        # Calling dot(args, kwargs) (line 524)
        dot_call_result_427935 = invoke(stypy.reporting.localization.Localization(__file__, 524, 42), dot_427932, *[B_427933], **kwargs_427934)
        
        # Processing the call keyword arguments (line 524)
        kwargs_427936 = {}
        # Getting the type of 'assert_allclose' (line 524)
        assert_allclose_427921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 524)
        assert_allclose_call_result_427937 = invoke(stypy.reporting.localization.Localization(__file__, 524, 12), assert_allclose_427921, *[matmat_call_result_427926, dot_call_result_427935], **kwargs_427936)
        
        
        # Call to assert_allclose(...): (line 525)
        # Processing the call arguments (line 525)
        
        # Call to matmat(...): (line 525)
        # Processing the call arguments (line 525)
        # Getting the type of 'B' (line 525)
        B_427942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 40), 'B', False)
        # Processing the call keyword arguments (line 525)
        kwargs_427943 = {}
        # Getting the type of 'op' (line 525)
        op_427939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 28), 'op', False)
        # Obtaining the member 'T' of a type (line 525)
        T_427940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 28), op_427939, 'T')
        # Obtaining the member 'matmat' of a type (line 525)
        matmat_427941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 28), T_427940, 'matmat')
        # Calling matmat(args, kwargs) (line 525)
        matmat_call_result_427944 = invoke(stypy.reporting.localization.Localization(__file__, 525, 28), matmat_427941, *[B_427942], **kwargs_427943)
        
        
        # Call to dot(...): (line 525)
        # Processing the call arguments (line 525)
        # Getting the type of 'B' (line 525)
        B_427952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 69), 'B', False)
        # Processing the call keyword arguments (line 525)
        kwargs_427953 = {}
        
        # Call to matrix_power(...): (line 525)
        # Processing the call arguments (line 525)
        # Getting the type of 'A' (line 525)
        A_427946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 57), 'A', False)
        # Getting the type of 'p' (line 525)
        p_427947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 60), 'p', False)
        # Processing the call keyword arguments (line 525)
        kwargs_427948 = {}
        # Getting the type of 'matrix_power' (line 525)
        matrix_power_427945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 44), 'matrix_power', False)
        # Calling matrix_power(args, kwargs) (line 525)
        matrix_power_call_result_427949 = invoke(stypy.reporting.localization.Localization(__file__, 525, 44), matrix_power_427945, *[A_427946, p_427947], **kwargs_427948)
        
        # Obtaining the member 'T' of a type (line 525)
        T_427950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 44), matrix_power_call_result_427949, 'T')
        # Obtaining the member 'dot' of a type (line 525)
        dot_427951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 44), T_427950, 'dot')
        # Calling dot(args, kwargs) (line 525)
        dot_call_result_427954 = invoke(stypy.reporting.localization.Localization(__file__, 525, 44), dot_427951, *[B_427952], **kwargs_427953)
        
        # Processing the call keyword arguments (line 525)
        kwargs_427955 = {}
        # Getting the type of 'assert_allclose' (line 525)
        assert_allclose_427938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 525)
        assert_allclose_call_result_427956 = invoke(stypy.reporting.localization.Localization(__file__, 525, 12), assert_allclose_427938, *[matmat_call_result_427944, dot_call_result_427954], **kwargs_427955)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_matrix_power_operator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_matrix_power_operator' in the type store
        # Getting the type of 'stypy_return_type' (line 514)
        stypy_return_type_427957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_427957)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_matrix_power_operator'
        return stypy_return_type_427957


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 498, 0, False)
        # Assigning a type to the variable 'self' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOperators.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestOperators' (line 498)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 0), 'TestOperators', TestOperators)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
