
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import threading
4: 
5: import numpy as np
6: from numpy import array, finfo, arange, eye, all, unique, ones, dot, matrix
7: import numpy.random as random
8: from numpy.testing import (
9:         assert_array_almost_equal, assert_almost_equal,
10:         assert_equal, assert_array_equal, assert_, assert_allclose,
11:         assert_warns)
12: import pytest
13: from pytest import raises as assert_raises
14: 
15: import scipy.linalg
16: from scipy.linalg import norm, inv
17: from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
18:         csr_matrix, identity, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
19: from scipy.sparse.linalg import SuperLU
20: from scipy.sparse.linalg.dsolve import (spsolve, use_solver, splu, spilu,
21:         MatrixRankWarning, _superlu, spsolve_triangular, factorized)
22: 
23: from scipy._lib._numpy_compat import suppress_warnings
24: 
25: 
26: sup_sparse_efficiency = suppress_warnings()
27: sup_sparse_efficiency.filter(SparseEfficiencyWarning)
28: 
29: # scikits.umfpack is not a SciPy dependency but it is optionally used in
30: # dsolve, so check whether it's available
31: try:
32:     import scikits.umfpack as umfpack
33:     has_umfpack = True
34: except ImportError:
35:     has_umfpack = False
36: 
37: def toarray(a):
38:     if isspmatrix(a):
39:         return a.toarray()
40:     else:
41:         return a
42: 
43: 
44: class TestFactorized(object):
45:     def setup_method(self):
46:         n = 5
47:         d = arange(n) + 1
48:         self.n = n
49:         self.A = spdiags((d, 2*d, d[::-1]), (-3, 0, 5), n, n).tocsc()
50:         random.seed(1234)
51: 
52:     def _check_singular(self):
53:         A = csc_matrix((5,5), dtype='d')
54:         b = ones(5)
55:         assert_array_almost_equal(0. * b, factorized(A)(b))
56: 
57:     def _check_non_singular(self):
58:         # Make a diagonal dominant, to make sure it is not singular
59:         n = 5
60:         a = csc_matrix(random.rand(n, n))
61:         b = ones(n)
62: 
63:         expected = splu(a).solve(b)
64:         assert_array_almost_equal(factorized(a)(b), expected)
65: 
66:     def test_singular_without_umfpack(self):
67:         use_solver(useUmfpack=False)
68:         with assert_raises(RuntimeError, message="Factor is exactly singular"):
69:             self._check_singular()
70: 
71:     @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
72:     def test_singular_with_umfpack(self):
73:         use_solver(useUmfpack=True)
74:         with suppress_warnings() as sup:
75:             sup.filter(RuntimeWarning, "divide by zero encountered in double_scalars")
76:             assert_warns(umfpack.UmfpackWarning, self._check_singular)
77: 
78:     def test_non_singular_without_umfpack(self):
79:         use_solver(useUmfpack=False)
80:         self._check_non_singular()
81: 
82:     @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
83:     def test_non_singular_with_umfpack(self):
84:         use_solver(useUmfpack=True)
85:         self._check_non_singular()
86: 
87:     def test_cannot_factorize_nonsquare_matrix_without_umfpack(self):
88:         use_solver(useUmfpack=False)
89:         msg = "can only factor square matrices"
90:         with assert_raises(ValueError, message=msg):
91:             factorized(self.A[:, :4])
92: 
93:     @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
94:     def test_factorizes_nonsquare_matrix_with_umfpack(self):
95:         use_solver(useUmfpack=True)
96:         # does not raise
97:         factorized(self.A[:,:4])
98: 
99:     def test_call_with_incorrectly_sized_matrix_without_umfpack(self):
100:         use_solver(useUmfpack=False)
101:         solve = factorized(self.A)
102:         b = random.rand(4)
103:         B = random.rand(4, 3)
104:         BB = random.rand(self.n, 3, 9)
105: 
106:         with assert_raises(ValueError, message="is of incompatible size"):
107:             solve(b)
108:         with assert_raises(ValueError, message="is of incompatible size"):
109:             solve(B)
110:         with assert_raises(ValueError,
111:                            message="object too deep for desired array"):
112:             solve(BB)
113: 
114:     @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
115:     def test_call_with_incorrectly_sized_matrix_with_umfpack(self):
116:         use_solver(useUmfpack=True)
117:         solve = factorized(self.A)
118:         b = random.rand(4)
119:         B = random.rand(4, 3)
120:         BB = random.rand(self.n, 3, 9)
121: 
122:         # does not raise
123:         solve(b)
124:         msg = "object too deep for desired array"
125:         with assert_raises(ValueError, message=msg):
126:             solve(B)
127:         with assert_raises(ValueError, message=msg):
128:             solve(BB)
129: 
130:     def test_call_with_cast_to_complex_without_umfpack(self):
131:         use_solver(useUmfpack=False)
132:         solve = factorized(self.A)
133:         b = random.rand(4)
134:         for t in [np.complex64, np.complex128]:
135:             with assert_raises(TypeError, message="Cannot cast array data"):
136:                 solve(b.astype(t))
137: 
138:     @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
139:     def test_call_with_cast_to_complex_with_umfpack(self):
140:         use_solver(useUmfpack=True)
141:         solve = factorized(self.A)
142:         b = random.rand(4)
143:         for t in [np.complex64, np.complex128]:
144:             assert_warns(np.ComplexWarning, solve, b.astype(t))
145: 
146:     @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
147:     def test_assume_sorted_indices_flag(self):
148:         # a sparse matrix with unsorted indices
149:         unsorted_inds = np.array([2, 0, 1, 0])
150:         data = np.array([10, 16, 5, 0.4])
151:         indptr = np.array([0, 1, 2, 4])
152:         A = csc_matrix((data, unsorted_inds, indptr), (3, 3))
153:         b = ones(3)
154: 
155:         # should raise when incorrectly assuming indices are sorted
156:         use_solver(useUmfpack=True, assumeSortedIndices=True)
157:         with assert_raises(RuntimeError,
158:                            message="UMFPACK_ERROR_invalid_matrix"):
159:             factorized(A)
160: 
161:         # should sort indices and succeed when not assuming indices are sorted
162:         use_solver(useUmfpack=True, assumeSortedIndices=False)
163:         expected = splu(A.copy()).solve(b)
164: 
165:         assert_equal(A.has_sorted_indices, 0)
166:         assert_array_almost_equal(factorized(A)(b), expected)
167:         assert_equal(A.has_sorted_indices, 1)
168: 
169: 
170: class TestLinsolve(object):
171:     def setup_method(self):
172:         use_solver(useUmfpack=False)
173: 
174:     def test_singular(self):
175:         A = csc_matrix((5,5), dtype='d')
176:         b = array([1, 2, 3, 4, 5],dtype='d')
177:         with suppress_warnings() as sup:
178:             sup.filter(MatrixRankWarning, "Matrix is exactly singular")
179:             x = spsolve(A, b)
180:         assert_(not np.isfinite(x).any())
181: 
182:     def test_singular_gh_3312(self):
183:         # "Bad" test case that leads SuperLU to call LAPACK with invalid
184:         # arguments. Check that it fails moderately gracefully.
185:         ij = np.array([(17, 0), (17, 6), (17, 12), (10, 13)], dtype=np.int32)
186:         v = np.array([0.284213, 0.94933781, 0.15767017, 0.38797296])
187:         A = csc_matrix((v, ij.T), shape=(20, 20))
188:         b = np.arange(20)
189: 
190:         try:
191:             # should either raise a runtimeerror or return value
192:             # appropriate for singular input
193:             x = spsolve(A, b)
194:             assert_(not np.isfinite(x).any())
195:         except RuntimeError:
196:             pass
197: 
198:     def test_twodiags(self):
199:         A = spdiags([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]], [0, 1], 5, 5)
200:         b = array([1, 2, 3, 4, 5])
201: 
202:         # condition number of A
203:         cond_A = norm(A.todense(),2) * norm(inv(A.todense()),2)
204: 
205:         for t in ['f','d','F','D']:
206:             eps = finfo(t).eps  # floating point epsilon
207:             b = b.astype(t)
208: 
209:             for format in ['csc','csr']:
210:                 Asp = A.astype(t).asformat(format)
211: 
212:                 x = spsolve(Asp,b)
213: 
214:                 assert_(norm(b - Asp*x) < 10 * cond_A * eps)
215: 
216:     def test_bvector_smoketest(self):
217:         Adense = matrix([[0., 1., 1.],
218:                          [1., 0., 1.],
219:                          [0., 0., 1.]])
220:         As = csc_matrix(Adense)
221:         random.seed(1234)
222:         x = random.randn(3)
223:         b = As*x
224:         x2 = spsolve(As, b)
225: 
226:         assert_array_almost_equal(x, x2)
227: 
228:     def test_bmatrix_smoketest(self):
229:         Adense = matrix([[0., 1., 1.],
230:                          [1., 0., 1.],
231:                          [0., 0., 1.]])
232:         As = csc_matrix(Adense)
233:         random.seed(1234)
234:         x = random.randn(3, 4)
235:         Bdense = As.dot(x)
236:         Bs = csc_matrix(Bdense)
237:         x2 = spsolve(As, Bs)
238:         assert_array_almost_equal(x, x2.todense())
239: 
240:     @sup_sparse_efficiency
241:     def test_non_square(self):
242:         # A is not square.
243:         A = ones((3, 4))
244:         b = ones((4, 1))
245:         assert_raises(ValueError, spsolve, A, b)
246:         # A2 and b2 have incompatible shapes.
247:         A2 = csc_matrix(eye(3))
248:         b2 = array([1.0, 2.0])
249:         assert_raises(ValueError, spsolve, A2, b2)
250: 
251:     @sup_sparse_efficiency
252:     def test_example_comparison(self):
253:         row = array([0,0,1,2,2,2])
254:         col = array([0,2,2,0,1,2])
255:         data = array([1,2,3,-4,5,6])
256:         sM = csr_matrix((data,(row,col)), shape=(3,3), dtype=float)
257:         M = sM.todense()
258: 
259:         row = array([0,0,1,1,0,0])
260:         col = array([0,2,1,1,0,0])
261:         data = array([1,1,1,1,1,1])
262:         sN = csr_matrix((data, (row,col)), shape=(3,3), dtype=float)
263:         N = sN.todense()
264: 
265:         sX = spsolve(sM, sN)
266:         X = scipy.linalg.solve(M, N)
267: 
268:         assert_array_almost_equal(X, sX.todense())
269: 
270:     @sup_sparse_efficiency
271:     @pytest.mark.skipif(not has_umfpack, reason="umfpack not available")
272:     def test_shape_compatibility(self):
273:         use_solver(useUmfpack=True)
274:         A = csc_matrix([[1., 0], [0, 2]])
275:         bs = [
276:             [1, 6],
277:             array([1, 6]),
278:             [[1], [6]],
279:             array([[1], [6]]),
280:             csc_matrix([[1], [6]]),
281:             csr_matrix([[1], [6]]),
282:             dok_matrix([[1], [6]]),
283:             bsr_matrix([[1], [6]]),
284:             array([[1., 2., 3.], [6., 8., 10.]]),
285:             csc_matrix([[1., 2., 3.], [6., 8., 10.]]),
286:             csr_matrix([[1., 2., 3.], [6., 8., 10.]]),
287:             dok_matrix([[1., 2., 3.], [6., 8., 10.]]),
288:             bsr_matrix([[1., 2., 3.], [6., 8., 10.]]),
289:             ]
290: 
291:         for b in bs:
292:             x = np.linalg.solve(A.toarray(), toarray(b))
293:             for spmattype in [csc_matrix, csr_matrix, dok_matrix, lil_matrix]:
294:                 x1 = spsolve(spmattype(A), b, use_umfpack=True)
295:                 x2 = spsolve(spmattype(A), b, use_umfpack=False)
296: 
297:                 # check solution
298:                 if x.ndim == 2 and x.shape[1] == 1:
299:                     # interprets also these as "vectors"
300:                     x = x.ravel()
301: 
302:                 assert_array_almost_equal(toarray(x1), x, err_msg=repr((b, spmattype, 1)))
303:                 assert_array_almost_equal(toarray(x2), x, err_msg=repr((b, spmattype, 2)))
304: 
305:                 # dense vs. sparse output  ("vectors" are always dense)
306:                 if isspmatrix(b) and x.ndim > 1:
307:                     assert_(isspmatrix(x1), repr((b, spmattype, 1)))
308:                     assert_(isspmatrix(x2), repr((b, spmattype, 2)))
309:                 else:
310:                     assert_(isinstance(x1, np.ndarray), repr((b, spmattype, 1)))
311:                     assert_(isinstance(x2, np.ndarray), repr((b, spmattype, 2)))
312: 
313:                 # check output shape
314:                 if x.ndim == 1:
315:                     # "vector"
316:                     assert_equal(x1.shape, (A.shape[1],))
317:                     assert_equal(x2.shape, (A.shape[1],))
318:                 else:
319:                     # "matrix"
320:                     assert_equal(x1.shape, x.shape)
321:                     assert_equal(x2.shape, x.shape)
322: 
323:         A = csc_matrix((3, 3))
324:         b = csc_matrix((1, 3))
325:         assert_raises(ValueError, spsolve, A, b)
326: 
327:     @sup_sparse_efficiency
328:     def test_ndarray_support(self):
329:         A = array([[1., 2.], [2., 0.]])
330:         x = array([[1., 1.], [0.5, -0.5]])
331:         b = array([[2., 0.], [2., 2.]])
332: 
333:         assert_array_almost_equal(x, spsolve(A, b))
334: 
335:     def test_gssv_badinput(self):
336:         N = 10
337:         d = arange(N) + 1.0
338:         A = spdiags((d, 2*d, d[::-1]), (-3, 0, 5), N, N)
339: 
340:         for spmatrix in (csc_matrix, csr_matrix):
341:             A = spmatrix(A)
342:             b = np.arange(N)
343: 
344:             def not_c_contig(x):
345:                 return x.repeat(2)[::2]
346: 
347:             def not_1dim(x):
348:                 return x[:,None]
349: 
350:             def bad_type(x):
351:                 return x.astype(bool)
352: 
353:             def too_short(x):
354:                 return x[:-1]
355: 
356:             badops = [not_c_contig, not_1dim, bad_type, too_short]
357: 
358:             for badop in badops:
359:                 msg = "%r %r" % (spmatrix, badop)
360:                 # Not C-contiguous
361:                 assert_raises((ValueError, TypeError), _superlu.gssv,
362:                               N, A.nnz, badop(A.data), A.indices, A.indptr,
363:                               b, int(spmatrix == csc_matrix), err_msg=msg)
364:                 assert_raises((ValueError, TypeError), _superlu.gssv,
365:                               N, A.nnz, A.data, badop(A.indices), A.indptr,
366:                               b, int(spmatrix == csc_matrix), err_msg=msg)
367:                 assert_raises((ValueError, TypeError), _superlu.gssv,
368:                               N, A.nnz, A.data, A.indices, badop(A.indptr),
369:                               b, int(spmatrix == csc_matrix), err_msg=msg)
370: 
371:     def test_sparsity_preservation(self):
372:         ident = csc_matrix([
373:             [1, 0, 0],
374:             [0, 1, 0],
375:             [0, 0, 1]])
376:         b = csc_matrix([
377:             [0, 1],
378:             [1, 0],
379:             [0, 0]])
380:         x = spsolve(ident, b)
381:         assert_equal(ident.nnz, 3)
382:         assert_equal(b.nnz, 2)
383:         assert_equal(x.nnz, 2)
384:         assert_allclose(x.A, b.A, atol=1e-12, rtol=1e-12)
385: 
386:     def test_dtype_cast(self):
387:         A_real = scipy.sparse.csr_matrix([[1, 2, 0],
388:                                           [0, 0, 3],
389:                                           [4, 0, 5]])
390:         A_complex = scipy.sparse.csr_matrix([[1, 2, 0],
391:                                              [0, 0, 3],
392:                                              [4, 0, 5 + 1j]])
393:         b_real = np.array([1,1,1])
394:         b_complex = np.array([1,1,1]) + 1j*np.array([1,1,1])
395:         x = spsolve(A_real, b_real)
396:         assert_(np.issubdtype(x.dtype, np.floating))
397:         x = spsolve(A_real, b_complex)
398:         assert_(np.issubdtype(x.dtype, np.complexfloating))
399:         x = spsolve(A_complex, b_real)
400:         assert_(np.issubdtype(x.dtype, np.complexfloating))
401:         x = spsolve(A_complex, b_complex)
402:         assert_(np.issubdtype(x.dtype, np.complexfloating))
403: 
404: 
405: class TestSplu(object):
406:     def setup_method(self):
407:         use_solver(useUmfpack=False)
408:         n = 40
409:         d = arange(n) + 1
410:         self.n = n
411:         self.A = spdiags((d, 2*d, d[::-1]), (-3, 0, 5), n, n)
412:         random.seed(1234)
413: 
414:     def _smoketest(self, spxlu, check, dtype):
415:         if np.issubdtype(dtype, np.complexfloating):
416:             A = self.A + 1j*self.A.T
417:         else:
418:             A = self.A
419: 
420:         A = A.astype(dtype)
421:         lu = spxlu(A)
422: 
423:         rng = random.RandomState(1234)
424: 
425:         # Input shapes
426:         for k in [None, 1, 2, self.n, self.n+2]:
427:             msg = "k=%r" % (k,)
428: 
429:             if k is None:
430:                 b = rng.rand(self.n)
431:             else:
432:                 b = rng.rand(self.n, k)
433: 
434:             if np.issubdtype(dtype, np.complexfloating):
435:                 b = b + 1j*rng.rand(*b.shape)
436:             b = b.astype(dtype)
437: 
438:             x = lu.solve(b)
439:             check(A, b, x, msg)
440: 
441:             x = lu.solve(b, 'T')
442:             check(A.T, b, x, msg)
443: 
444:             x = lu.solve(b, 'H')
445:             check(A.T.conj(), b, x, msg)
446: 
447:     @sup_sparse_efficiency
448:     def test_splu_smoketest(self):
449:         self._internal_test_splu_smoketest()
450: 
451:     def _internal_test_splu_smoketest(self):
452:         # Check that splu works at all
453:         def check(A, b, x, msg=""):
454:             eps = np.finfo(A.dtype).eps
455:             r = A * x
456:             assert_(abs(r - b).max() < 1e3*eps, msg)
457: 
458:         self._smoketest(splu, check, np.float32)
459:         self._smoketest(splu, check, np.float64)
460:         self._smoketest(splu, check, np.complex64)
461:         self._smoketest(splu, check, np.complex128)
462: 
463:     @sup_sparse_efficiency
464:     def test_spilu_smoketest(self):
465:         self._internal_test_spilu_smoketest()
466: 
467:     def _internal_test_spilu_smoketest(self):
468:         errors = []
469: 
470:         def check(A, b, x, msg=""):
471:             r = A * x
472:             err = abs(r - b).max()
473:             assert_(err < 1e-2, msg)
474:             if b.dtype in (np.float64, np.complex128):
475:                 errors.append(err)
476: 
477:         self._smoketest(spilu, check, np.float32)
478:         self._smoketest(spilu, check, np.float64)
479:         self._smoketest(spilu, check, np.complex64)
480:         self._smoketest(spilu, check, np.complex128)
481: 
482:         assert_(max(errors) > 1e-5)
483: 
484:     @sup_sparse_efficiency
485:     def test_spilu_drop_rule(self):
486:         # Test passing in the drop_rule argument to spilu.
487:         A = identity(2)
488: 
489:         rules = [
490:             b'basic,area'.decode('ascii'),  # unicode
491:             b'basic,area',  # ascii
492:             [b'basic', b'area'.decode('ascii')]
493:         ]
494:         for rule in rules:
495:             # Argument should be accepted
496:             assert_(isinstance(spilu(A, drop_rule=rule), SuperLU))
497: 
498:     def test_splu_nnz0(self):
499:         A = csc_matrix((5,5), dtype='d')
500:         assert_raises(RuntimeError, splu, A)
501: 
502:     def test_spilu_nnz0(self):
503:         A = csc_matrix((5,5), dtype='d')
504:         assert_raises(RuntimeError, spilu, A)
505: 
506:     def test_splu_basic(self):
507:         # Test basic splu functionality.
508:         n = 30
509:         rng = random.RandomState(12)
510:         a = rng.rand(n, n)
511:         a[a < 0.95] = 0
512:         # First test with a singular matrix
513:         a[:, 0] = 0
514:         a_ = csc_matrix(a)
515:         # Matrix is exactly singular
516:         assert_raises(RuntimeError, splu, a_)
517: 
518:         # Make a diagonal dominant, to make sure it is not singular
519:         a += 4*eye(n)
520:         a_ = csc_matrix(a)
521:         lu = splu(a_)
522:         b = ones(n)
523:         x = lu.solve(b)
524:         assert_almost_equal(dot(a, x), b)
525: 
526:     def test_splu_perm(self):
527:         # Test the permutation vectors exposed by splu.
528:         n = 30
529:         a = random.random((n, n))
530:         a[a < 0.95] = 0
531:         # Make a diagonal dominant, to make sure it is not singular
532:         a += 4*eye(n)
533:         a_ = csc_matrix(a)
534:         lu = splu(a_)
535:         # Check that the permutation indices do belong to [0, n-1].
536:         for perm in (lu.perm_r, lu.perm_c):
537:             assert_(all(perm > -1))
538:             assert_(all(perm < n))
539:             assert_equal(len(unique(perm)), len(perm))
540: 
541:         # Now make a symmetric, and test that the two permutation vectors are
542:         # the same
543:         # Note: a += a.T relies on undefined behavior.
544:         a = a + a.T
545:         a_ = csc_matrix(a)
546:         lu = splu(a_)
547:         assert_array_equal(lu.perm_r, lu.perm_c)
548: 
549:     def test_lu_refcount(self):
550:         # Test that we are keeping track of the reference count with splu.
551:         n = 30
552:         a = random.random((n, n))
553:         a[a < 0.95] = 0
554:         # Make a diagonal dominant, to make sure it is not singular
555:         a += 4*eye(n)
556:         a_ = csc_matrix(a)
557:         lu = splu(a_)
558: 
559:         # And now test that we don't have a refcount bug
560:         import sys
561:         rc = sys.getrefcount(lu)
562:         for attr in ('perm_r', 'perm_c'):
563:             perm = getattr(lu, attr)
564:             assert_equal(sys.getrefcount(lu), rc + 1)
565:             del perm
566:             assert_equal(sys.getrefcount(lu), rc)
567: 
568:     def test_bad_inputs(self):
569:         A = self.A.tocsc()
570: 
571:         assert_raises(ValueError, splu, A[:,:4])
572:         assert_raises(ValueError, spilu, A[:,:4])
573: 
574:         for lu in [splu(A), spilu(A)]:
575:             b = random.rand(42)
576:             B = random.rand(42, 3)
577:             BB = random.rand(self.n, 3, 9)
578:             assert_raises(ValueError, lu.solve, b)
579:             assert_raises(ValueError, lu.solve, B)
580:             assert_raises(ValueError, lu.solve, BB)
581:             assert_raises(TypeError, lu.solve,
582:                           b.astype(np.complex64))
583:             assert_raises(TypeError, lu.solve,
584:                           b.astype(np.complex128))
585: 
586:     @sup_sparse_efficiency
587:     def test_superlu_dlamch_i386_nan(self):
588:         # SuperLU 4.3 calls some functions returning floats without
589:         # declaring them. On i386@linux call convention, this fails to
590:         # clear floating point registers after call. As a result, NaN
591:         # can appear in the next floating point operation made.
592:         #
593:         # Here's a test case that triggered the issue.
594:         n = 8
595:         d = np.arange(n) + 1
596:         A = spdiags((d, 2*d, d[::-1]), (-3, 0, 5), n, n)
597:         A = A.astype(np.float32)
598:         spilu(A)
599:         A = A + 1j*A
600:         B = A.A
601:         assert_(not np.isnan(B).any())
602: 
603:     @sup_sparse_efficiency
604:     def test_lu_attr(self):
605: 
606:         def check(dtype, complex_2=False):
607:             A = self.A.astype(dtype)
608: 
609:             if complex_2:
610:                 A = A + 1j*A.T
611: 
612:             n = A.shape[0]
613:             lu = splu(A)
614: 
615:             # Check that the decomposition is as advertized
616: 
617:             Pc = np.zeros((n, n))
618:             Pc[np.arange(n), lu.perm_c] = 1
619: 
620:             Pr = np.zeros((n, n))
621:             Pr[lu.perm_r, np.arange(n)] = 1
622: 
623:             Ad = A.toarray()
624:             lhs = Pr.dot(Ad).dot(Pc)
625:             rhs = (lu.L * lu.U).toarray()
626: 
627:             eps = np.finfo(dtype).eps
628: 
629:             assert_allclose(lhs, rhs, atol=100*eps)
630: 
631:         check(np.float32)
632:         check(np.float64)
633:         check(np.complex64)
634:         check(np.complex128)
635:         check(np.complex64, True)
636:         check(np.complex128, True)
637: 
638:     @sup_sparse_efficiency
639:     def test_threads_parallel(self):
640:         oks = []
641: 
642:         def worker():
643:             try:
644:                 self.test_splu_basic()
645:                 self._internal_test_splu_smoketest()
646:                 self._internal_test_spilu_smoketest()
647:                 oks.append(True)
648:             except:
649:                 pass
650: 
651:         threads = [threading.Thread(target=worker)
652:                    for k in range(20)]
653:         for t in threads:
654:             t.start()
655:         for t in threads:
656:             t.join()
657: 
658:         assert_equal(len(oks), 20)
659: 
660: 
661: class TestSpsolveTriangular(object):
662:     def setup_method(self):
663:         use_solver(useUmfpack=False)
664: 
665:     def test_singular(self):
666:         n = 5
667:         A = csr_matrix((n, n))
668:         b = np.arange(n)
669:         for lower in (True, False):
670:             assert_raises(scipy.linalg.LinAlgError, spsolve_triangular, A, b, lower=lower)
671: 
672:     @sup_sparse_efficiency
673:     def test_bad_shape(self):
674:         # A is not square.
675:         A = np.zeros((3, 4))
676:         b = ones((4, 1))
677:         assert_raises(ValueError, spsolve_triangular, A, b)
678:         # A2 and b2 have incompatible shapes.
679:         A2 = csr_matrix(eye(3))
680:         b2 = array([1.0, 2.0])
681:         assert_raises(ValueError, spsolve_triangular, A2, b2)
682: 
683:     @sup_sparse_efficiency
684:     def test_input_types(self):
685:         A = array([[1., 0.], [1., 2.]])
686:         b = array([[2., 0.], [2., 2.]])
687:         for matrix_type in (array, csc_matrix, csr_matrix):
688:             x = spsolve_triangular(matrix_type(A), b, lower=True)
689:             assert_array_almost_equal(A.dot(x), b)
690: 
691:     @sup_sparse_efficiency
692:     def test_random(self):
693:         def random_triangle_matrix(n, lower=True):
694:             A = scipy.sparse.random(n, n, density=0.1, format='coo')
695:             if lower:
696:                 A = scipy.sparse.tril(A)
697:             else:
698:                 A = scipy.sparse.triu(A)
699:             A = A.tocsr(copy=False)
700:             for i in range(n):
701:                 A[i, i] = np.random.rand() + 1
702:             return A
703: 
704:         np.random.seed(1234)
705:         for lower in (True, False):
706:             for n in (10, 10**2, 10**3):
707:                 A = random_triangle_matrix(n, lower=lower)
708:                 for m in (1, 10):
709:                     for b in (np.random.rand(n, m),
710:                               np.random.randint(-9, 9, (n, m)),
711:                               np.random.randint(-9, 9, (n, m)) +
712:                               np.random.randint(-9, 9, (n, m)) * 1j):
713:                         x = spsolve_triangular(A, b, lower=lower)
714:                         assert_array_almost_equal(A.dot(x), b)
715: 
716: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import threading' statement (line 3)
import threading

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'threading', threading, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')
import_392932 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_392932) is not StypyTypeError):

    if (import_392932 != 'pyd_module'):
        __import__(import_392932)
        sys_modules_392933 = sys.modules[import_392932]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_392933.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_392932)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy import array, finfo, arange, eye, all, unique, ones, dot, matrix' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')
import_392934 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_392934) is not StypyTypeError):

    if (import_392934 != 'pyd_module'):
        __import__(import_392934)
        sys_modules_392935 = sys.modules[import_392934]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', sys_modules_392935.module_type_store, module_type_store, ['array', 'finfo', 'arange', 'eye', 'all', 'unique', 'ones', 'dot', 'matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_392935, sys_modules_392935.module_type_store, module_type_store)
    else:
        from numpy import array, finfo, arange, eye, all, unique, ones, dot, matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', None, module_type_store, ['array', 'finfo', 'arange', 'eye', 'all', 'unique', 'ones', 'dot', 'matrix'], [array, finfo, arange, eye, all, unique, ones, dot, matrix])

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_392934)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy.random' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')
import_392936 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.random')

if (type(import_392936) is not StypyTypeError):

    if (import_392936 != 'pyd_module'):
        __import__(import_392936)
        sys_modules_392937 = sys.modules[import_392936]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'random', sys_modules_392937.module_type_store, module_type_store)
    else:
        import numpy.random as random

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'random', numpy.random, module_type_store)

else:
    # Assigning a type to the variable 'numpy.random' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.random', import_392936)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.testing import assert_array_almost_equal, assert_almost_equal, assert_equal, assert_array_equal, assert_, assert_allclose, assert_warns' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')
import_392938 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing')

if (type(import_392938) is not StypyTypeError):

    if (import_392938 != 'pyd_module'):
        __import__(import_392938)
        sys_modules_392939 = sys.modules[import_392938]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', sys_modules_392939.module_type_store, module_type_store, ['assert_array_almost_equal', 'assert_almost_equal', 'assert_equal', 'assert_array_equal', 'assert_', 'assert_allclose', 'assert_warns'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_392939, sys_modules_392939.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal, assert_almost_equal, assert_equal, assert_array_equal, assert_, assert_allclose, assert_warns

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal', 'assert_almost_equal', 'assert_equal', 'assert_array_equal', 'assert_', 'assert_allclose', 'assert_warns'], [assert_array_almost_equal, assert_almost_equal, assert_equal, assert_array_equal, assert_, assert_allclose, assert_warns])

else:
    # Assigning a type to the variable 'numpy.testing' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', import_392938)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import pytest' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')
import_392940 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'pytest')

if (type(import_392940) is not StypyTypeError):

    if (import_392940 != 'pyd_module'):
        __import__(import_392940)
        sys_modules_392941 = sys.modules[import_392940]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'pytest', sys_modules_392941.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'pytest', import_392940)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from pytest import assert_raises' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')
import_392942 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'pytest')

if (type(import_392942) is not StypyTypeError):

    if (import_392942 != 'pyd_module'):
        __import__(import_392942)
        sys_modules_392943 = sys.modules[import_392942]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'pytest', sys_modules_392943.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_392943, sys_modules_392943.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'pytest', import_392942)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import scipy.linalg' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')
import_392944 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg')

if (type(import_392944) is not StypyTypeError):

    if (import_392944 != 'pyd_module'):
        __import__(import_392944)
        sys_modules_392945 = sys.modules[import_392944]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg', sys_modules_392945.module_type_store, module_type_store)
    else:
        import scipy.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg', scipy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.linalg' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg', import_392944)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.linalg import norm, inv' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')
import_392946 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.linalg')

if (type(import_392946) is not StypyTypeError):

    if (import_392946 != 'pyd_module'):
        __import__(import_392946)
        sys_modules_392947 = sys.modules[import_392946]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.linalg', sys_modules_392947.module_type_store, module_type_store, ['norm', 'inv'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_392947, sys_modules_392947.module_type_store, module_type_store)
    else:
        from scipy.linalg import norm, inv

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.linalg', None, module_type_store, ['norm', 'inv'], [norm, inv])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.linalg', import_392946)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.sparse import spdiags, SparseEfficiencyWarning, csc_matrix, csr_matrix, identity, isspmatrix, dok_matrix, lil_matrix, bsr_matrix' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')
import_392948 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse')

if (type(import_392948) is not StypyTypeError):

    if (import_392948 != 'pyd_module'):
        __import__(import_392948)
        sys_modules_392949 = sys.modules[import_392948]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse', sys_modules_392949.module_type_store, module_type_store, ['spdiags', 'SparseEfficiencyWarning', 'csc_matrix', 'csr_matrix', 'identity', 'isspmatrix', 'dok_matrix', 'lil_matrix', 'bsr_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_392949, sys_modules_392949.module_type_store, module_type_store)
    else:
        from scipy.sparse import spdiags, SparseEfficiencyWarning, csc_matrix, csr_matrix, identity, isspmatrix, dok_matrix, lil_matrix, bsr_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse', None, module_type_store, ['spdiags', 'SparseEfficiencyWarning', 'csc_matrix', 'csr_matrix', 'identity', 'isspmatrix', 'dok_matrix', 'lil_matrix', 'bsr_matrix'], [spdiags, SparseEfficiencyWarning, csc_matrix, csr_matrix, identity, isspmatrix, dok_matrix, lil_matrix, bsr_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse', import_392948)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy.sparse.linalg import SuperLU' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')
import_392950 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.sparse.linalg')

if (type(import_392950) is not StypyTypeError):

    if (import_392950 != 'pyd_module'):
        __import__(import_392950)
        sys_modules_392951 = sys.modules[import_392950]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.sparse.linalg', sys_modules_392951.module_type_store, module_type_store, ['SuperLU'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_392951, sys_modules_392951.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import SuperLU

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.sparse.linalg', None, module_type_store, ['SuperLU'], [SuperLU])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.sparse.linalg', import_392950)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy.sparse.linalg.dsolve import spsolve, use_solver, splu, spilu, MatrixRankWarning, _superlu, spsolve_triangular, factorized' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')
import_392952 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.linalg.dsolve')

if (type(import_392952) is not StypyTypeError):

    if (import_392952 != 'pyd_module'):
        __import__(import_392952)
        sys_modules_392953 = sys.modules[import_392952]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.linalg.dsolve', sys_modules_392953.module_type_store, module_type_store, ['spsolve', 'use_solver', 'splu', 'spilu', 'MatrixRankWarning', '_superlu', 'spsolve_triangular', 'factorized'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_392953, sys_modules_392953.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.dsolve import spsolve, use_solver, splu, spilu, MatrixRankWarning, _superlu, spsolve_triangular, factorized

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.linalg.dsolve', None, module_type_store, ['spsolve', 'use_solver', 'splu', 'spilu', 'MatrixRankWarning', '_superlu', 'spsolve_triangular', 'factorized'], [spsolve, use_solver, splu, spilu, MatrixRankWarning, _superlu, spsolve_triangular, factorized])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.dsolve' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.linalg.dsolve', import_392952)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')
import_392954 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy._lib._numpy_compat')

if (type(import_392954) is not StypyTypeError):

    if (import_392954 != 'pyd_module'):
        __import__(import_392954)
        sys_modules_392955 = sys.modules[import_392954]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy._lib._numpy_compat', sys_modules_392955.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_392955, sys_modules_392955.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy._lib._numpy_compat', import_392954)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')


# Assigning a Call to a Name (line 26):

# Call to suppress_warnings(...): (line 26)
# Processing the call keyword arguments (line 26)
kwargs_392957 = {}
# Getting the type of 'suppress_warnings' (line 26)
suppress_warnings_392956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 24), 'suppress_warnings', False)
# Calling suppress_warnings(args, kwargs) (line 26)
suppress_warnings_call_result_392958 = invoke(stypy.reporting.localization.Localization(__file__, 26, 24), suppress_warnings_392956, *[], **kwargs_392957)

# Assigning a type to the variable 'sup_sparse_efficiency' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'sup_sparse_efficiency', suppress_warnings_call_result_392958)

# Call to filter(...): (line 27)
# Processing the call arguments (line 27)
# Getting the type of 'SparseEfficiencyWarning' (line 27)
SparseEfficiencyWarning_392961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), 'SparseEfficiencyWarning', False)
# Processing the call keyword arguments (line 27)
kwargs_392962 = {}
# Getting the type of 'sup_sparse_efficiency' (line 27)
sup_sparse_efficiency_392959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'sup_sparse_efficiency', False)
# Obtaining the member 'filter' of a type (line 27)
filter_392960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 0), sup_sparse_efficiency_392959, 'filter')
# Calling filter(args, kwargs) (line 27)
filter_call_result_392963 = invoke(stypy.reporting.localization.Localization(__file__, 27, 0), filter_392960, *[SparseEfficiencyWarning_392961], **kwargs_392962)



# SSA begins for try-except statement (line 31)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 4))

# 'import scikits.umfpack' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')
import_392964 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 4), 'scikits.umfpack')

if (type(import_392964) is not StypyTypeError):

    if (import_392964 != 'pyd_module'):
        __import__(import_392964)
        sys_modules_392965 = sys.modules[import_392964]
        import_module(stypy.reporting.localization.Localization(__file__, 32, 4), 'umfpack', sys_modules_392965.module_type_store, module_type_store)
    else:
        import scikits.umfpack as umfpack

        import_module(stypy.reporting.localization.Localization(__file__, 32, 4), 'umfpack', scikits.umfpack, module_type_store)

else:
    # Assigning a type to the variable 'scikits.umfpack' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'scikits.umfpack', import_392964)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/tests/')


# Assigning a Name to a Name (line 33):
# Getting the type of 'True' (line 33)
True_392966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 18), 'True')
# Assigning a type to the variable 'has_umfpack' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'has_umfpack', True_392966)
# SSA branch for the except part of a try statement (line 31)
# SSA branch for the except 'ImportError' branch of a try statement (line 31)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 35):
# Getting the type of 'False' (line 35)
False_392967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 'False')
# Assigning a type to the variable 'has_umfpack' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'has_umfpack', False_392967)
# SSA join for try-except statement (line 31)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def toarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'toarray'
    module_type_store = module_type_store.open_function_context('toarray', 37, 0, False)
    
    # Passed parameters checking function
    toarray.stypy_localization = localization
    toarray.stypy_type_of_self = None
    toarray.stypy_type_store = module_type_store
    toarray.stypy_function_name = 'toarray'
    toarray.stypy_param_names_list = ['a']
    toarray.stypy_varargs_param_name = None
    toarray.stypy_kwargs_param_name = None
    toarray.stypy_call_defaults = defaults
    toarray.stypy_call_varargs = varargs
    toarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'toarray', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'toarray', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'toarray(...)' code ##################

    
    
    # Call to isspmatrix(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'a' (line 38)
    a_392969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 18), 'a', False)
    # Processing the call keyword arguments (line 38)
    kwargs_392970 = {}
    # Getting the type of 'isspmatrix' (line 38)
    isspmatrix_392968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 7), 'isspmatrix', False)
    # Calling isspmatrix(args, kwargs) (line 38)
    isspmatrix_call_result_392971 = invoke(stypy.reporting.localization.Localization(__file__, 38, 7), isspmatrix_392968, *[a_392969], **kwargs_392970)
    
    # Testing the type of an if condition (line 38)
    if_condition_392972 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 4), isspmatrix_call_result_392971)
    # Assigning a type to the variable 'if_condition_392972' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'if_condition_392972', if_condition_392972)
    # SSA begins for if statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to toarray(...): (line 39)
    # Processing the call keyword arguments (line 39)
    kwargs_392975 = {}
    # Getting the type of 'a' (line 39)
    a_392973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'a', False)
    # Obtaining the member 'toarray' of a type (line 39)
    toarray_392974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 15), a_392973, 'toarray')
    # Calling toarray(args, kwargs) (line 39)
    toarray_call_result_392976 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), toarray_392974, *[], **kwargs_392975)
    
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', toarray_call_result_392976)
    # SSA branch for the else part of an if statement (line 38)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'a' (line 41)
    a_392977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', a_392977)
    # SSA join for if statement (line 38)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'toarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'toarray' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_392978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_392978)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'toarray'
    return stypy_return_type_392978

# Assigning a type to the variable 'toarray' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'toarray', toarray)
# Declaration of the 'TestFactorized' class

class TestFactorized(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized.setup_method.__dict__.__setitem__('stypy_function_name', 'TestFactorized.setup_method')
        TestFactorized.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Num to a Name (line 46):
        int_392979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 12), 'int')
        # Assigning a type to the variable 'n' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'n', int_392979)
        
        # Assigning a BinOp to a Name (line 47):
        
        # Call to arange(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'n' (line 47)
        n_392981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'n', False)
        # Processing the call keyword arguments (line 47)
        kwargs_392982 = {}
        # Getting the type of 'arange' (line 47)
        arange_392980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 47)
        arange_call_result_392983 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), arange_392980, *[n_392981], **kwargs_392982)
        
        int_392984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 24), 'int')
        # Applying the binary operator '+' (line 47)
        result_add_392985 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 12), '+', arange_call_result_392983, int_392984)
        
        # Assigning a type to the variable 'd' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'd', result_add_392985)
        
        # Assigning a Name to a Attribute (line 48):
        # Getting the type of 'n' (line 48)
        n_392986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'n')
        # Getting the type of 'self' (line 48)
        self_392987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member 'n' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_392987, 'n', n_392986)
        
        # Assigning a Call to a Attribute (line 49):
        
        # Call to tocsc(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_393008 = {}
        
        # Call to spdiags(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Obtaining an instance of the builtin type 'tuple' (line 49)
        tuple_392989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 49)
        # Adding element type (line 49)
        # Getting the type of 'd' (line 49)
        d_392990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 26), tuple_392989, d_392990)
        # Adding element type (line 49)
        int_392991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 29), 'int')
        # Getting the type of 'd' (line 49)
        d_392992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 31), 'd', False)
        # Applying the binary operator '*' (line 49)
        result_mul_392993 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 29), '*', int_392991, d_392992)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 26), tuple_392989, result_mul_392993)
        # Adding element type (line 49)
        
        # Obtaining the type of the subscript
        int_392994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 38), 'int')
        slice_392995 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 49, 34), None, None, int_392994)
        # Getting the type of 'd' (line 49)
        d_392996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'd', False)
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___392997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 34), d_392996, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_392998 = invoke(stypy.reporting.localization.Localization(__file__, 49, 34), getitem___392997, slice_392995)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 26), tuple_392989, subscript_call_result_392998)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 49)
        tuple_392999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 49)
        # Adding element type (line 49)
        int_393000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 45), tuple_392999, int_393000)
        # Adding element type (line 49)
        int_393001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 45), tuple_392999, int_393001)
        # Adding element type (line 49)
        int_393002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 45), tuple_392999, int_393002)
        
        # Getting the type of 'n' (line 49)
        n_393003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 56), 'n', False)
        # Getting the type of 'n' (line 49)
        n_393004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 59), 'n', False)
        # Processing the call keyword arguments (line 49)
        kwargs_393005 = {}
        # Getting the type of 'spdiags' (line 49)
        spdiags_392988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'spdiags', False)
        # Calling spdiags(args, kwargs) (line 49)
        spdiags_call_result_393006 = invoke(stypy.reporting.localization.Localization(__file__, 49, 17), spdiags_392988, *[tuple_392989, tuple_392999, n_393003, n_393004], **kwargs_393005)
        
        # Obtaining the member 'tocsc' of a type (line 49)
        tocsc_393007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 17), spdiags_call_result_393006, 'tocsc')
        # Calling tocsc(args, kwargs) (line 49)
        tocsc_call_result_393009 = invoke(stypy.reporting.localization.Localization(__file__, 49, 17), tocsc_393007, *[], **kwargs_393008)
        
        # Getting the type of 'self' (line 49)
        self_393010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self')
        # Setting the type of the member 'A' of a type (line 49)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_393010, 'A', tocsc_call_result_393009)
        
        # Call to seed(...): (line 50)
        # Processing the call arguments (line 50)
        int_393013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 20), 'int')
        # Processing the call keyword arguments (line 50)
        kwargs_393014 = {}
        # Getting the type of 'random' (line 50)
        random_393011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'random', False)
        # Obtaining the member 'seed' of a type (line 50)
        seed_393012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), random_393011, 'seed')
        # Calling seed(args, kwargs) (line 50)
        seed_call_result_393015 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), seed_393012, *[int_393013], **kwargs_393014)
        
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_393016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393016)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_393016


    @norecursion
    def _check_singular(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_singular'
        module_type_store = module_type_store.open_function_context('_check_singular', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized._check_singular.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized._check_singular.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized._check_singular.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized._check_singular.__dict__.__setitem__('stypy_function_name', 'TestFactorized._check_singular')
        TestFactorized._check_singular.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized._check_singular.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized._check_singular.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized._check_singular.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized._check_singular.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized._check_singular.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized._check_singular.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized._check_singular', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_singular', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_singular(...)' code ##################

        
        # Assigning a Call to a Name (line 53):
        
        # Call to csc_matrix(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_393018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        int_393019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 24), tuple_393018, int_393019)
        # Adding element type (line 53)
        int_393020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 24), tuple_393018, int_393020)
        
        # Processing the call keyword arguments (line 53)
        str_393021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 36), 'str', 'd')
        keyword_393022 = str_393021
        kwargs_393023 = {'dtype': keyword_393022}
        # Getting the type of 'csc_matrix' (line 53)
        csc_matrix_393017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 53)
        csc_matrix_call_result_393024 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), csc_matrix_393017, *[tuple_393018], **kwargs_393023)
        
        # Assigning a type to the variable 'A' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'A', csc_matrix_call_result_393024)
        
        # Assigning a Call to a Name (line 54):
        
        # Call to ones(...): (line 54)
        # Processing the call arguments (line 54)
        int_393026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 17), 'int')
        # Processing the call keyword arguments (line 54)
        kwargs_393027 = {}
        # Getting the type of 'ones' (line 54)
        ones_393025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 54)
        ones_call_result_393028 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), ones_393025, *[int_393026], **kwargs_393027)
        
        # Assigning a type to the variable 'b' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'b', ones_call_result_393028)
        
        # Call to assert_array_almost_equal(...): (line 55)
        # Processing the call arguments (line 55)
        float_393030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'float')
        # Getting the type of 'b' (line 55)
        b_393031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 39), 'b', False)
        # Applying the binary operator '*' (line 55)
        result_mul_393032 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 34), '*', float_393030, b_393031)
        
        
        # Call to (...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'b' (line 55)
        b_393037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 56), 'b', False)
        # Processing the call keyword arguments (line 55)
        kwargs_393038 = {}
        
        # Call to factorized(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'A' (line 55)
        A_393034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 53), 'A', False)
        # Processing the call keyword arguments (line 55)
        kwargs_393035 = {}
        # Getting the type of 'factorized' (line 55)
        factorized_393033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 42), 'factorized', False)
        # Calling factorized(args, kwargs) (line 55)
        factorized_call_result_393036 = invoke(stypy.reporting.localization.Localization(__file__, 55, 42), factorized_393033, *[A_393034], **kwargs_393035)
        
        # Calling (args, kwargs) (line 55)
        _call_result_393039 = invoke(stypy.reporting.localization.Localization(__file__, 55, 42), factorized_call_result_393036, *[b_393037], **kwargs_393038)
        
        # Processing the call keyword arguments (line 55)
        kwargs_393040 = {}
        # Getting the type of 'assert_array_almost_equal' (line 55)
        assert_array_almost_equal_393029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 55)
        assert_array_almost_equal_call_result_393041 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), assert_array_almost_equal_393029, *[result_mul_393032, _call_result_393039], **kwargs_393040)
        
        
        # ################# End of '_check_singular(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_singular' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_393042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393042)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_singular'
        return stypy_return_type_393042


    @norecursion
    def _check_non_singular(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_non_singular'
        module_type_store = module_type_store.open_function_context('_check_non_singular', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized._check_non_singular.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized._check_non_singular.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized._check_non_singular.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized._check_non_singular.__dict__.__setitem__('stypy_function_name', 'TestFactorized._check_non_singular')
        TestFactorized._check_non_singular.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized._check_non_singular.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized._check_non_singular.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized._check_non_singular.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized._check_non_singular.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized._check_non_singular.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized._check_non_singular.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized._check_non_singular', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_non_singular', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_non_singular(...)' code ##################

        
        # Assigning a Num to a Name (line 59):
        int_393043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 12), 'int')
        # Assigning a type to the variable 'n' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'n', int_393043)
        
        # Assigning a Call to a Name (line 60):
        
        # Call to csc_matrix(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to rand(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'n' (line 60)
        n_393047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 35), 'n', False)
        # Getting the type of 'n' (line 60)
        n_393048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'n', False)
        # Processing the call keyword arguments (line 60)
        kwargs_393049 = {}
        # Getting the type of 'random' (line 60)
        random_393045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'random', False)
        # Obtaining the member 'rand' of a type (line 60)
        rand_393046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 23), random_393045, 'rand')
        # Calling rand(args, kwargs) (line 60)
        rand_call_result_393050 = invoke(stypy.reporting.localization.Localization(__file__, 60, 23), rand_393046, *[n_393047, n_393048], **kwargs_393049)
        
        # Processing the call keyword arguments (line 60)
        kwargs_393051 = {}
        # Getting the type of 'csc_matrix' (line 60)
        csc_matrix_393044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 60)
        csc_matrix_call_result_393052 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), csc_matrix_393044, *[rand_call_result_393050], **kwargs_393051)
        
        # Assigning a type to the variable 'a' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'a', csc_matrix_call_result_393052)
        
        # Assigning a Call to a Name (line 61):
        
        # Call to ones(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'n' (line 61)
        n_393054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'n', False)
        # Processing the call keyword arguments (line 61)
        kwargs_393055 = {}
        # Getting the type of 'ones' (line 61)
        ones_393053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 61)
        ones_call_result_393056 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), ones_393053, *[n_393054], **kwargs_393055)
        
        # Assigning a type to the variable 'b' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'b', ones_call_result_393056)
        
        # Assigning a Call to a Name (line 63):
        
        # Call to solve(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'b' (line 63)
        b_393062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 33), 'b', False)
        # Processing the call keyword arguments (line 63)
        kwargs_393063 = {}
        
        # Call to splu(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'a' (line 63)
        a_393058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'a', False)
        # Processing the call keyword arguments (line 63)
        kwargs_393059 = {}
        # Getting the type of 'splu' (line 63)
        splu_393057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'splu', False)
        # Calling splu(args, kwargs) (line 63)
        splu_call_result_393060 = invoke(stypy.reporting.localization.Localization(__file__, 63, 19), splu_393057, *[a_393058], **kwargs_393059)
        
        # Obtaining the member 'solve' of a type (line 63)
        solve_393061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 19), splu_call_result_393060, 'solve')
        # Calling solve(args, kwargs) (line 63)
        solve_call_result_393064 = invoke(stypy.reporting.localization.Localization(__file__, 63, 19), solve_393061, *[b_393062], **kwargs_393063)
        
        # Assigning a type to the variable 'expected' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'expected', solve_call_result_393064)
        
        # Call to assert_array_almost_equal(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to (...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'b' (line 64)
        b_393070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 48), 'b', False)
        # Processing the call keyword arguments (line 64)
        kwargs_393071 = {}
        
        # Call to factorized(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'a' (line 64)
        a_393067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 45), 'a', False)
        # Processing the call keyword arguments (line 64)
        kwargs_393068 = {}
        # Getting the type of 'factorized' (line 64)
        factorized_393066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'factorized', False)
        # Calling factorized(args, kwargs) (line 64)
        factorized_call_result_393069 = invoke(stypy.reporting.localization.Localization(__file__, 64, 34), factorized_393066, *[a_393067], **kwargs_393068)
        
        # Calling (args, kwargs) (line 64)
        _call_result_393072 = invoke(stypy.reporting.localization.Localization(__file__, 64, 34), factorized_call_result_393069, *[b_393070], **kwargs_393071)
        
        # Getting the type of 'expected' (line 64)
        expected_393073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 52), 'expected', False)
        # Processing the call keyword arguments (line 64)
        kwargs_393074 = {}
        # Getting the type of 'assert_array_almost_equal' (line 64)
        assert_array_almost_equal_393065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 64)
        assert_array_almost_equal_call_result_393075 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assert_array_almost_equal_393065, *[_call_result_393072, expected_393073], **kwargs_393074)
        
        
        # ################# End of '_check_non_singular(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_non_singular' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_393076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393076)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_non_singular'
        return stypy_return_type_393076


    @norecursion
    def test_singular_without_umfpack(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_singular_without_umfpack'
        module_type_store = module_type_store.open_function_context('test_singular_without_umfpack', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized.test_singular_without_umfpack.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized.test_singular_without_umfpack.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized.test_singular_without_umfpack.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized.test_singular_without_umfpack.__dict__.__setitem__('stypy_function_name', 'TestFactorized.test_singular_without_umfpack')
        TestFactorized.test_singular_without_umfpack.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized.test_singular_without_umfpack.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized.test_singular_without_umfpack.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized.test_singular_without_umfpack.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized.test_singular_without_umfpack.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized.test_singular_without_umfpack.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized.test_singular_without_umfpack.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized.test_singular_without_umfpack', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_singular_without_umfpack', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_singular_without_umfpack(...)' code ##################

        
        # Call to use_solver(...): (line 67)
        # Processing the call keyword arguments (line 67)
        # Getting the type of 'False' (line 67)
        False_393078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 30), 'False', False)
        keyword_393079 = False_393078
        kwargs_393080 = {'useUmfpack': keyword_393079}
        # Getting the type of 'use_solver' (line 67)
        use_solver_393077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 67)
        use_solver_call_result_393081 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), use_solver_393077, *[], **kwargs_393080)
        
        
        # Call to assert_raises(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'RuntimeError' (line 68)
        RuntimeError_393083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'RuntimeError', False)
        # Processing the call keyword arguments (line 68)
        str_393084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 49), 'str', 'Factor is exactly singular')
        keyword_393085 = str_393084
        kwargs_393086 = {'message': keyword_393085}
        # Getting the type of 'assert_raises' (line 68)
        assert_raises_393082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 68)
        assert_raises_call_result_393087 = invoke(stypy.reporting.localization.Localization(__file__, 68, 13), assert_raises_393082, *[RuntimeError_393083], **kwargs_393086)
        
        with_393088 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 68, 13), assert_raises_call_result_393087, 'with parameter', '__enter__', '__exit__')

        if with_393088:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 68)
            enter___393089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 13), assert_raises_call_result_393087, '__enter__')
            with_enter_393090 = invoke(stypy.reporting.localization.Localization(__file__, 68, 13), enter___393089)
            
            # Call to _check_singular(...): (line 69)
            # Processing the call keyword arguments (line 69)
            kwargs_393093 = {}
            # Getting the type of 'self' (line 69)
            self_393091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'self', False)
            # Obtaining the member '_check_singular' of a type (line 69)
            _check_singular_393092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), self_393091, '_check_singular')
            # Calling _check_singular(args, kwargs) (line 69)
            _check_singular_call_result_393094 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), _check_singular_393092, *[], **kwargs_393093)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 68)
            exit___393095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 13), assert_raises_call_result_393087, '__exit__')
            with_exit_393096 = invoke(stypy.reporting.localization.Localization(__file__, 68, 13), exit___393095, None, None, None)

        
        # ################# End of 'test_singular_without_umfpack(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_singular_without_umfpack' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_393097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393097)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_singular_without_umfpack'
        return stypy_return_type_393097


    @norecursion
    def test_singular_with_umfpack(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_singular_with_umfpack'
        module_type_store = module_type_store.open_function_context('test_singular_with_umfpack', 71, 4, False)
        # Assigning a type to the variable 'self' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized.test_singular_with_umfpack.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized.test_singular_with_umfpack.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized.test_singular_with_umfpack.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized.test_singular_with_umfpack.__dict__.__setitem__('stypy_function_name', 'TestFactorized.test_singular_with_umfpack')
        TestFactorized.test_singular_with_umfpack.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized.test_singular_with_umfpack.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized.test_singular_with_umfpack.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized.test_singular_with_umfpack.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized.test_singular_with_umfpack.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized.test_singular_with_umfpack.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized.test_singular_with_umfpack.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized.test_singular_with_umfpack', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_singular_with_umfpack', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_singular_with_umfpack(...)' code ##################

        
        # Call to use_solver(...): (line 73)
        # Processing the call keyword arguments (line 73)
        # Getting the type of 'True' (line 73)
        True_393099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'True', False)
        keyword_393100 = True_393099
        kwargs_393101 = {'useUmfpack': keyword_393100}
        # Getting the type of 'use_solver' (line 73)
        use_solver_393098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 73)
        use_solver_call_result_393102 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), use_solver_393098, *[], **kwargs_393101)
        
        
        # Call to suppress_warnings(...): (line 74)
        # Processing the call keyword arguments (line 74)
        kwargs_393104 = {}
        # Getting the type of 'suppress_warnings' (line 74)
        suppress_warnings_393103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 74)
        suppress_warnings_call_result_393105 = invoke(stypy.reporting.localization.Localization(__file__, 74, 13), suppress_warnings_393103, *[], **kwargs_393104)
        
        with_393106 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 74, 13), suppress_warnings_call_result_393105, 'with parameter', '__enter__', '__exit__')

        if with_393106:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 74)
            enter___393107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), suppress_warnings_call_result_393105, '__enter__')
            with_enter_393108 = invoke(stypy.reporting.localization.Localization(__file__, 74, 13), enter___393107)
            # Assigning a type to the variable 'sup' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'sup', with_enter_393108)
            
            # Call to filter(...): (line 75)
            # Processing the call arguments (line 75)
            # Getting the type of 'RuntimeWarning' (line 75)
            RuntimeWarning_393111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'RuntimeWarning', False)
            str_393112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 39), 'str', 'divide by zero encountered in double_scalars')
            # Processing the call keyword arguments (line 75)
            kwargs_393113 = {}
            # Getting the type of 'sup' (line 75)
            sup_393109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 75)
            filter_393110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), sup_393109, 'filter')
            # Calling filter(args, kwargs) (line 75)
            filter_call_result_393114 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), filter_393110, *[RuntimeWarning_393111, str_393112], **kwargs_393113)
            
            
            # Call to assert_warns(...): (line 76)
            # Processing the call arguments (line 76)
            # Getting the type of 'umfpack' (line 76)
            umfpack_393116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'umfpack', False)
            # Obtaining the member 'UmfpackWarning' of a type (line 76)
            UmfpackWarning_393117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 25), umfpack_393116, 'UmfpackWarning')
            # Getting the type of 'self' (line 76)
            self_393118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 49), 'self', False)
            # Obtaining the member '_check_singular' of a type (line 76)
            _check_singular_393119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 49), self_393118, '_check_singular')
            # Processing the call keyword arguments (line 76)
            kwargs_393120 = {}
            # Getting the type of 'assert_warns' (line 76)
            assert_warns_393115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'assert_warns', False)
            # Calling assert_warns(args, kwargs) (line 76)
            assert_warns_call_result_393121 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), assert_warns_393115, *[UmfpackWarning_393117, _check_singular_393119], **kwargs_393120)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 74)
            exit___393122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), suppress_warnings_call_result_393105, '__exit__')
            with_exit_393123 = invoke(stypy.reporting.localization.Localization(__file__, 74, 13), exit___393122, None, None, None)

        
        # ################# End of 'test_singular_with_umfpack(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_singular_with_umfpack' in the type store
        # Getting the type of 'stypy_return_type' (line 71)
        stypy_return_type_393124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393124)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_singular_with_umfpack'
        return stypy_return_type_393124


    @norecursion
    def test_non_singular_without_umfpack(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_non_singular_without_umfpack'
        module_type_store = module_type_store.open_function_context('test_non_singular_without_umfpack', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized.test_non_singular_without_umfpack.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized.test_non_singular_without_umfpack.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized.test_non_singular_without_umfpack.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized.test_non_singular_without_umfpack.__dict__.__setitem__('stypy_function_name', 'TestFactorized.test_non_singular_without_umfpack')
        TestFactorized.test_non_singular_without_umfpack.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized.test_non_singular_without_umfpack.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized.test_non_singular_without_umfpack.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized.test_non_singular_without_umfpack.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized.test_non_singular_without_umfpack.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized.test_non_singular_without_umfpack.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized.test_non_singular_without_umfpack.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized.test_non_singular_without_umfpack', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_non_singular_without_umfpack', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_non_singular_without_umfpack(...)' code ##################

        
        # Call to use_solver(...): (line 79)
        # Processing the call keyword arguments (line 79)
        # Getting the type of 'False' (line 79)
        False_393126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'False', False)
        keyword_393127 = False_393126
        kwargs_393128 = {'useUmfpack': keyword_393127}
        # Getting the type of 'use_solver' (line 79)
        use_solver_393125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 79)
        use_solver_call_result_393129 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), use_solver_393125, *[], **kwargs_393128)
        
        
        # Call to _check_non_singular(...): (line 80)
        # Processing the call keyword arguments (line 80)
        kwargs_393132 = {}
        # Getting the type of 'self' (line 80)
        self_393130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self', False)
        # Obtaining the member '_check_non_singular' of a type (line 80)
        _check_non_singular_393131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_393130, '_check_non_singular')
        # Calling _check_non_singular(args, kwargs) (line 80)
        _check_non_singular_call_result_393133 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), _check_non_singular_393131, *[], **kwargs_393132)
        
        
        # ################# End of 'test_non_singular_without_umfpack(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_non_singular_without_umfpack' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_393134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393134)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_non_singular_without_umfpack'
        return stypy_return_type_393134


    @norecursion
    def test_non_singular_with_umfpack(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_non_singular_with_umfpack'
        module_type_store = module_type_store.open_function_context('test_non_singular_with_umfpack', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized.test_non_singular_with_umfpack.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized.test_non_singular_with_umfpack.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized.test_non_singular_with_umfpack.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized.test_non_singular_with_umfpack.__dict__.__setitem__('stypy_function_name', 'TestFactorized.test_non_singular_with_umfpack')
        TestFactorized.test_non_singular_with_umfpack.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized.test_non_singular_with_umfpack.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized.test_non_singular_with_umfpack.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized.test_non_singular_with_umfpack.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized.test_non_singular_with_umfpack.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized.test_non_singular_with_umfpack.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized.test_non_singular_with_umfpack.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized.test_non_singular_with_umfpack', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_non_singular_with_umfpack', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_non_singular_with_umfpack(...)' code ##################

        
        # Call to use_solver(...): (line 84)
        # Processing the call keyword arguments (line 84)
        # Getting the type of 'True' (line 84)
        True_393136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'True', False)
        keyword_393137 = True_393136
        kwargs_393138 = {'useUmfpack': keyword_393137}
        # Getting the type of 'use_solver' (line 84)
        use_solver_393135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 84)
        use_solver_call_result_393139 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), use_solver_393135, *[], **kwargs_393138)
        
        
        # Call to _check_non_singular(...): (line 85)
        # Processing the call keyword arguments (line 85)
        kwargs_393142 = {}
        # Getting the type of 'self' (line 85)
        self_393140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self', False)
        # Obtaining the member '_check_non_singular' of a type (line 85)
        _check_non_singular_393141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_393140, '_check_non_singular')
        # Calling _check_non_singular(args, kwargs) (line 85)
        _check_non_singular_call_result_393143 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), _check_non_singular_393141, *[], **kwargs_393142)
        
        
        # ################# End of 'test_non_singular_with_umfpack(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_non_singular_with_umfpack' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_393144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393144)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_non_singular_with_umfpack'
        return stypy_return_type_393144


    @norecursion
    def test_cannot_factorize_nonsquare_matrix_without_umfpack(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cannot_factorize_nonsquare_matrix_without_umfpack'
        module_type_store = module_type_store.open_function_context('test_cannot_factorize_nonsquare_matrix_without_umfpack', 87, 4, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized.test_cannot_factorize_nonsquare_matrix_without_umfpack.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized.test_cannot_factorize_nonsquare_matrix_without_umfpack.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized.test_cannot_factorize_nonsquare_matrix_without_umfpack.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized.test_cannot_factorize_nonsquare_matrix_without_umfpack.__dict__.__setitem__('stypy_function_name', 'TestFactorized.test_cannot_factorize_nonsquare_matrix_without_umfpack')
        TestFactorized.test_cannot_factorize_nonsquare_matrix_without_umfpack.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized.test_cannot_factorize_nonsquare_matrix_without_umfpack.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized.test_cannot_factorize_nonsquare_matrix_without_umfpack.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized.test_cannot_factorize_nonsquare_matrix_without_umfpack.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized.test_cannot_factorize_nonsquare_matrix_without_umfpack.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized.test_cannot_factorize_nonsquare_matrix_without_umfpack.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized.test_cannot_factorize_nonsquare_matrix_without_umfpack.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized.test_cannot_factorize_nonsquare_matrix_without_umfpack', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cannot_factorize_nonsquare_matrix_without_umfpack', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cannot_factorize_nonsquare_matrix_without_umfpack(...)' code ##################

        
        # Call to use_solver(...): (line 88)
        # Processing the call keyword arguments (line 88)
        # Getting the type of 'False' (line 88)
        False_393146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 30), 'False', False)
        keyword_393147 = False_393146
        kwargs_393148 = {'useUmfpack': keyword_393147}
        # Getting the type of 'use_solver' (line 88)
        use_solver_393145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 88)
        use_solver_call_result_393149 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), use_solver_393145, *[], **kwargs_393148)
        
        
        # Assigning a Str to a Name (line 89):
        str_393150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 14), 'str', 'can only factor square matrices')
        # Assigning a type to the variable 'msg' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'msg', str_393150)
        
        # Call to assert_raises(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'ValueError' (line 90)
        ValueError_393152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 90)
        # Getting the type of 'msg' (line 90)
        msg_393153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 47), 'msg', False)
        keyword_393154 = msg_393153
        kwargs_393155 = {'message': keyword_393154}
        # Getting the type of 'assert_raises' (line 90)
        assert_raises_393151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 90)
        assert_raises_call_result_393156 = invoke(stypy.reporting.localization.Localization(__file__, 90, 13), assert_raises_393151, *[ValueError_393152], **kwargs_393155)
        
        with_393157 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 90, 13), assert_raises_call_result_393156, 'with parameter', '__enter__', '__exit__')

        if with_393157:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 90)
            enter___393158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 13), assert_raises_call_result_393156, '__enter__')
            with_enter_393159 = invoke(stypy.reporting.localization.Localization(__file__, 90, 13), enter___393158)
            
            # Call to factorized(...): (line 91)
            # Processing the call arguments (line 91)
            
            # Obtaining the type of the subscript
            slice_393161 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 91, 23), None, None, None)
            int_393162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 34), 'int')
            slice_393163 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 91, 23), None, int_393162, None)
            # Getting the type of 'self' (line 91)
            self_393164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 23), 'self', False)
            # Obtaining the member 'A' of a type (line 91)
            A_393165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 23), self_393164, 'A')
            # Obtaining the member '__getitem__' of a type (line 91)
            getitem___393166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 23), A_393165, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 91)
            subscript_call_result_393167 = invoke(stypy.reporting.localization.Localization(__file__, 91, 23), getitem___393166, (slice_393161, slice_393163))
            
            # Processing the call keyword arguments (line 91)
            kwargs_393168 = {}
            # Getting the type of 'factorized' (line 91)
            factorized_393160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'factorized', False)
            # Calling factorized(args, kwargs) (line 91)
            factorized_call_result_393169 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), factorized_393160, *[subscript_call_result_393167], **kwargs_393168)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 90)
            exit___393170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 13), assert_raises_call_result_393156, '__exit__')
            with_exit_393171 = invoke(stypy.reporting.localization.Localization(__file__, 90, 13), exit___393170, None, None, None)

        
        # ################# End of 'test_cannot_factorize_nonsquare_matrix_without_umfpack(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cannot_factorize_nonsquare_matrix_without_umfpack' in the type store
        # Getting the type of 'stypy_return_type' (line 87)
        stypy_return_type_393172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393172)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cannot_factorize_nonsquare_matrix_without_umfpack'
        return stypy_return_type_393172


    @norecursion
    def test_factorizes_nonsquare_matrix_with_umfpack(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_factorizes_nonsquare_matrix_with_umfpack'
        module_type_store = module_type_store.open_function_context('test_factorizes_nonsquare_matrix_with_umfpack', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized.test_factorizes_nonsquare_matrix_with_umfpack.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized.test_factorizes_nonsquare_matrix_with_umfpack.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized.test_factorizes_nonsquare_matrix_with_umfpack.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized.test_factorizes_nonsquare_matrix_with_umfpack.__dict__.__setitem__('stypy_function_name', 'TestFactorized.test_factorizes_nonsquare_matrix_with_umfpack')
        TestFactorized.test_factorizes_nonsquare_matrix_with_umfpack.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized.test_factorizes_nonsquare_matrix_with_umfpack.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized.test_factorizes_nonsquare_matrix_with_umfpack.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized.test_factorizes_nonsquare_matrix_with_umfpack.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized.test_factorizes_nonsquare_matrix_with_umfpack.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized.test_factorizes_nonsquare_matrix_with_umfpack.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized.test_factorizes_nonsquare_matrix_with_umfpack.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized.test_factorizes_nonsquare_matrix_with_umfpack', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_factorizes_nonsquare_matrix_with_umfpack', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_factorizes_nonsquare_matrix_with_umfpack(...)' code ##################

        
        # Call to use_solver(...): (line 95)
        # Processing the call keyword arguments (line 95)
        # Getting the type of 'True' (line 95)
        True_393174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 30), 'True', False)
        keyword_393175 = True_393174
        kwargs_393176 = {'useUmfpack': keyword_393175}
        # Getting the type of 'use_solver' (line 95)
        use_solver_393173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 95)
        use_solver_call_result_393177 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), use_solver_393173, *[], **kwargs_393176)
        
        
        # Call to factorized(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Obtaining the type of the subscript
        slice_393179 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 19), None, None, None)
        int_393180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 29), 'int')
        slice_393181 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 19), None, int_393180, None)
        # Getting the type of 'self' (line 97)
        self_393182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'self', False)
        # Obtaining the member 'A' of a type (line 97)
        A_393183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 19), self_393182, 'A')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___393184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 19), A_393183, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_393185 = invoke(stypy.reporting.localization.Localization(__file__, 97, 19), getitem___393184, (slice_393179, slice_393181))
        
        # Processing the call keyword arguments (line 97)
        kwargs_393186 = {}
        # Getting the type of 'factorized' (line 97)
        factorized_393178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'factorized', False)
        # Calling factorized(args, kwargs) (line 97)
        factorized_call_result_393187 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), factorized_393178, *[subscript_call_result_393185], **kwargs_393186)
        
        
        # ################# End of 'test_factorizes_nonsquare_matrix_with_umfpack(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_factorizes_nonsquare_matrix_with_umfpack' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_393188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393188)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_factorizes_nonsquare_matrix_with_umfpack'
        return stypy_return_type_393188


    @norecursion
    def test_call_with_incorrectly_sized_matrix_without_umfpack(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_call_with_incorrectly_sized_matrix_without_umfpack'
        module_type_store = module_type_store.open_function_context('test_call_with_incorrectly_sized_matrix_without_umfpack', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized.test_call_with_incorrectly_sized_matrix_without_umfpack.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized.test_call_with_incorrectly_sized_matrix_without_umfpack.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized.test_call_with_incorrectly_sized_matrix_without_umfpack.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized.test_call_with_incorrectly_sized_matrix_without_umfpack.__dict__.__setitem__('stypy_function_name', 'TestFactorized.test_call_with_incorrectly_sized_matrix_without_umfpack')
        TestFactorized.test_call_with_incorrectly_sized_matrix_without_umfpack.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized.test_call_with_incorrectly_sized_matrix_without_umfpack.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized.test_call_with_incorrectly_sized_matrix_without_umfpack.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized.test_call_with_incorrectly_sized_matrix_without_umfpack.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized.test_call_with_incorrectly_sized_matrix_without_umfpack.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized.test_call_with_incorrectly_sized_matrix_without_umfpack.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized.test_call_with_incorrectly_sized_matrix_without_umfpack.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized.test_call_with_incorrectly_sized_matrix_without_umfpack', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_call_with_incorrectly_sized_matrix_without_umfpack', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_call_with_incorrectly_sized_matrix_without_umfpack(...)' code ##################

        
        # Call to use_solver(...): (line 100)
        # Processing the call keyword arguments (line 100)
        # Getting the type of 'False' (line 100)
        False_393190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'False', False)
        keyword_393191 = False_393190
        kwargs_393192 = {'useUmfpack': keyword_393191}
        # Getting the type of 'use_solver' (line 100)
        use_solver_393189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 100)
        use_solver_call_result_393193 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), use_solver_393189, *[], **kwargs_393192)
        
        
        # Assigning a Call to a Name (line 101):
        
        # Call to factorized(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'self' (line 101)
        self_393195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'self', False)
        # Obtaining the member 'A' of a type (line 101)
        A_393196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 27), self_393195, 'A')
        # Processing the call keyword arguments (line 101)
        kwargs_393197 = {}
        # Getting the type of 'factorized' (line 101)
        factorized_393194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'factorized', False)
        # Calling factorized(args, kwargs) (line 101)
        factorized_call_result_393198 = invoke(stypy.reporting.localization.Localization(__file__, 101, 16), factorized_393194, *[A_393196], **kwargs_393197)
        
        # Assigning a type to the variable 'solve' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'solve', factorized_call_result_393198)
        
        # Assigning a Call to a Name (line 102):
        
        # Call to rand(...): (line 102)
        # Processing the call arguments (line 102)
        int_393201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 24), 'int')
        # Processing the call keyword arguments (line 102)
        kwargs_393202 = {}
        # Getting the type of 'random' (line 102)
        random_393199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'random', False)
        # Obtaining the member 'rand' of a type (line 102)
        rand_393200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), random_393199, 'rand')
        # Calling rand(args, kwargs) (line 102)
        rand_call_result_393203 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), rand_393200, *[int_393201], **kwargs_393202)
        
        # Assigning a type to the variable 'b' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'b', rand_call_result_393203)
        
        # Assigning a Call to a Name (line 103):
        
        # Call to rand(...): (line 103)
        # Processing the call arguments (line 103)
        int_393206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 24), 'int')
        int_393207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 27), 'int')
        # Processing the call keyword arguments (line 103)
        kwargs_393208 = {}
        # Getting the type of 'random' (line 103)
        random_393204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'random', False)
        # Obtaining the member 'rand' of a type (line 103)
        rand_393205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), random_393204, 'rand')
        # Calling rand(args, kwargs) (line 103)
        rand_call_result_393209 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), rand_393205, *[int_393206, int_393207], **kwargs_393208)
        
        # Assigning a type to the variable 'B' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'B', rand_call_result_393209)
        
        # Assigning a Call to a Name (line 104):
        
        # Call to rand(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'self' (line 104)
        self_393212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'self', False)
        # Obtaining the member 'n' of a type (line 104)
        n_393213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 25), self_393212, 'n')
        int_393214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 33), 'int')
        int_393215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 36), 'int')
        # Processing the call keyword arguments (line 104)
        kwargs_393216 = {}
        # Getting the type of 'random' (line 104)
        random_393210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'random', False)
        # Obtaining the member 'rand' of a type (line 104)
        rand_393211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 13), random_393210, 'rand')
        # Calling rand(args, kwargs) (line 104)
        rand_call_result_393217 = invoke(stypy.reporting.localization.Localization(__file__, 104, 13), rand_393211, *[n_393213, int_393214, int_393215], **kwargs_393216)
        
        # Assigning a type to the variable 'BB' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'BB', rand_call_result_393217)
        
        # Call to assert_raises(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'ValueError' (line 106)
        ValueError_393219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 106)
        str_393220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 47), 'str', 'is of incompatible size')
        keyword_393221 = str_393220
        kwargs_393222 = {'message': keyword_393221}
        # Getting the type of 'assert_raises' (line 106)
        assert_raises_393218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 106)
        assert_raises_call_result_393223 = invoke(stypy.reporting.localization.Localization(__file__, 106, 13), assert_raises_393218, *[ValueError_393219], **kwargs_393222)
        
        with_393224 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 106, 13), assert_raises_call_result_393223, 'with parameter', '__enter__', '__exit__')

        if with_393224:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 106)
            enter___393225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 13), assert_raises_call_result_393223, '__enter__')
            with_enter_393226 = invoke(stypy.reporting.localization.Localization(__file__, 106, 13), enter___393225)
            
            # Call to solve(...): (line 107)
            # Processing the call arguments (line 107)
            # Getting the type of 'b' (line 107)
            b_393228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 18), 'b', False)
            # Processing the call keyword arguments (line 107)
            kwargs_393229 = {}
            # Getting the type of 'solve' (line 107)
            solve_393227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'solve', False)
            # Calling solve(args, kwargs) (line 107)
            solve_call_result_393230 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), solve_393227, *[b_393228], **kwargs_393229)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 106)
            exit___393231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 13), assert_raises_call_result_393223, '__exit__')
            with_exit_393232 = invoke(stypy.reporting.localization.Localization(__file__, 106, 13), exit___393231, None, None, None)

        
        # Call to assert_raises(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'ValueError' (line 108)
        ValueError_393234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 108)
        str_393235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 47), 'str', 'is of incompatible size')
        keyword_393236 = str_393235
        kwargs_393237 = {'message': keyword_393236}
        # Getting the type of 'assert_raises' (line 108)
        assert_raises_393233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 108)
        assert_raises_call_result_393238 = invoke(stypy.reporting.localization.Localization(__file__, 108, 13), assert_raises_393233, *[ValueError_393234], **kwargs_393237)
        
        with_393239 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 108, 13), assert_raises_call_result_393238, 'with parameter', '__enter__', '__exit__')

        if with_393239:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 108)
            enter___393240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 13), assert_raises_call_result_393238, '__enter__')
            with_enter_393241 = invoke(stypy.reporting.localization.Localization(__file__, 108, 13), enter___393240)
            
            # Call to solve(...): (line 109)
            # Processing the call arguments (line 109)
            # Getting the type of 'B' (line 109)
            B_393243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 18), 'B', False)
            # Processing the call keyword arguments (line 109)
            kwargs_393244 = {}
            # Getting the type of 'solve' (line 109)
            solve_393242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'solve', False)
            # Calling solve(args, kwargs) (line 109)
            solve_call_result_393245 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), solve_393242, *[B_393243], **kwargs_393244)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 108)
            exit___393246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 13), assert_raises_call_result_393238, '__exit__')
            with_exit_393247 = invoke(stypy.reporting.localization.Localization(__file__, 108, 13), exit___393246, None, None, None)

        
        # Call to assert_raises(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'ValueError' (line 110)
        ValueError_393249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 110)
        str_393250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 35), 'str', 'object too deep for desired array')
        keyword_393251 = str_393250
        kwargs_393252 = {'message': keyword_393251}
        # Getting the type of 'assert_raises' (line 110)
        assert_raises_393248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 110)
        assert_raises_call_result_393253 = invoke(stypy.reporting.localization.Localization(__file__, 110, 13), assert_raises_393248, *[ValueError_393249], **kwargs_393252)
        
        with_393254 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 110, 13), assert_raises_call_result_393253, 'with parameter', '__enter__', '__exit__')

        if with_393254:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 110)
            enter___393255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 13), assert_raises_call_result_393253, '__enter__')
            with_enter_393256 = invoke(stypy.reporting.localization.Localization(__file__, 110, 13), enter___393255)
            
            # Call to solve(...): (line 112)
            # Processing the call arguments (line 112)
            # Getting the type of 'BB' (line 112)
            BB_393258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'BB', False)
            # Processing the call keyword arguments (line 112)
            kwargs_393259 = {}
            # Getting the type of 'solve' (line 112)
            solve_393257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'solve', False)
            # Calling solve(args, kwargs) (line 112)
            solve_call_result_393260 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), solve_393257, *[BB_393258], **kwargs_393259)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 110)
            exit___393261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 13), assert_raises_call_result_393253, '__exit__')
            with_exit_393262 = invoke(stypy.reporting.localization.Localization(__file__, 110, 13), exit___393261, None, None, None)

        
        # ################# End of 'test_call_with_incorrectly_sized_matrix_without_umfpack(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_call_with_incorrectly_sized_matrix_without_umfpack' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_393263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393263)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_call_with_incorrectly_sized_matrix_without_umfpack'
        return stypy_return_type_393263


    @norecursion
    def test_call_with_incorrectly_sized_matrix_with_umfpack(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_call_with_incorrectly_sized_matrix_with_umfpack'
        module_type_store = module_type_store.open_function_context('test_call_with_incorrectly_sized_matrix_with_umfpack', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized.test_call_with_incorrectly_sized_matrix_with_umfpack.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized.test_call_with_incorrectly_sized_matrix_with_umfpack.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized.test_call_with_incorrectly_sized_matrix_with_umfpack.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized.test_call_with_incorrectly_sized_matrix_with_umfpack.__dict__.__setitem__('stypy_function_name', 'TestFactorized.test_call_with_incorrectly_sized_matrix_with_umfpack')
        TestFactorized.test_call_with_incorrectly_sized_matrix_with_umfpack.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized.test_call_with_incorrectly_sized_matrix_with_umfpack.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized.test_call_with_incorrectly_sized_matrix_with_umfpack.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized.test_call_with_incorrectly_sized_matrix_with_umfpack.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized.test_call_with_incorrectly_sized_matrix_with_umfpack.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized.test_call_with_incorrectly_sized_matrix_with_umfpack.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized.test_call_with_incorrectly_sized_matrix_with_umfpack.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized.test_call_with_incorrectly_sized_matrix_with_umfpack', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_call_with_incorrectly_sized_matrix_with_umfpack', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_call_with_incorrectly_sized_matrix_with_umfpack(...)' code ##################

        
        # Call to use_solver(...): (line 116)
        # Processing the call keyword arguments (line 116)
        # Getting the type of 'True' (line 116)
        True_393265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 30), 'True', False)
        keyword_393266 = True_393265
        kwargs_393267 = {'useUmfpack': keyword_393266}
        # Getting the type of 'use_solver' (line 116)
        use_solver_393264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 116)
        use_solver_call_result_393268 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), use_solver_393264, *[], **kwargs_393267)
        
        
        # Assigning a Call to a Name (line 117):
        
        # Call to factorized(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'self' (line 117)
        self_393270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'self', False)
        # Obtaining the member 'A' of a type (line 117)
        A_393271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 27), self_393270, 'A')
        # Processing the call keyword arguments (line 117)
        kwargs_393272 = {}
        # Getting the type of 'factorized' (line 117)
        factorized_393269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'factorized', False)
        # Calling factorized(args, kwargs) (line 117)
        factorized_call_result_393273 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), factorized_393269, *[A_393271], **kwargs_393272)
        
        # Assigning a type to the variable 'solve' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'solve', factorized_call_result_393273)
        
        # Assigning a Call to a Name (line 118):
        
        # Call to rand(...): (line 118)
        # Processing the call arguments (line 118)
        int_393276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 24), 'int')
        # Processing the call keyword arguments (line 118)
        kwargs_393277 = {}
        # Getting the type of 'random' (line 118)
        random_393274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'random', False)
        # Obtaining the member 'rand' of a type (line 118)
        rand_393275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), random_393274, 'rand')
        # Calling rand(args, kwargs) (line 118)
        rand_call_result_393278 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), rand_393275, *[int_393276], **kwargs_393277)
        
        # Assigning a type to the variable 'b' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'b', rand_call_result_393278)
        
        # Assigning a Call to a Name (line 119):
        
        # Call to rand(...): (line 119)
        # Processing the call arguments (line 119)
        int_393281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 24), 'int')
        int_393282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 27), 'int')
        # Processing the call keyword arguments (line 119)
        kwargs_393283 = {}
        # Getting the type of 'random' (line 119)
        random_393279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'random', False)
        # Obtaining the member 'rand' of a type (line 119)
        rand_393280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), random_393279, 'rand')
        # Calling rand(args, kwargs) (line 119)
        rand_call_result_393284 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), rand_393280, *[int_393281, int_393282], **kwargs_393283)
        
        # Assigning a type to the variable 'B' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'B', rand_call_result_393284)
        
        # Assigning a Call to a Name (line 120):
        
        # Call to rand(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'self' (line 120)
        self_393287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), 'self', False)
        # Obtaining the member 'n' of a type (line 120)
        n_393288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 25), self_393287, 'n')
        int_393289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 33), 'int')
        int_393290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 36), 'int')
        # Processing the call keyword arguments (line 120)
        kwargs_393291 = {}
        # Getting the type of 'random' (line 120)
        random_393285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 13), 'random', False)
        # Obtaining the member 'rand' of a type (line 120)
        rand_393286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 13), random_393285, 'rand')
        # Calling rand(args, kwargs) (line 120)
        rand_call_result_393292 = invoke(stypy.reporting.localization.Localization(__file__, 120, 13), rand_393286, *[n_393288, int_393289, int_393290], **kwargs_393291)
        
        # Assigning a type to the variable 'BB' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'BB', rand_call_result_393292)
        
        # Call to solve(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'b' (line 123)
        b_393294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), 'b', False)
        # Processing the call keyword arguments (line 123)
        kwargs_393295 = {}
        # Getting the type of 'solve' (line 123)
        solve_393293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'solve', False)
        # Calling solve(args, kwargs) (line 123)
        solve_call_result_393296 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), solve_393293, *[b_393294], **kwargs_393295)
        
        
        # Assigning a Str to a Name (line 124):
        str_393297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 14), 'str', 'object too deep for desired array')
        # Assigning a type to the variable 'msg' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'msg', str_393297)
        
        # Call to assert_raises(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'ValueError' (line 125)
        ValueError_393299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 125)
        # Getting the type of 'msg' (line 125)
        msg_393300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 47), 'msg', False)
        keyword_393301 = msg_393300
        kwargs_393302 = {'message': keyword_393301}
        # Getting the type of 'assert_raises' (line 125)
        assert_raises_393298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 125)
        assert_raises_call_result_393303 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), assert_raises_393298, *[ValueError_393299], **kwargs_393302)
        
        with_393304 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 125, 13), assert_raises_call_result_393303, 'with parameter', '__enter__', '__exit__')

        if with_393304:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 125)
            enter___393305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), assert_raises_call_result_393303, '__enter__')
            with_enter_393306 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), enter___393305)
            
            # Call to solve(...): (line 126)
            # Processing the call arguments (line 126)
            # Getting the type of 'B' (line 126)
            B_393308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 18), 'B', False)
            # Processing the call keyword arguments (line 126)
            kwargs_393309 = {}
            # Getting the type of 'solve' (line 126)
            solve_393307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'solve', False)
            # Calling solve(args, kwargs) (line 126)
            solve_call_result_393310 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), solve_393307, *[B_393308], **kwargs_393309)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 125)
            exit___393311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), assert_raises_call_result_393303, '__exit__')
            with_exit_393312 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), exit___393311, None, None, None)

        
        # Call to assert_raises(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'ValueError' (line 127)
        ValueError_393314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 127)
        # Getting the type of 'msg' (line 127)
        msg_393315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 47), 'msg', False)
        keyword_393316 = msg_393315
        kwargs_393317 = {'message': keyword_393316}
        # Getting the type of 'assert_raises' (line 127)
        assert_raises_393313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 127)
        assert_raises_call_result_393318 = invoke(stypy.reporting.localization.Localization(__file__, 127, 13), assert_raises_393313, *[ValueError_393314], **kwargs_393317)
        
        with_393319 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 127, 13), assert_raises_call_result_393318, 'with parameter', '__enter__', '__exit__')

        if with_393319:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 127)
            enter___393320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 13), assert_raises_call_result_393318, '__enter__')
            with_enter_393321 = invoke(stypy.reporting.localization.Localization(__file__, 127, 13), enter___393320)
            
            # Call to solve(...): (line 128)
            # Processing the call arguments (line 128)
            # Getting the type of 'BB' (line 128)
            BB_393323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 18), 'BB', False)
            # Processing the call keyword arguments (line 128)
            kwargs_393324 = {}
            # Getting the type of 'solve' (line 128)
            solve_393322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'solve', False)
            # Calling solve(args, kwargs) (line 128)
            solve_call_result_393325 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), solve_393322, *[BB_393323], **kwargs_393324)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 127)
            exit___393326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 13), assert_raises_call_result_393318, '__exit__')
            with_exit_393327 = invoke(stypy.reporting.localization.Localization(__file__, 127, 13), exit___393326, None, None, None)

        
        # ################# End of 'test_call_with_incorrectly_sized_matrix_with_umfpack(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_call_with_incorrectly_sized_matrix_with_umfpack' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_393328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393328)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_call_with_incorrectly_sized_matrix_with_umfpack'
        return stypy_return_type_393328


    @norecursion
    def test_call_with_cast_to_complex_without_umfpack(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_call_with_cast_to_complex_without_umfpack'
        module_type_store = module_type_store.open_function_context('test_call_with_cast_to_complex_without_umfpack', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized.test_call_with_cast_to_complex_without_umfpack.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized.test_call_with_cast_to_complex_without_umfpack.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized.test_call_with_cast_to_complex_without_umfpack.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized.test_call_with_cast_to_complex_without_umfpack.__dict__.__setitem__('stypy_function_name', 'TestFactorized.test_call_with_cast_to_complex_without_umfpack')
        TestFactorized.test_call_with_cast_to_complex_without_umfpack.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized.test_call_with_cast_to_complex_without_umfpack.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized.test_call_with_cast_to_complex_without_umfpack.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized.test_call_with_cast_to_complex_without_umfpack.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized.test_call_with_cast_to_complex_without_umfpack.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized.test_call_with_cast_to_complex_without_umfpack.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized.test_call_with_cast_to_complex_without_umfpack.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized.test_call_with_cast_to_complex_without_umfpack', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_call_with_cast_to_complex_without_umfpack', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_call_with_cast_to_complex_without_umfpack(...)' code ##################

        
        # Call to use_solver(...): (line 131)
        # Processing the call keyword arguments (line 131)
        # Getting the type of 'False' (line 131)
        False_393330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'False', False)
        keyword_393331 = False_393330
        kwargs_393332 = {'useUmfpack': keyword_393331}
        # Getting the type of 'use_solver' (line 131)
        use_solver_393329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 131)
        use_solver_call_result_393333 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), use_solver_393329, *[], **kwargs_393332)
        
        
        # Assigning a Call to a Name (line 132):
        
        # Call to factorized(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'self' (line 132)
        self_393335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'self', False)
        # Obtaining the member 'A' of a type (line 132)
        A_393336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 27), self_393335, 'A')
        # Processing the call keyword arguments (line 132)
        kwargs_393337 = {}
        # Getting the type of 'factorized' (line 132)
        factorized_393334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'factorized', False)
        # Calling factorized(args, kwargs) (line 132)
        factorized_call_result_393338 = invoke(stypy.reporting.localization.Localization(__file__, 132, 16), factorized_393334, *[A_393336], **kwargs_393337)
        
        # Assigning a type to the variable 'solve' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'solve', factorized_call_result_393338)
        
        # Assigning a Call to a Name (line 133):
        
        # Call to rand(...): (line 133)
        # Processing the call arguments (line 133)
        int_393341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 24), 'int')
        # Processing the call keyword arguments (line 133)
        kwargs_393342 = {}
        # Getting the type of 'random' (line 133)
        random_393339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'random', False)
        # Obtaining the member 'rand' of a type (line 133)
        rand_393340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), random_393339, 'rand')
        # Calling rand(args, kwargs) (line 133)
        rand_call_result_393343 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), rand_393340, *[int_393341], **kwargs_393342)
        
        # Assigning a type to the variable 'b' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'b', rand_call_result_393343)
        
        
        # Obtaining an instance of the builtin type 'list' (line 134)
        list_393344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 134)
        # Adding element type (line 134)
        # Getting the type of 'np' (line 134)
        np_393345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'np')
        # Obtaining the member 'complex64' of a type (line 134)
        complex64_393346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 18), np_393345, 'complex64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 17), list_393344, complex64_393346)
        # Adding element type (line 134)
        # Getting the type of 'np' (line 134)
        np_393347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 32), 'np')
        # Obtaining the member 'complex128' of a type (line 134)
        complex128_393348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 32), np_393347, 'complex128')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 17), list_393344, complex128_393348)
        
        # Testing the type of a for loop iterable (line 134)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 134, 8), list_393344)
        # Getting the type of the for loop variable (line 134)
        for_loop_var_393349 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 134, 8), list_393344)
        # Assigning a type to the variable 't' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 't', for_loop_var_393349)
        # SSA begins for a for statement (line 134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_raises(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'TypeError' (line 135)
        TypeError_393351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'TypeError', False)
        # Processing the call keyword arguments (line 135)
        str_393352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 50), 'str', 'Cannot cast array data')
        keyword_393353 = str_393352
        kwargs_393354 = {'message': keyword_393353}
        # Getting the type of 'assert_raises' (line 135)
        assert_raises_393350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 135)
        assert_raises_call_result_393355 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), assert_raises_393350, *[TypeError_393351], **kwargs_393354)
        
        with_393356 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 135, 17), assert_raises_call_result_393355, 'with parameter', '__enter__', '__exit__')

        if with_393356:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 135)
            enter___393357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 17), assert_raises_call_result_393355, '__enter__')
            with_enter_393358 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), enter___393357)
            
            # Call to solve(...): (line 136)
            # Processing the call arguments (line 136)
            
            # Call to astype(...): (line 136)
            # Processing the call arguments (line 136)
            # Getting the type of 't' (line 136)
            t_393362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 31), 't', False)
            # Processing the call keyword arguments (line 136)
            kwargs_393363 = {}
            # Getting the type of 'b' (line 136)
            b_393360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 22), 'b', False)
            # Obtaining the member 'astype' of a type (line 136)
            astype_393361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 22), b_393360, 'astype')
            # Calling astype(args, kwargs) (line 136)
            astype_call_result_393364 = invoke(stypy.reporting.localization.Localization(__file__, 136, 22), astype_393361, *[t_393362], **kwargs_393363)
            
            # Processing the call keyword arguments (line 136)
            kwargs_393365 = {}
            # Getting the type of 'solve' (line 136)
            solve_393359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'solve', False)
            # Calling solve(args, kwargs) (line 136)
            solve_call_result_393366 = invoke(stypy.reporting.localization.Localization(__file__, 136, 16), solve_393359, *[astype_call_result_393364], **kwargs_393365)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 135)
            exit___393367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 17), assert_raises_call_result_393355, '__exit__')
            with_exit_393368 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), exit___393367, None, None, None)

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_call_with_cast_to_complex_without_umfpack(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_call_with_cast_to_complex_without_umfpack' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_393369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393369)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_call_with_cast_to_complex_without_umfpack'
        return stypy_return_type_393369


    @norecursion
    def test_call_with_cast_to_complex_with_umfpack(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_call_with_cast_to_complex_with_umfpack'
        module_type_store = module_type_store.open_function_context('test_call_with_cast_to_complex_with_umfpack', 138, 4, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized.test_call_with_cast_to_complex_with_umfpack.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized.test_call_with_cast_to_complex_with_umfpack.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized.test_call_with_cast_to_complex_with_umfpack.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized.test_call_with_cast_to_complex_with_umfpack.__dict__.__setitem__('stypy_function_name', 'TestFactorized.test_call_with_cast_to_complex_with_umfpack')
        TestFactorized.test_call_with_cast_to_complex_with_umfpack.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized.test_call_with_cast_to_complex_with_umfpack.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized.test_call_with_cast_to_complex_with_umfpack.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized.test_call_with_cast_to_complex_with_umfpack.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized.test_call_with_cast_to_complex_with_umfpack.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized.test_call_with_cast_to_complex_with_umfpack.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized.test_call_with_cast_to_complex_with_umfpack.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized.test_call_with_cast_to_complex_with_umfpack', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_call_with_cast_to_complex_with_umfpack', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_call_with_cast_to_complex_with_umfpack(...)' code ##################

        
        # Call to use_solver(...): (line 140)
        # Processing the call keyword arguments (line 140)
        # Getting the type of 'True' (line 140)
        True_393371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'True', False)
        keyword_393372 = True_393371
        kwargs_393373 = {'useUmfpack': keyword_393372}
        # Getting the type of 'use_solver' (line 140)
        use_solver_393370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 140)
        use_solver_call_result_393374 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), use_solver_393370, *[], **kwargs_393373)
        
        
        # Assigning a Call to a Name (line 141):
        
        # Call to factorized(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'self' (line 141)
        self_393376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 27), 'self', False)
        # Obtaining the member 'A' of a type (line 141)
        A_393377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 27), self_393376, 'A')
        # Processing the call keyword arguments (line 141)
        kwargs_393378 = {}
        # Getting the type of 'factorized' (line 141)
        factorized_393375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'factorized', False)
        # Calling factorized(args, kwargs) (line 141)
        factorized_call_result_393379 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), factorized_393375, *[A_393377], **kwargs_393378)
        
        # Assigning a type to the variable 'solve' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'solve', factorized_call_result_393379)
        
        # Assigning a Call to a Name (line 142):
        
        # Call to rand(...): (line 142)
        # Processing the call arguments (line 142)
        int_393382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 24), 'int')
        # Processing the call keyword arguments (line 142)
        kwargs_393383 = {}
        # Getting the type of 'random' (line 142)
        random_393380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'random', False)
        # Obtaining the member 'rand' of a type (line 142)
        rand_393381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), random_393380, 'rand')
        # Calling rand(args, kwargs) (line 142)
        rand_call_result_393384 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), rand_393381, *[int_393382], **kwargs_393383)
        
        # Assigning a type to the variable 'b' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'b', rand_call_result_393384)
        
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_393385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        # Adding element type (line 143)
        # Getting the type of 'np' (line 143)
        np_393386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 18), 'np')
        # Obtaining the member 'complex64' of a type (line 143)
        complex64_393387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 18), np_393386, 'complex64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 17), list_393385, complex64_393387)
        # Adding element type (line 143)
        # Getting the type of 'np' (line 143)
        np_393388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 32), 'np')
        # Obtaining the member 'complex128' of a type (line 143)
        complex128_393389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 32), np_393388, 'complex128')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 17), list_393385, complex128_393389)
        
        # Testing the type of a for loop iterable (line 143)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 143, 8), list_393385)
        # Getting the type of the for loop variable (line 143)
        for_loop_var_393390 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 143, 8), list_393385)
        # Assigning a type to the variable 't' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 't', for_loop_var_393390)
        # SSA begins for a for statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_warns(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'np' (line 144)
        np_393392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'np', False)
        # Obtaining the member 'ComplexWarning' of a type (line 144)
        ComplexWarning_393393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 25), np_393392, 'ComplexWarning')
        # Getting the type of 'solve' (line 144)
        solve_393394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 44), 'solve', False)
        
        # Call to astype(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 't' (line 144)
        t_393397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 60), 't', False)
        # Processing the call keyword arguments (line 144)
        kwargs_393398 = {}
        # Getting the type of 'b' (line 144)
        b_393395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 51), 'b', False)
        # Obtaining the member 'astype' of a type (line 144)
        astype_393396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 51), b_393395, 'astype')
        # Calling astype(args, kwargs) (line 144)
        astype_call_result_393399 = invoke(stypy.reporting.localization.Localization(__file__, 144, 51), astype_393396, *[t_393397], **kwargs_393398)
        
        # Processing the call keyword arguments (line 144)
        kwargs_393400 = {}
        # Getting the type of 'assert_warns' (line 144)
        assert_warns_393391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'assert_warns', False)
        # Calling assert_warns(args, kwargs) (line 144)
        assert_warns_call_result_393401 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), assert_warns_393391, *[ComplexWarning_393393, solve_393394, astype_call_result_393399], **kwargs_393400)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_call_with_cast_to_complex_with_umfpack(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_call_with_cast_to_complex_with_umfpack' in the type store
        # Getting the type of 'stypy_return_type' (line 138)
        stypy_return_type_393402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393402)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_call_with_cast_to_complex_with_umfpack'
        return stypy_return_type_393402


    @norecursion
    def test_assume_sorted_indices_flag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_assume_sorted_indices_flag'
        module_type_store = module_type_store.open_function_context('test_assume_sorted_indices_flag', 146, 4, False)
        # Assigning a type to the variable 'self' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFactorized.test_assume_sorted_indices_flag.__dict__.__setitem__('stypy_localization', localization)
        TestFactorized.test_assume_sorted_indices_flag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFactorized.test_assume_sorted_indices_flag.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFactorized.test_assume_sorted_indices_flag.__dict__.__setitem__('stypy_function_name', 'TestFactorized.test_assume_sorted_indices_flag')
        TestFactorized.test_assume_sorted_indices_flag.__dict__.__setitem__('stypy_param_names_list', [])
        TestFactorized.test_assume_sorted_indices_flag.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFactorized.test_assume_sorted_indices_flag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFactorized.test_assume_sorted_indices_flag.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFactorized.test_assume_sorted_indices_flag.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFactorized.test_assume_sorted_indices_flag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFactorized.test_assume_sorted_indices_flag.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized.test_assume_sorted_indices_flag', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_assume_sorted_indices_flag', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_assume_sorted_indices_flag(...)' code ##################

        
        # Assigning a Call to a Name (line 149):
        
        # Call to array(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Obtaining an instance of the builtin type 'list' (line 149)
        list_393405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 149)
        # Adding element type (line 149)
        int_393406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 33), list_393405, int_393406)
        # Adding element type (line 149)
        int_393407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 33), list_393405, int_393407)
        # Adding element type (line 149)
        int_393408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 33), list_393405, int_393408)
        # Adding element type (line 149)
        int_393409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 33), list_393405, int_393409)
        
        # Processing the call keyword arguments (line 149)
        kwargs_393410 = {}
        # Getting the type of 'np' (line 149)
        np_393403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'np', False)
        # Obtaining the member 'array' of a type (line 149)
        array_393404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), np_393403, 'array')
        # Calling array(args, kwargs) (line 149)
        array_call_result_393411 = invoke(stypy.reporting.localization.Localization(__file__, 149, 24), array_393404, *[list_393405], **kwargs_393410)
        
        # Assigning a type to the variable 'unsorted_inds' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'unsorted_inds', array_call_result_393411)
        
        # Assigning a Call to a Name (line 150):
        
        # Call to array(...): (line 150)
        # Processing the call arguments (line 150)
        
        # Obtaining an instance of the builtin type 'list' (line 150)
        list_393414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 150)
        # Adding element type (line 150)
        int_393415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 24), list_393414, int_393415)
        # Adding element type (line 150)
        int_393416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 24), list_393414, int_393416)
        # Adding element type (line 150)
        int_393417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 24), list_393414, int_393417)
        # Adding element type (line 150)
        float_393418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 24), list_393414, float_393418)
        
        # Processing the call keyword arguments (line 150)
        kwargs_393419 = {}
        # Getting the type of 'np' (line 150)
        np_393412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 150)
        array_393413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 15), np_393412, 'array')
        # Calling array(args, kwargs) (line 150)
        array_call_result_393420 = invoke(stypy.reporting.localization.Localization(__file__, 150, 15), array_393413, *[list_393414], **kwargs_393419)
        
        # Assigning a type to the variable 'data' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'data', array_call_result_393420)
        
        # Assigning a Call to a Name (line 151):
        
        # Call to array(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_393423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        int_393424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 26), list_393423, int_393424)
        # Adding element type (line 151)
        int_393425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 26), list_393423, int_393425)
        # Adding element type (line 151)
        int_393426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 26), list_393423, int_393426)
        # Adding element type (line 151)
        int_393427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 26), list_393423, int_393427)
        
        # Processing the call keyword arguments (line 151)
        kwargs_393428 = {}
        # Getting the type of 'np' (line 151)
        np_393421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 151)
        array_393422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 17), np_393421, 'array')
        # Calling array(args, kwargs) (line 151)
        array_call_result_393429 = invoke(stypy.reporting.localization.Localization(__file__, 151, 17), array_393422, *[list_393423], **kwargs_393428)
        
        # Assigning a type to the variable 'indptr' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'indptr', array_call_result_393429)
        
        # Assigning a Call to a Name (line 152):
        
        # Call to csc_matrix(...): (line 152)
        # Processing the call arguments (line 152)
        
        # Obtaining an instance of the builtin type 'tuple' (line 152)
        tuple_393431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 152)
        # Adding element type (line 152)
        # Getting the type of 'data' (line 152)
        data_393432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 24), tuple_393431, data_393432)
        # Adding element type (line 152)
        # Getting the type of 'unsorted_inds' (line 152)
        unsorted_inds_393433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'unsorted_inds', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 24), tuple_393431, unsorted_inds_393433)
        # Adding element type (line 152)
        # Getting the type of 'indptr' (line 152)
        indptr_393434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 45), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 24), tuple_393431, indptr_393434)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 152)
        tuple_393435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 152)
        # Adding element type (line 152)
        int_393436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 55), tuple_393435, int_393436)
        # Adding element type (line 152)
        int_393437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 55), tuple_393435, int_393437)
        
        # Processing the call keyword arguments (line 152)
        kwargs_393438 = {}
        # Getting the type of 'csc_matrix' (line 152)
        csc_matrix_393430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 152)
        csc_matrix_call_result_393439 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), csc_matrix_393430, *[tuple_393431, tuple_393435], **kwargs_393438)
        
        # Assigning a type to the variable 'A' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'A', csc_matrix_call_result_393439)
        
        # Assigning a Call to a Name (line 153):
        
        # Call to ones(...): (line 153)
        # Processing the call arguments (line 153)
        int_393441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 17), 'int')
        # Processing the call keyword arguments (line 153)
        kwargs_393442 = {}
        # Getting the type of 'ones' (line 153)
        ones_393440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 153)
        ones_call_result_393443 = invoke(stypy.reporting.localization.Localization(__file__, 153, 12), ones_393440, *[int_393441], **kwargs_393442)
        
        # Assigning a type to the variable 'b' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'b', ones_call_result_393443)
        
        # Call to use_solver(...): (line 156)
        # Processing the call keyword arguments (line 156)
        # Getting the type of 'True' (line 156)
        True_393445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 30), 'True', False)
        keyword_393446 = True_393445
        # Getting the type of 'True' (line 156)
        True_393447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 56), 'True', False)
        keyword_393448 = True_393447
        kwargs_393449 = {'assumeSortedIndices': keyword_393448, 'useUmfpack': keyword_393446}
        # Getting the type of 'use_solver' (line 156)
        use_solver_393444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 156)
        use_solver_call_result_393450 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), use_solver_393444, *[], **kwargs_393449)
        
        
        # Call to assert_raises(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'RuntimeError' (line 157)
        RuntimeError_393452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'RuntimeError', False)
        # Processing the call keyword arguments (line 157)
        str_393453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 35), 'str', 'UMFPACK_ERROR_invalid_matrix')
        keyword_393454 = str_393453
        kwargs_393455 = {'message': keyword_393454}
        # Getting the type of 'assert_raises' (line 157)
        assert_raises_393451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 157)
        assert_raises_call_result_393456 = invoke(stypy.reporting.localization.Localization(__file__, 157, 13), assert_raises_393451, *[RuntimeError_393452], **kwargs_393455)
        
        with_393457 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 157, 13), assert_raises_call_result_393456, 'with parameter', '__enter__', '__exit__')

        if with_393457:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 157)
            enter___393458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 13), assert_raises_call_result_393456, '__enter__')
            with_enter_393459 = invoke(stypy.reporting.localization.Localization(__file__, 157, 13), enter___393458)
            
            # Call to factorized(...): (line 159)
            # Processing the call arguments (line 159)
            # Getting the type of 'A' (line 159)
            A_393461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'A', False)
            # Processing the call keyword arguments (line 159)
            kwargs_393462 = {}
            # Getting the type of 'factorized' (line 159)
            factorized_393460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'factorized', False)
            # Calling factorized(args, kwargs) (line 159)
            factorized_call_result_393463 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), factorized_393460, *[A_393461], **kwargs_393462)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 157)
            exit___393464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 13), assert_raises_call_result_393456, '__exit__')
            with_exit_393465 = invoke(stypy.reporting.localization.Localization(__file__, 157, 13), exit___393464, None, None, None)

        
        # Call to use_solver(...): (line 162)
        # Processing the call keyword arguments (line 162)
        # Getting the type of 'True' (line 162)
        True_393467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 30), 'True', False)
        keyword_393468 = True_393467
        # Getting the type of 'False' (line 162)
        False_393469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 56), 'False', False)
        keyword_393470 = False_393469
        kwargs_393471 = {'assumeSortedIndices': keyword_393470, 'useUmfpack': keyword_393468}
        # Getting the type of 'use_solver' (line 162)
        use_solver_393466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 162)
        use_solver_call_result_393472 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), use_solver_393466, *[], **kwargs_393471)
        
        
        # Assigning a Call to a Name (line 163):
        
        # Call to solve(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'b' (line 163)
        b_393481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 40), 'b', False)
        # Processing the call keyword arguments (line 163)
        kwargs_393482 = {}
        
        # Call to splu(...): (line 163)
        # Processing the call arguments (line 163)
        
        # Call to copy(...): (line 163)
        # Processing the call keyword arguments (line 163)
        kwargs_393476 = {}
        # Getting the type of 'A' (line 163)
        A_393474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 24), 'A', False)
        # Obtaining the member 'copy' of a type (line 163)
        copy_393475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 24), A_393474, 'copy')
        # Calling copy(args, kwargs) (line 163)
        copy_call_result_393477 = invoke(stypy.reporting.localization.Localization(__file__, 163, 24), copy_393475, *[], **kwargs_393476)
        
        # Processing the call keyword arguments (line 163)
        kwargs_393478 = {}
        # Getting the type of 'splu' (line 163)
        splu_393473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'splu', False)
        # Calling splu(args, kwargs) (line 163)
        splu_call_result_393479 = invoke(stypy.reporting.localization.Localization(__file__, 163, 19), splu_393473, *[copy_call_result_393477], **kwargs_393478)
        
        # Obtaining the member 'solve' of a type (line 163)
        solve_393480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), splu_call_result_393479, 'solve')
        # Calling solve(args, kwargs) (line 163)
        solve_call_result_393483 = invoke(stypy.reporting.localization.Localization(__file__, 163, 19), solve_393480, *[b_393481], **kwargs_393482)
        
        # Assigning a type to the variable 'expected' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'expected', solve_call_result_393483)
        
        # Call to assert_equal(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'A' (line 165)
        A_393485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 21), 'A', False)
        # Obtaining the member 'has_sorted_indices' of a type (line 165)
        has_sorted_indices_393486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 21), A_393485, 'has_sorted_indices')
        int_393487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 43), 'int')
        # Processing the call keyword arguments (line 165)
        kwargs_393488 = {}
        # Getting the type of 'assert_equal' (line 165)
        assert_equal_393484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 165)
        assert_equal_call_result_393489 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), assert_equal_393484, *[has_sorted_indices_393486, int_393487], **kwargs_393488)
        
        
        # Call to assert_array_almost_equal(...): (line 166)
        # Processing the call arguments (line 166)
        
        # Call to (...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'b' (line 166)
        b_393495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 48), 'b', False)
        # Processing the call keyword arguments (line 166)
        kwargs_393496 = {}
        
        # Call to factorized(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'A' (line 166)
        A_393492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 45), 'A', False)
        # Processing the call keyword arguments (line 166)
        kwargs_393493 = {}
        # Getting the type of 'factorized' (line 166)
        factorized_393491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 34), 'factorized', False)
        # Calling factorized(args, kwargs) (line 166)
        factorized_call_result_393494 = invoke(stypy.reporting.localization.Localization(__file__, 166, 34), factorized_393491, *[A_393492], **kwargs_393493)
        
        # Calling (args, kwargs) (line 166)
        _call_result_393497 = invoke(stypy.reporting.localization.Localization(__file__, 166, 34), factorized_call_result_393494, *[b_393495], **kwargs_393496)
        
        # Getting the type of 'expected' (line 166)
        expected_393498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 52), 'expected', False)
        # Processing the call keyword arguments (line 166)
        kwargs_393499 = {}
        # Getting the type of 'assert_array_almost_equal' (line 166)
        assert_array_almost_equal_393490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 166)
        assert_array_almost_equal_call_result_393500 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), assert_array_almost_equal_393490, *[_call_result_393497, expected_393498], **kwargs_393499)
        
        
        # Call to assert_equal(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'A' (line 167)
        A_393502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'A', False)
        # Obtaining the member 'has_sorted_indices' of a type (line 167)
        has_sorted_indices_393503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 21), A_393502, 'has_sorted_indices')
        int_393504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 43), 'int')
        # Processing the call keyword arguments (line 167)
        kwargs_393505 = {}
        # Getting the type of 'assert_equal' (line 167)
        assert_equal_393501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 167)
        assert_equal_call_result_393506 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), assert_equal_393501, *[has_sorted_indices_393503, int_393504], **kwargs_393505)
        
        
        # ################# End of 'test_assume_sorted_indices_flag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_assume_sorted_indices_flag' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_393507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393507)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_assume_sorted_indices_flag'
        return stypy_return_type_393507


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 44, 0, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFactorized.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestFactorized' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'TestFactorized', TestFactorized)
# Declaration of the 'TestLinsolve' class

class TestLinsolve(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinsolve.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestLinsolve.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinsolve.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinsolve.setup_method.__dict__.__setitem__('stypy_function_name', 'TestLinsolve.setup_method')
        TestLinsolve.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinsolve.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinsolve.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinsolve.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinsolve.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinsolve.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinsolve.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to use_solver(...): (line 172)
        # Processing the call keyword arguments (line 172)
        # Getting the type of 'False' (line 172)
        False_393509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 30), 'False', False)
        keyword_393510 = False_393509
        kwargs_393511 = {'useUmfpack': keyword_393510}
        # Getting the type of 'use_solver' (line 172)
        use_solver_393508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 172)
        use_solver_call_result_393512 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), use_solver_393508, *[], **kwargs_393511)
        
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_393513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393513)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_393513


    @norecursion
    def test_singular(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_singular'
        module_type_store = module_type_store.open_function_context('test_singular', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinsolve.test_singular.__dict__.__setitem__('stypy_localization', localization)
        TestLinsolve.test_singular.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinsolve.test_singular.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinsolve.test_singular.__dict__.__setitem__('stypy_function_name', 'TestLinsolve.test_singular')
        TestLinsolve.test_singular.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinsolve.test_singular.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinsolve.test_singular.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinsolve.test_singular.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinsolve.test_singular.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinsolve.test_singular.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinsolve.test_singular.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.test_singular', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_singular', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_singular(...)' code ##################

        
        # Assigning a Call to a Name (line 175):
        
        # Call to csc_matrix(...): (line 175)
        # Processing the call arguments (line 175)
        
        # Obtaining an instance of the builtin type 'tuple' (line 175)
        tuple_393515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 175)
        # Adding element type (line 175)
        int_393516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 24), tuple_393515, int_393516)
        # Adding element type (line 175)
        int_393517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 24), tuple_393515, int_393517)
        
        # Processing the call keyword arguments (line 175)
        str_393518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 36), 'str', 'd')
        keyword_393519 = str_393518
        kwargs_393520 = {'dtype': keyword_393519}
        # Getting the type of 'csc_matrix' (line 175)
        csc_matrix_393514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 175)
        csc_matrix_call_result_393521 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), csc_matrix_393514, *[tuple_393515], **kwargs_393520)
        
        # Assigning a type to the variable 'A' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'A', csc_matrix_call_result_393521)
        
        # Assigning a Call to a Name (line 176):
        
        # Call to array(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_393523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        int_393524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 18), list_393523, int_393524)
        # Adding element type (line 176)
        int_393525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 18), list_393523, int_393525)
        # Adding element type (line 176)
        int_393526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 18), list_393523, int_393526)
        # Adding element type (line 176)
        int_393527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 18), list_393523, int_393527)
        # Adding element type (line 176)
        int_393528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 18), list_393523, int_393528)
        
        # Processing the call keyword arguments (line 176)
        str_393529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 40), 'str', 'd')
        keyword_393530 = str_393529
        kwargs_393531 = {'dtype': keyword_393530}
        # Getting the type of 'array' (line 176)
        array_393522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'array', False)
        # Calling array(args, kwargs) (line 176)
        array_call_result_393532 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), array_393522, *[list_393523], **kwargs_393531)
        
        # Assigning a type to the variable 'b' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'b', array_call_result_393532)
        
        # Call to suppress_warnings(...): (line 177)
        # Processing the call keyword arguments (line 177)
        kwargs_393534 = {}
        # Getting the type of 'suppress_warnings' (line 177)
        suppress_warnings_393533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 177)
        suppress_warnings_call_result_393535 = invoke(stypy.reporting.localization.Localization(__file__, 177, 13), suppress_warnings_393533, *[], **kwargs_393534)
        
        with_393536 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 177, 13), suppress_warnings_call_result_393535, 'with parameter', '__enter__', '__exit__')

        if with_393536:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 177)
            enter___393537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 13), suppress_warnings_call_result_393535, '__enter__')
            with_enter_393538 = invoke(stypy.reporting.localization.Localization(__file__, 177, 13), enter___393537)
            # Assigning a type to the variable 'sup' (line 177)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), 'sup', with_enter_393538)
            
            # Call to filter(...): (line 178)
            # Processing the call arguments (line 178)
            # Getting the type of 'MatrixRankWarning' (line 178)
            MatrixRankWarning_393541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'MatrixRankWarning', False)
            str_393542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 42), 'str', 'Matrix is exactly singular')
            # Processing the call keyword arguments (line 178)
            kwargs_393543 = {}
            # Getting the type of 'sup' (line 178)
            sup_393539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 178)
            filter_393540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), sup_393539, 'filter')
            # Calling filter(args, kwargs) (line 178)
            filter_call_result_393544 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), filter_393540, *[MatrixRankWarning_393541, str_393542], **kwargs_393543)
            
            
            # Assigning a Call to a Name (line 179):
            
            # Call to spsolve(...): (line 179)
            # Processing the call arguments (line 179)
            # Getting the type of 'A' (line 179)
            A_393546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 24), 'A', False)
            # Getting the type of 'b' (line 179)
            b_393547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 27), 'b', False)
            # Processing the call keyword arguments (line 179)
            kwargs_393548 = {}
            # Getting the type of 'spsolve' (line 179)
            spsolve_393545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'spsolve', False)
            # Calling spsolve(args, kwargs) (line 179)
            spsolve_call_result_393549 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), spsolve_393545, *[A_393546, b_393547], **kwargs_393548)
            
            # Assigning a type to the variable 'x' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'x', spsolve_call_result_393549)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 177)
            exit___393550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 13), suppress_warnings_call_result_393535, '__exit__')
            with_exit_393551 = invoke(stypy.reporting.localization.Localization(__file__, 177, 13), exit___393550, None, None, None)

        
        # Call to assert_(...): (line 180)
        # Processing the call arguments (line 180)
        
        
        # Call to any(...): (line 180)
        # Processing the call keyword arguments (line 180)
        kwargs_393559 = {}
        
        # Call to isfinite(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'x' (line 180)
        x_393555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 32), 'x', False)
        # Processing the call keyword arguments (line 180)
        kwargs_393556 = {}
        # Getting the type of 'np' (line 180)
        np_393553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 180)
        isfinite_393554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 20), np_393553, 'isfinite')
        # Calling isfinite(args, kwargs) (line 180)
        isfinite_call_result_393557 = invoke(stypy.reporting.localization.Localization(__file__, 180, 20), isfinite_393554, *[x_393555], **kwargs_393556)
        
        # Obtaining the member 'any' of a type (line 180)
        any_393558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 20), isfinite_call_result_393557, 'any')
        # Calling any(args, kwargs) (line 180)
        any_call_result_393560 = invoke(stypy.reporting.localization.Localization(__file__, 180, 20), any_393558, *[], **kwargs_393559)
        
        # Applying the 'not' unary operator (line 180)
        result_not__393561 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 16), 'not', any_call_result_393560)
        
        # Processing the call keyword arguments (line 180)
        kwargs_393562 = {}
        # Getting the type of 'assert_' (line 180)
        assert__393552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 180)
        assert__call_result_393563 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), assert__393552, *[result_not__393561], **kwargs_393562)
        
        
        # ################# End of 'test_singular(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_singular' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_393564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393564)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_singular'
        return stypy_return_type_393564


    @norecursion
    def test_singular_gh_3312(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_singular_gh_3312'
        module_type_store = module_type_store.open_function_context('test_singular_gh_3312', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinsolve.test_singular_gh_3312.__dict__.__setitem__('stypy_localization', localization)
        TestLinsolve.test_singular_gh_3312.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinsolve.test_singular_gh_3312.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinsolve.test_singular_gh_3312.__dict__.__setitem__('stypy_function_name', 'TestLinsolve.test_singular_gh_3312')
        TestLinsolve.test_singular_gh_3312.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinsolve.test_singular_gh_3312.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinsolve.test_singular_gh_3312.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinsolve.test_singular_gh_3312.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinsolve.test_singular_gh_3312.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinsolve.test_singular_gh_3312.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinsolve.test_singular_gh_3312.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.test_singular_gh_3312', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_singular_gh_3312', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_singular_gh_3312(...)' code ##################

        
        # Assigning a Call to a Name (line 185):
        
        # Call to array(...): (line 185)
        # Processing the call arguments (line 185)
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_393567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_393568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)
        int_393569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 24), tuple_393568, int_393569)
        # Adding element type (line 185)
        int_393570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 24), tuple_393568, int_393570)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 22), list_393567, tuple_393568)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_393571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)
        int_393572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 33), tuple_393571, int_393572)
        # Adding element type (line 185)
        int_393573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 33), tuple_393571, int_393573)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 22), list_393567, tuple_393571)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_393574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)
        int_393575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 42), tuple_393574, int_393575)
        # Adding element type (line 185)
        int_393576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 42), tuple_393574, int_393576)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 22), list_393567, tuple_393574)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_393577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)
        int_393578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 52), tuple_393577, int_393578)
        # Adding element type (line 185)
        int_393579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 52), tuple_393577, int_393579)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 22), list_393567, tuple_393577)
        
        # Processing the call keyword arguments (line 185)
        # Getting the type of 'np' (line 185)
        np_393580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 68), 'np', False)
        # Obtaining the member 'int32' of a type (line 185)
        int32_393581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 68), np_393580, 'int32')
        keyword_393582 = int32_393581
        kwargs_393583 = {'dtype': keyword_393582}
        # Getting the type of 'np' (line 185)
        np_393565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 185)
        array_393566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 13), np_393565, 'array')
        # Calling array(args, kwargs) (line 185)
        array_call_result_393584 = invoke(stypy.reporting.localization.Localization(__file__, 185, 13), array_393566, *[list_393567], **kwargs_393583)
        
        # Assigning a type to the variable 'ij' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'ij', array_call_result_393584)
        
        # Assigning a Call to a Name (line 186):
        
        # Call to array(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Obtaining an instance of the builtin type 'list' (line 186)
        list_393587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 186)
        # Adding element type (line 186)
        float_393588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 21), list_393587, float_393588)
        # Adding element type (line 186)
        float_393589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 21), list_393587, float_393589)
        # Adding element type (line 186)
        float_393590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 21), list_393587, float_393590)
        # Adding element type (line 186)
        float_393591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 21), list_393587, float_393591)
        
        # Processing the call keyword arguments (line 186)
        kwargs_393592 = {}
        # Getting the type of 'np' (line 186)
        np_393585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 186)
        array_393586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), np_393585, 'array')
        # Calling array(args, kwargs) (line 186)
        array_call_result_393593 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), array_393586, *[list_393587], **kwargs_393592)
        
        # Assigning a type to the variable 'v' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'v', array_call_result_393593)
        
        # Assigning a Call to a Name (line 187):
        
        # Call to csc_matrix(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Obtaining an instance of the builtin type 'tuple' (line 187)
        tuple_393595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 187)
        # Adding element type (line 187)
        # Getting the type of 'v' (line 187)
        v_393596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'v', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 24), tuple_393595, v_393596)
        # Adding element type (line 187)
        # Getting the type of 'ij' (line 187)
        ij_393597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 27), 'ij', False)
        # Obtaining the member 'T' of a type (line 187)
        T_393598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 27), ij_393597, 'T')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 24), tuple_393595, T_393598)
        
        # Processing the call keyword arguments (line 187)
        
        # Obtaining an instance of the builtin type 'tuple' (line 187)
        tuple_393599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 187)
        # Adding element type (line 187)
        int_393600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 41), tuple_393599, int_393600)
        # Adding element type (line 187)
        int_393601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 41), tuple_393599, int_393601)
        
        keyword_393602 = tuple_393599
        kwargs_393603 = {'shape': keyword_393602}
        # Getting the type of 'csc_matrix' (line 187)
        csc_matrix_393594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 187)
        csc_matrix_call_result_393604 = invoke(stypy.reporting.localization.Localization(__file__, 187, 12), csc_matrix_393594, *[tuple_393595], **kwargs_393603)
        
        # Assigning a type to the variable 'A' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'A', csc_matrix_call_result_393604)
        
        # Assigning a Call to a Name (line 188):
        
        # Call to arange(...): (line 188)
        # Processing the call arguments (line 188)
        int_393607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 22), 'int')
        # Processing the call keyword arguments (line 188)
        kwargs_393608 = {}
        # Getting the type of 'np' (line 188)
        np_393605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 188)
        arange_393606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), np_393605, 'arange')
        # Calling arange(args, kwargs) (line 188)
        arange_call_result_393609 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), arange_393606, *[int_393607], **kwargs_393608)
        
        # Assigning a type to the variable 'b' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'b', arange_call_result_393609)
        
        
        # SSA begins for try-except statement (line 190)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 193):
        
        # Call to spsolve(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'A' (line 193)
        A_393611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 24), 'A', False)
        # Getting the type of 'b' (line 193)
        b_393612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 27), 'b', False)
        # Processing the call keyword arguments (line 193)
        kwargs_393613 = {}
        # Getting the type of 'spsolve' (line 193)
        spsolve_393610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'spsolve', False)
        # Calling spsolve(args, kwargs) (line 193)
        spsolve_call_result_393614 = invoke(stypy.reporting.localization.Localization(__file__, 193, 16), spsolve_393610, *[A_393611, b_393612], **kwargs_393613)
        
        # Assigning a type to the variable 'x' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'x', spsolve_call_result_393614)
        
        # Call to assert_(...): (line 194)
        # Processing the call arguments (line 194)
        
        
        # Call to any(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_393622 = {}
        
        # Call to isfinite(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'x' (line 194)
        x_393618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 36), 'x', False)
        # Processing the call keyword arguments (line 194)
        kwargs_393619 = {}
        # Getting the type of 'np' (line 194)
        np_393616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 24), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 194)
        isfinite_393617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 24), np_393616, 'isfinite')
        # Calling isfinite(args, kwargs) (line 194)
        isfinite_call_result_393620 = invoke(stypy.reporting.localization.Localization(__file__, 194, 24), isfinite_393617, *[x_393618], **kwargs_393619)
        
        # Obtaining the member 'any' of a type (line 194)
        any_393621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 24), isfinite_call_result_393620, 'any')
        # Calling any(args, kwargs) (line 194)
        any_call_result_393623 = invoke(stypy.reporting.localization.Localization(__file__, 194, 24), any_393621, *[], **kwargs_393622)
        
        # Applying the 'not' unary operator (line 194)
        result_not__393624 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 20), 'not', any_call_result_393623)
        
        # Processing the call keyword arguments (line 194)
        kwargs_393625 = {}
        # Getting the type of 'assert_' (line 194)
        assert__393615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 194)
        assert__call_result_393626 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), assert__393615, *[result_not__393624], **kwargs_393625)
        
        # SSA branch for the except part of a try statement (line 190)
        # SSA branch for the except 'RuntimeError' branch of a try statement (line 190)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 190)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_singular_gh_3312(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_singular_gh_3312' in the type store
        # Getting the type of 'stypy_return_type' (line 182)
        stypy_return_type_393627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393627)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_singular_gh_3312'
        return stypy_return_type_393627


    @norecursion
    def test_twodiags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_twodiags'
        module_type_store = module_type_store.open_function_context('test_twodiags', 198, 4, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinsolve.test_twodiags.__dict__.__setitem__('stypy_localization', localization)
        TestLinsolve.test_twodiags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinsolve.test_twodiags.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinsolve.test_twodiags.__dict__.__setitem__('stypy_function_name', 'TestLinsolve.test_twodiags')
        TestLinsolve.test_twodiags.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinsolve.test_twodiags.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinsolve.test_twodiags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinsolve.test_twodiags.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinsolve.test_twodiags.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinsolve.test_twodiags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinsolve.test_twodiags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.test_twodiags', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_twodiags', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_twodiags(...)' code ##################

        
        # Assigning a Call to a Name (line 199):
        
        # Call to spdiags(...): (line 199)
        # Processing the call arguments (line 199)
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_393629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_393630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        int_393631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_393630, int_393631)
        # Adding element type (line 199)
        int_393632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_393630, int_393632)
        # Adding element type (line 199)
        int_393633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_393630, int_393633)
        # Adding element type (line 199)
        int_393634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_393630, int_393634)
        # Adding element type (line 199)
        int_393635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_393630, int_393635)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 20), list_393629, list_393630)
        # Adding element type (line 199)
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_393636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        int_393637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 38), list_393636, int_393637)
        # Adding element type (line 199)
        int_393638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 38), list_393636, int_393638)
        # Adding element type (line 199)
        int_393639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 38), list_393636, int_393639)
        # Adding element type (line 199)
        int_393640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 38), list_393636, int_393640)
        # Adding element type (line 199)
        int_393641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 38), list_393636, int_393641)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 20), list_393629, list_393636)
        
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_393642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        int_393643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 57), list_393642, int_393643)
        # Adding element type (line 199)
        int_393644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 57), list_393642, int_393644)
        
        int_393645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 65), 'int')
        int_393646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 68), 'int')
        # Processing the call keyword arguments (line 199)
        kwargs_393647 = {}
        # Getting the type of 'spdiags' (line 199)
        spdiags_393628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'spdiags', False)
        # Calling spdiags(args, kwargs) (line 199)
        spdiags_call_result_393648 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), spdiags_393628, *[list_393629, list_393642, int_393645, int_393646], **kwargs_393647)
        
        # Assigning a type to the variable 'A' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'A', spdiags_call_result_393648)
        
        # Assigning a Call to a Name (line 200):
        
        # Call to array(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Obtaining an instance of the builtin type 'list' (line 200)
        list_393650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 200)
        # Adding element type (line 200)
        int_393651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 18), list_393650, int_393651)
        # Adding element type (line 200)
        int_393652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 18), list_393650, int_393652)
        # Adding element type (line 200)
        int_393653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 18), list_393650, int_393653)
        # Adding element type (line 200)
        int_393654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 18), list_393650, int_393654)
        # Adding element type (line 200)
        int_393655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 18), list_393650, int_393655)
        
        # Processing the call keyword arguments (line 200)
        kwargs_393656 = {}
        # Getting the type of 'array' (line 200)
        array_393649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'array', False)
        # Calling array(args, kwargs) (line 200)
        array_call_result_393657 = invoke(stypy.reporting.localization.Localization(__file__, 200, 12), array_393649, *[list_393650], **kwargs_393656)
        
        # Assigning a type to the variable 'b' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'b', array_call_result_393657)
        
        # Assigning a BinOp to a Name (line 203):
        
        # Call to norm(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Call to todense(...): (line 203)
        # Processing the call keyword arguments (line 203)
        kwargs_393661 = {}
        # Getting the type of 'A' (line 203)
        A_393659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 22), 'A', False)
        # Obtaining the member 'todense' of a type (line 203)
        todense_393660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 22), A_393659, 'todense')
        # Calling todense(args, kwargs) (line 203)
        todense_call_result_393662 = invoke(stypy.reporting.localization.Localization(__file__, 203, 22), todense_393660, *[], **kwargs_393661)
        
        int_393663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 34), 'int')
        # Processing the call keyword arguments (line 203)
        kwargs_393664 = {}
        # Getting the type of 'norm' (line 203)
        norm_393658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 17), 'norm', False)
        # Calling norm(args, kwargs) (line 203)
        norm_call_result_393665 = invoke(stypy.reporting.localization.Localization(__file__, 203, 17), norm_393658, *[todense_call_result_393662, int_393663], **kwargs_393664)
        
        
        # Call to norm(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Call to inv(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Call to todense(...): (line 203)
        # Processing the call keyword arguments (line 203)
        kwargs_393670 = {}
        # Getting the type of 'A' (line 203)
        A_393668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 48), 'A', False)
        # Obtaining the member 'todense' of a type (line 203)
        todense_393669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 48), A_393668, 'todense')
        # Calling todense(args, kwargs) (line 203)
        todense_call_result_393671 = invoke(stypy.reporting.localization.Localization(__file__, 203, 48), todense_393669, *[], **kwargs_393670)
        
        # Processing the call keyword arguments (line 203)
        kwargs_393672 = {}
        # Getting the type of 'inv' (line 203)
        inv_393667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 44), 'inv', False)
        # Calling inv(args, kwargs) (line 203)
        inv_call_result_393673 = invoke(stypy.reporting.localization.Localization(__file__, 203, 44), inv_393667, *[todense_call_result_393671], **kwargs_393672)
        
        int_393674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 61), 'int')
        # Processing the call keyword arguments (line 203)
        kwargs_393675 = {}
        # Getting the type of 'norm' (line 203)
        norm_393666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 39), 'norm', False)
        # Calling norm(args, kwargs) (line 203)
        norm_call_result_393676 = invoke(stypy.reporting.localization.Localization(__file__, 203, 39), norm_393666, *[inv_call_result_393673, int_393674], **kwargs_393675)
        
        # Applying the binary operator '*' (line 203)
        result_mul_393677 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 17), '*', norm_call_result_393665, norm_call_result_393676)
        
        # Assigning a type to the variable 'cond_A' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'cond_A', result_mul_393677)
        
        
        # Obtaining an instance of the builtin type 'list' (line 205)
        list_393678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 205)
        # Adding element type (line 205)
        str_393679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 18), 'str', 'f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 17), list_393678, str_393679)
        # Adding element type (line 205)
        str_393680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 22), 'str', 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 17), list_393678, str_393680)
        # Adding element type (line 205)
        str_393681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 26), 'str', 'F')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 17), list_393678, str_393681)
        # Adding element type (line 205)
        str_393682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 30), 'str', 'D')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 17), list_393678, str_393682)
        
        # Testing the type of a for loop iterable (line 205)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 205, 8), list_393678)
        # Getting the type of the for loop variable (line 205)
        for_loop_var_393683 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 205, 8), list_393678)
        # Assigning a type to the variable 't' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 't', for_loop_var_393683)
        # SSA begins for a for statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Attribute to a Name (line 206):
        
        # Call to finfo(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 't' (line 206)
        t_393685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 't', False)
        # Processing the call keyword arguments (line 206)
        kwargs_393686 = {}
        # Getting the type of 'finfo' (line 206)
        finfo_393684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 18), 'finfo', False)
        # Calling finfo(args, kwargs) (line 206)
        finfo_call_result_393687 = invoke(stypy.reporting.localization.Localization(__file__, 206, 18), finfo_393684, *[t_393685], **kwargs_393686)
        
        # Obtaining the member 'eps' of a type (line 206)
        eps_393688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 18), finfo_call_result_393687, 'eps')
        # Assigning a type to the variable 'eps' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'eps', eps_393688)
        
        # Assigning a Call to a Name (line 207):
        
        # Call to astype(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 't' (line 207)
        t_393691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 25), 't', False)
        # Processing the call keyword arguments (line 207)
        kwargs_393692 = {}
        # Getting the type of 'b' (line 207)
        b_393689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'b', False)
        # Obtaining the member 'astype' of a type (line 207)
        astype_393690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), b_393689, 'astype')
        # Calling astype(args, kwargs) (line 207)
        astype_call_result_393693 = invoke(stypy.reporting.localization.Localization(__file__, 207, 16), astype_393690, *[t_393691], **kwargs_393692)
        
        # Assigning a type to the variable 'b' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'b', astype_call_result_393693)
        
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_393694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        # Adding element type (line 209)
        str_393695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 27), 'str', 'csc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 26), list_393694, str_393695)
        # Adding element type (line 209)
        str_393696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 33), 'str', 'csr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 26), list_393694, str_393696)
        
        # Testing the type of a for loop iterable (line 209)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 209, 12), list_393694)
        # Getting the type of the for loop variable (line 209)
        for_loop_var_393697 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 209, 12), list_393694)
        # Assigning a type to the variable 'format' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'format', for_loop_var_393697)
        # SSA begins for a for statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 210):
        
        # Call to asformat(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'format' (line 210)
        format_393704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 43), 'format', False)
        # Processing the call keyword arguments (line 210)
        kwargs_393705 = {}
        
        # Call to astype(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 't' (line 210)
        t_393700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 31), 't', False)
        # Processing the call keyword arguments (line 210)
        kwargs_393701 = {}
        # Getting the type of 'A' (line 210)
        A_393698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'A', False)
        # Obtaining the member 'astype' of a type (line 210)
        astype_393699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 22), A_393698, 'astype')
        # Calling astype(args, kwargs) (line 210)
        astype_call_result_393702 = invoke(stypy.reporting.localization.Localization(__file__, 210, 22), astype_393699, *[t_393700], **kwargs_393701)
        
        # Obtaining the member 'asformat' of a type (line 210)
        asformat_393703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 22), astype_call_result_393702, 'asformat')
        # Calling asformat(args, kwargs) (line 210)
        asformat_call_result_393706 = invoke(stypy.reporting.localization.Localization(__file__, 210, 22), asformat_393703, *[format_393704], **kwargs_393705)
        
        # Assigning a type to the variable 'Asp' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'Asp', asformat_call_result_393706)
        
        # Assigning a Call to a Name (line 212):
        
        # Call to spsolve(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'Asp' (line 212)
        Asp_393708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 28), 'Asp', False)
        # Getting the type of 'b' (line 212)
        b_393709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 32), 'b', False)
        # Processing the call keyword arguments (line 212)
        kwargs_393710 = {}
        # Getting the type of 'spsolve' (line 212)
        spsolve_393707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'spsolve', False)
        # Calling spsolve(args, kwargs) (line 212)
        spsolve_call_result_393711 = invoke(stypy.reporting.localization.Localization(__file__, 212, 20), spsolve_393707, *[Asp_393708, b_393709], **kwargs_393710)
        
        # Assigning a type to the variable 'x' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'x', spsolve_call_result_393711)
        
        # Call to assert_(...): (line 214)
        # Processing the call arguments (line 214)
        
        
        # Call to norm(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'b' (line 214)
        b_393714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 29), 'b', False)
        # Getting the type of 'Asp' (line 214)
        Asp_393715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 33), 'Asp', False)
        # Getting the type of 'x' (line 214)
        x_393716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 37), 'x', False)
        # Applying the binary operator '*' (line 214)
        result_mul_393717 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 33), '*', Asp_393715, x_393716)
        
        # Applying the binary operator '-' (line 214)
        result_sub_393718 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 29), '-', b_393714, result_mul_393717)
        
        # Processing the call keyword arguments (line 214)
        kwargs_393719 = {}
        # Getting the type of 'norm' (line 214)
        norm_393713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'norm', False)
        # Calling norm(args, kwargs) (line 214)
        norm_call_result_393720 = invoke(stypy.reporting.localization.Localization(__file__, 214, 24), norm_393713, *[result_sub_393718], **kwargs_393719)
        
        int_393721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 42), 'int')
        # Getting the type of 'cond_A' (line 214)
        cond_A_393722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 47), 'cond_A', False)
        # Applying the binary operator '*' (line 214)
        result_mul_393723 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 42), '*', int_393721, cond_A_393722)
        
        # Getting the type of 'eps' (line 214)
        eps_393724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 56), 'eps', False)
        # Applying the binary operator '*' (line 214)
        result_mul_393725 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 54), '*', result_mul_393723, eps_393724)
        
        # Applying the binary operator '<' (line 214)
        result_lt_393726 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 24), '<', norm_call_result_393720, result_mul_393725)
        
        # Processing the call keyword arguments (line 214)
        kwargs_393727 = {}
        # Getting the type of 'assert_' (line 214)
        assert__393712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'assert_', False)
        # Calling assert_(args, kwargs) (line 214)
        assert__call_result_393728 = invoke(stypy.reporting.localization.Localization(__file__, 214, 16), assert__393712, *[result_lt_393726], **kwargs_393727)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_twodiags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_twodiags' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_393729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393729)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_twodiags'
        return stypy_return_type_393729


    @norecursion
    def test_bvector_smoketest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bvector_smoketest'
        module_type_store = module_type_store.open_function_context('test_bvector_smoketest', 216, 4, False)
        # Assigning a type to the variable 'self' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinsolve.test_bvector_smoketest.__dict__.__setitem__('stypy_localization', localization)
        TestLinsolve.test_bvector_smoketest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinsolve.test_bvector_smoketest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinsolve.test_bvector_smoketest.__dict__.__setitem__('stypy_function_name', 'TestLinsolve.test_bvector_smoketest')
        TestLinsolve.test_bvector_smoketest.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinsolve.test_bvector_smoketest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinsolve.test_bvector_smoketest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinsolve.test_bvector_smoketest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinsolve.test_bvector_smoketest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinsolve.test_bvector_smoketest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinsolve.test_bvector_smoketest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.test_bvector_smoketest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bvector_smoketest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bvector_smoketest(...)' code ##################

        
        # Assigning a Call to a Name (line 217):
        
        # Call to matrix(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_393731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_393732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        float_393733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 25), list_393732, float_393733)
        # Adding element type (line 217)
        float_393734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 25), list_393732, float_393734)
        # Adding element type (line 217)
        float_393735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 25), list_393732, float_393735)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 24), list_393731, list_393732)
        # Adding element type (line 217)
        
        # Obtaining an instance of the builtin type 'list' (line 218)
        list_393736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 218)
        # Adding element type (line 218)
        float_393737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 25), list_393736, float_393737)
        # Adding element type (line 218)
        float_393738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 25), list_393736, float_393738)
        # Adding element type (line 218)
        float_393739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 25), list_393736, float_393739)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 24), list_393731, list_393736)
        # Adding element type (line 217)
        
        # Obtaining an instance of the builtin type 'list' (line 219)
        list_393740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 219)
        # Adding element type (line 219)
        float_393741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 25), list_393740, float_393741)
        # Adding element type (line 219)
        float_393742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 25), list_393740, float_393742)
        # Adding element type (line 219)
        float_393743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 25), list_393740, float_393743)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 24), list_393731, list_393740)
        
        # Processing the call keyword arguments (line 217)
        kwargs_393744 = {}
        # Getting the type of 'matrix' (line 217)
        matrix_393730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 17), 'matrix', False)
        # Calling matrix(args, kwargs) (line 217)
        matrix_call_result_393745 = invoke(stypy.reporting.localization.Localization(__file__, 217, 17), matrix_393730, *[list_393731], **kwargs_393744)
        
        # Assigning a type to the variable 'Adense' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'Adense', matrix_call_result_393745)
        
        # Assigning a Call to a Name (line 220):
        
        # Call to csc_matrix(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'Adense' (line 220)
        Adense_393747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 24), 'Adense', False)
        # Processing the call keyword arguments (line 220)
        kwargs_393748 = {}
        # Getting the type of 'csc_matrix' (line 220)
        csc_matrix_393746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 13), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 220)
        csc_matrix_call_result_393749 = invoke(stypy.reporting.localization.Localization(__file__, 220, 13), csc_matrix_393746, *[Adense_393747], **kwargs_393748)
        
        # Assigning a type to the variable 'As' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'As', csc_matrix_call_result_393749)
        
        # Call to seed(...): (line 221)
        # Processing the call arguments (line 221)
        int_393752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 20), 'int')
        # Processing the call keyword arguments (line 221)
        kwargs_393753 = {}
        # Getting the type of 'random' (line 221)
        random_393750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'random', False)
        # Obtaining the member 'seed' of a type (line 221)
        seed_393751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), random_393750, 'seed')
        # Calling seed(args, kwargs) (line 221)
        seed_call_result_393754 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), seed_393751, *[int_393752], **kwargs_393753)
        
        
        # Assigning a Call to a Name (line 222):
        
        # Call to randn(...): (line 222)
        # Processing the call arguments (line 222)
        int_393757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 25), 'int')
        # Processing the call keyword arguments (line 222)
        kwargs_393758 = {}
        # Getting the type of 'random' (line 222)
        random_393755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'random', False)
        # Obtaining the member 'randn' of a type (line 222)
        randn_393756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), random_393755, 'randn')
        # Calling randn(args, kwargs) (line 222)
        randn_call_result_393759 = invoke(stypy.reporting.localization.Localization(__file__, 222, 12), randn_393756, *[int_393757], **kwargs_393758)
        
        # Assigning a type to the variable 'x' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'x', randn_call_result_393759)
        
        # Assigning a BinOp to a Name (line 223):
        # Getting the type of 'As' (line 223)
        As_393760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'As')
        # Getting the type of 'x' (line 223)
        x_393761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'x')
        # Applying the binary operator '*' (line 223)
        result_mul_393762 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 12), '*', As_393760, x_393761)
        
        # Assigning a type to the variable 'b' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'b', result_mul_393762)
        
        # Assigning a Call to a Name (line 224):
        
        # Call to spsolve(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'As' (line 224)
        As_393764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 21), 'As', False)
        # Getting the type of 'b' (line 224)
        b_393765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 25), 'b', False)
        # Processing the call keyword arguments (line 224)
        kwargs_393766 = {}
        # Getting the type of 'spsolve' (line 224)
        spsolve_393763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 13), 'spsolve', False)
        # Calling spsolve(args, kwargs) (line 224)
        spsolve_call_result_393767 = invoke(stypy.reporting.localization.Localization(__file__, 224, 13), spsolve_393763, *[As_393764, b_393765], **kwargs_393766)
        
        # Assigning a type to the variable 'x2' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'x2', spsolve_call_result_393767)
        
        # Call to assert_array_almost_equal(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'x' (line 226)
        x_393769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 34), 'x', False)
        # Getting the type of 'x2' (line 226)
        x2_393770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 37), 'x2', False)
        # Processing the call keyword arguments (line 226)
        kwargs_393771 = {}
        # Getting the type of 'assert_array_almost_equal' (line 226)
        assert_array_almost_equal_393768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 226)
        assert_array_almost_equal_call_result_393772 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), assert_array_almost_equal_393768, *[x_393769, x2_393770], **kwargs_393771)
        
        
        # ################# End of 'test_bvector_smoketest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bvector_smoketest' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_393773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393773)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bvector_smoketest'
        return stypy_return_type_393773


    @norecursion
    def test_bmatrix_smoketest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bmatrix_smoketest'
        module_type_store = module_type_store.open_function_context('test_bmatrix_smoketest', 228, 4, False)
        # Assigning a type to the variable 'self' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinsolve.test_bmatrix_smoketest.__dict__.__setitem__('stypy_localization', localization)
        TestLinsolve.test_bmatrix_smoketest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinsolve.test_bmatrix_smoketest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinsolve.test_bmatrix_smoketest.__dict__.__setitem__('stypy_function_name', 'TestLinsolve.test_bmatrix_smoketest')
        TestLinsolve.test_bmatrix_smoketest.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinsolve.test_bmatrix_smoketest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinsolve.test_bmatrix_smoketest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinsolve.test_bmatrix_smoketest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinsolve.test_bmatrix_smoketest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinsolve.test_bmatrix_smoketest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinsolve.test_bmatrix_smoketest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.test_bmatrix_smoketest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bmatrix_smoketest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bmatrix_smoketest(...)' code ##################

        
        # Assigning a Call to a Name (line 229):
        
        # Call to matrix(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Obtaining an instance of the builtin type 'list' (line 229)
        list_393775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 229)
        # Adding element type (line 229)
        
        # Obtaining an instance of the builtin type 'list' (line 229)
        list_393776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 229)
        # Adding element type (line 229)
        float_393777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 25), list_393776, float_393777)
        # Adding element type (line 229)
        float_393778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 25), list_393776, float_393778)
        # Adding element type (line 229)
        float_393779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 25), list_393776, float_393779)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 24), list_393775, list_393776)
        # Adding element type (line 229)
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_393780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        float_393781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 25), list_393780, float_393781)
        # Adding element type (line 230)
        float_393782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 25), list_393780, float_393782)
        # Adding element type (line 230)
        float_393783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 25), list_393780, float_393783)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 24), list_393775, list_393780)
        # Adding element type (line 229)
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_393784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        float_393785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 25), list_393784, float_393785)
        # Adding element type (line 231)
        float_393786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 25), list_393784, float_393786)
        # Adding element type (line 231)
        float_393787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 25), list_393784, float_393787)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 24), list_393775, list_393784)
        
        # Processing the call keyword arguments (line 229)
        kwargs_393788 = {}
        # Getting the type of 'matrix' (line 229)
        matrix_393774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 17), 'matrix', False)
        # Calling matrix(args, kwargs) (line 229)
        matrix_call_result_393789 = invoke(stypy.reporting.localization.Localization(__file__, 229, 17), matrix_393774, *[list_393775], **kwargs_393788)
        
        # Assigning a type to the variable 'Adense' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'Adense', matrix_call_result_393789)
        
        # Assigning a Call to a Name (line 232):
        
        # Call to csc_matrix(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'Adense' (line 232)
        Adense_393791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 'Adense', False)
        # Processing the call keyword arguments (line 232)
        kwargs_393792 = {}
        # Getting the type of 'csc_matrix' (line 232)
        csc_matrix_393790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 13), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 232)
        csc_matrix_call_result_393793 = invoke(stypy.reporting.localization.Localization(__file__, 232, 13), csc_matrix_393790, *[Adense_393791], **kwargs_393792)
        
        # Assigning a type to the variable 'As' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'As', csc_matrix_call_result_393793)
        
        # Call to seed(...): (line 233)
        # Processing the call arguments (line 233)
        int_393796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 20), 'int')
        # Processing the call keyword arguments (line 233)
        kwargs_393797 = {}
        # Getting the type of 'random' (line 233)
        random_393794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'random', False)
        # Obtaining the member 'seed' of a type (line 233)
        seed_393795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), random_393794, 'seed')
        # Calling seed(args, kwargs) (line 233)
        seed_call_result_393798 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), seed_393795, *[int_393796], **kwargs_393797)
        
        
        # Assigning a Call to a Name (line 234):
        
        # Call to randn(...): (line 234)
        # Processing the call arguments (line 234)
        int_393801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 25), 'int')
        int_393802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 28), 'int')
        # Processing the call keyword arguments (line 234)
        kwargs_393803 = {}
        # Getting the type of 'random' (line 234)
        random_393799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'random', False)
        # Obtaining the member 'randn' of a type (line 234)
        randn_393800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), random_393799, 'randn')
        # Calling randn(args, kwargs) (line 234)
        randn_call_result_393804 = invoke(stypy.reporting.localization.Localization(__file__, 234, 12), randn_393800, *[int_393801, int_393802], **kwargs_393803)
        
        # Assigning a type to the variable 'x' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'x', randn_call_result_393804)
        
        # Assigning a Call to a Name (line 235):
        
        # Call to dot(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'x' (line 235)
        x_393807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 24), 'x', False)
        # Processing the call keyword arguments (line 235)
        kwargs_393808 = {}
        # Getting the type of 'As' (line 235)
        As_393805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 17), 'As', False)
        # Obtaining the member 'dot' of a type (line 235)
        dot_393806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 17), As_393805, 'dot')
        # Calling dot(args, kwargs) (line 235)
        dot_call_result_393809 = invoke(stypy.reporting.localization.Localization(__file__, 235, 17), dot_393806, *[x_393807], **kwargs_393808)
        
        # Assigning a type to the variable 'Bdense' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'Bdense', dot_call_result_393809)
        
        # Assigning a Call to a Name (line 236):
        
        # Call to csc_matrix(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'Bdense' (line 236)
        Bdense_393811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 24), 'Bdense', False)
        # Processing the call keyword arguments (line 236)
        kwargs_393812 = {}
        # Getting the type of 'csc_matrix' (line 236)
        csc_matrix_393810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 13), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 236)
        csc_matrix_call_result_393813 = invoke(stypy.reporting.localization.Localization(__file__, 236, 13), csc_matrix_393810, *[Bdense_393811], **kwargs_393812)
        
        # Assigning a type to the variable 'Bs' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'Bs', csc_matrix_call_result_393813)
        
        # Assigning a Call to a Name (line 237):
        
        # Call to spsolve(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'As' (line 237)
        As_393815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 21), 'As', False)
        # Getting the type of 'Bs' (line 237)
        Bs_393816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 25), 'Bs', False)
        # Processing the call keyword arguments (line 237)
        kwargs_393817 = {}
        # Getting the type of 'spsolve' (line 237)
        spsolve_393814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 13), 'spsolve', False)
        # Calling spsolve(args, kwargs) (line 237)
        spsolve_call_result_393818 = invoke(stypy.reporting.localization.Localization(__file__, 237, 13), spsolve_393814, *[As_393815, Bs_393816], **kwargs_393817)
        
        # Assigning a type to the variable 'x2' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'x2', spsolve_call_result_393818)
        
        # Call to assert_array_almost_equal(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'x' (line 238)
        x_393820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 34), 'x', False)
        
        # Call to todense(...): (line 238)
        # Processing the call keyword arguments (line 238)
        kwargs_393823 = {}
        # Getting the type of 'x2' (line 238)
        x2_393821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 37), 'x2', False)
        # Obtaining the member 'todense' of a type (line 238)
        todense_393822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 37), x2_393821, 'todense')
        # Calling todense(args, kwargs) (line 238)
        todense_call_result_393824 = invoke(stypy.reporting.localization.Localization(__file__, 238, 37), todense_393822, *[], **kwargs_393823)
        
        # Processing the call keyword arguments (line 238)
        kwargs_393825 = {}
        # Getting the type of 'assert_array_almost_equal' (line 238)
        assert_array_almost_equal_393819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 238)
        assert_array_almost_equal_call_result_393826 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), assert_array_almost_equal_393819, *[x_393820, todense_call_result_393824], **kwargs_393825)
        
        
        # ################# End of 'test_bmatrix_smoketest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bmatrix_smoketest' in the type store
        # Getting the type of 'stypy_return_type' (line 228)
        stypy_return_type_393827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393827)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bmatrix_smoketest'
        return stypy_return_type_393827


    @norecursion
    def test_non_square(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_non_square'
        module_type_store = module_type_store.open_function_context('test_non_square', 240, 4, False)
        # Assigning a type to the variable 'self' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinsolve.test_non_square.__dict__.__setitem__('stypy_localization', localization)
        TestLinsolve.test_non_square.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinsolve.test_non_square.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinsolve.test_non_square.__dict__.__setitem__('stypy_function_name', 'TestLinsolve.test_non_square')
        TestLinsolve.test_non_square.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinsolve.test_non_square.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinsolve.test_non_square.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinsolve.test_non_square.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinsolve.test_non_square.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinsolve.test_non_square.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinsolve.test_non_square.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.test_non_square', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_non_square', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_non_square(...)' code ##################

        
        # Assigning a Call to a Name (line 243):
        
        # Call to ones(...): (line 243)
        # Processing the call arguments (line 243)
        
        # Obtaining an instance of the builtin type 'tuple' (line 243)
        tuple_393829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 243)
        # Adding element type (line 243)
        int_393830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 18), tuple_393829, int_393830)
        # Adding element type (line 243)
        int_393831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 18), tuple_393829, int_393831)
        
        # Processing the call keyword arguments (line 243)
        kwargs_393832 = {}
        # Getting the type of 'ones' (line 243)
        ones_393828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 243)
        ones_call_result_393833 = invoke(stypy.reporting.localization.Localization(__file__, 243, 12), ones_393828, *[tuple_393829], **kwargs_393832)
        
        # Assigning a type to the variable 'A' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'A', ones_call_result_393833)
        
        # Assigning a Call to a Name (line 244):
        
        # Call to ones(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Obtaining an instance of the builtin type 'tuple' (line 244)
        tuple_393835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 244)
        # Adding element type (line 244)
        int_393836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 18), tuple_393835, int_393836)
        # Adding element type (line 244)
        int_393837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 18), tuple_393835, int_393837)
        
        # Processing the call keyword arguments (line 244)
        kwargs_393838 = {}
        # Getting the type of 'ones' (line 244)
        ones_393834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 244)
        ones_call_result_393839 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), ones_393834, *[tuple_393835], **kwargs_393838)
        
        # Assigning a type to the variable 'b' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'b', ones_call_result_393839)
        
        # Call to assert_raises(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'ValueError' (line 245)
        ValueError_393841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 22), 'ValueError', False)
        # Getting the type of 'spsolve' (line 245)
        spsolve_393842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 34), 'spsolve', False)
        # Getting the type of 'A' (line 245)
        A_393843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 43), 'A', False)
        # Getting the type of 'b' (line 245)
        b_393844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 46), 'b', False)
        # Processing the call keyword arguments (line 245)
        kwargs_393845 = {}
        # Getting the type of 'assert_raises' (line 245)
        assert_raises_393840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 245)
        assert_raises_call_result_393846 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), assert_raises_393840, *[ValueError_393841, spsolve_393842, A_393843, b_393844], **kwargs_393845)
        
        
        # Assigning a Call to a Name (line 247):
        
        # Call to csc_matrix(...): (line 247)
        # Processing the call arguments (line 247)
        
        # Call to eye(...): (line 247)
        # Processing the call arguments (line 247)
        int_393849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 28), 'int')
        # Processing the call keyword arguments (line 247)
        kwargs_393850 = {}
        # Getting the type of 'eye' (line 247)
        eye_393848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 24), 'eye', False)
        # Calling eye(args, kwargs) (line 247)
        eye_call_result_393851 = invoke(stypy.reporting.localization.Localization(__file__, 247, 24), eye_393848, *[int_393849], **kwargs_393850)
        
        # Processing the call keyword arguments (line 247)
        kwargs_393852 = {}
        # Getting the type of 'csc_matrix' (line 247)
        csc_matrix_393847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 13), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 247)
        csc_matrix_call_result_393853 = invoke(stypy.reporting.localization.Localization(__file__, 247, 13), csc_matrix_393847, *[eye_call_result_393851], **kwargs_393852)
        
        # Assigning a type to the variable 'A2' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'A2', csc_matrix_call_result_393853)
        
        # Assigning a Call to a Name (line 248):
        
        # Call to array(...): (line 248)
        # Processing the call arguments (line 248)
        
        # Obtaining an instance of the builtin type 'list' (line 248)
        list_393855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 248)
        # Adding element type (line 248)
        float_393856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 19), list_393855, float_393856)
        # Adding element type (line 248)
        float_393857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 19), list_393855, float_393857)
        
        # Processing the call keyword arguments (line 248)
        kwargs_393858 = {}
        # Getting the type of 'array' (line 248)
        array_393854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 13), 'array', False)
        # Calling array(args, kwargs) (line 248)
        array_call_result_393859 = invoke(stypy.reporting.localization.Localization(__file__, 248, 13), array_393854, *[list_393855], **kwargs_393858)
        
        # Assigning a type to the variable 'b2' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'b2', array_call_result_393859)
        
        # Call to assert_raises(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'ValueError' (line 249)
        ValueError_393861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 22), 'ValueError', False)
        # Getting the type of 'spsolve' (line 249)
        spsolve_393862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 34), 'spsolve', False)
        # Getting the type of 'A2' (line 249)
        A2_393863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 43), 'A2', False)
        # Getting the type of 'b2' (line 249)
        b2_393864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 47), 'b2', False)
        # Processing the call keyword arguments (line 249)
        kwargs_393865 = {}
        # Getting the type of 'assert_raises' (line 249)
        assert_raises_393860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 249)
        assert_raises_call_result_393866 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), assert_raises_393860, *[ValueError_393861, spsolve_393862, A2_393863, b2_393864], **kwargs_393865)
        
        
        # ################# End of 'test_non_square(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_non_square' in the type store
        # Getting the type of 'stypy_return_type' (line 240)
        stypy_return_type_393867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393867)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_non_square'
        return stypy_return_type_393867


    @norecursion
    def test_example_comparison(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_example_comparison'
        module_type_store = module_type_store.open_function_context('test_example_comparison', 251, 4, False)
        # Assigning a type to the variable 'self' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinsolve.test_example_comparison.__dict__.__setitem__('stypy_localization', localization)
        TestLinsolve.test_example_comparison.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinsolve.test_example_comparison.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinsolve.test_example_comparison.__dict__.__setitem__('stypy_function_name', 'TestLinsolve.test_example_comparison')
        TestLinsolve.test_example_comparison.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinsolve.test_example_comparison.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinsolve.test_example_comparison.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinsolve.test_example_comparison.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinsolve.test_example_comparison.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinsolve.test_example_comparison.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinsolve.test_example_comparison.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.test_example_comparison', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_example_comparison', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_example_comparison(...)' code ##################

        
        # Assigning a Call to a Name (line 253):
        
        # Call to array(...): (line 253)
        # Processing the call arguments (line 253)
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_393869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        int_393870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 20), list_393869, int_393870)
        # Adding element type (line 253)
        int_393871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 20), list_393869, int_393871)
        # Adding element type (line 253)
        int_393872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 20), list_393869, int_393872)
        # Adding element type (line 253)
        int_393873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 20), list_393869, int_393873)
        # Adding element type (line 253)
        int_393874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 20), list_393869, int_393874)
        # Adding element type (line 253)
        int_393875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 20), list_393869, int_393875)
        
        # Processing the call keyword arguments (line 253)
        kwargs_393876 = {}
        # Getting the type of 'array' (line 253)
        array_393868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 14), 'array', False)
        # Calling array(args, kwargs) (line 253)
        array_call_result_393877 = invoke(stypy.reporting.localization.Localization(__file__, 253, 14), array_393868, *[list_393869], **kwargs_393876)
        
        # Assigning a type to the variable 'row' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'row', array_call_result_393877)
        
        # Assigning a Call to a Name (line 254):
        
        # Call to array(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_393879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        int_393880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 20), list_393879, int_393880)
        # Adding element type (line 254)
        int_393881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 20), list_393879, int_393881)
        # Adding element type (line 254)
        int_393882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 20), list_393879, int_393882)
        # Adding element type (line 254)
        int_393883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 20), list_393879, int_393883)
        # Adding element type (line 254)
        int_393884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 20), list_393879, int_393884)
        # Adding element type (line 254)
        int_393885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 20), list_393879, int_393885)
        
        # Processing the call keyword arguments (line 254)
        kwargs_393886 = {}
        # Getting the type of 'array' (line 254)
        array_393878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 14), 'array', False)
        # Calling array(args, kwargs) (line 254)
        array_call_result_393887 = invoke(stypy.reporting.localization.Localization(__file__, 254, 14), array_393878, *[list_393879], **kwargs_393886)
        
        # Assigning a type to the variable 'col' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'col', array_call_result_393887)
        
        # Assigning a Call to a Name (line 255):
        
        # Call to array(...): (line 255)
        # Processing the call arguments (line 255)
        
        # Obtaining an instance of the builtin type 'list' (line 255)
        list_393889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 255)
        # Adding element type (line 255)
        int_393890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 21), list_393889, int_393890)
        # Adding element type (line 255)
        int_393891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 21), list_393889, int_393891)
        # Adding element type (line 255)
        int_393892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 21), list_393889, int_393892)
        # Adding element type (line 255)
        int_393893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 21), list_393889, int_393893)
        # Adding element type (line 255)
        int_393894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 21), list_393889, int_393894)
        # Adding element type (line 255)
        int_393895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 21), list_393889, int_393895)
        
        # Processing the call keyword arguments (line 255)
        kwargs_393896 = {}
        # Getting the type of 'array' (line 255)
        array_393888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'array', False)
        # Calling array(args, kwargs) (line 255)
        array_call_result_393897 = invoke(stypy.reporting.localization.Localization(__file__, 255, 15), array_393888, *[list_393889], **kwargs_393896)
        
        # Assigning a type to the variable 'data' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'data', array_call_result_393897)
        
        # Assigning a Call to a Name (line 256):
        
        # Call to csr_matrix(...): (line 256)
        # Processing the call arguments (line 256)
        
        # Obtaining an instance of the builtin type 'tuple' (line 256)
        tuple_393899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 256)
        # Adding element type (line 256)
        # Getting the type of 'data' (line 256)
        data_393900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 25), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 25), tuple_393899, data_393900)
        # Adding element type (line 256)
        
        # Obtaining an instance of the builtin type 'tuple' (line 256)
        tuple_393901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 256)
        # Adding element type (line 256)
        # Getting the type of 'row' (line 256)
        row_393902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 31), 'row', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 31), tuple_393901, row_393902)
        # Adding element type (line 256)
        # Getting the type of 'col' (line 256)
        col_393903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 35), 'col', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 31), tuple_393901, col_393903)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 25), tuple_393899, tuple_393901)
        
        # Processing the call keyword arguments (line 256)
        
        # Obtaining an instance of the builtin type 'tuple' (line 256)
        tuple_393904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 256)
        # Adding element type (line 256)
        int_393905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 49), tuple_393904, int_393905)
        # Adding element type (line 256)
        int_393906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 49), tuple_393904, int_393906)
        
        keyword_393907 = tuple_393904
        # Getting the type of 'float' (line 256)
        float_393908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 61), 'float', False)
        keyword_393909 = float_393908
        kwargs_393910 = {'dtype': keyword_393909, 'shape': keyword_393907}
        # Getting the type of 'csr_matrix' (line 256)
        csr_matrix_393898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 13), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 256)
        csr_matrix_call_result_393911 = invoke(stypy.reporting.localization.Localization(__file__, 256, 13), csr_matrix_393898, *[tuple_393899], **kwargs_393910)
        
        # Assigning a type to the variable 'sM' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'sM', csr_matrix_call_result_393911)
        
        # Assigning a Call to a Name (line 257):
        
        # Call to todense(...): (line 257)
        # Processing the call keyword arguments (line 257)
        kwargs_393914 = {}
        # Getting the type of 'sM' (line 257)
        sM_393912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'sM', False)
        # Obtaining the member 'todense' of a type (line 257)
        todense_393913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), sM_393912, 'todense')
        # Calling todense(args, kwargs) (line 257)
        todense_call_result_393915 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), todense_393913, *[], **kwargs_393914)
        
        # Assigning a type to the variable 'M' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'M', todense_call_result_393915)
        
        # Assigning a Call to a Name (line 259):
        
        # Call to array(...): (line 259)
        # Processing the call arguments (line 259)
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_393917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        int_393918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 20), list_393917, int_393918)
        # Adding element type (line 259)
        int_393919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 20), list_393917, int_393919)
        # Adding element type (line 259)
        int_393920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 20), list_393917, int_393920)
        # Adding element type (line 259)
        int_393921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 20), list_393917, int_393921)
        # Adding element type (line 259)
        int_393922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 20), list_393917, int_393922)
        # Adding element type (line 259)
        int_393923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 20), list_393917, int_393923)
        
        # Processing the call keyword arguments (line 259)
        kwargs_393924 = {}
        # Getting the type of 'array' (line 259)
        array_393916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 14), 'array', False)
        # Calling array(args, kwargs) (line 259)
        array_call_result_393925 = invoke(stypy.reporting.localization.Localization(__file__, 259, 14), array_393916, *[list_393917], **kwargs_393924)
        
        # Assigning a type to the variable 'row' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'row', array_call_result_393925)
        
        # Assigning a Call to a Name (line 260):
        
        # Call to array(...): (line 260)
        # Processing the call arguments (line 260)
        
        # Obtaining an instance of the builtin type 'list' (line 260)
        list_393927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 260)
        # Adding element type (line 260)
        int_393928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 20), list_393927, int_393928)
        # Adding element type (line 260)
        int_393929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 20), list_393927, int_393929)
        # Adding element type (line 260)
        int_393930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 20), list_393927, int_393930)
        # Adding element type (line 260)
        int_393931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 20), list_393927, int_393931)
        # Adding element type (line 260)
        int_393932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 20), list_393927, int_393932)
        # Adding element type (line 260)
        int_393933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 20), list_393927, int_393933)
        
        # Processing the call keyword arguments (line 260)
        kwargs_393934 = {}
        # Getting the type of 'array' (line 260)
        array_393926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 14), 'array', False)
        # Calling array(args, kwargs) (line 260)
        array_call_result_393935 = invoke(stypy.reporting.localization.Localization(__file__, 260, 14), array_393926, *[list_393927], **kwargs_393934)
        
        # Assigning a type to the variable 'col' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'col', array_call_result_393935)
        
        # Assigning a Call to a Name (line 261):
        
        # Call to array(...): (line 261)
        # Processing the call arguments (line 261)
        
        # Obtaining an instance of the builtin type 'list' (line 261)
        list_393937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 261)
        # Adding element type (line 261)
        int_393938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 21), list_393937, int_393938)
        # Adding element type (line 261)
        int_393939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 21), list_393937, int_393939)
        # Adding element type (line 261)
        int_393940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 21), list_393937, int_393940)
        # Adding element type (line 261)
        int_393941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 21), list_393937, int_393941)
        # Adding element type (line 261)
        int_393942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 21), list_393937, int_393942)
        # Adding element type (line 261)
        int_393943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 21), list_393937, int_393943)
        
        # Processing the call keyword arguments (line 261)
        kwargs_393944 = {}
        # Getting the type of 'array' (line 261)
        array_393936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'array', False)
        # Calling array(args, kwargs) (line 261)
        array_call_result_393945 = invoke(stypy.reporting.localization.Localization(__file__, 261, 15), array_393936, *[list_393937], **kwargs_393944)
        
        # Assigning a type to the variable 'data' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'data', array_call_result_393945)
        
        # Assigning a Call to a Name (line 262):
        
        # Call to csr_matrix(...): (line 262)
        # Processing the call arguments (line 262)
        
        # Obtaining an instance of the builtin type 'tuple' (line 262)
        tuple_393947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 262)
        # Adding element type (line 262)
        # Getting the type of 'data' (line 262)
        data_393948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 25), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 25), tuple_393947, data_393948)
        # Adding element type (line 262)
        
        # Obtaining an instance of the builtin type 'tuple' (line 262)
        tuple_393949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 262)
        # Adding element type (line 262)
        # Getting the type of 'row' (line 262)
        row_393950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 32), 'row', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 32), tuple_393949, row_393950)
        # Adding element type (line 262)
        # Getting the type of 'col' (line 262)
        col_393951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 36), 'col', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 32), tuple_393949, col_393951)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 25), tuple_393947, tuple_393949)
        
        # Processing the call keyword arguments (line 262)
        
        # Obtaining an instance of the builtin type 'tuple' (line 262)
        tuple_393952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 262)
        # Adding element type (line 262)
        int_393953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 50), tuple_393952, int_393953)
        # Adding element type (line 262)
        int_393954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 50), tuple_393952, int_393954)
        
        keyword_393955 = tuple_393952
        # Getting the type of 'float' (line 262)
        float_393956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 62), 'float', False)
        keyword_393957 = float_393956
        kwargs_393958 = {'dtype': keyword_393957, 'shape': keyword_393955}
        # Getting the type of 'csr_matrix' (line 262)
        csr_matrix_393946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 13), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 262)
        csr_matrix_call_result_393959 = invoke(stypy.reporting.localization.Localization(__file__, 262, 13), csr_matrix_393946, *[tuple_393947], **kwargs_393958)
        
        # Assigning a type to the variable 'sN' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'sN', csr_matrix_call_result_393959)
        
        # Assigning a Call to a Name (line 263):
        
        # Call to todense(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_393962 = {}
        # Getting the type of 'sN' (line 263)
        sN_393960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'sN', False)
        # Obtaining the member 'todense' of a type (line 263)
        todense_393961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 12), sN_393960, 'todense')
        # Calling todense(args, kwargs) (line 263)
        todense_call_result_393963 = invoke(stypy.reporting.localization.Localization(__file__, 263, 12), todense_393961, *[], **kwargs_393962)
        
        # Assigning a type to the variable 'N' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'N', todense_call_result_393963)
        
        # Assigning a Call to a Name (line 265):
        
        # Call to spsolve(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'sM' (line 265)
        sM_393965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 21), 'sM', False)
        # Getting the type of 'sN' (line 265)
        sN_393966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 25), 'sN', False)
        # Processing the call keyword arguments (line 265)
        kwargs_393967 = {}
        # Getting the type of 'spsolve' (line 265)
        spsolve_393964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 13), 'spsolve', False)
        # Calling spsolve(args, kwargs) (line 265)
        spsolve_call_result_393968 = invoke(stypy.reporting.localization.Localization(__file__, 265, 13), spsolve_393964, *[sM_393965, sN_393966], **kwargs_393967)
        
        # Assigning a type to the variable 'sX' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'sX', spsolve_call_result_393968)
        
        # Assigning a Call to a Name (line 266):
        
        # Call to solve(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'M' (line 266)
        M_393972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 31), 'M', False)
        # Getting the type of 'N' (line 266)
        N_393973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 34), 'N', False)
        # Processing the call keyword arguments (line 266)
        kwargs_393974 = {}
        # Getting the type of 'scipy' (line 266)
        scipy_393969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 266)
        linalg_393970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), scipy_393969, 'linalg')
        # Obtaining the member 'solve' of a type (line 266)
        solve_393971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), linalg_393970, 'solve')
        # Calling solve(args, kwargs) (line 266)
        solve_call_result_393975 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), solve_393971, *[M_393972, N_393973], **kwargs_393974)
        
        # Assigning a type to the variable 'X' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'X', solve_call_result_393975)
        
        # Call to assert_array_almost_equal(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'X' (line 268)
        X_393977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 34), 'X', False)
        
        # Call to todense(...): (line 268)
        # Processing the call keyword arguments (line 268)
        kwargs_393980 = {}
        # Getting the type of 'sX' (line 268)
        sX_393978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 37), 'sX', False)
        # Obtaining the member 'todense' of a type (line 268)
        todense_393979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 37), sX_393978, 'todense')
        # Calling todense(args, kwargs) (line 268)
        todense_call_result_393981 = invoke(stypy.reporting.localization.Localization(__file__, 268, 37), todense_393979, *[], **kwargs_393980)
        
        # Processing the call keyword arguments (line 268)
        kwargs_393982 = {}
        # Getting the type of 'assert_array_almost_equal' (line 268)
        assert_array_almost_equal_393976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 268)
        assert_array_almost_equal_call_result_393983 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), assert_array_almost_equal_393976, *[X_393977, todense_call_result_393981], **kwargs_393982)
        
        
        # ################# End of 'test_example_comparison(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_example_comparison' in the type store
        # Getting the type of 'stypy_return_type' (line 251)
        stypy_return_type_393984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393984)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_example_comparison'
        return stypy_return_type_393984


    @norecursion
    def test_shape_compatibility(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_shape_compatibility'
        module_type_store = module_type_store.open_function_context('test_shape_compatibility', 270, 4, False)
        # Assigning a type to the variable 'self' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinsolve.test_shape_compatibility.__dict__.__setitem__('stypy_localization', localization)
        TestLinsolve.test_shape_compatibility.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinsolve.test_shape_compatibility.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinsolve.test_shape_compatibility.__dict__.__setitem__('stypy_function_name', 'TestLinsolve.test_shape_compatibility')
        TestLinsolve.test_shape_compatibility.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinsolve.test_shape_compatibility.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinsolve.test_shape_compatibility.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinsolve.test_shape_compatibility.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinsolve.test_shape_compatibility.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinsolve.test_shape_compatibility.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinsolve.test_shape_compatibility.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.test_shape_compatibility', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_shape_compatibility', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_shape_compatibility(...)' code ##################

        
        # Call to use_solver(...): (line 273)
        # Processing the call keyword arguments (line 273)
        # Getting the type of 'True' (line 273)
        True_393986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 30), 'True', False)
        keyword_393987 = True_393986
        kwargs_393988 = {'useUmfpack': keyword_393987}
        # Getting the type of 'use_solver' (line 273)
        use_solver_393985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 273)
        use_solver_call_result_393989 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), use_solver_393985, *[], **kwargs_393988)
        
        
        # Assigning a Call to a Name (line 274):
        
        # Call to csc_matrix(...): (line 274)
        # Processing the call arguments (line 274)
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_393991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_393992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        float_393993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 24), list_393992, float_393993)
        # Adding element type (line 274)
        int_393994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 24), list_393992, int_393994)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 23), list_393991, list_393992)
        # Adding element type (line 274)
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_393995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        int_393996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 33), list_393995, int_393996)
        # Adding element type (line 274)
        int_393997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 33), list_393995, int_393997)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 23), list_393991, list_393995)
        
        # Processing the call keyword arguments (line 274)
        kwargs_393998 = {}
        # Getting the type of 'csc_matrix' (line 274)
        csc_matrix_393990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 274)
        csc_matrix_call_result_393999 = invoke(stypy.reporting.localization.Localization(__file__, 274, 12), csc_matrix_393990, *[list_393991], **kwargs_393998)
        
        # Assigning a type to the variable 'A' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'A', csc_matrix_call_result_393999)
        
        # Assigning a List to a Name (line 275):
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_394000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_394001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        # Adding element type (line 276)
        int_394002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 12), list_394001, int_394002)
        # Adding element type (line 276)
        int_394003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 12), list_394001, int_394003)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 13), list_394000, list_394001)
        # Adding element type (line 275)
        
        # Call to array(...): (line 277)
        # Processing the call arguments (line 277)
        
        # Obtaining an instance of the builtin type 'list' (line 277)
        list_394005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 277)
        # Adding element type (line 277)
        int_394006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 18), list_394005, int_394006)
        # Adding element type (line 277)
        int_394007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 18), list_394005, int_394007)
        
        # Processing the call keyword arguments (line 277)
        kwargs_394008 = {}
        # Getting the type of 'array' (line 277)
        array_394004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'array', False)
        # Calling array(args, kwargs) (line 277)
        array_call_result_394009 = invoke(stypy.reporting.localization.Localization(__file__, 277, 12), array_394004, *[list_394005], **kwargs_394008)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 13), list_394000, array_call_result_394009)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 278)
        list_394010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 278)
        # Adding element type (line 278)
        
        # Obtaining an instance of the builtin type 'list' (line 278)
        list_394011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 278)
        # Adding element type (line 278)
        int_394012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 13), list_394011, int_394012)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 12), list_394010, list_394011)
        # Adding element type (line 278)
        
        # Obtaining an instance of the builtin type 'list' (line 278)
        list_394013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 278)
        # Adding element type (line 278)
        int_394014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 18), list_394013, int_394014)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 12), list_394010, list_394013)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 13), list_394000, list_394010)
        # Adding element type (line 275)
        
        # Call to array(...): (line 279)
        # Processing the call arguments (line 279)
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_394016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        # Adding element type (line 279)
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_394017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        # Adding element type (line 279)
        int_394018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 19), list_394017, int_394018)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 18), list_394016, list_394017)
        # Adding element type (line 279)
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_394019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        # Adding element type (line 279)
        int_394020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 24), list_394019, int_394020)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 18), list_394016, list_394019)
        
        # Processing the call keyword arguments (line 279)
        kwargs_394021 = {}
        # Getting the type of 'array' (line 279)
        array_394015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'array', False)
        # Calling array(args, kwargs) (line 279)
        array_call_result_394022 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), array_394015, *[list_394016], **kwargs_394021)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 13), list_394000, array_call_result_394022)
        # Adding element type (line 275)
        
        # Call to csc_matrix(...): (line 280)
        # Processing the call arguments (line 280)
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_394024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_394025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        int_394026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 24), list_394025, int_394026)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 23), list_394024, list_394025)
        # Adding element type (line 280)
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_394027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        int_394028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 29), list_394027, int_394028)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 23), list_394024, list_394027)
        
        # Processing the call keyword arguments (line 280)
        kwargs_394029 = {}
        # Getting the type of 'csc_matrix' (line 280)
        csc_matrix_394023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 280)
        csc_matrix_call_result_394030 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), csc_matrix_394023, *[list_394024], **kwargs_394029)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 13), list_394000, csc_matrix_call_result_394030)
        # Adding element type (line 275)
        
        # Call to csr_matrix(...): (line 281)
        # Processing the call arguments (line 281)
        
        # Obtaining an instance of the builtin type 'list' (line 281)
        list_394032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 281)
        # Adding element type (line 281)
        
        # Obtaining an instance of the builtin type 'list' (line 281)
        list_394033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 281)
        # Adding element type (line 281)
        int_394034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 24), list_394033, int_394034)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 23), list_394032, list_394033)
        # Adding element type (line 281)
        
        # Obtaining an instance of the builtin type 'list' (line 281)
        list_394035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 281)
        # Adding element type (line 281)
        int_394036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 29), list_394035, int_394036)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 23), list_394032, list_394035)
        
        # Processing the call keyword arguments (line 281)
        kwargs_394037 = {}
        # Getting the type of 'csr_matrix' (line 281)
        csr_matrix_394031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 281)
        csr_matrix_call_result_394038 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), csr_matrix_394031, *[list_394032], **kwargs_394037)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 13), list_394000, csr_matrix_call_result_394038)
        # Adding element type (line 275)
        
        # Call to dok_matrix(...): (line 282)
        # Processing the call arguments (line 282)
        
        # Obtaining an instance of the builtin type 'list' (line 282)
        list_394040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 282)
        # Adding element type (line 282)
        
        # Obtaining an instance of the builtin type 'list' (line 282)
        list_394041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 282)
        # Adding element type (line 282)
        int_394042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 24), list_394041, int_394042)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 23), list_394040, list_394041)
        # Adding element type (line 282)
        
        # Obtaining an instance of the builtin type 'list' (line 282)
        list_394043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 282)
        # Adding element type (line 282)
        int_394044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 29), list_394043, int_394044)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 23), list_394040, list_394043)
        
        # Processing the call keyword arguments (line 282)
        kwargs_394045 = {}
        # Getting the type of 'dok_matrix' (line 282)
        dok_matrix_394039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 282)
        dok_matrix_call_result_394046 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), dok_matrix_394039, *[list_394040], **kwargs_394045)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 13), list_394000, dok_matrix_call_result_394046)
        # Adding element type (line 275)
        
        # Call to bsr_matrix(...): (line 283)
        # Processing the call arguments (line 283)
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_394048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_394049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        int_394050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 24), list_394049, int_394050)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 23), list_394048, list_394049)
        # Adding element type (line 283)
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_394051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        int_394052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 29), list_394051, int_394052)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 23), list_394048, list_394051)
        
        # Processing the call keyword arguments (line 283)
        kwargs_394053 = {}
        # Getting the type of 'bsr_matrix' (line 283)
        bsr_matrix_394047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'bsr_matrix', False)
        # Calling bsr_matrix(args, kwargs) (line 283)
        bsr_matrix_call_result_394054 = invoke(stypy.reporting.localization.Localization(__file__, 283, 12), bsr_matrix_394047, *[list_394048], **kwargs_394053)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 13), list_394000, bsr_matrix_call_result_394054)
        # Adding element type (line 275)
        
        # Call to array(...): (line 284)
        # Processing the call arguments (line 284)
        
        # Obtaining an instance of the builtin type 'list' (line 284)
        list_394056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 284)
        # Adding element type (line 284)
        
        # Obtaining an instance of the builtin type 'list' (line 284)
        list_394057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 284)
        # Adding element type (line 284)
        float_394058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 19), list_394057, float_394058)
        # Adding element type (line 284)
        float_394059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 19), list_394057, float_394059)
        # Adding element type (line 284)
        float_394060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 19), list_394057, float_394060)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 18), list_394056, list_394057)
        # Adding element type (line 284)
        
        # Obtaining an instance of the builtin type 'list' (line 284)
        list_394061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 284)
        # Adding element type (line 284)
        float_394062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 33), list_394061, float_394062)
        # Adding element type (line 284)
        float_394063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 33), list_394061, float_394063)
        # Adding element type (line 284)
        float_394064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 33), list_394061, float_394064)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 18), list_394056, list_394061)
        
        # Processing the call keyword arguments (line 284)
        kwargs_394065 = {}
        # Getting the type of 'array' (line 284)
        array_394055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'array', False)
        # Calling array(args, kwargs) (line 284)
        array_call_result_394066 = invoke(stypy.reporting.localization.Localization(__file__, 284, 12), array_394055, *[list_394056], **kwargs_394065)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 13), list_394000, array_call_result_394066)
        # Adding element type (line 275)
        
        # Call to csc_matrix(...): (line 285)
        # Processing the call arguments (line 285)
        
        # Obtaining an instance of the builtin type 'list' (line 285)
        list_394068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 285)
        # Adding element type (line 285)
        
        # Obtaining an instance of the builtin type 'list' (line 285)
        list_394069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 285)
        # Adding element type (line 285)
        float_394070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 24), list_394069, float_394070)
        # Adding element type (line 285)
        float_394071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 24), list_394069, float_394071)
        # Adding element type (line 285)
        float_394072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 24), list_394069, float_394072)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 23), list_394068, list_394069)
        # Adding element type (line 285)
        
        # Obtaining an instance of the builtin type 'list' (line 285)
        list_394073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 285)
        # Adding element type (line 285)
        float_394074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 38), list_394073, float_394074)
        # Adding element type (line 285)
        float_394075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 38), list_394073, float_394075)
        # Adding element type (line 285)
        float_394076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 38), list_394073, float_394076)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 23), list_394068, list_394073)
        
        # Processing the call keyword arguments (line 285)
        kwargs_394077 = {}
        # Getting the type of 'csc_matrix' (line 285)
        csc_matrix_394067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 285)
        csc_matrix_call_result_394078 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), csc_matrix_394067, *[list_394068], **kwargs_394077)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 13), list_394000, csc_matrix_call_result_394078)
        # Adding element type (line 275)
        
        # Call to csr_matrix(...): (line 286)
        # Processing the call arguments (line 286)
        
        # Obtaining an instance of the builtin type 'list' (line 286)
        list_394080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 286)
        # Adding element type (line 286)
        
        # Obtaining an instance of the builtin type 'list' (line 286)
        list_394081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 286)
        # Adding element type (line 286)
        float_394082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 24), list_394081, float_394082)
        # Adding element type (line 286)
        float_394083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 24), list_394081, float_394083)
        # Adding element type (line 286)
        float_394084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 24), list_394081, float_394084)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 23), list_394080, list_394081)
        # Adding element type (line 286)
        
        # Obtaining an instance of the builtin type 'list' (line 286)
        list_394085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 286)
        # Adding element type (line 286)
        float_394086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 38), list_394085, float_394086)
        # Adding element type (line 286)
        float_394087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 38), list_394085, float_394087)
        # Adding element type (line 286)
        float_394088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 38), list_394085, float_394088)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 23), list_394080, list_394085)
        
        # Processing the call keyword arguments (line 286)
        kwargs_394089 = {}
        # Getting the type of 'csr_matrix' (line 286)
        csr_matrix_394079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 286)
        csr_matrix_call_result_394090 = invoke(stypy.reporting.localization.Localization(__file__, 286, 12), csr_matrix_394079, *[list_394080], **kwargs_394089)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 13), list_394000, csr_matrix_call_result_394090)
        # Adding element type (line 275)
        
        # Call to dok_matrix(...): (line 287)
        # Processing the call arguments (line 287)
        
        # Obtaining an instance of the builtin type 'list' (line 287)
        list_394092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 287)
        # Adding element type (line 287)
        
        # Obtaining an instance of the builtin type 'list' (line 287)
        list_394093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 287)
        # Adding element type (line 287)
        float_394094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 24), list_394093, float_394094)
        # Adding element type (line 287)
        float_394095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 24), list_394093, float_394095)
        # Adding element type (line 287)
        float_394096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 24), list_394093, float_394096)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 23), list_394092, list_394093)
        # Adding element type (line 287)
        
        # Obtaining an instance of the builtin type 'list' (line 287)
        list_394097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 287)
        # Adding element type (line 287)
        float_394098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 38), list_394097, float_394098)
        # Adding element type (line 287)
        float_394099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 38), list_394097, float_394099)
        # Adding element type (line 287)
        float_394100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 38), list_394097, float_394100)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 23), list_394092, list_394097)
        
        # Processing the call keyword arguments (line 287)
        kwargs_394101 = {}
        # Getting the type of 'dok_matrix' (line 287)
        dok_matrix_394091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 287)
        dok_matrix_call_result_394102 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), dok_matrix_394091, *[list_394092], **kwargs_394101)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 13), list_394000, dok_matrix_call_result_394102)
        # Adding element type (line 275)
        
        # Call to bsr_matrix(...): (line 288)
        # Processing the call arguments (line 288)
        
        # Obtaining an instance of the builtin type 'list' (line 288)
        list_394104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 288)
        # Adding element type (line 288)
        
        # Obtaining an instance of the builtin type 'list' (line 288)
        list_394105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 288)
        # Adding element type (line 288)
        float_394106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 24), list_394105, float_394106)
        # Adding element type (line 288)
        float_394107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 24), list_394105, float_394107)
        # Adding element type (line 288)
        float_394108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 24), list_394105, float_394108)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 23), list_394104, list_394105)
        # Adding element type (line 288)
        
        # Obtaining an instance of the builtin type 'list' (line 288)
        list_394109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 288)
        # Adding element type (line 288)
        float_394110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 38), list_394109, float_394110)
        # Adding element type (line 288)
        float_394111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 38), list_394109, float_394111)
        # Adding element type (line 288)
        float_394112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 38), list_394109, float_394112)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 23), list_394104, list_394109)
        
        # Processing the call keyword arguments (line 288)
        kwargs_394113 = {}
        # Getting the type of 'bsr_matrix' (line 288)
        bsr_matrix_394103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'bsr_matrix', False)
        # Calling bsr_matrix(args, kwargs) (line 288)
        bsr_matrix_call_result_394114 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), bsr_matrix_394103, *[list_394104], **kwargs_394113)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 13), list_394000, bsr_matrix_call_result_394114)
        
        # Assigning a type to the variable 'bs' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'bs', list_394000)
        
        # Getting the type of 'bs' (line 291)
        bs_394115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 17), 'bs')
        # Testing the type of a for loop iterable (line 291)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 291, 8), bs_394115)
        # Getting the type of the for loop variable (line 291)
        for_loop_var_394116 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 291, 8), bs_394115)
        # Assigning a type to the variable 'b' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'b', for_loop_var_394116)
        # SSA begins for a for statement (line 291)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 292):
        
        # Call to solve(...): (line 292)
        # Processing the call arguments (line 292)
        
        # Call to toarray(...): (line 292)
        # Processing the call keyword arguments (line 292)
        kwargs_394122 = {}
        # Getting the type of 'A' (line 292)
        A_394120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 32), 'A', False)
        # Obtaining the member 'toarray' of a type (line 292)
        toarray_394121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 32), A_394120, 'toarray')
        # Calling toarray(args, kwargs) (line 292)
        toarray_call_result_394123 = invoke(stypy.reporting.localization.Localization(__file__, 292, 32), toarray_394121, *[], **kwargs_394122)
        
        
        # Call to toarray(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'b' (line 292)
        b_394125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 53), 'b', False)
        # Processing the call keyword arguments (line 292)
        kwargs_394126 = {}
        # Getting the type of 'toarray' (line 292)
        toarray_394124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 45), 'toarray', False)
        # Calling toarray(args, kwargs) (line 292)
        toarray_call_result_394127 = invoke(stypy.reporting.localization.Localization(__file__, 292, 45), toarray_394124, *[b_394125], **kwargs_394126)
        
        # Processing the call keyword arguments (line 292)
        kwargs_394128 = {}
        # Getting the type of 'np' (line 292)
        np_394117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'np', False)
        # Obtaining the member 'linalg' of a type (line 292)
        linalg_394118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 16), np_394117, 'linalg')
        # Obtaining the member 'solve' of a type (line 292)
        solve_394119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 16), linalg_394118, 'solve')
        # Calling solve(args, kwargs) (line 292)
        solve_call_result_394129 = invoke(stypy.reporting.localization.Localization(__file__, 292, 16), solve_394119, *[toarray_call_result_394123, toarray_call_result_394127], **kwargs_394128)
        
        # Assigning a type to the variable 'x' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'x', solve_call_result_394129)
        
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_394130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        # Getting the type of 'csc_matrix' (line 293)
        csc_matrix_394131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 30), 'csc_matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 29), list_394130, csc_matrix_394131)
        # Adding element type (line 293)
        # Getting the type of 'csr_matrix' (line 293)
        csr_matrix_394132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 42), 'csr_matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 29), list_394130, csr_matrix_394132)
        # Adding element type (line 293)
        # Getting the type of 'dok_matrix' (line 293)
        dok_matrix_394133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 54), 'dok_matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 29), list_394130, dok_matrix_394133)
        # Adding element type (line 293)
        # Getting the type of 'lil_matrix' (line 293)
        lil_matrix_394134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 66), 'lil_matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 29), list_394130, lil_matrix_394134)
        
        # Testing the type of a for loop iterable (line 293)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 293, 12), list_394130)
        # Getting the type of the for loop variable (line 293)
        for_loop_var_394135 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 293, 12), list_394130)
        # Assigning a type to the variable 'spmattype' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'spmattype', for_loop_var_394135)
        # SSA begins for a for statement (line 293)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 294):
        
        # Call to spsolve(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Call to spmattype(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'A' (line 294)
        A_394138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 39), 'A', False)
        # Processing the call keyword arguments (line 294)
        kwargs_394139 = {}
        # Getting the type of 'spmattype' (line 294)
        spmattype_394137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 29), 'spmattype', False)
        # Calling spmattype(args, kwargs) (line 294)
        spmattype_call_result_394140 = invoke(stypy.reporting.localization.Localization(__file__, 294, 29), spmattype_394137, *[A_394138], **kwargs_394139)
        
        # Getting the type of 'b' (line 294)
        b_394141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 43), 'b', False)
        # Processing the call keyword arguments (line 294)
        # Getting the type of 'True' (line 294)
        True_394142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 58), 'True', False)
        keyword_394143 = True_394142
        kwargs_394144 = {'use_umfpack': keyword_394143}
        # Getting the type of 'spsolve' (line 294)
        spsolve_394136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'spsolve', False)
        # Calling spsolve(args, kwargs) (line 294)
        spsolve_call_result_394145 = invoke(stypy.reporting.localization.Localization(__file__, 294, 21), spsolve_394136, *[spmattype_call_result_394140, b_394141], **kwargs_394144)
        
        # Assigning a type to the variable 'x1' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'x1', spsolve_call_result_394145)
        
        # Assigning a Call to a Name (line 295):
        
        # Call to spsolve(...): (line 295)
        # Processing the call arguments (line 295)
        
        # Call to spmattype(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'A' (line 295)
        A_394148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 39), 'A', False)
        # Processing the call keyword arguments (line 295)
        kwargs_394149 = {}
        # Getting the type of 'spmattype' (line 295)
        spmattype_394147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 29), 'spmattype', False)
        # Calling spmattype(args, kwargs) (line 295)
        spmattype_call_result_394150 = invoke(stypy.reporting.localization.Localization(__file__, 295, 29), spmattype_394147, *[A_394148], **kwargs_394149)
        
        # Getting the type of 'b' (line 295)
        b_394151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 43), 'b', False)
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'False' (line 295)
        False_394152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 58), 'False', False)
        keyword_394153 = False_394152
        kwargs_394154 = {'use_umfpack': keyword_394153}
        # Getting the type of 'spsolve' (line 295)
        spsolve_394146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 21), 'spsolve', False)
        # Calling spsolve(args, kwargs) (line 295)
        spsolve_call_result_394155 = invoke(stypy.reporting.localization.Localization(__file__, 295, 21), spsolve_394146, *[spmattype_call_result_394150, b_394151], **kwargs_394154)
        
        # Assigning a type to the variable 'x2' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'x2', spsolve_call_result_394155)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 298)
        x_394156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 19), 'x')
        # Obtaining the member 'ndim' of a type (line 298)
        ndim_394157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 19), x_394156, 'ndim')
        int_394158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 29), 'int')
        # Applying the binary operator '==' (line 298)
        result_eq_394159 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 19), '==', ndim_394157, int_394158)
        
        
        
        # Obtaining the type of the subscript
        int_394160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 43), 'int')
        # Getting the type of 'x' (line 298)
        x_394161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 35), 'x')
        # Obtaining the member 'shape' of a type (line 298)
        shape_394162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 35), x_394161, 'shape')
        # Obtaining the member '__getitem__' of a type (line 298)
        getitem___394163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 35), shape_394162, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 298)
        subscript_call_result_394164 = invoke(stypy.reporting.localization.Localization(__file__, 298, 35), getitem___394163, int_394160)
        
        int_394165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 49), 'int')
        # Applying the binary operator '==' (line 298)
        result_eq_394166 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 35), '==', subscript_call_result_394164, int_394165)
        
        # Applying the binary operator 'and' (line 298)
        result_and_keyword_394167 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 19), 'and', result_eq_394159, result_eq_394166)
        
        # Testing the type of an if condition (line 298)
        if_condition_394168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 16), result_and_keyword_394167)
        # Assigning a type to the variable 'if_condition_394168' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'if_condition_394168', if_condition_394168)
        # SSA begins for if statement (line 298)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 300):
        
        # Call to ravel(...): (line 300)
        # Processing the call keyword arguments (line 300)
        kwargs_394171 = {}
        # Getting the type of 'x' (line 300)
        x_394169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 24), 'x', False)
        # Obtaining the member 'ravel' of a type (line 300)
        ravel_394170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 24), x_394169, 'ravel')
        # Calling ravel(args, kwargs) (line 300)
        ravel_call_result_394172 = invoke(stypy.reporting.localization.Localization(__file__, 300, 24), ravel_394170, *[], **kwargs_394171)
        
        # Assigning a type to the variable 'x' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'x', ravel_call_result_394172)
        # SSA join for if statement (line 298)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_array_almost_equal(...): (line 302)
        # Processing the call arguments (line 302)
        
        # Call to toarray(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'x1' (line 302)
        x1_394175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 50), 'x1', False)
        # Processing the call keyword arguments (line 302)
        kwargs_394176 = {}
        # Getting the type of 'toarray' (line 302)
        toarray_394174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 42), 'toarray', False)
        # Calling toarray(args, kwargs) (line 302)
        toarray_call_result_394177 = invoke(stypy.reporting.localization.Localization(__file__, 302, 42), toarray_394174, *[x1_394175], **kwargs_394176)
        
        # Getting the type of 'x' (line 302)
        x_394178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 55), 'x', False)
        # Processing the call keyword arguments (line 302)
        
        # Call to repr(...): (line 302)
        # Processing the call arguments (line 302)
        
        # Obtaining an instance of the builtin type 'tuple' (line 302)
        tuple_394180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 72), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 302)
        # Adding element type (line 302)
        # Getting the type of 'b' (line 302)
        b_394181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 72), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 72), tuple_394180, b_394181)
        # Adding element type (line 302)
        # Getting the type of 'spmattype' (line 302)
        spmattype_394182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 75), 'spmattype', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 72), tuple_394180, spmattype_394182)
        # Adding element type (line 302)
        int_394183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 86), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 72), tuple_394180, int_394183)
        
        # Processing the call keyword arguments (line 302)
        kwargs_394184 = {}
        # Getting the type of 'repr' (line 302)
        repr_394179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 66), 'repr', False)
        # Calling repr(args, kwargs) (line 302)
        repr_call_result_394185 = invoke(stypy.reporting.localization.Localization(__file__, 302, 66), repr_394179, *[tuple_394180], **kwargs_394184)
        
        keyword_394186 = repr_call_result_394185
        kwargs_394187 = {'err_msg': keyword_394186}
        # Getting the type of 'assert_array_almost_equal' (line 302)
        assert_array_almost_equal_394173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 302)
        assert_array_almost_equal_call_result_394188 = invoke(stypy.reporting.localization.Localization(__file__, 302, 16), assert_array_almost_equal_394173, *[toarray_call_result_394177, x_394178], **kwargs_394187)
        
        
        # Call to assert_array_almost_equal(...): (line 303)
        # Processing the call arguments (line 303)
        
        # Call to toarray(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'x2' (line 303)
        x2_394191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 50), 'x2', False)
        # Processing the call keyword arguments (line 303)
        kwargs_394192 = {}
        # Getting the type of 'toarray' (line 303)
        toarray_394190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 42), 'toarray', False)
        # Calling toarray(args, kwargs) (line 303)
        toarray_call_result_394193 = invoke(stypy.reporting.localization.Localization(__file__, 303, 42), toarray_394190, *[x2_394191], **kwargs_394192)
        
        # Getting the type of 'x' (line 303)
        x_394194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 55), 'x', False)
        # Processing the call keyword arguments (line 303)
        
        # Call to repr(...): (line 303)
        # Processing the call arguments (line 303)
        
        # Obtaining an instance of the builtin type 'tuple' (line 303)
        tuple_394196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 72), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 303)
        # Adding element type (line 303)
        # Getting the type of 'b' (line 303)
        b_394197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 72), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 72), tuple_394196, b_394197)
        # Adding element type (line 303)
        # Getting the type of 'spmattype' (line 303)
        spmattype_394198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 75), 'spmattype', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 72), tuple_394196, spmattype_394198)
        # Adding element type (line 303)
        int_394199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 86), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 72), tuple_394196, int_394199)
        
        # Processing the call keyword arguments (line 303)
        kwargs_394200 = {}
        # Getting the type of 'repr' (line 303)
        repr_394195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 66), 'repr', False)
        # Calling repr(args, kwargs) (line 303)
        repr_call_result_394201 = invoke(stypy.reporting.localization.Localization(__file__, 303, 66), repr_394195, *[tuple_394196], **kwargs_394200)
        
        keyword_394202 = repr_call_result_394201
        kwargs_394203 = {'err_msg': keyword_394202}
        # Getting the type of 'assert_array_almost_equal' (line 303)
        assert_array_almost_equal_394189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 303)
        assert_array_almost_equal_call_result_394204 = invoke(stypy.reporting.localization.Localization(__file__, 303, 16), assert_array_almost_equal_394189, *[toarray_call_result_394193, x_394194], **kwargs_394203)
        
        
        
        # Evaluating a boolean operation
        
        # Call to isspmatrix(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'b' (line 306)
        b_394206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 30), 'b', False)
        # Processing the call keyword arguments (line 306)
        kwargs_394207 = {}
        # Getting the type of 'isspmatrix' (line 306)
        isspmatrix_394205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 306)
        isspmatrix_call_result_394208 = invoke(stypy.reporting.localization.Localization(__file__, 306, 19), isspmatrix_394205, *[b_394206], **kwargs_394207)
        
        
        # Getting the type of 'x' (line 306)
        x_394209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 37), 'x')
        # Obtaining the member 'ndim' of a type (line 306)
        ndim_394210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 37), x_394209, 'ndim')
        int_394211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 46), 'int')
        # Applying the binary operator '>' (line 306)
        result_gt_394212 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 37), '>', ndim_394210, int_394211)
        
        # Applying the binary operator 'and' (line 306)
        result_and_keyword_394213 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 19), 'and', isspmatrix_call_result_394208, result_gt_394212)
        
        # Testing the type of an if condition (line 306)
        if_condition_394214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 16), result_and_keyword_394213)
        # Assigning a type to the variable 'if_condition_394214' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'if_condition_394214', if_condition_394214)
        # SSA begins for if statement (line 306)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_(...): (line 307)
        # Processing the call arguments (line 307)
        
        # Call to isspmatrix(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'x1' (line 307)
        x1_394217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 39), 'x1', False)
        # Processing the call keyword arguments (line 307)
        kwargs_394218 = {}
        # Getting the type of 'isspmatrix' (line 307)
        isspmatrix_394216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 28), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 307)
        isspmatrix_call_result_394219 = invoke(stypy.reporting.localization.Localization(__file__, 307, 28), isspmatrix_394216, *[x1_394217], **kwargs_394218)
        
        
        # Call to repr(...): (line 307)
        # Processing the call arguments (line 307)
        
        # Obtaining an instance of the builtin type 'tuple' (line 307)
        tuple_394221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 307)
        # Adding element type (line 307)
        # Getting the type of 'b' (line 307)
        b_394222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 50), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 50), tuple_394221, b_394222)
        # Adding element type (line 307)
        # Getting the type of 'spmattype' (line 307)
        spmattype_394223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 53), 'spmattype', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 50), tuple_394221, spmattype_394223)
        # Adding element type (line 307)
        int_394224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 50), tuple_394221, int_394224)
        
        # Processing the call keyword arguments (line 307)
        kwargs_394225 = {}
        # Getting the type of 'repr' (line 307)
        repr_394220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 44), 'repr', False)
        # Calling repr(args, kwargs) (line 307)
        repr_call_result_394226 = invoke(stypy.reporting.localization.Localization(__file__, 307, 44), repr_394220, *[tuple_394221], **kwargs_394225)
        
        # Processing the call keyword arguments (line 307)
        kwargs_394227 = {}
        # Getting the type of 'assert_' (line 307)
        assert__394215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 20), 'assert_', False)
        # Calling assert_(args, kwargs) (line 307)
        assert__call_result_394228 = invoke(stypy.reporting.localization.Localization(__file__, 307, 20), assert__394215, *[isspmatrix_call_result_394219, repr_call_result_394226], **kwargs_394227)
        
        
        # Call to assert_(...): (line 308)
        # Processing the call arguments (line 308)
        
        # Call to isspmatrix(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'x2' (line 308)
        x2_394231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 39), 'x2', False)
        # Processing the call keyword arguments (line 308)
        kwargs_394232 = {}
        # Getting the type of 'isspmatrix' (line 308)
        isspmatrix_394230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 28), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 308)
        isspmatrix_call_result_394233 = invoke(stypy.reporting.localization.Localization(__file__, 308, 28), isspmatrix_394230, *[x2_394231], **kwargs_394232)
        
        
        # Call to repr(...): (line 308)
        # Processing the call arguments (line 308)
        
        # Obtaining an instance of the builtin type 'tuple' (line 308)
        tuple_394235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 308)
        # Adding element type (line 308)
        # Getting the type of 'b' (line 308)
        b_394236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 50), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 50), tuple_394235, b_394236)
        # Adding element type (line 308)
        # Getting the type of 'spmattype' (line 308)
        spmattype_394237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 53), 'spmattype', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 50), tuple_394235, spmattype_394237)
        # Adding element type (line 308)
        int_394238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 50), tuple_394235, int_394238)
        
        # Processing the call keyword arguments (line 308)
        kwargs_394239 = {}
        # Getting the type of 'repr' (line 308)
        repr_394234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 44), 'repr', False)
        # Calling repr(args, kwargs) (line 308)
        repr_call_result_394240 = invoke(stypy.reporting.localization.Localization(__file__, 308, 44), repr_394234, *[tuple_394235], **kwargs_394239)
        
        # Processing the call keyword arguments (line 308)
        kwargs_394241 = {}
        # Getting the type of 'assert_' (line 308)
        assert__394229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 20), 'assert_', False)
        # Calling assert_(args, kwargs) (line 308)
        assert__call_result_394242 = invoke(stypy.reporting.localization.Localization(__file__, 308, 20), assert__394229, *[isspmatrix_call_result_394233, repr_call_result_394240], **kwargs_394241)
        
        # SSA branch for the else part of an if statement (line 306)
        module_type_store.open_ssa_branch('else')
        
        # Call to assert_(...): (line 310)
        # Processing the call arguments (line 310)
        
        # Call to isinstance(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'x1' (line 310)
        x1_394245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 39), 'x1', False)
        # Getting the type of 'np' (line 310)
        np_394246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 43), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 310)
        ndarray_394247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 43), np_394246, 'ndarray')
        # Processing the call keyword arguments (line 310)
        kwargs_394248 = {}
        # Getting the type of 'isinstance' (line 310)
        isinstance_394244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 28), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 310)
        isinstance_call_result_394249 = invoke(stypy.reporting.localization.Localization(__file__, 310, 28), isinstance_394244, *[x1_394245, ndarray_394247], **kwargs_394248)
        
        
        # Call to repr(...): (line 310)
        # Processing the call arguments (line 310)
        
        # Obtaining an instance of the builtin type 'tuple' (line 310)
        tuple_394251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 62), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 310)
        # Adding element type (line 310)
        # Getting the type of 'b' (line 310)
        b_394252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 62), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 62), tuple_394251, b_394252)
        # Adding element type (line 310)
        # Getting the type of 'spmattype' (line 310)
        spmattype_394253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 65), 'spmattype', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 62), tuple_394251, spmattype_394253)
        # Adding element type (line 310)
        int_394254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 76), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 62), tuple_394251, int_394254)
        
        # Processing the call keyword arguments (line 310)
        kwargs_394255 = {}
        # Getting the type of 'repr' (line 310)
        repr_394250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 56), 'repr', False)
        # Calling repr(args, kwargs) (line 310)
        repr_call_result_394256 = invoke(stypy.reporting.localization.Localization(__file__, 310, 56), repr_394250, *[tuple_394251], **kwargs_394255)
        
        # Processing the call keyword arguments (line 310)
        kwargs_394257 = {}
        # Getting the type of 'assert_' (line 310)
        assert__394243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 20), 'assert_', False)
        # Calling assert_(args, kwargs) (line 310)
        assert__call_result_394258 = invoke(stypy.reporting.localization.Localization(__file__, 310, 20), assert__394243, *[isinstance_call_result_394249, repr_call_result_394256], **kwargs_394257)
        
        
        # Call to assert_(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Call to isinstance(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'x2' (line 311)
        x2_394261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 39), 'x2', False)
        # Getting the type of 'np' (line 311)
        np_394262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 43), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 311)
        ndarray_394263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 43), np_394262, 'ndarray')
        # Processing the call keyword arguments (line 311)
        kwargs_394264 = {}
        # Getting the type of 'isinstance' (line 311)
        isinstance_394260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 28), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 311)
        isinstance_call_result_394265 = invoke(stypy.reporting.localization.Localization(__file__, 311, 28), isinstance_394260, *[x2_394261, ndarray_394263], **kwargs_394264)
        
        
        # Call to repr(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Obtaining an instance of the builtin type 'tuple' (line 311)
        tuple_394267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 62), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 311)
        # Adding element type (line 311)
        # Getting the type of 'b' (line 311)
        b_394268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 62), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 62), tuple_394267, b_394268)
        # Adding element type (line 311)
        # Getting the type of 'spmattype' (line 311)
        spmattype_394269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 65), 'spmattype', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 62), tuple_394267, spmattype_394269)
        # Adding element type (line 311)
        int_394270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 76), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 62), tuple_394267, int_394270)
        
        # Processing the call keyword arguments (line 311)
        kwargs_394271 = {}
        # Getting the type of 'repr' (line 311)
        repr_394266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 56), 'repr', False)
        # Calling repr(args, kwargs) (line 311)
        repr_call_result_394272 = invoke(stypy.reporting.localization.Localization(__file__, 311, 56), repr_394266, *[tuple_394267], **kwargs_394271)
        
        # Processing the call keyword arguments (line 311)
        kwargs_394273 = {}
        # Getting the type of 'assert_' (line 311)
        assert__394259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 20), 'assert_', False)
        # Calling assert_(args, kwargs) (line 311)
        assert__call_result_394274 = invoke(stypy.reporting.localization.Localization(__file__, 311, 20), assert__394259, *[isinstance_call_result_394265, repr_call_result_394272], **kwargs_394273)
        
        # SSA join for if statement (line 306)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'x' (line 314)
        x_394275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 19), 'x')
        # Obtaining the member 'ndim' of a type (line 314)
        ndim_394276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 19), x_394275, 'ndim')
        int_394277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 29), 'int')
        # Applying the binary operator '==' (line 314)
        result_eq_394278 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 19), '==', ndim_394276, int_394277)
        
        # Testing the type of an if condition (line 314)
        if_condition_394279 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 16), result_eq_394278)
        # Assigning a type to the variable 'if_condition_394279' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'if_condition_394279', if_condition_394279)
        # SSA begins for if statement (line 314)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_equal(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'x1' (line 316)
        x1_394281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 33), 'x1', False)
        # Obtaining the member 'shape' of a type (line 316)
        shape_394282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 33), x1_394281, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 316)
        tuple_394283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 316)
        # Adding element type (line 316)
        
        # Obtaining the type of the subscript
        int_394284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 52), 'int')
        # Getting the type of 'A' (line 316)
        A_394285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 44), 'A', False)
        # Obtaining the member 'shape' of a type (line 316)
        shape_394286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 44), A_394285, 'shape')
        # Obtaining the member '__getitem__' of a type (line 316)
        getitem___394287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 44), shape_394286, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 316)
        subscript_call_result_394288 = invoke(stypy.reporting.localization.Localization(__file__, 316, 44), getitem___394287, int_394284)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 44), tuple_394283, subscript_call_result_394288)
        
        # Processing the call keyword arguments (line 316)
        kwargs_394289 = {}
        # Getting the type of 'assert_equal' (line 316)
        assert_equal_394280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 20), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 316)
        assert_equal_call_result_394290 = invoke(stypy.reporting.localization.Localization(__file__, 316, 20), assert_equal_394280, *[shape_394282, tuple_394283], **kwargs_394289)
        
        
        # Call to assert_equal(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'x2' (line 317)
        x2_394292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 33), 'x2', False)
        # Obtaining the member 'shape' of a type (line 317)
        shape_394293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 33), x2_394292, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 317)
        tuple_394294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 317)
        # Adding element type (line 317)
        
        # Obtaining the type of the subscript
        int_394295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 52), 'int')
        # Getting the type of 'A' (line 317)
        A_394296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 44), 'A', False)
        # Obtaining the member 'shape' of a type (line 317)
        shape_394297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 44), A_394296, 'shape')
        # Obtaining the member '__getitem__' of a type (line 317)
        getitem___394298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 44), shape_394297, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 317)
        subscript_call_result_394299 = invoke(stypy.reporting.localization.Localization(__file__, 317, 44), getitem___394298, int_394295)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 44), tuple_394294, subscript_call_result_394299)
        
        # Processing the call keyword arguments (line 317)
        kwargs_394300 = {}
        # Getting the type of 'assert_equal' (line 317)
        assert_equal_394291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 317)
        assert_equal_call_result_394301 = invoke(stypy.reporting.localization.Localization(__file__, 317, 20), assert_equal_394291, *[shape_394293, tuple_394294], **kwargs_394300)
        
        # SSA branch for the else part of an if statement (line 314)
        module_type_store.open_ssa_branch('else')
        
        # Call to assert_equal(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'x1' (line 320)
        x1_394303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 33), 'x1', False)
        # Obtaining the member 'shape' of a type (line 320)
        shape_394304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 33), x1_394303, 'shape')
        # Getting the type of 'x' (line 320)
        x_394305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 43), 'x', False)
        # Obtaining the member 'shape' of a type (line 320)
        shape_394306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 43), x_394305, 'shape')
        # Processing the call keyword arguments (line 320)
        kwargs_394307 = {}
        # Getting the type of 'assert_equal' (line 320)
        assert_equal_394302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 20), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 320)
        assert_equal_call_result_394308 = invoke(stypy.reporting.localization.Localization(__file__, 320, 20), assert_equal_394302, *[shape_394304, shape_394306], **kwargs_394307)
        
        
        # Call to assert_equal(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'x2' (line 321)
        x2_394310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 33), 'x2', False)
        # Obtaining the member 'shape' of a type (line 321)
        shape_394311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 33), x2_394310, 'shape')
        # Getting the type of 'x' (line 321)
        x_394312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 43), 'x', False)
        # Obtaining the member 'shape' of a type (line 321)
        shape_394313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 43), x_394312, 'shape')
        # Processing the call keyword arguments (line 321)
        kwargs_394314 = {}
        # Getting the type of 'assert_equal' (line 321)
        assert_equal_394309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 20), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 321)
        assert_equal_call_result_394315 = invoke(stypy.reporting.localization.Localization(__file__, 321, 20), assert_equal_394309, *[shape_394311, shape_394313], **kwargs_394314)
        
        # SSA join for if statement (line 314)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 323):
        
        # Call to csc_matrix(...): (line 323)
        # Processing the call arguments (line 323)
        
        # Obtaining an instance of the builtin type 'tuple' (line 323)
        tuple_394317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 323)
        # Adding element type (line 323)
        int_394318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 24), tuple_394317, int_394318)
        # Adding element type (line 323)
        int_394319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 24), tuple_394317, int_394319)
        
        # Processing the call keyword arguments (line 323)
        kwargs_394320 = {}
        # Getting the type of 'csc_matrix' (line 323)
        csc_matrix_394316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 323)
        csc_matrix_call_result_394321 = invoke(stypy.reporting.localization.Localization(__file__, 323, 12), csc_matrix_394316, *[tuple_394317], **kwargs_394320)
        
        # Assigning a type to the variable 'A' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'A', csc_matrix_call_result_394321)
        
        # Assigning a Call to a Name (line 324):
        
        # Call to csc_matrix(...): (line 324)
        # Processing the call arguments (line 324)
        
        # Obtaining an instance of the builtin type 'tuple' (line 324)
        tuple_394323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 324)
        # Adding element type (line 324)
        int_394324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 24), tuple_394323, int_394324)
        # Adding element type (line 324)
        int_394325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 24), tuple_394323, int_394325)
        
        # Processing the call keyword arguments (line 324)
        kwargs_394326 = {}
        # Getting the type of 'csc_matrix' (line 324)
        csc_matrix_394322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 324)
        csc_matrix_call_result_394327 = invoke(stypy.reporting.localization.Localization(__file__, 324, 12), csc_matrix_394322, *[tuple_394323], **kwargs_394326)
        
        # Assigning a type to the variable 'b' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'b', csc_matrix_call_result_394327)
        
        # Call to assert_raises(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'ValueError' (line 325)
        ValueError_394329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 22), 'ValueError', False)
        # Getting the type of 'spsolve' (line 325)
        spsolve_394330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 34), 'spsolve', False)
        # Getting the type of 'A' (line 325)
        A_394331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 43), 'A', False)
        # Getting the type of 'b' (line 325)
        b_394332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 46), 'b', False)
        # Processing the call keyword arguments (line 325)
        kwargs_394333 = {}
        # Getting the type of 'assert_raises' (line 325)
        assert_raises_394328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 325)
        assert_raises_call_result_394334 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), assert_raises_394328, *[ValueError_394329, spsolve_394330, A_394331, b_394332], **kwargs_394333)
        
        
        # ################# End of 'test_shape_compatibility(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_shape_compatibility' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_394335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_394335)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_shape_compatibility'
        return stypy_return_type_394335


    @norecursion
    def test_ndarray_support(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ndarray_support'
        module_type_store = module_type_store.open_function_context('test_ndarray_support', 327, 4, False)
        # Assigning a type to the variable 'self' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinsolve.test_ndarray_support.__dict__.__setitem__('stypy_localization', localization)
        TestLinsolve.test_ndarray_support.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinsolve.test_ndarray_support.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinsolve.test_ndarray_support.__dict__.__setitem__('stypy_function_name', 'TestLinsolve.test_ndarray_support')
        TestLinsolve.test_ndarray_support.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinsolve.test_ndarray_support.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinsolve.test_ndarray_support.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinsolve.test_ndarray_support.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinsolve.test_ndarray_support.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinsolve.test_ndarray_support.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinsolve.test_ndarray_support.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.test_ndarray_support', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ndarray_support', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ndarray_support(...)' code ##################

        
        # Assigning a Call to a Name (line 329):
        
        # Call to array(...): (line 329)
        # Processing the call arguments (line 329)
        
        # Obtaining an instance of the builtin type 'list' (line 329)
        list_394337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 329)
        # Adding element type (line 329)
        
        # Obtaining an instance of the builtin type 'list' (line 329)
        list_394338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 329)
        # Adding element type (line 329)
        float_394339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 19), list_394338, float_394339)
        # Adding element type (line 329)
        float_394340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 19), list_394338, float_394340)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 18), list_394337, list_394338)
        # Adding element type (line 329)
        
        # Obtaining an instance of the builtin type 'list' (line 329)
        list_394341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 329)
        # Adding element type (line 329)
        float_394342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 29), list_394341, float_394342)
        # Adding element type (line 329)
        float_394343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 29), list_394341, float_394343)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 18), list_394337, list_394341)
        
        # Processing the call keyword arguments (line 329)
        kwargs_394344 = {}
        # Getting the type of 'array' (line 329)
        array_394336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'array', False)
        # Calling array(args, kwargs) (line 329)
        array_call_result_394345 = invoke(stypy.reporting.localization.Localization(__file__, 329, 12), array_394336, *[list_394337], **kwargs_394344)
        
        # Assigning a type to the variable 'A' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'A', array_call_result_394345)
        
        # Assigning a Call to a Name (line 330):
        
        # Call to array(...): (line 330)
        # Processing the call arguments (line 330)
        
        # Obtaining an instance of the builtin type 'list' (line 330)
        list_394347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 330)
        # Adding element type (line 330)
        
        # Obtaining an instance of the builtin type 'list' (line 330)
        list_394348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 330)
        # Adding element type (line 330)
        float_394349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 19), list_394348, float_394349)
        # Adding element type (line 330)
        float_394350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 19), list_394348, float_394350)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 18), list_394347, list_394348)
        # Adding element type (line 330)
        
        # Obtaining an instance of the builtin type 'list' (line 330)
        list_394351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 330)
        # Adding element type (line 330)
        float_394352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 29), list_394351, float_394352)
        # Adding element type (line 330)
        float_394353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 29), list_394351, float_394353)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 18), list_394347, list_394351)
        
        # Processing the call keyword arguments (line 330)
        kwargs_394354 = {}
        # Getting the type of 'array' (line 330)
        array_394346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'array', False)
        # Calling array(args, kwargs) (line 330)
        array_call_result_394355 = invoke(stypy.reporting.localization.Localization(__file__, 330, 12), array_394346, *[list_394347], **kwargs_394354)
        
        # Assigning a type to the variable 'x' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'x', array_call_result_394355)
        
        # Assigning a Call to a Name (line 331):
        
        # Call to array(...): (line 331)
        # Processing the call arguments (line 331)
        
        # Obtaining an instance of the builtin type 'list' (line 331)
        list_394357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 331)
        # Adding element type (line 331)
        
        # Obtaining an instance of the builtin type 'list' (line 331)
        list_394358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 331)
        # Adding element type (line 331)
        float_394359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 19), list_394358, float_394359)
        # Adding element type (line 331)
        float_394360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 19), list_394358, float_394360)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 18), list_394357, list_394358)
        # Adding element type (line 331)
        
        # Obtaining an instance of the builtin type 'list' (line 331)
        list_394361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 331)
        # Adding element type (line 331)
        float_394362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 29), list_394361, float_394362)
        # Adding element type (line 331)
        float_394363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 29), list_394361, float_394363)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 18), list_394357, list_394361)
        
        # Processing the call keyword arguments (line 331)
        kwargs_394364 = {}
        # Getting the type of 'array' (line 331)
        array_394356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'array', False)
        # Calling array(args, kwargs) (line 331)
        array_call_result_394365 = invoke(stypy.reporting.localization.Localization(__file__, 331, 12), array_394356, *[list_394357], **kwargs_394364)
        
        # Assigning a type to the variable 'b' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'b', array_call_result_394365)
        
        # Call to assert_array_almost_equal(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'x' (line 333)
        x_394367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 34), 'x', False)
        
        # Call to spsolve(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'A' (line 333)
        A_394369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 45), 'A', False)
        # Getting the type of 'b' (line 333)
        b_394370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 48), 'b', False)
        # Processing the call keyword arguments (line 333)
        kwargs_394371 = {}
        # Getting the type of 'spsolve' (line 333)
        spsolve_394368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 37), 'spsolve', False)
        # Calling spsolve(args, kwargs) (line 333)
        spsolve_call_result_394372 = invoke(stypy.reporting.localization.Localization(__file__, 333, 37), spsolve_394368, *[A_394369, b_394370], **kwargs_394371)
        
        # Processing the call keyword arguments (line 333)
        kwargs_394373 = {}
        # Getting the type of 'assert_array_almost_equal' (line 333)
        assert_array_almost_equal_394366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 333)
        assert_array_almost_equal_call_result_394374 = invoke(stypy.reporting.localization.Localization(__file__, 333, 8), assert_array_almost_equal_394366, *[x_394367, spsolve_call_result_394372], **kwargs_394373)
        
        
        # ################# End of 'test_ndarray_support(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ndarray_support' in the type store
        # Getting the type of 'stypy_return_type' (line 327)
        stypy_return_type_394375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_394375)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ndarray_support'
        return stypy_return_type_394375


    @norecursion
    def test_gssv_badinput(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_gssv_badinput'
        module_type_store = module_type_store.open_function_context('test_gssv_badinput', 335, 4, False)
        # Assigning a type to the variable 'self' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinsolve.test_gssv_badinput.__dict__.__setitem__('stypy_localization', localization)
        TestLinsolve.test_gssv_badinput.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinsolve.test_gssv_badinput.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinsolve.test_gssv_badinput.__dict__.__setitem__('stypy_function_name', 'TestLinsolve.test_gssv_badinput')
        TestLinsolve.test_gssv_badinput.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinsolve.test_gssv_badinput.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinsolve.test_gssv_badinput.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinsolve.test_gssv_badinput.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinsolve.test_gssv_badinput.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinsolve.test_gssv_badinput.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinsolve.test_gssv_badinput.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.test_gssv_badinput', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_gssv_badinput', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_gssv_badinput(...)' code ##################

        
        # Assigning a Num to a Name (line 336):
        int_394376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 12), 'int')
        # Assigning a type to the variable 'N' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'N', int_394376)
        
        # Assigning a BinOp to a Name (line 337):
        
        # Call to arange(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'N' (line 337)
        N_394378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 19), 'N', False)
        # Processing the call keyword arguments (line 337)
        kwargs_394379 = {}
        # Getting the type of 'arange' (line 337)
        arange_394377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 337)
        arange_call_result_394380 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), arange_394377, *[N_394378], **kwargs_394379)
        
        float_394381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 24), 'float')
        # Applying the binary operator '+' (line 337)
        result_add_394382 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 12), '+', arange_call_result_394380, float_394381)
        
        # Assigning a type to the variable 'd' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'd', result_add_394382)
        
        # Assigning a Call to a Name (line 338):
        
        # Call to spdiags(...): (line 338)
        # Processing the call arguments (line 338)
        
        # Obtaining an instance of the builtin type 'tuple' (line 338)
        tuple_394384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 338)
        # Adding element type (line 338)
        # Getting the type of 'd' (line 338)
        d_394385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 21), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 21), tuple_394384, d_394385)
        # Adding element type (line 338)
        int_394386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 24), 'int')
        # Getting the type of 'd' (line 338)
        d_394387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 26), 'd', False)
        # Applying the binary operator '*' (line 338)
        result_mul_394388 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 24), '*', int_394386, d_394387)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 21), tuple_394384, result_mul_394388)
        # Adding element type (line 338)
        
        # Obtaining the type of the subscript
        int_394389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 33), 'int')
        slice_394390 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 338, 29), None, None, int_394389)
        # Getting the type of 'd' (line 338)
        d_394391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 29), 'd', False)
        # Obtaining the member '__getitem__' of a type (line 338)
        getitem___394392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 29), d_394391, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 338)
        subscript_call_result_394393 = invoke(stypy.reporting.localization.Localization(__file__, 338, 29), getitem___394392, slice_394390)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 21), tuple_394384, subscript_call_result_394393)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 338)
        tuple_394394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 338)
        # Adding element type (line 338)
        int_394395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 40), tuple_394394, int_394395)
        # Adding element type (line 338)
        int_394396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 40), tuple_394394, int_394396)
        # Adding element type (line 338)
        int_394397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 40), tuple_394394, int_394397)
        
        # Getting the type of 'N' (line 338)
        N_394398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 51), 'N', False)
        # Getting the type of 'N' (line 338)
        N_394399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 54), 'N', False)
        # Processing the call keyword arguments (line 338)
        kwargs_394400 = {}
        # Getting the type of 'spdiags' (line 338)
        spdiags_394383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'spdiags', False)
        # Calling spdiags(args, kwargs) (line 338)
        spdiags_call_result_394401 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), spdiags_394383, *[tuple_394384, tuple_394394, N_394398, N_394399], **kwargs_394400)
        
        # Assigning a type to the variable 'A' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'A', spdiags_call_result_394401)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 340)
        tuple_394402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 340)
        # Adding element type (line 340)
        # Getting the type of 'csc_matrix' (line 340)
        csc_matrix_394403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 25), 'csc_matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 25), tuple_394402, csc_matrix_394403)
        # Adding element type (line 340)
        # Getting the type of 'csr_matrix' (line 340)
        csr_matrix_394404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 37), 'csr_matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 25), tuple_394402, csr_matrix_394404)
        
        # Testing the type of a for loop iterable (line 340)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 340, 8), tuple_394402)
        # Getting the type of the for loop variable (line 340)
        for_loop_var_394405 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 340, 8), tuple_394402)
        # Assigning a type to the variable 'spmatrix' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'spmatrix', for_loop_var_394405)
        # SSA begins for a for statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 341):
        
        # Call to spmatrix(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'A' (line 341)
        A_394407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'A', False)
        # Processing the call keyword arguments (line 341)
        kwargs_394408 = {}
        # Getting the type of 'spmatrix' (line 341)
        spmatrix_394406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'spmatrix', False)
        # Calling spmatrix(args, kwargs) (line 341)
        spmatrix_call_result_394409 = invoke(stypy.reporting.localization.Localization(__file__, 341, 16), spmatrix_394406, *[A_394407], **kwargs_394408)
        
        # Assigning a type to the variable 'A' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'A', spmatrix_call_result_394409)
        
        # Assigning a Call to a Name (line 342):
        
        # Call to arange(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'N' (line 342)
        N_394412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 26), 'N', False)
        # Processing the call keyword arguments (line 342)
        kwargs_394413 = {}
        # Getting the type of 'np' (line 342)
        np_394410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'np', False)
        # Obtaining the member 'arange' of a type (line 342)
        arange_394411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 16), np_394410, 'arange')
        # Calling arange(args, kwargs) (line 342)
        arange_call_result_394414 = invoke(stypy.reporting.localization.Localization(__file__, 342, 16), arange_394411, *[N_394412], **kwargs_394413)
        
        # Assigning a type to the variable 'b' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'b', arange_call_result_394414)

        @norecursion
        def not_c_contig(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'not_c_contig'
            module_type_store = module_type_store.open_function_context('not_c_contig', 344, 12, False)
            
            # Passed parameters checking function
            not_c_contig.stypy_localization = localization
            not_c_contig.stypy_type_of_self = None
            not_c_contig.stypy_type_store = module_type_store
            not_c_contig.stypy_function_name = 'not_c_contig'
            not_c_contig.stypy_param_names_list = ['x']
            not_c_contig.stypy_varargs_param_name = None
            not_c_contig.stypy_kwargs_param_name = None
            not_c_contig.stypy_call_defaults = defaults
            not_c_contig.stypy_call_varargs = varargs
            not_c_contig.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'not_c_contig', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'not_c_contig', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'not_c_contig(...)' code ##################

            
            # Obtaining the type of the subscript
            int_394415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 37), 'int')
            slice_394416 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 345, 23), None, None, int_394415)
            
            # Call to repeat(...): (line 345)
            # Processing the call arguments (line 345)
            int_394419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 32), 'int')
            # Processing the call keyword arguments (line 345)
            kwargs_394420 = {}
            # Getting the type of 'x' (line 345)
            x_394417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 'x', False)
            # Obtaining the member 'repeat' of a type (line 345)
            repeat_394418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 23), x_394417, 'repeat')
            # Calling repeat(args, kwargs) (line 345)
            repeat_call_result_394421 = invoke(stypy.reporting.localization.Localization(__file__, 345, 23), repeat_394418, *[int_394419], **kwargs_394420)
            
            # Obtaining the member '__getitem__' of a type (line 345)
            getitem___394422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 23), repeat_call_result_394421, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 345)
            subscript_call_result_394423 = invoke(stypy.reporting.localization.Localization(__file__, 345, 23), getitem___394422, slice_394416)
            
            # Assigning a type to the variable 'stypy_return_type' (line 345)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 16), 'stypy_return_type', subscript_call_result_394423)
            
            # ################# End of 'not_c_contig(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'not_c_contig' in the type store
            # Getting the type of 'stypy_return_type' (line 344)
            stypy_return_type_394424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_394424)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'not_c_contig'
            return stypy_return_type_394424

        # Assigning a type to the variable 'not_c_contig' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'not_c_contig', not_c_contig)

        @norecursion
        def not_1dim(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'not_1dim'
            module_type_store = module_type_store.open_function_context('not_1dim', 347, 12, False)
            
            # Passed parameters checking function
            not_1dim.stypy_localization = localization
            not_1dim.stypy_type_of_self = None
            not_1dim.stypy_type_store = module_type_store
            not_1dim.stypy_function_name = 'not_1dim'
            not_1dim.stypy_param_names_list = ['x']
            not_1dim.stypy_varargs_param_name = None
            not_1dim.stypy_kwargs_param_name = None
            not_1dim.stypy_call_defaults = defaults
            not_1dim.stypy_call_varargs = varargs
            not_1dim.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'not_1dim', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'not_1dim', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'not_1dim(...)' code ##################

            
            # Obtaining the type of the subscript
            slice_394425 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 348, 23), None, None, None)
            # Getting the type of 'None' (line 348)
            None_394426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 27), 'None')
            # Getting the type of 'x' (line 348)
            x_394427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 23), 'x')
            # Obtaining the member '__getitem__' of a type (line 348)
            getitem___394428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 23), x_394427, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 348)
            subscript_call_result_394429 = invoke(stypy.reporting.localization.Localization(__file__, 348, 23), getitem___394428, (slice_394425, None_394426))
            
            # Assigning a type to the variable 'stypy_return_type' (line 348)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'stypy_return_type', subscript_call_result_394429)
            
            # ################# End of 'not_1dim(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'not_1dim' in the type store
            # Getting the type of 'stypy_return_type' (line 347)
            stypy_return_type_394430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_394430)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'not_1dim'
            return stypy_return_type_394430

        # Assigning a type to the variable 'not_1dim' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'not_1dim', not_1dim)

        @norecursion
        def bad_type(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'bad_type'
            module_type_store = module_type_store.open_function_context('bad_type', 350, 12, False)
            
            # Passed parameters checking function
            bad_type.stypy_localization = localization
            bad_type.stypy_type_of_self = None
            bad_type.stypy_type_store = module_type_store
            bad_type.stypy_function_name = 'bad_type'
            bad_type.stypy_param_names_list = ['x']
            bad_type.stypy_varargs_param_name = None
            bad_type.stypy_kwargs_param_name = None
            bad_type.stypy_call_defaults = defaults
            bad_type.stypy_call_varargs = varargs
            bad_type.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'bad_type', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'bad_type', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'bad_type(...)' code ##################

            
            # Call to astype(...): (line 351)
            # Processing the call arguments (line 351)
            # Getting the type of 'bool' (line 351)
            bool_394433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 32), 'bool', False)
            # Processing the call keyword arguments (line 351)
            kwargs_394434 = {}
            # Getting the type of 'x' (line 351)
            x_394431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 23), 'x', False)
            # Obtaining the member 'astype' of a type (line 351)
            astype_394432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 23), x_394431, 'astype')
            # Calling astype(args, kwargs) (line 351)
            astype_call_result_394435 = invoke(stypy.reporting.localization.Localization(__file__, 351, 23), astype_394432, *[bool_394433], **kwargs_394434)
            
            # Assigning a type to the variable 'stypy_return_type' (line 351)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 16), 'stypy_return_type', astype_call_result_394435)
            
            # ################# End of 'bad_type(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'bad_type' in the type store
            # Getting the type of 'stypy_return_type' (line 350)
            stypy_return_type_394436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_394436)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'bad_type'
            return stypy_return_type_394436

        # Assigning a type to the variable 'bad_type' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'bad_type', bad_type)

        @norecursion
        def too_short(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'too_short'
            module_type_store = module_type_store.open_function_context('too_short', 353, 12, False)
            
            # Passed parameters checking function
            too_short.stypy_localization = localization
            too_short.stypy_type_of_self = None
            too_short.stypy_type_store = module_type_store
            too_short.stypy_function_name = 'too_short'
            too_short.stypy_param_names_list = ['x']
            too_short.stypy_varargs_param_name = None
            too_short.stypy_kwargs_param_name = None
            too_short.stypy_call_defaults = defaults
            too_short.stypy_call_varargs = varargs
            too_short.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'too_short', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'too_short', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'too_short(...)' code ##################

            
            # Obtaining the type of the subscript
            int_394437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 26), 'int')
            slice_394438 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 354, 23), None, int_394437, None)
            # Getting the type of 'x' (line 354)
            x_394439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 23), 'x')
            # Obtaining the member '__getitem__' of a type (line 354)
            getitem___394440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 23), x_394439, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 354)
            subscript_call_result_394441 = invoke(stypy.reporting.localization.Localization(__file__, 354, 23), getitem___394440, slice_394438)
            
            # Assigning a type to the variable 'stypy_return_type' (line 354)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'stypy_return_type', subscript_call_result_394441)
            
            # ################# End of 'too_short(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'too_short' in the type store
            # Getting the type of 'stypy_return_type' (line 353)
            stypy_return_type_394442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_394442)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'too_short'
            return stypy_return_type_394442

        # Assigning a type to the variable 'too_short' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'too_short', too_short)
        
        # Assigning a List to a Name (line 356):
        
        # Obtaining an instance of the builtin type 'list' (line 356)
        list_394443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 356)
        # Adding element type (line 356)
        # Getting the type of 'not_c_contig' (line 356)
        not_c_contig_394444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 22), 'not_c_contig')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 21), list_394443, not_c_contig_394444)
        # Adding element type (line 356)
        # Getting the type of 'not_1dim' (line 356)
        not_1dim_394445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 36), 'not_1dim')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 21), list_394443, not_1dim_394445)
        # Adding element type (line 356)
        # Getting the type of 'bad_type' (line 356)
        bad_type_394446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 46), 'bad_type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 21), list_394443, bad_type_394446)
        # Adding element type (line 356)
        # Getting the type of 'too_short' (line 356)
        too_short_394447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 56), 'too_short')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 21), list_394443, too_short_394447)
        
        # Assigning a type to the variable 'badops' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'badops', list_394443)
        
        # Getting the type of 'badops' (line 358)
        badops_394448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 25), 'badops')
        # Testing the type of a for loop iterable (line 358)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 358, 12), badops_394448)
        # Getting the type of the for loop variable (line 358)
        for_loop_var_394449 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 358, 12), badops_394448)
        # Assigning a type to the variable 'badop' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'badop', for_loop_var_394449)
        # SSA begins for a for statement (line 358)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 359):
        str_394450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 22), 'str', '%r %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 359)
        tuple_394451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 359)
        # Adding element type (line 359)
        # Getting the type of 'spmatrix' (line 359)
        spmatrix_394452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 33), 'spmatrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 33), tuple_394451, spmatrix_394452)
        # Adding element type (line 359)
        # Getting the type of 'badop' (line 359)
        badop_394453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 43), 'badop')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 33), tuple_394451, badop_394453)
        
        # Applying the binary operator '%' (line 359)
        result_mod_394454 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 22), '%', str_394450, tuple_394451)
        
        # Assigning a type to the variable 'msg' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'msg', result_mod_394454)
        
        # Call to assert_raises(...): (line 361)
        # Processing the call arguments (line 361)
        
        # Obtaining an instance of the builtin type 'tuple' (line 361)
        tuple_394456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 361)
        # Adding element type (line 361)
        # Getting the type of 'ValueError' (line 361)
        ValueError_394457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 31), 'ValueError', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 31), tuple_394456, ValueError_394457)
        # Adding element type (line 361)
        # Getting the type of 'TypeError' (line 361)
        TypeError_394458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 43), 'TypeError', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 31), tuple_394456, TypeError_394458)
        
        # Getting the type of '_superlu' (line 361)
        _superlu_394459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 55), '_superlu', False)
        # Obtaining the member 'gssv' of a type (line 361)
        gssv_394460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 55), _superlu_394459, 'gssv')
        # Getting the type of 'N' (line 362)
        N_394461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 30), 'N', False)
        # Getting the type of 'A' (line 362)
        A_394462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 33), 'A', False)
        # Obtaining the member 'nnz' of a type (line 362)
        nnz_394463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 33), A_394462, 'nnz')
        
        # Call to badop(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'A' (line 362)
        A_394465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 46), 'A', False)
        # Obtaining the member 'data' of a type (line 362)
        data_394466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 46), A_394465, 'data')
        # Processing the call keyword arguments (line 362)
        kwargs_394467 = {}
        # Getting the type of 'badop' (line 362)
        badop_394464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 40), 'badop', False)
        # Calling badop(args, kwargs) (line 362)
        badop_call_result_394468 = invoke(stypy.reporting.localization.Localization(__file__, 362, 40), badop_394464, *[data_394466], **kwargs_394467)
        
        # Getting the type of 'A' (line 362)
        A_394469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 55), 'A', False)
        # Obtaining the member 'indices' of a type (line 362)
        indices_394470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 55), A_394469, 'indices')
        # Getting the type of 'A' (line 362)
        A_394471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 66), 'A', False)
        # Obtaining the member 'indptr' of a type (line 362)
        indptr_394472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 66), A_394471, 'indptr')
        # Getting the type of 'b' (line 363)
        b_394473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 30), 'b', False)
        
        # Call to int(...): (line 363)
        # Processing the call arguments (line 363)
        
        # Getting the type of 'spmatrix' (line 363)
        spmatrix_394475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 37), 'spmatrix', False)
        # Getting the type of 'csc_matrix' (line 363)
        csc_matrix_394476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 49), 'csc_matrix', False)
        # Applying the binary operator '==' (line 363)
        result_eq_394477 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 37), '==', spmatrix_394475, csc_matrix_394476)
        
        # Processing the call keyword arguments (line 363)
        kwargs_394478 = {}
        # Getting the type of 'int' (line 363)
        int_394474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 33), 'int', False)
        # Calling int(args, kwargs) (line 363)
        int_call_result_394479 = invoke(stypy.reporting.localization.Localization(__file__, 363, 33), int_394474, *[result_eq_394477], **kwargs_394478)
        
        # Processing the call keyword arguments (line 361)
        # Getting the type of 'msg' (line 363)
        msg_394480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 70), 'msg', False)
        keyword_394481 = msg_394480
        kwargs_394482 = {'err_msg': keyword_394481}
        # Getting the type of 'assert_raises' (line 361)
        assert_raises_394455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 361)
        assert_raises_call_result_394483 = invoke(stypy.reporting.localization.Localization(__file__, 361, 16), assert_raises_394455, *[tuple_394456, gssv_394460, N_394461, nnz_394463, badop_call_result_394468, indices_394470, indptr_394472, b_394473, int_call_result_394479], **kwargs_394482)
        
        
        # Call to assert_raises(...): (line 364)
        # Processing the call arguments (line 364)
        
        # Obtaining an instance of the builtin type 'tuple' (line 364)
        tuple_394485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 364)
        # Adding element type (line 364)
        # Getting the type of 'ValueError' (line 364)
        ValueError_394486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 31), 'ValueError', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 31), tuple_394485, ValueError_394486)
        # Adding element type (line 364)
        # Getting the type of 'TypeError' (line 364)
        TypeError_394487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 43), 'TypeError', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 31), tuple_394485, TypeError_394487)
        
        # Getting the type of '_superlu' (line 364)
        _superlu_394488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 55), '_superlu', False)
        # Obtaining the member 'gssv' of a type (line 364)
        gssv_394489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 55), _superlu_394488, 'gssv')
        # Getting the type of 'N' (line 365)
        N_394490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 30), 'N', False)
        # Getting the type of 'A' (line 365)
        A_394491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 33), 'A', False)
        # Obtaining the member 'nnz' of a type (line 365)
        nnz_394492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 33), A_394491, 'nnz')
        # Getting the type of 'A' (line 365)
        A_394493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 40), 'A', False)
        # Obtaining the member 'data' of a type (line 365)
        data_394494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 40), A_394493, 'data')
        
        # Call to badop(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'A' (line 365)
        A_394496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 54), 'A', False)
        # Obtaining the member 'indices' of a type (line 365)
        indices_394497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 54), A_394496, 'indices')
        # Processing the call keyword arguments (line 365)
        kwargs_394498 = {}
        # Getting the type of 'badop' (line 365)
        badop_394495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 48), 'badop', False)
        # Calling badop(args, kwargs) (line 365)
        badop_call_result_394499 = invoke(stypy.reporting.localization.Localization(__file__, 365, 48), badop_394495, *[indices_394497], **kwargs_394498)
        
        # Getting the type of 'A' (line 365)
        A_394500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 66), 'A', False)
        # Obtaining the member 'indptr' of a type (line 365)
        indptr_394501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 66), A_394500, 'indptr')
        # Getting the type of 'b' (line 366)
        b_394502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 30), 'b', False)
        
        # Call to int(...): (line 366)
        # Processing the call arguments (line 366)
        
        # Getting the type of 'spmatrix' (line 366)
        spmatrix_394504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 37), 'spmatrix', False)
        # Getting the type of 'csc_matrix' (line 366)
        csc_matrix_394505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 49), 'csc_matrix', False)
        # Applying the binary operator '==' (line 366)
        result_eq_394506 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 37), '==', spmatrix_394504, csc_matrix_394505)
        
        # Processing the call keyword arguments (line 366)
        kwargs_394507 = {}
        # Getting the type of 'int' (line 366)
        int_394503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 33), 'int', False)
        # Calling int(args, kwargs) (line 366)
        int_call_result_394508 = invoke(stypy.reporting.localization.Localization(__file__, 366, 33), int_394503, *[result_eq_394506], **kwargs_394507)
        
        # Processing the call keyword arguments (line 364)
        # Getting the type of 'msg' (line 366)
        msg_394509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 70), 'msg', False)
        keyword_394510 = msg_394509
        kwargs_394511 = {'err_msg': keyword_394510}
        # Getting the type of 'assert_raises' (line 364)
        assert_raises_394484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 364)
        assert_raises_call_result_394512 = invoke(stypy.reporting.localization.Localization(__file__, 364, 16), assert_raises_394484, *[tuple_394485, gssv_394489, N_394490, nnz_394492, data_394494, badop_call_result_394499, indptr_394501, b_394502, int_call_result_394508], **kwargs_394511)
        
        
        # Call to assert_raises(...): (line 367)
        # Processing the call arguments (line 367)
        
        # Obtaining an instance of the builtin type 'tuple' (line 367)
        tuple_394514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 367)
        # Adding element type (line 367)
        # Getting the type of 'ValueError' (line 367)
        ValueError_394515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 31), 'ValueError', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 31), tuple_394514, ValueError_394515)
        # Adding element type (line 367)
        # Getting the type of 'TypeError' (line 367)
        TypeError_394516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 43), 'TypeError', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 31), tuple_394514, TypeError_394516)
        
        # Getting the type of '_superlu' (line 367)
        _superlu_394517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 55), '_superlu', False)
        # Obtaining the member 'gssv' of a type (line 367)
        gssv_394518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 55), _superlu_394517, 'gssv')
        # Getting the type of 'N' (line 368)
        N_394519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 30), 'N', False)
        # Getting the type of 'A' (line 368)
        A_394520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 33), 'A', False)
        # Obtaining the member 'nnz' of a type (line 368)
        nnz_394521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 33), A_394520, 'nnz')
        # Getting the type of 'A' (line 368)
        A_394522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 40), 'A', False)
        # Obtaining the member 'data' of a type (line 368)
        data_394523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 40), A_394522, 'data')
        # Getting the type of 'A' (line 368)
        A_394524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 48), 'A', False)
        # Obtaining the member 'indices' of a type (line 368)
        indices_394525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 48), A_394524, 'indices')
        
        # Call to badop(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'A' (line 368)
        A_394527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 65), 'A', False)
        # Obtaining the member 'indptr' of a type (line 368)
        indptr_394528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 65), A_394527, 'indptr')
        # Processing the call keyword arguments (line 368)
        kwargs_394529 = {}
        # Getting the type of 'badop' (line 368)
        badop_394526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 59), 'badop', False)
        # Calling badop(args, kwargs) (line 368)
        badop_call_result_394530 = invoke(stypy.reporting.localization.Localization(__file__, 368, 59), badop_394526, *[indptr_394528], **kwargs_394529)
        
        # Getting the type of 'b' (line 369)
        b_394531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 30), 'b', False)
        
        # Call to int(...): (line 369)
        # Processing the call arguments (line 369)
        
        # Getting the type of 'spmatrix' (line 369)
        spmatrix_394533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 37), 'spmatrix', False)
        # Getting the type of 'csc_matrix' (line 369)
        csc_matrix_394534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 49), 'csc_matrix', False)
        # Applying the binary operator '==' (line 369)
        result_eq_394535 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 37), '==', spmatrix_394533, csc_matrix_394534)
        
        # Processing the call keyword arguments (line 369)
        kwargs_394536 = {}
        # Getting the type of 'int' (line 369)
        int_394532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 33), 'int', False)
        # Calling int(args, kwargs) (line 369)
        int_call_result_394537 = invoke(stypy.reporting.localization.Localization(__file__, 369, 33), int_394532, *[result_eq_394535], **kwargs_394536)
        
        # Processing the call keyword arguments (line 367)
        # Getting the type of 'msg' (line 369)
        msg_394538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 70), 'msg', False)
        keyword_394539 = msg_394538
        kwargs_394540 = {'err_msg': keyword_394539}
        # Getting the type of 'assert_raises' (line 367)
        assert_raises_394513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 367)
        assert_raises_call_result_394541 = invoke(stypy.reporting.localization.Localization(__file__, 367, 16), assert_raises_394513, *[tuple_394514, gssv_394518, N_394519, nnz_394521, data_394523, indices_394525, badop_call_result_394530, b_394531, int_call_result_394537], **kwargs_394540)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_gssv_badinput(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_gssv_badinput' in the type store
        # Getting the type of 'stypy_return_type' (line 335)
        stypy_return_type_394542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_394542)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_gssv_badinput'
        return stypy_return_type_394542


    @norecursion
    def test_sparsity_preservation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sparsity_preservation'
        module_type_store = module_type_store.open_function_context('test_sparsity_preservation', 371, 4, False)
        # Assigning a type to the variable 'self' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinsolve.test_sparsity_preservation.__dict__.__setitem__('stypy_localization', localization)
        TestLinsolve.test_sparsity_preservation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinsolve.test_sparsity_preservation.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinsolve.test_sparsity_preservation.__dict__.__setitem__('stypy_function_name', 'TestLinsolve.test_sparsity_preservation')
        TestLinsolve.test_sparsity_preservation.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinsolve.test_sparsity_preservation.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinsolve.test_sparsity_preservation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinsolve.test_sparsity_preservation.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinsolve.test_sparsity_preservation.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinsolve.test_sparsity_preservation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinsolve.test_sparsity_preservation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.test_sparsity_preservation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sparsity_preservation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sparsity_preservation(...)' code ##################

        
        # Assigning a Call to a Name (line 372):
        
        # Call to csc_matrix(...): (line 372)
        # Processing the call arguments (line 372)
        
        # Obtaining an instance of the builtin type 'list' (line 372)
        list_394544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 372)
        # Adding element type (line 372)
        
        # Obtaining an instance of the builtin type 'list' (line 373)
        list_394545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 373)
        # Adding element type (line 373)
        int_394546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 12), list_394545, int_394546)
        # Adding element type (line 373)
        int_394547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 12), list_394545, int_394547)
        # Adding element type (line 373)
        int_394548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 12), list_394545, int_394548)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 27), list_394544, list_394545)
        # Adding element type (line 372)
        
        # Obtaining an instance of the builtin type 'list' (line 374)
        list_394549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 374)
        # Adding element type (line 374)
        int_394550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 12), list_394549, int_394550)
        # Adding element type (line 374)
        int_394551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 12), list_394549, int_394551)
        # Adding element type (line 374)
        int_394552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 12), list_394549, int_394552)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 27), list_394544, list_394549)
        # Adding element type (line 372)
        
        # Obtaining an instance of the builtin type 'list' (line 375)
        list_394553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 375)
        # Adding element type (line 375)
        int_394554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 12), list_394553, int_394554)
        # Adding element type (line 375)
        int_394555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 12), list_394553, int_394555)
        # Adding element type (line 375)
        int_394556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 12), list_394553, int_394556)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 27), list_394544, list_394553)
        
        # Processing the call keyword arguments (line 372)
        kwargs_394557 = {}
        # Getting the type of 'csc_matrix' (line 372)
        csc_matrix_394543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 372)
        csc_matrix_call_result_394558 = invoke(stypy.reporting.localization.Localization(__file__, 372, 16), csc_matrix_394543, *[list_394544], **kwargs_394557)
        
        # Assigning a type to the variable 'ident' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'ident', csc_matrix_call_result_394558)
        
        # Assigning a Call to a Name (line 376):
        
        # Call to csc_matrix(...): (line 376)
        # Processing the call arguments (line 376)
        
        # Obtaining an instance of the builtin type 'list' (line 376)
        list_394560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 376)
        # Adding element type (line 376)
        
        # Obtaining an instance of the builtin type 'list' (line 377)
        list_394561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 377)
        # Adding element type (line 377)
        int_394562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 12), list_394561, int_394562)
        # Adding element type (line 377)
        int_394563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 12), list_394561, int_394563)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 23), list_394560, list_394561)
        # Adding element type (line 376)
        
        # Obtaining an instance of the builtin type 'list' (line 378)
        list_394564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 378)
        # Adding element type (line 378)
        int_394565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 12), list_394564, int_394565)
        # Adding element type (line 378)
        int_394566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 12), list_394564, int_394566)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 23), list_394560, list_394564)
        # Adding element type (line 376)
        
        # Obtaining an instance of the builtin type 'list' (line 379)
        list_394567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 379)
        # Adding element type (line 379)
        int_394568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 12), list_394567, int_394568)
        # Adding element type (line 379)
        int_394569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 12), list_394567, int_394569)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 23), list_394560, list_394567)
        
        # Processing the call keyword arguments (line 376)
        kwargs_394570 = {}
        # Getting the type of 'csc_matrix' (line 376)
        csc_matrix_394559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 376)
        csc_matrix_call_result_394571 = invoke(stypy.reporting.localization.Localization(__file__, 376, 12), csc_matrix_394559, *[list_394560], **kwargs_394570)
        
        # Assigning a type to the variable 'b' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'b', csc_matrix_call_result_394571)
        
        # Assigning a Call to a Name (line 380):
        
        # Call to spsolve(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'ident' (line 380)
        ident_394573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 20), 'ident', False)
        # Getting the type of 'b' (line 380)
        b_394574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 27), 'b', False)
        # Processing the call keyword arguments (line 380)
        kwargs_394575 = {}
        # Getting the type of 'spsolve' (line 380)
        spsolve_394572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'spsolve', False)
        # Calling spsolve(args, kwargs) (line 380)
        spsolve_call_result_394576 = invoke(stypy.reporting.localization.Localization(__file__, 380, 12), spsolve_394572, *[ident_394573, b_394574], **kwargs_394575)
        
        # Assigning a type to the variable 'x' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'x', spsolve_call_result_394576)
        
        # Call to assert_equal(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'ident' (line 381)
        ident_394578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'ident', False)
        # Obtaining the member 'nnz' of a type (line 381)
        nnz_394579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 21), ident_394578, 'nnz')
        int_394580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 32), 'int')
        # Processing the call keyword arguments (line 381)
        kwargs_394581 = {}
        # Getting the type of 'assert_equal' (line 381)
        assert_equal_394577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 381)
        assert_equal_call_result_394582 = invoke(stypy.reporting.localization.Localization(__file__, 381, 8), assert_equal_394577, *[nnz_394579, int_394580], **kwargs_394581)
        
        
        # Call to assert_equal(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'b' (line 382)
        b_394584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 21), 'b', False)
        # Obtaining the member 'nnz' of a type (line 382)
        nnz_394585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 21), b_394584, 'nnz')
        int_394586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 28), 'int')
        # Processing the call keyword arguments (line 382)
        kwargs_394587 = {}
        # Getting the type of 'assert_equal' (line 382)
        assert_equal_394583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 382)
        assert_equal_call_result_394588 = invoke(stypy.reporting.localization.Localization(__file__, 382, 8), assert_equal_394583, *[nnz_394585, int_394586], **kwargs_394587)
        
        
        # Call to assert_equal(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'x' (line 383)
        x_394590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 21), 'x', False)
        # Obtaining the member 'nnz' of a type (line 383)
        nnz_394591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 21), x_394590, 'nnz')
        int_394592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 28), 'int')
        # Processing the call keyword arguments (line 383)
        kwargs_394593 = {}
        # Getting the type of 'assert_equal' (line 383)
        assert_equal_394589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 383)
        assert_equal_call_result_394594 = invoke(stypy.reporting.localization.Localization(__file__, 383, 8), assert_equal_394589, *[nnz_394591, int_394592], **kwargs_394593)
        
        
        # Call to assert_allclose(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'x' (line 384)
        x_394596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 24), 'x', False)
        # Obtaining the member 'A' of a type (line 384)
        A_394597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 24), x_394596, 'A')
        # Getting the type of 'b' (line 384)
        b_394598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 29), 'b', False)
        # Obtaining the member 'A' of a type (line 384)
        A_394599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 29), b_394598, 'A')
        # Processing the call keyword arguments (line 384)
        float_394600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 39), 'float')
        keyword_394601 = float_394600
        float_394602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 51), 'float')
        keyword_394603 = float_394602
        kwargs_394604 = {'rtol': keyword_394603, 'atol': keyword_394601}
        # Getting the type of 'assert_allclose' (line 384)
        assert_allclose_394595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 384)
        assert_allclose_call_result_394605 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), assert_allclose_394595, *[A_394597, A_394599], **kwargs_394604)
        
        
        # ################# End of 'test_sparsity_preservation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sparsity_preservation' in the type store
        # Getting the type of 'stypy_return_type' (line 371)
        stypy_return_type_394606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_394606)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sparsity_preservation'
        return stypy_return_type_394606


    @norecursion
    def test_dtype_cast(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dtype_cast'
        module_type_store = module_type_store.open_function_context('test_dtype_cast', 386, 4, False)
        # Assigning a type to the variable 'self' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinsolve.test_dtype_cast.__dict__.__setitem__('stypy_localization', localization)
        TestLinsolve.test_dtype_cast.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinsolve.test_dtype_cast.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinsolve.test_dtype_cast.__dict__.__setitem__('stypy_function_name', 'TestLinsolve.test_dtype_cast')
        TestLinsolve.test_dtype_cast.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinsolve.test_dtype_cast.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinsolve.test_dtype_cast.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinsolve.test_dtype_cast.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinsolve.test_dtype_cast.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinsolve.test_dtype_cast.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinsolve.test_dtype_cast.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.test_dtype_cast', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dtype_cast', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dtype_cast(...)' code ##################

        
        # Assigning a Call to a Name (line 387):
        
        # Call to csr_matrix(...): (line 387)
        # Processing the call arguments (line 387)
        
        # Obtaining an instance of the builtin type 'list' (line 387)
        list_394610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 387)
        # Adding element type (line 387)
        
        # Obtaining an instance of the builtin type 'list' (line 387)
        list_394611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 387)
        # Adding element type (line 387)
        int_394612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 42), list_394611, int_394612)
        # Adding element type (line 387)
        int_394613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 42), list_394611, int_394613)
        # Adding element type (line 387)
        int_394614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 42), list_394611, int_394614)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 41), list_394610, list_394611)
        # Adding element type (line 387)
        
        # Obtaining an instance of the builtin type 'list' (line 388)
        list_394615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 388)
        # Adding element type (line 388)
        int_394616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 42), list_394615, int_394616)
        # Adding element type (line 388)
        int_394617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 42), list_394615, int_394617)
        # Adding element type (line 388)
        int_394618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 42), list_394615, int_394618)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 41), list_394610, list_394615)
        # Adding element type (line 387)
        
        # Obtaining an instance of the builtin type 'list' (line 389)
        list_394619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 389)
        # Adding element type (line 389)
        int_394620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 42), list_394619, int_394620)
        # Adding element type (line 389)
        int_394621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 42), list_394619, int_394621)
        # Adding element type (line 389)
        int_394622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 42), list_394619, int_394622)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 41), list_394610, list_394619)
        
        # Processing the call keyword arguments (line 387)
        kwargs_394623 = {}
        # Getting the type of 'scipy' (line 387)
        scipy_394607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 17), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 387)
        sparse_394608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 17), scipy_394607, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 387)
        csr_matrix_394609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 17), sparse_394608, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 387)
        csr_matrix_call_result_394624 = invoke(stypy.reporting.localization.Localization(__file__, 387, 17), csr_matrix_394609, *[list_394610], **kwargs_394623)
        
        # Assigning a type to the variable 'A_real' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'A_real', csr_matrix_call_result_394624)
        
        # Assigning a Call to a Name (line 390):
        
        # Call to csr_matrix(...): (line 390)
        # Processing the call arguments (line 390)
        
        # Obtaining an instance of the builtin type 'list' (line 390)
        list_394628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 390)
        # Adding element type (line 390)
        
        # Obtaining an instance of the builtin type 'list' (line 390)
        list_394629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 390)
        # Adding element type (line 390)
        int_394630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 45), list_394629, int_394630)
        # Adding element type (line 390)
        int_394631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 45), list_394629, int_394631)
        # Adding element type (line 390)
        int_394632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 45), list_394629, int_394632)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 44), list_394628, list_394629)
        # Adding element type (line 390)
        
        # Obtaining an instance of the builtin type 'list' (line 391)
        list_394633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 391)
        # Adding element type (line 391)
        int_394634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 45), list_394633, int_394634)
        # Adding element type (line 391)
        int_394635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 45), list_394633, int_394635)
        # Adding element type (line 391)
        int_394636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 45), list_394633, int_394636)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 44), list_394628, list_394633)
        # Adding element type (line 390)
        
        # Obtaining an instance of the builtin type 'list' (line 392)
        list_394637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 392)
        # Adding element type (line 392)
        int_394638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 45), list_394637, int_394638)
        # Adding element type (line 392)
        int_394639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 45), list_394637, int_394639)
        # Adding element type (line 392)
        int_394640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 52), 'int')
        complex_394641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 56), 'complex')
        # Applying the binary operator '+' (line 392)
        result_add_394642 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 52), '+', int_394640, complex_394641)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 45), list_394637, result_add_394642)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 44), list_394628, list_394637)
        
        # Processing the call keyword arguments (line 390)
        kwargs_394643 = {}
        # Getting the type of 'scipy' (line 390)
        scipy_394625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 20), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 390)
        sparse_394626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 20), scipy_394625, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 390)
        csr_matrix_394627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 20), sparse_394626, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 390)
        csr_matrix_call_result_394644 = invoke(stypy.reporting.localization.Localization(__file__, 390, 20), csr_matrix_394627, *[list_394628], **kwargs_394643)
        
        # Assigning a type to the variable 'A_complex' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'A_complex', csr_matrix_call_result_394644)
        
        # Assigning a Call to a Name (line 393):
        
        # Call to array(...): (line 393)
        # Processing the call arguments (line 393)
        
        # Obtaining an instance of the builtin type 'list' (line 393)
        list_394647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 393)
        # Adding element type (line 393)
        int_394648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 26), list_394647, int_394648)
        # Adding element type (line 393)
        int_394649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 26), list_394647, int_394649)
        # Adding element type (line 393)
        int_394650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 26), list_394647, int_394650)
        
        # Processing the call keyword arguments (line 393)
        kwargs_394651 = {}
        # Getting the type of 'np' (line 393)
        np_394645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 393)
        array_394646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 17), np_394645, 'array')
        # Calling array(args, kwargs) (line 393)
        array_call_result_394652 = invoke(stypy.reporting.localization.Localization(__file__, 393, 17), array_394646, *[list_394647], **kwargs_394651)
        
        # Assigning a type to the variable 'b_real' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'b_real', array_call_result_394652)
        
        # Assigning a BinOp to a Name (line 394):
        
        # Call to array(...): (line 394)
        # Processing the call arguments (line 394)
        
        # Obtaining an instance of the builtin type 'list' (line 394)
        list_394655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 394)
        # Adding element type (line 394)
        int_394656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 29), list_394655, int_394656)
        # Adding element type (line 394)
        int_394657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 29), list_394655, int_394657)
        # Adding element type (line 394)
        int_394658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 29), list_394655, int_394658)
        
        # Processing the call keyword arguments (line 394)
        kwargs_394659 = {}
        # Getting the type of 'np' (line 394)
        np_394653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 20), 'np', False)
        # Obtaining the member 'array' of a type (line 394)
        array_394654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 20), np_394653, 'array')
        # Calling array(args, kwargs) (line 394)
        array_call_result_394660 = invoke(stypy.reporting.localization.Localization(__file__, 394, 20), array_394654, *[list_394655], **kwargs_394659)
        
        complex_394661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 40), 'complex')
        
        # Call to array(...): (line 394)
        # Processing the call arguments (line 394)
        
        # Obtaining an instance of the builtin type 'list' (line 394)
        list_394664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 394)
        # Adding element type (line 394)
        int_394665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 52), list_394664, int_394665)
        # Adding element type (line 394)
        int_394666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 52), list_394664, int_394666)
        # Adding element type (line 394)
        int_394667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 52), list_394664, int_394667)
        
        # Processing the call keyword arguments (line 394)
        kwargs_394668 = {}
        # Getting the type of 'np' (line 394)
        np_394662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 43), 'np', False)
        # Obtaining the member 'array' of a type (line 394)
        array_394663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 43), np_394662, 'array')
        # Calling array(args, kwargs) (line 394)
        array_call_result_394669 = invoke(stypy.reporting.localization.Localization(__file__, 394, 43), array_394663, *[list_394664], **kwargs_394668)
        
        # Applying the binary operator '*' (line 394)
        result_mul_394670 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 40), '*', complex_394661, array_call_result_394669)
        
        # Applying the binary operator '+' (line 394)
        result_add_394671 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 20), '+', array_call_result_394660, result_mul_394670)
        
        # Assigning a type to the variable 'b_complex' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'b_complex', result_add_394671)
        
        # Assigning a Call to a Name (line 395):
        
        # Call to spsolve(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'A_real' (line 395)
        A_real_394673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 20), 'A_real', False)
        # Getting the type of 'b_real' (line 395)
        b_real_394674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 28), 'b_real', False)
        # Processing the call keyword arguments (line 395)
        kwargs_394675 = {}
        # Getting the type of 'spsolve' (line 395)
        spsolve_394672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'spsolve', False)
        # Calling spsolve(args, kwargs) (line 395)
        spsolve_call_result_394676 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), spsolve_394672, *[A_real_394673, b_real_394674], **kwargs_394675)
        
        # Assigning a type to the variable 'x' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'x', spsolve_call_result_394676)
        
        # Call to assert_(...): (line 396)
        # Processing the call arguments (line 396)
        
        # Call to issubdtype(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'x' (line 396)
        x_394680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 30), 'x', False)
        # Obtaining the member 'dtype' of a type (line 396)
        dtype_394681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 30), x_394680, 'dtype')
        # Getting the type of 'np' (line 396)
        np_394682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 39), 'np', False)
        # Obtaining the member 'floating' of a type (line 396)
        floating_394683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 39), np_394682, 'floating')
        # Processing the call keyword arguments (line 396)
        kwargs_394684 = {}
        # Getting the type of 'np' (line 396)
        np_394678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 16), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 396)
        issubdtype_394679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 16), np_394678, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 396)
        issubdtype_call_result_394685 = invoke(stypy.reporting.localization.Localization(__file__, 396, 16), issubdtype_394679, *[dtype_394681, floating_394683], **kwargs_394684)
        
        # Processing the call keyword arguments (line 396)
        kwargs_394686 = {}
        # Getting the type of 'assert_' (line 396)
        assert__394677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 396)
        assert__call_result_394687 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), assert__394677, *[issubdtype_call_result_394685], **kwargs_394686)
        
        
        # Assigning a Call to a Name (line 397):
        
        # Call to spsolve(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'A_real' (line 397)
        A_real_394689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 20), 'A_real', False)
        # Getting the type of 'b_complex' (line 397)
        b_complex_394690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 28), 'b_complex', False)
        # Processing the call keyword arguments (line 397)
        kwargs_394691 = {}
        # Getting the type of 'spsolve' (line 397)
        spsolve_394688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'spsolve', False)
        # Calling spsolve(args, kwargs) (line 397)
        spsolve_call_result_394692 = invoke(stypy.reporting.localization.Localization(__file__, 397, 12), spsolve_394688, *[A_real_394689, b_complex_394690], **kwargs_394691)
        
        # Assigning a type to the variable 'x' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'x', spsolve_call_result_394692)
        
        # Call to assert_(...): (line 398)
        # Processing the call arguments (line 398)
        
        # Call to issubdtype(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'x' (line 398)
        x_394696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 30), 'x', False)
        # Obtaining the member 'dtype' of a type (line 398)
        dtype_394697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 30), x_394696, 'dtype')
        # Getting the type of 'np' (line 398)
        np_394698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 39), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 398)
        complexfloating_394699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 39), np_394698, 'complexfloating')
        # Processing the call keyword arguments (line 398)
        kwargs_394700 = {}
        # Getting the type of 'np' (line 398)
        np_394694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 16), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 398)
        issubdtype_394695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 16), np_394694, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 398)
        issubdtype_call_result_394701 = invoke(stypy.reporting.localization.Localization(__file__, 398, 16), issubdtype_394695, *[dtype_394697, complexfloating_394699], **kwargs_394700)
        
        # Processing the call keyword arguments (line 398)
        kwargs_394702 = {}
        # Getting the type of 'assert_' (line 398)
        assert__394693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 398)
        assert__call_result_394703 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), assert__394693, *[issubdtype_call_result_394701], **kwargs_394702)
        
        
        # Assigning a Call to a Name (line 399):
        
        # Call to spsolve(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'A_complex' (line 399)
        A_complex_394705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 20), 'A_complex', False)
        # Getting the type of 'b_real' (line 399)
        b_real_394706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'b_real', False)
        # Processing the call keyword arguments (line 399)
        kwargs_394707 = {}
        # Getting the type of 'spsolve' (line 399)
        spsolve_394704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'spsolve', False)
        # Calling spsolve(args, kwargs) (line 399)
        spsolve_call_result_394708 = invoke(stypy.reporting.localization.Localization(__file__, 399, 12), spsolve_394704, *[A_complex_394705, b_real_394706], **kwargs_394707)
        
        # Assigning a type to the variable 'x' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'x', spsolve_call_result_394708)
        
        # Call to assert_(...): (line 400)
        # Processing the call arguments (line 400)
        
        # Call to issubdtype(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'x' (line 400)
        x_394712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 30), 'x', False)
        # Obtaining the member 'dtype' of a type (line 400)
        dtype_394713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 30), x_394712, 'dtype')
        # Getting the type of 'np' (line 400)
        np_394714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 39), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 400)
        complexfloating_394715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 39), np_394714, 'complexfloating')
        # Processing the call keyword arguments (line 400)
        kwargs_394716 = {}
        # Getting the type of 'np' (line 400)
        np_394710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 16), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 400)
        issubdtype_394711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 16), np_394710, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 400)
        issubdtype_call_result_394717 = invoke(stypy.reporting.localization.Localization(__file__, 400, 16), issubdtype_394711, *[dtype_394713, complexfloating_394715], **kwargs_394716)
        
        # Processing the call keyword arguments (line 400)
        kwargs_394718 = {}
        # Getting the type of 'assert_' (line 400)
        assert__394709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 400)
        assert__call_result_394719 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), assert__394709, *[issubdtype_call_result_394717], **kwargs_394718)
        
        
        # Assigning a Call to a Name (line 401):
        
        # Call to spsolve(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'A_complex' (line 401)
        A_complex_394721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 20), 'A_complex', False)
        # Getting the type of 'b_complex' (line 401)
        b_complex_394722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 31), 'b_complex', False)
        # Processing the call keyword arguments (line 401)
        kwargs_394723 = {}
        # Getting the type of 'spsolve' (line 401)
        spsolve_394720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'spsolve', False)
        # Calling spsolve(args, kwargs) (line 401)
        spsolve_call_result_394724 = invoke(stypy.reporting.localization.Localization(__file__, 401, 12), spsolve_394720, *[A_complex_394721, b_complex_394722], **kwargs_394723)
        
        # Assigning a type to the variable 'x' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'x', spsolve_call_result_394724)
        
        # Call to assert_(...): (line 402)
        # Processing the call arguments (line 402)
        
        # Call to issubdtype(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'x' (line 402)
        x_394728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 30), 'x', False)
        # Obtaining the member 'dtype' of a type (line 402)
        dtype_394729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 30), x_394728, 'dtype')
        # Getting the type of 'np' (line 402)
        np_394730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 39), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 402)
        complexfloating_394731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 39), np_394730, 'complexfloating')
        # Processing the call keyword arguments (line 402)
        kwargs_394732 = {}
        # Getting the type of 'np' (line 402)
        np_394726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 16), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 402)
        issubdtype_394727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 16), np_394726, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 402)
        issubdtype_call_result_394733 = invoke(stypy.reporting.localization.Localization(__file__, 402, 16), issubdtype_394727, *[dtype_394729, complexfloating_394731], **kwargs_394732)
        
        # Processing the call keyword arguments (line 402)
        kwargs_394734 = {}
        # Getting the type of 'assert_' (line 402)
        assert__394725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 402)
        assert__call_result_394735 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), assert__394725, *[issubdtype_call_result_394733], **kwargs_394734)
        
        
        # ################# End of 'test_dtype_cast(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dtype_cast' in the type store
        # Getting the type of 'stypy_return_type' (line 386)
        stypy_return_type_394736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_394736)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dtype_cast'
        return stypy_return_type_394736


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 170, 0, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinsolve.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLinsolve' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'TestLinsolve', TestLinsolve)
# Declaration of the 'TestSplu' class

class TestSplu(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 406, 4, False)
        # Assigning a type to the variable 'self' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestSplu.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu.setup_method.__dict__.__setitem__('stypy_function_name', 'TestSplu.setup_method')
        TestSplu.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to use_solver(...): (line 407)
        # Processing the call keyword arguments (line 407)
        # Getting the type of 'False' (line 407)
        False_394738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 30), 'False', False)
        keyword_394739 = False_394738
        kwargs_394740 = {'useUmfpack': keyword_394739}
        # Getting the type of 'use_solver' (line 407)
        use_solver_394737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 407)
        use_solver_call_result_394741 = invoke(stypy.reporting.localization.Localization(__file__, 407, 8), use_solver_394737, *[], **kwargs_394740)
        
        
        # Assigning a Num to a Name (line 408):
        int_394742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 12), 'int')
        # Assigning a type to the variable 'n' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'n', int_394742)
        
        # Assigning a BinOp to a Name (line 409):
        
        # Call to arange(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'n' (line 409)
        n_394744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 19), 'n', False)
        # Processing the call keyword arguments (line 409)
        kwargs_394745 = {}
        # Getting the type of 'arange' (line 409)
        arange_394743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 409)
        arange_call_result_394746 = invoke(stypy.reporting.localization.Localization(__file__, 409, 12), arange_394743, *[n_394744], **kwargs_394745)
        
        int_394747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 24), 'int')
        # Applying the binary operator '+' (line 409)
        result_add_394748 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 12), '+', arange_call_result_394746, int_394747)
        
        # Assigning a type to the variable 'd' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'd', result_add_394748)
        
        # Assigning a Name to a Attribute (line 410):
        # Getting the type of 'n' (line 410)
        n_394749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 17), 'n')
        # Getting the type of 'self' (line 410)
        self_394750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'self')
        # Setting the type of the member 'n' of a type (line 410)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 8), self_394750, 'n', n_394749)
        
        # Assigning a Call to a Attribute (line 411):
        
        # Call to spdiags(...): (line 411)
        # Processing the call arguments (line 411)
        
        # Obtaining an instance of the builtin type 'tuple' (line 411)
        tuple_394752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 411)
        # Adding element type (line 411)
        # Getting the type of 'd' (line 411)
        d_394753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 26), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 26), tuple_394752, d_394753)
        # Adding element type (line 411)
        int_394754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 29), 'int')
        # Getting the type of 'd' (line 411)
        d_394755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 31), 'd', False)
        # Applying the binary operator '*' (line 411)
        result_mul_394756 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 29), '*', int_394754, d_394755)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 26), tuple_394752, result_mul_394756)
        # Adding element type (line 411)
        
        # Obtaining the type of the subscript
        int_394757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 38), 'int')
        slice_394758 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 411, 34), None, None, int_394757)
        # Getting the type of 'd' (line 411)
        d_394759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 34), 'd', False)
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___394760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 34), d_394759, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_394761 = invoke(stypy.reporting.localization.Localization(__file__, 411, 34), getitem___394760, slice_394758)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 26), tuple_394752, subscript_call_result_394761)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 411)
        tuple_394762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 411)
        # Adding element type (line 411)
        int_394763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 45), tuple_394762, int_394763)
        # Adding element type (line 411)
        int_394764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 45), tuple_394762, int_394764)
        # Adding element type (line 411)
        int_394765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 45), tuple_394762, int_394765)
        
        # Getting the type of 'n' (line 411)
        n_394766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 56), 'n', False)
        # Getting the type of 'n' (line 411)
        n_394767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 59), 'n', False)
        # Processing the call keyword arguments (line 411)
        kwargs_394768 = {}
        # Getting the type of 'spdiags' (line 411)
        spdiags_394751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 17), 'spdiags', False)
        # Calling spdiags(args, kwargs) (line 411)
        spdiags_call_result_394769 = invoke(stypy.reporting.localization.Localization(__file__, 411, 17), spdiags_394751, *[tuple_394752, tuple_394762, n_394766, n_394767], **kwargs_394768)
        
        # Getting the type of 'self' (line 411)
        self_394770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'self')
        # Setting the type of the member 'A' of a type (line 411)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), self_394770, 'A', spdiags_call_result_394769)
        
        # Call to seed(...): (line 412)
        # Processing the call arguments (line 412)
        int_394773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 20), 'int')
        # Processing the call keyword arguments (line 412)
        kwargs_394774 = {}
        # Getting the type of 'random' (line 412)
        random_394771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'random', False)
        # Obtaining the member 'seed' of a type (line 412)
        seed_394772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 8), random_394771, 'seed')
        # Calling seed(args, kwargs) (line 412)
        seed_call_result_394775 = invoke(stypy.reporting.localization.Localization(__file__, 412, 8), seed_394772, *[int_394773], **kwargs_394774)
        
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 406)
        stypy_return_type_394776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_394776)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_394776


    @norecursion
    def _smoketest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_smoketest'
        module_type_store = module_type_store.open_function_context('_smoketest', 414, 4, False)
        # Assigning a type to the variable 'self' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu._smoketest.__dict__.__setitem__('stypy_localization', localization)
        TestSplu._smoketest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu._smoketest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu._smoketest.__dict__.__setitem__('stypy_function_name', 'TestSplu._smoketest')
        TestSplu._smoketest.__dict__.__setitem__('stypy_param_names_list', ['spxlu', 'check', 'dtype'])
        TestSplu._smoketest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu._smoketest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu._smoketest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu._smoketest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu._smoketest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu._smoketest.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu._smoketest', ['spxlu', 'check', 'dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_smoketest', localization, ['spxlu', 'check', 'dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_smoketest(...)' code ##################

        
        
        # Call to issubdtype(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'dtype' (line 415)
        dtype_394779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 25), 'dtype', False)
        # Getting the type of 'np' (line 415)
        np_394780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 32), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 415)
        complexfloating_394781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 32), np_394780, 'complexfloating')
        # Processing the call keyword arguments (line 415)
        kwargs_394782 = {}
        # Getting the type of 'np' (line 415)
        np_394777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 11), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 415)
        issubdtype_394778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 11), np_394777, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 415)
        issubdtype_call_result_394783 = invoke(stypy.reporting.localization.Localization(__file__, 415, 11), issubdtype_394778, *[dtype_394779, complexfloating_394781], **kwargs_394782)
        
        # Testing the type of an if condition (line 415)
        if_condition_394784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 415, 8), issubdtype_call_result_394783)
        # Assigning a type to the variable 'if_condition_394784' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'if_condition_394784', if_condition_394784)
        # SSA begins for if statement (line 415)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 416):
        # Getting the type of 'self' (line 416)
        self_394785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 16), 'self')
        # Obtaining the member 'A' of a type (line 416)
        A_394786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 16), self_394785, 'A')
        complex_394787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 25), 'complex')
        # Getting the type of 'self' (line 416)
        self_394788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 28), 'self')
        # Obtaining the member 'A' of a type (line 416)
        A_394789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 28), self_394788, 'A')
        # Obtaining the member 'T' of a type (line 416)
        T_394790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 28), A_394789, 'T')
        # Applying the binary operator '*' (line 416)
        result_mul_394791 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 25), '*', complex_394787, T_394790)
        
        # Applying the binary operator '+' (line 416)
        result_add_394792 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 16), '+', A_394786, result_mul_394791)
        
        # Assigning a type to the variable 'A' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'A', result_add_394792)
        # SSA branch for the else part of an if statement (line 415)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 418):
        # Getting the type of 'self' (line 418)
        self_394793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'self')
        # Obtaining the member 'A' of a type (line 418)
        A_394794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 16), self_394793, 'A')
        # Assigning a type to the variable 'A' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'A', A_394794)
        # SSA join for if statement (line 415)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 420):
        
        # Call to astype(...): (line 420)
        # Processing the call arguments (line 420)
        # Getting the type of 'dtype' (line 420)
        dtype_394797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 21), 'dtype', False)
        # Processing the call keyword arguments (line 420)
        kwargs_394798 = {}
        # Getting the type of 'A' (line 420)
        A_394795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'A', False)
        # Obtaining the member 'astype' of a type (line 420)
        astype_394796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 12), A_394795, 'astype')
        # Calling astype(args, kwargs) (line 420)
        astype_call_result_394799 = invoke(stypy.reporting.localization.Localization(__file__, 420, 12), astype_394796, *[dtype_394797], **kwargs_394798)
        
        # Assigning a type to the variable 'A' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'A', astype_call_result_394799)
        
        # Assigning a Call to a Name (line 421):
        
        # Call to spxlu(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'A' (line 421)
        A_394801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 19), 'A', False)
        # Processing the call keyword arguments (line 421)
        kwargs_394802 = {}
        # Getting the type of 'spxlu' (line 421)
        spxlu_394800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 13), 'spxlu', False)
        # Calling spxlu(args, kwargs) (line 421)
        spxlu_call_result_394803 = invoke(stypy.reporting.localization.Localization(__file__, 421, 13), spxlu_394800, *[A_394801], **kwargs_394802)
        
        # Assigning a type to the variable 'lu' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'lu', spxlu_call_result_394803)
        
        # Assigning a Call to a Name (line 423):
        
        # Call to RandomState(...): (line 423)
        # Processing the call arguments (line 423)
        int_394806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 33), 'int')
        # Processing the call keyword arguments (line 423)
        kwargs_394807 = {}
        # Getting the type of 'random' (line 423)
        random_394804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 14), 'random', False)
        # Obtaining the member 'RandomState' of a type (line 423)
        RandomState_394805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 14), random_394804, 'RandomState')
        # Calling RandomState(args, kwargs) (line 423)
        RandomState_call_result_394808 = invoke(stypy.reporting.localization.Localization(__file__, 423, 14), RandomState_394805, *[int_394806], **kwargs_394807)
        
        # Assigning a type to the variable 'rng' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'rng', RandomState_call_result_394808)
        
        
        # Obtaining an instance of the builtin type 'list' (line 426)
        list_394809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 426)
        # Adding element type (line 426)
        # Getting the type of 'None' (line 426)
        None_394810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 18), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 17), list_394809, None_394810)
        # Adding element type (line 426)
        int_394811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 17), list_394809, int_394811)
        # Adding element type (line 426)
        int_394812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 17), list_394809, int_394812)
        # Adding element type (line 426)
        # Getting the type of 'self' (line 426)
        self_394813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 30), 'self')
        # Obtaining the member 'n' of a type (line 426)
        n_394814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 30), self_394813, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 17), list_394809, n_394814)
        # Adding element type (line 426)
        # Getting the type of 'self' (line 426)
        self_394815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 38), 'self')
        # Obtaining the member 'n' of a type (line 426)
        n_394816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 38), self_394815, 'n')
        int_394817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 45), 'int')
        # Applying the binary operator '+' (line 426)
        result_add_394818 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 38), '+', n_394816, int_394817)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 17), list_394809, result_add_394818)
        
        # Testing the type of a for loop iterable (line 426)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 426, 8), list_394809)
        # Getting the type of the for loop variable (line 426)
        for_loop_var_394819 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 426, 8), list_394809)
        # Assigning a type to the variable 'k' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'k', for_loop_var_394819)
        # SSA begins for a for statement (line 426)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 427):
        str_394820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 18), 'str', 'k=%r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 427)
        tuple_394821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 427)
        # Adding element type (line 427)
        # Getting the type of 'k' (line 427)
        k_394822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 28), 'k')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 28), tuple_394821, k_394822)
        
        # Applying the binary operator '%' (line 427)
        result_mod_394823 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 18), '%', str_394820, tuple_394821)
        
        # Assigning a type to the variable 'msg' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'msg', result_mod_394823)
        
        # Type idiom detected: calculating its left and rigth part (line 429)
        # Getting the type of 'k' (line 429)
        k_394824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 15), 'k')
        # Getting the type of 'None' (line 429)
        None_394825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 20), 'None')
        
        (may_be_394826, more_types_in_union_394827) = may_be_none(k_394824, None_394825)

        if may_be_394826:

            if more_types_in_union_394827:
                # Runtime conditional SSA (line 429)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 430):
            
            # Call to rand(...): (line 430)
            # Processing the call arguments (line 430)
            # Getting the type of 'self' (line 430)
            self_394830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 29), 'self', False)
            # Obtaining the member 'n' of a type (line 430)
            n_394831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 29), self_394830, 'n')
            # Processing the call keyword arguments (line 430)
            kwargs_394832 = {}
            # Getting the type of 'rng' (line 430)
            rng_394828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 20), 'rng', False)
            # Obtaining the member 'rand' of a type (line 430)
            rand_394829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 20), rng_394828, 'rand')
            # Calling rand(args, kwargs) (line 430)
            rand_call_result_394833 = invoke(stypy.reporting.localization.Localization(__file__, 430, 20), rand_394829, *[n_394831], **kwargs_394832)
            
            # Assigning a type to the variable 'b' (line 430)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'b', rand_call_result_394833)

            if more_types_in_union_394827:
                # Runtime conditional SSA for else branch (line 429)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_394826) or more_types_in_union_394827):
            
            # Assigning a Call to a Name (line 432):
            
            # Call to rand(...): (line 432)
            # Processing the call arguments (line 432)
            # Getting the type of 'self' (line 432)
            self_394836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 29), 'self', False)
            # Obtaining the member 'n' of a type (line 432)
            n_394837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 29), self_394836, 'n')
            # Getting the type of 'k' (line 432)
            k_394838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 37), 'k', False)
            # Processing the call keyword arguments (line 432)
            kwargs_394839 = {}
            # Getting the type of 'rng' (line 432)
            rng_394834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 20), 'rng', False)
            # Obtaining the member 'rand' of a type (line 432)
            rand_394835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 20), rng_394834, 'rand')
            # Calling rand(args, kwargs) (line 432)
            rand_call_result_394840 = invoke(stypy.reporting.localization.Localization(__file__, 432, 20), rand_394835, *[n_394837, k_394838], **kwargs_394839)
            
            # Assigning a type to the variable 'b' (line 432)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 16), 'b', rand_call_result_394840)

            if (may_be_394826 and more_types_in_union_394827):
                # SSA join for if statement (line 429)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to issubdtype(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'dtype' (line 434)
        dtype_394843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 29), 'dtype', False)
        # Getting the type of 'np' (line 434)
        np_394844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 36), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 434)
        complexfloating_394845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 36), np_394844, 'complexfloating')
        # Processing the call keyword arguments (line 434)
        kwargs_394846 = {}
        # Getting the type of 'np' (line 434)
        np_394841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 15), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 434)
        issubdtype_394842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 15), np_394841, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 434)
        issubdtype_call_result_394847 = invoke(stypy.reporting.localization.Localization(__file__, 434, 15), issubdtype_394842, *[dtype_394843, complexfloating_394845], **kwargs_394846)
        
        # Testing the type of an if condition (line 434)
        if_condition_394848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 434, 12), issubdtype_call_result_394847)
        # Assigning a type to the variable 'if_condition_394848' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'if_condition_394848', if_condition_394848)
        # SSA begins for if statement (line 434)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 435):
        # Getting the type of 'b' (line 435)
        b_394849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'b')
        complex_394850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 24), 'complex')
        
        # Call to rand(...): (line 435)
        # Getting the type of 'b' (line 435)
        b_394853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 37), 'b', False)
        # Obtaining the member 'shape' of a type (line 435)
        shape_394854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 37), b_394853, 'shape')
        # Processing the call keyword arguments (line 435)
        kwargs_394855 = {}
        # Getting the type of 'rng' (line 435)
        rng_394851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 27), 'rng', False)
        # Obtaining the member 'rand' of a type (line 435)
        rand_394852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 27), rng_394851, 'rand')
        # Calling rand(args, kwargs) (line 435)
        rand_call_result_394856 = invoke(stypy.reporting.localization.Localization(__file__, 435, 27), rand_394852, *[shape_394854], **kwargs_394855)
        
        # Applying the binary operator '*' (line 435)
        result_mul_394857 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 24), '*', complex_394850, rand_call_result_394856)
        
        # Applying the binary operator '+' (line 435)
        result_add_394858 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 20), '+', b_394849, result_mul_394857)
        
        # Assigning a type to the variable 'b' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 16), 'b', result_add_394858)
        # SSA join for if statement (line 434)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 436):
        
        # Call to astype(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'dtype' (line 436)
        dtype_394861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 25), 'dtype', False)
        # Processing the call keyword arguments (line 436)
        kwargs_394862 = {}
        # Getting the type of 'b' (line 436)
        b_394859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'b', False)
        # Obtaining the member 'astype' of a type (line 436)
        astype_394860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 16), b_394859, 'astype')
        # Calling astype(args, kwargs) (line 436)
        astype_call_result_394863 = invoke(stypy.reporting.localization.Localization(__file__, 436, 16), astype_394860, *[dtype_394861], **kwargs_394862)
        
        # Assigning a type to the variable 'b' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'b', astype_call_result_394863)
        
        # Assigning a Call to a Name (line 438):
        
        # Call to solve(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'b' (line 438)
        b_394866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 25), 'b', False)
        # Processing the call keyword arguments (line 438)
        kwargs_394867 = {}
        # Getting the type of 'lu' (line 438)
        lu_394864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'lu', False)
        # Obtaining the member 'solve' of a type (line 438)
        solve_394865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 16), lu_394864, 'solve')
        # Calling solve(args, kwargs) (line 438)
        solve_call_result_394868 = invoke(stypy.reporting.localization.Localization(__file__, 438, 16), solve_394865, *[b_394866], **kwargs_394867)
        
        # Assigning a type to the variable 'x' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'x', solve_call_result_394868)
        
        # Call to check(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'A' (line 439)
        A_394870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 18), 'A', False)
        # Getting the type of 'b' (line 439)
        b_394871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 21), 'b', False)
        # Getting the type of 'x' (line 439)
        x_394872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 24), 'x', False)
        # Getting the type of 'msg' (line 439)
        msg_394873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 27), 'msg', False)
        # Processing the call keyword arguments (line 439)
        kwargs_394874 = {}
        # Getting the type of 'check' (line 439)
        check_394869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'check', False)
        # Calling check(args, kwargs) (line 439)
        check_call_result_394875 = invoke(stypy.reporting.localization.Localization(__file__, 439, 12), check_394869, *[A_394870, b_394871, x_394872, msg_394873], **kwargs_394874)
        
        
        # Assigning a Call to a Name (line 441):
        
        # Call to solve(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'b' (line 441)
        b_394878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 25), 'b', False)
        str_394879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 28), 'str', 'T')
        # Processing the call keyword arguments (line 441)
        kwargs_394880 = {}
        # Getting the type of 'lu' (line 441)
        lu_394876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'lu', False)
        # Obtaining the member 'solve' of a type (line 441)
        solve_394877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 16), lu_394876, 'solve')
        # Calling solve(args, kwargs) (line 441)
        solve_call_result_394881 = invoke(stypy.reporting.localization.Localization(__file__, 441, 16), solve_394877, *[b_394878, str_394879], **kwargs_394880)
        
        # Assigning a type to the variable 'x' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'x', solve_call_result_394881)
        
        # Call to check(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 'A' (line 442)
        A_394883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 18), 'A', False)
        # Obtaining the member 'T' of a type (line 442)
        T_394884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 18), A_394883, 'T')
        # Getting the type of 'b' (line 442)
        b_394885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 23), 'b', False)
        # Getting the type of 'x' (line 442)
        x_394886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 26), 'x', False)
        # Getting the type of 'msg' (line 442)
        msg_394887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 29), 'msg', False)
        # Processing the call keyword arguments (line 442)
        kwargs_394888 = {}
        # Getting the type of 'check' (line 442)
        check_394882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'check', False)
        # Calling check(args, kwargs) (line 442)
        check_call_result_394889 = invoke(stypy.reporting.localization.Localization(__file__, 442, 12), check_394882, *[T_394884, b_394885, x_394886, msg_394887], **kwargs_394888)
        
        
        # Assigning a Call to a Name (line 444):
        
        # Call to solve(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of 'b' (line 444)
        b_394892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 25), 'b', False)
        str_394893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 28), 'str', 'H')
        # Processing the call keyword arguments (line 444)
        kwargs_394894 = {}
        # Getting the type of 'lu' (line 444)
        lu_394890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 'lu', False)
        # Obtaining the member 'solve' of a type (line 444)
        solve_394891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 16), lu_394890, 'solve')
        # Calling solve(args, kwargs) (line 444)
        solve_call_result_394895 = invoke(stypy.reporting.localization.Localization(__file__, 444, 16), solve_394891, *[b_394892, str_394893], **kwargs_394894)
        
        # Assigning a type to the variable 'x' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'x', solve_call_result_394895)
        
        # Call to check(...): (line 445)
        # Processing the call arguments (line 445)
        
        # Call to conj(...): (line 445)
        # Processing the call keyword arguments (line 445)
        kwargs_394900 = {}
        # Getting the type of 'A' (line 445)
        A_394897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 18), 'A', False)
        # Obtaining the member 'T' of a type (line 445)
        T_394898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 18), A_394897, 'T')
        # Obtaining the member 'conj' of a type (line 445)
        conj_394899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 18), T_394898, 'conj')
        # Calling conj(args, kwargs) (line 445)
        conj_call_result_394901 = invoke(stypy.reporting.localization.Localization(__file__, 445, 18), conj_394899, *[], **kwargs_394900)
        
        # Getting the type of 'b' (line 445)
        b_394902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 30), 'b', False)
        # Getting the type of 'x' (line 445)
        x_394903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 33), 'x', False)
        # Getting the type of 'msg' (line 445)
        msg_394904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 36), 'msg', False)
        # Processing the call keyword arguments (line 445)
        kwargs_394905 = {}
        # Getting the type of 'check' (line 445)
        check_394896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'check', False)
        # Calling check(args, kwargs) (line 445)
        check_call_result_394906 = invoke(stypy.reporting.localization.Localization(__file__, 445, 12), check_394896, *[conj_call_result_394901, b_394902, x_394903, msg_394904], **kwargs_394905)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_smoketest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_smoketest' in the type store
        # Getting the type of 'stypy_return_type' (line 414)
        stypy_return_type_394907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_394907)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_smoketest'
        return stypy_return_type_394907


    @norecursion
    def test_splu_smoketest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_splu_smoketest'
        module_type_store = module_type_store.open_function_context('test_splu_smoketest', 447, 4, False)
        # Assigning a type to the variable 'self' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu.test_splu_smoketest.__dict__.__setitem__('stypy_localization', localization)
        TestSplu.test_splu_smoketest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu.test_splu_smoketest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu.test_splu_smoketest.__dict__.__setitem__('stypy_function_name', 'TestSplu.test_splu_smoketest')
        TestSplu.test_splu_smoketest.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu.test_splu_smoketest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu.test_splu_smoketest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu.test_splu_smoketest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu.test_splu_smoketest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu.test_splu_smoketest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu.test_splu_smoketest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.test_splu_smoketest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_splu_smoketest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_splu_smoketest(...)' code ##################

        
        # Call to _internal_test_splu_smoketest(...): (line 449)
        # Processing the call keyword arguments (line 449)
        kwargs_394910 = {}
        # Getting the type of 'self' (line 449)
        self_394908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'self', False)
        # Obtaining the member '_internal_test_splu_smoketest' of a type (line 449)
        _internal_test_splu_smoketest_394909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 8), self_394908, '_internal_test_splu_smoketest')
        # Calling _internal_test_splu_smoketest(args, kwargs) (line 449)
        _internal_test_splu_smoketest_call_result_394911 = invoke(stypy.reporting.localization.Localization(__file__, 449, 8), _internal_test_splu_smoketest_394909, *[], **kwargs_394910)
        
        
        # ################# End of 'test_splu_smoketest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_splu_smoketest' in the type store
        # Getting the type of 'stypy_return_type' (line 447)
        stypy_return_type_394912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_394912)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_splu_smoketest'
        return stypy_return_type_394912


    @norecursion
    def _internal_test_splu_smoketest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_internal_test_splu_smoketest'
        module_type_store = module_type_store.open_function_context('_internal_test_splu_smoketest', 451, 4, False)
        # Assigning a type to the variable 'self' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu._internal_test_splu_smoketest.__dict__.__setitem__('stypy_localization', localization)
        TestSplu._internal_test_splu_smoketest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu._internal_test_splu_smoketest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu._internal_test_splu_smoketest.__dict__.__setitem__('stypy_function_name', 'TestSplu._internal_test_splu_smoketest')
        TestSplu._internal_test_splu_smoketest.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu._internal_test_splu_smoketest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu._internal_test_splu_smoketest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu._internal_test_splu_smoketest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu._internal_test_splu_smoketest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu._internal_test_splu_smoketest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu._internal_test_splu_smoketest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu._internal_test_splu_smoketest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_internal_test_splu_smoketest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_internal_test_splu_smoketest(...)' code ##################


        @norecursion
        def check(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            str_394913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 31), 'str', '')
            defaults = [str_394913]
            # Create a new context for function 'check'
            module_type_store = module_type_store.open_function_context('check', 453, 8, False)
            
            # Passed parameters checking function
            check.stypy_localization = localization
            check.stypy_type_of_self = None
            check.stypy_type_store = module_type_store
            check.stypy_function_name = 'check'
            check.stypy_param_names_list = ['A', 'b', 'x', 'msg']
            check.stypy_varargs_param_name = None
            check.stypy_kwargs_param_name = None
            check.stypy_call_defaults = defaults
            check.stypy_call_varargs = varargs
            check.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'check', ['A', 'b', 'x', 'msg'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'check', localization, ['A', 'b', 'x', 'msg'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'check(...)' code ##################

            
            # Assigning a Attribute to a Name (line 454):
            
            # Call to finfo(...): (line 454)
            # Processing the call arguments (line 454)
            # Getting the type of 'A' (line 454)
            A_394916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 27), 'A', False)
            # Obtaining the member 'dtype' of a type (line 454)
            dtype_394917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 27), A_394916, 'dtype')
            # Processing the call keyword arguments (line 454)
            kwargs_394918 = {}
            # Getting the type of 'np' (line 454)
            np_394914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 18), 'np', False)
            # Obtaining the member 'finfo' of a type (line 454)
            finfo_394915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 18), np_394914, 'finfo')
            # Calling finfo(args, kwargs) (line 454)
            finfo_call_result_394919 = invoke(stypy.reporting.localization.Localization(__file__, 454, 18), finfo_394915, *[dtype_394917], **kwargs_394918)
            
            # Obtaining the member 'eps' of a type (line 454)
            eps_394920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 18), finfo_call_result_394919, 'eps')
            # Assigning a type to the variable 'eps' (line 454)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'eps', eps_394920)
            
            # Assigning a BinOp to a Name (line 455):
            # Getting the type of 'A' (line 455)
            A_394921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'A')
            # Getting the type of 'x' (line 455)
            x_394922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 20), 'x')
            # Applying the binary operator '*' (line 455)
            result_mul_394923 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 16), '*', A_394921, x_394922)
            
            # Assigning a type to the variable 'r' (line 455)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'r', result_mul_394923)
            
            # Call to assert_(...): (line 456)
            # Processing the call arguments (line 456)
            
            
            # Call to max(...): (line 456)
            # Processing the call keyword arguments (line 456)
            kwargs_394932 = {}
            
            # Call to abs(...): (line 456)
            # Processing the call arguments (line 456)
            # Getting the type of 'r' (line 456)
            r_394926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 24), 'r', False)
            # Getting the type of 'b' (line 456)
            b_394927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 28), 'b', False)
            # Applying the binary operator '-' (line 456)
            result_sub_394928 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 24), '-', r_394926, b_394927)
            
            # Processing the call keyword arguments (line 456)
            kwargs_394929 = {}
            # Getting the type of 'abs' (line 456)
            abs_394925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 20), 'abs', False)
            # Calling abs(args, kwargs) (line 456)
            abs_call_result_394930 = invoke(stypy.reporting.localization.Localization(__file__, 456, 20), abs_394925, *[result_sub_394928], **kwargs_394929)
            
            # Obtaining the member 'max' of a type (line 456)
            max_394931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 20), abs_call_result_394930, 'max')
            # Calling max(args, kwargs) (line 456)
            max_call_result_394933 = invoke(stypy.reporting.localization.Localization(__file__, 456, 20), max_394931, *[], **kwargs_394932)
            
            float_394934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 39), 'float')
            # Getting the type of 'eps' (line 456)
            eps_394935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 43), 'eps', False)
            # Applying the binary operator '*' (line 456)
            result_mul_394936 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 39), '*', float_394934, eps_394935)
            
            # Applying the binary operator '<' (line 456)
            result_lt_394937 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 20), '<', max_call_result_394933, result_mul_394936)
            
            # Getting the type of 'msg' (line 456)
            msg_394938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 48), 'msg', False)
            # Processing the call keyword arguments (line 456)
            kwargs_394939 = {}
            # Getting the type of 'assert_' (line 456)
            assert__394924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'assert_', False)
            # Calling assert_(args, kwargs) (line 456)
            assert__call_result_394940 = invoke(stypy.reporting.localization.Localization(__file__, 456, 12), assert__394924, *[result_lt_394937, msg_394938], **kwargs_394939)
            
            
            # ################# End of 'check(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'check' in the type store
            # Getting the type of 'stypy_return_type' (line 453)
            stypy_return_type_394941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_394941)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'check'
            return stypy_return_type_394941

        # Assigning a type to the variable 'check' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'check', check)
        
        # Call to _smoketest(...): (line 458)
        # Processing the call arguments (line 458)
        # Getting the type of 'splu' (line 458)
        splu_394944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 24), 'splu', False)
        # Getting the type of 'check' (line 458)
        check_394945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 30), 'check', False)
        # Getting the type of 'np' (line 458)
        np_394946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 37), 'np', False)
        # Obtaining the member 'float32' of a type (line 458)
        float32_394947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 37), np_394946, 'float32')
        # Processing the call keyword arguments (line 458)
        kwargs_394948 = {}
        # Getting the type of 'self' (line 458)
        self_394942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'self', False)
        # Obtaining the member '_smoketest' of a type (line 458)
        _smoketest_394943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), self_394942, '_smoketest')
        # Calling _smoketest(args, kwargs) (line 458)
        _smoketest_call_result_394949 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), _smoketest_394943, *[splu_394944, check_394945, float32_394947], **kwargs_394948)
        
        
        # Call to _smoketest(...): (line 459)
        # Processing the call arguments (line 459)
        # Getting the type of 'splu' (line 459)
        splu_394952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 24), 'splu', False)
        # Getting the type of 'check' (line 459)
        check_394953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 30), 'check', False)
        # Getting the type of 'np' (line 459)
        np_394954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 37), 'np', False)
        # Obtaining the member 'float64' of a type (line 459)
        float64_394955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 37), np_394954, 'float64')
        # Processing the call keyword arguments (line 459)
        kwargs_394956 = {}
        # Getting the type of 'self' (line 459)
        self_394950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'self', False)
        # Obtaining the member '_smoketest' of a type (line 459)
        _smoketest_394951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), self_394950, '_smoketest')
        # Calling _smoketest(args, kwargs) (line 459)
        _smoketest_call_result_394957 = invoke(stypy.reporting.localization.Localization(__file__, 459, 8), _smoketest_394951, *[splu_394952, check_394953, float64_394955], **kwargs_394956)
        
        
        # Call to _smoketest(...): (line 460)
        # Processing the call arguments (line 460)
        # Getting the type of 'splu' (line 460)
        splu_394960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 24), 'splu', False)
        # Getting the type of 'check' (line 460)
        check_394961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 30), 'check', False)
        # Getting the type of 'np' (line 460)
        np_394962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 37), 'np', False)
        # Obtaining the member 'complex64' of a type (line 460)
        complex64_394963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 37), np_394962, 'complex64')
        # Processing the call keyword arguments (line 460)
        kwargs_394964 = {}
        # Getting the type of 'self' (line 460)
        self_394958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'self', False)
        # Obtaining the member '_smoketest' of a type (line 460)
        _smoketest_394959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 8), self_394958, '_smoketest')
        # Calling _smoketest(args, kwargs) (line 460)
        _smoketest_call_result_394965 = invoke(stypy.reporting.localization.Localization(__file__, 460, 8), _smoketest_394959, *[splu_394960, check_394961, complex64_394963], **kwargs_394964)
        
        
        # Call to _smoketest(...): (line 461)
        # Processing the call arguments (line 461)
        # Getting the type of 'splu' (line 461)
        splu_394968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 24), 'splu', False)
        # Getting the type of 'check' (line 461)
        check_394969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 30), 'check', False)
        # Getting the type of 'np' (line 461)
        np_394970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 37), 'np', False)
        # Obtaining the member 'complex128' of a type (line 461)
        complex128_394971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 37), np_394970, 'complex128')
        # Processing the call keyword arguments (line 461)
        kwargs_394972 = {}
        # Getting the type of 'self' (line 461)
        self_394966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'self', False)
        # Obtaining the member '_smoketest' of a type (line 461)
        _smoketest_394967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 8), self_394966, '_smoketest')
        # Calling _smoketest(args, kwargs) (line 461)
        _smoketest_call_result_394973 = invoke(stypy.reporting.localization.Localization(__file__, 461, 8), _smoketest_394967, *[splu_394968, check_394969, complex128_394971], **kwargs_394972)
        
        
        # ################# End of '_internal_test_splu_smoketest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_internal_test_splu_smoketest' in the type store
        # Getting the type of 'stypy_return_type' (line 451)
        stypy_return_type_394974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_394974)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_internal_test_splu_smoketest'
        return stypy_return_type_394974


    @norecursion
    def test_spilu_smoketest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spilu_smoketest'
        module_type_store = module_type_store.open_function_context('test_spilu_smoketest', 463, 4, False)
        # Assigning a type to the variable 'self' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu.test_spilu_smoketest.__dict__.__setitem__('stypy_localization', localization)
        TestSplu.test_spilu_smoketest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu.test_spilu_smoketest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu.test_spilu_smoketest.__dict__.__setitem__('stypy_function_name', 'TestSplu.test_spilu_smoketest')
        TestSplu.test_spilu_smoketest.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu.test_spilu_smoketest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu.test_spilu_smoketest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu.test_spilu_smoketest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu.test_spilu_smoketest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu.test_spilu_smoketest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu.test_spilu_smoketest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.test_spilu_smoketest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spilu_smoketest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spilu_smoketest(...)' code ##################

        
        # Call to _internal_test_spilu_smoketest(...): (line 465)
        # Processing the call keyword arguments (line 465)
        kwargs_394977 = {}
        # Getting the type of 'self' (line 465)
        self_394975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'self', False)
        # Obtaining the member '_internal_test_spilu_smoketest' of a type (line 465)
        _internal_test_spilu_smoketest_394976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 8), self_394975, '_internal_test_spilu_smoketest')
        # Calling _internal_test_spilu_smoketest(args, kwargs) (line 465)
        _internal_test_spilu_smoketest_call_result_394978 = invoke(stypy.reporting.localization.Localization(__file__, 465, 8), _internal_test_spilu_smoketest_394976, *[], **kwargs_394977)
        
        
        # ################# End of 'test_spilu_smoketest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spilu_smoketest' in the type store
        # Getting the type of 'stypy_return_type' (line 463)
        stypy_return_type_394979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_394979)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spilu_smoketest'
        return stypy_return_type_394979


    @norecursion
    def _internal_test_spilu_smoketest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_internal_test_spilu_smoketest'
        module_type_store = module_type_store.open_function_context('_internal_test_spilu_smoketest', 467, 4, False)
        # Assigning a type to the variable 'self' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu._internal_test_spilu_smoketest.__dict__.__setitem__('stypy_localization', localization)
        TestSplu._internal_test_spilu_smoketest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu._internal_test_spilu_smoketest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu._internal_test_spilu_smoketest.__dict__.__setitem__('stypy_function_name', 'TestSplu._internal_test_spilu_smoketest')
        TestSplu._internal_test_spilu_smoketest.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu._internal_test_spilu_smoketest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu._internal_test_spilu_smoketest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu._internal_test_spilu_smoketest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu._internal_test_spilu_smoketest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu._internal_test_spilu_smoketest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu._internal_test_spilu_smoketest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu._internal_test_spilu_smoketest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_internal_test_spilu_smoketest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_internal_test_spilu_smoketest(...)' code ##################

        
        # Assigning a List to a Name (line 468):
        
        # Obtaining an instance of the builtin type 'list' (line 468)
        list_394980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 468)
        
        # Assigning a type to the variable 'errors' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'errors', list_394980)

        @norecursion
        def check(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            str_394981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 31), 'str', '')
            defaults = [str_394981]
            # Create a new context for function 'check'
            module_type_store = module_type_store.open_function_context('check', 470, 8, False)
            
            # Passed parameters checking function
            check.stypy_localization = localization
            check.stypy_type_of_self = None
            check.stypy_type_store = module_type_store
            check.stypy_function_name = 'check'
            check.stypy_param_names_list = ['A', 'b', 'x', 'msg']
            check.stypy_varargs_param_name = None
            check.stypy_kwargs_param_name = None
            check.stypy_call_defaults = defaults
            check.stypy_call_varargs = varargs
            check.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'check', ['A', 'b', 'x', 'msg'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'check', localization, ['A', 'b', 'x', 'msg'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'check(...)' code ##################

            
            # Assigning a BinOp to a Name (line 471):
            # Getting the type of 'A' (line 471)
            A_394982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'A')
            # Getting the type of 'x' (line 471)
            x_394983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 20), 'x')
            # Applying the binary operator '*' (line 471)
            result_mul_394984 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 16), '*', A_394982, x_394983)
            
            # Assigning a type to the variable 'r' (line 471)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'r', result_mul_394984)
            
            # Assigning a Call to a Name (line 472):
            
            # Call to max(...): (line 472)
            # Processing the call keyword arguments (line 472)
            kwargs_394992 = {}
            
            # Call to abs(...): (line 472)
            # Processing the call arguments (line 472)
            # Getting the type of 'r' (line 472)
            r_394986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 22), 'r', False)
            # Getting the type of 'b' (line 472)
            b_394987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 26), 'b', False)
            # Applying the binary operator '-' (line 472)
            result_sub_394988 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 22), '-', r_394986, b_394987)
            
            # Processing the call keyword arguments (line 472)
            kwargs_394989 = {}
            # Getting the type of 'abs' (line 472)
            abs_394985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 18), 'abs', False)
            # Calling abs(args, kwargs) (line 472)
            abs_call_result_394990 = invoke(stypy.reporting.localization.Localization(__file__, 472, 18), abs_394985, *[result_sub_394988], **kwargs_394989)
            
            # Obtaining the member 'max' of a type (line 472)
            max_394991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 18), abs_call_result_394990, 'max')
            # Calling max(args, kwargs) (line 472)
            max_call_result_394993 = invoke(stypy.reporting.localization.Localization(__file__, 472, 18), max_394991, *[], **kwargs_394992)
            
            # Assigning a type to the variable 'err' (line 472)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'err', max_call_result_394993)
            
            # Call to assert_(...): (line 473)
            # Processing the call arguments (line 473)
            
            # Getting the type of 'err' (line 473)
            err_394995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 20), 'err', False)
            float_394996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 26), 'float')
            # Applying the binary operator '<' (line 473)
            result_lt_394997 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 20), '<', err_394995, float_394996)
            
            # Getting the type of 'msg' (line 473)
            msg_394998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 32), 'msg', False)
            # Processing the call keyword arguments (line 473)
            kwargs_394999 = {}
            # Getting the type of 'assert_' (line 473)
            assert__394994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'assert_', False)
            # Calling assert_(args, kwargs) (line 473)
            assert__call_result_395000 = invoke(stypy.reporting.localization.Localization(__file__, 473, 12), assert__394994, *[result_lt_394997, msg_394998], **kwargs_394999)
            
            
            
            # Getting the type of 'b' (line 474)
            b_395001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 15), 'b')
            # Obtaining the member 'dtype' of a type (line 474)
            dtype_395002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 15), b_395001, 'dtype')
            
            # Obtaining an instance of the builtin type 'tuple' (line 474)
            tuple_395003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 474)
            # Adding element type (line 474)
            # Getting the type of 'np' (line 474)
            np_395004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 27), 'np')
            # Obtaining the member 'float64' of a type (line 474)
            float64_395005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 27), np_395004, 'float64')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 27), tuple_395003, float64_395005)
            # Adding element type (line 474)
            # Getting the type of 'np' (line 474)
            np_395006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 39), 'np')
            # Obtaining the member 'complex128' of a type (line 474)
            complex128_395007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 39), np_395006, 'complex128')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 27), tuple_395003, complex128_395007)
            
            # Applying the binary operator 'in' (line 474)
            result_contains_395008 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 15), 'in', dtype_395002, tuple_395003)
            
            # Testing the type of an if condition (line 474)
            if_condition_395009 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 474, 12), result_contains_395008)
            # Assigning a type to the variable 'if_condition_395009' (line 474)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 12), 'if_condition_395009', if_condition_395009)
            # SSA begins for if statement (line 474)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 475)
            # Processing the call arguments (line 475)
            # Getting the type of 'err' (line 475)
            err_395012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 30), 'err', False)
            # Processing the call keyword arguments (line 475)
            kwargs_395013 = {}
            # Getting the type of 'errors' (line 475)
            errors_395010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 16), 'errors', False)
            # Obtaining the member 'append' of a type (line 475)
            append_395011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 16), errors_395010, 'append')
            # Calling append(args, kwargs) (line 475)
            append_call_result_395014 = invoke(stypy.reporting.localization.Localization(__file__, 475, 16), append_395011, *[err_395012], **kwargs_395013)
            
            # SSA join for if statement (line 474)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'check(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'check' in the type store
            # Getting the type of 'stypy_return_type' (line 470)
            stypy_return_type_395015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_395015)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'check'
            return stypy_return_type_395015

        # Assigning a type to the variable 'check' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'check', check)
        
        # Call to _smoketest(...): (line 477)
        # Processing the call arguments (line 477)
        # Getting the type of 'spilu' (line 477)
        spilu_395018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 24), 'spilu', False)
        # Getting the type of 'check' (line 477)
        check_395019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 31), 'check', False)
        # Getting the type of 'np' (line 477)
        np_395020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 38), 'np', False)
        # Obtaining the member 'float32' of a type (line 477)
        float32_395021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 38), np_395020, 'float32')
        # Processing the call keyword arguments (line 477)
        kwargs_395022 = {}
        # Getting the type of 'self' (line 477)
        self_395016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'self', False)
        # Obtaining the member '_smoketest' of a type (line 477)
        _smoketest_395017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), self_395016, '_smoketest')
        # Calling _smoketest(args, kwargs) (line 477)
        _smoketest_call_result_395023 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), _smoketest_395017, *[spilu_395018, check_395019, float32_395021], **kwargs_395022)
        
        
        # Call to _smoketest(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'spilu' (line 478)
        spilu_395026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 24), 'spilu', False)
        # Getting the type of 'check' (line 478)
        check_395027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 31), 'check', False)
        # Getting the type of 'np' (line 478)
        np_395028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 38), 'np', False)
        # Obtaining the member 'float64' of a type (line 478)
        float64_395029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 38), np_395028, 'float64')
        # Processing the call keyword arguments (line 478)
        kwargs_395030 = {}
        # Getting the type of 'self' (line 478)
        self_395024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'self', False)
        # Obtaining the member '_smoketest' of a type (line 478)
        _smoketest_395025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 8), self_395024, '_smoketest')
        # Calling _smoketest(args, kwargs) (line 478)
        _smoketest_call_result_395031 = invoke(stypy.reporting.localization.Localization(__file__, 478, 8), _smoketest_395025, *[spilu_395026, check_395027, float64_395029], **kwargs_395030)
        
        
        # Call to _smoketest(...): (line 479)
        # Processing the call arguments (line 479)
        # Getting the type of 'spilu' (line 479)
        spilu_395034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 24), 'spilu', False)
        # Getting the type of 'check' (line 479)
        check_395035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 31), 'check', False)
        # Getting the type of 'np' (line 479)
        np_395036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 38), 'np', False)
        # Obtaining the member 'complex64' of a type (line 479)
        complex64_395037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 38), np_395036, 'complex64')
        # Processing the call keyword arguments (line 479)
        kwargs_395038 = {}
        # Getting the type of 'self' (line 479)
        self_395032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'self', False)
        # Obtaining the member '_smoketest' of a type (line 479)
        _smoketest_395033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 8), self_395032, '_smoketest')
        # Calling _smoketest(args, kwargs) (line 479)
        _smoketest_call_result_395039 = invoke(stypy.reporting.localization.Localization(__file__, 479, 8), _smoketest_395033, *[spilu_395034, check_395035, complex64_395037], **kwargs_395038)
        
        
        # Call to _smoketest(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'spilu' (line 480)
        spilu_395042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 24), 'spilu', False)
        # Getting the type of 'check' (line 480)
        check_395043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 31), 'check', False)
        # Getting the type of 'np' (line 480)
        np_395044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 38), 'np', False)
        # Obtaining the member 'complex128' of a type (line 480)
        complex128_395045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 38), np_395044, 'complex128')
        # Processing the call keyword arguments (line 480)
        kwargs_395046 = {}
        # Getting the type of 'self' (line 480)
        self_395040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'self', False)
        # Obtaining the member '_smoketest' of a type (line 480)
        _smoketest_395041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), self_395040, '_smoketest')
        # Calling _smoketest(args, kwargs) (line 480)
        _smoketest_call_result_395047 = invoke(stypy.reporting.localization.Localization(__file__, 480, 8), _smoketest_395041, *[spilu_395042, check_395043, complex128_395045], **kwargs_395046)
        
        
        # Call to assert_(...): (line 482)
        # Processing the call arguments (line 482)
        
        
        # Call to max(...): (line 482)
        # Processing the call arguments (line 482)
        # Getting the type of 'errors' (line 482)
        errors_395050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 20), 'errors', False)
        # Processing the call keyword arguments (line 482)
        kwargs_395051 = {}
        # Getting the type of 'max' (line 482)
        max_395049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 16), 'max', False)
        # Calling max(args, kwargs) (line 482)
        max_call_result_395052 = invoke(stypy.reporting.localization.Localization(__file__, 482, 16), max_395049, *[errors_395050], **kwargs_395051)
        
        float_395053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 30), 'float')
        # Applying the binary operator '>' (line 482)
        result_gt_395054 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 16), '>', max_call_result_395052, float_395053)
        
        # Processing the call keyword arguments (line 482)
        kwargs_395055 = {}
        # Getting the type of 'assert_' (line 482)
        assert__395048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 482)
        assert__call_result_395056 = invoke(stypy.reporting.localization.Localization(__file__, 482, 8), assert__395048, *[result_gt_395054], **kwargs_395055)
        
        
        # ################# End of '_internal_test_spilu_smoketest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_internal_test_spilu_smoketest' in the type store
        # Getting the type of 'stypy_return_type' (line 467)
        stypy_return_type_395057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395057)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_internal_test_spilu_smoketest'
        return stypy_return_type_395057


    @norecursion
    def test_spilu_drop_rule(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spilu_drop_rule'
        module_type_store = module_type_store.open_function_context('test_spilu_drop_rule', 484, 4, False)
        # Assigning a type to the variable 'self' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu.test_spilu_drop_rule.__dict__.__setitem__('stypy_localization', localization)
        TestSplu.test_spilu_drop_rule.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu.test_spilu_drop_rule.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu.test_spilu_drop_rule.__dict__.__setitem__('stypy_function_name', 'TestSplu.test_spilu_drop_rule')
        TestSplu.test_spilu_drop_rule.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu.test_spilu_drop_rule.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu.test_spilu_drop_rule.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu.test_spilu_drop_rule.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu.test_spilu_drop_rule.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu.test_spilu_drop_rule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu.test_spilu_drop_rule.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.test_spilu_drop_rule', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spilu_drop_rule', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spilu_drop_rule(...)' code ##################

        
        # Assigning a Call to a Name (line 487):
        
        # Call to identity(...): (line 487)
        # Processing the call arguments (line 487)
        int_395059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 21), 'int')
        # Processing the call keyword arguments (line 487)
        kwargs_395060 = {}
        # Getting the type of 'identity' (line 487)
        identity_395058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'identity', False)
        # Calling identity(args, kwargs) (line 487)
        identity_call_result_395061 = invoke(stypy.reporting.localization.Localization(__file__, 487, 12), identity_395058, *[int_395059], **kwargs_395060)
        
        # Assigning a type to the variable 'A' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'A', identity_call_result_395061)
        
        # Assigning a List to a Name (line 489):
        
        # Obtaining an instance of the builtin type 'list' (line 489)
        list_395062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 489)
        # Adding element type (line 489)
        
        # Call to decode(...): (line 490)
        # Processing the call arguments (line 490)
        str_395065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 33), 'str', 'ascii')
        # Processing the call keyword arguments (line 490)
        kwargs_395066 = {}
        str_395063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 12), 'str', 'basic,area')
        # Obtaining the member 'decode' of a type (line 490)
        decode_395064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 12), str_395063, 'decode')
        # Calling decode(args, kwargs) (line 490)
        decode_call_result_395067 = invoke(stypy.reporting.localization.Localization(__file__, 490, 12), decode_395064, *[str_395065], **kwargs_395066)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 16), list_395062, decode_call_result_395067)
        # Adding element type (line 489)
        str_395068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 12), 'str', 'basic,area')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 16), list_395062, str_395068)
        # Adding element type (line 489)
        
        # Obtaining an instance of the builtin type 'list' (line 492)
        list_395069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 492)
        # Adding element type (line 492)
        str_395070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 13), 'str', 'basic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 492, 12), list_395069, str_395070)
        # Adding element type (line 492)
        
        # Call to decode(...): (line 492)
        # Processing the call arguments (line 492)
        str_395073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 38), 'str', 'ascii')
        # Processing the call keyword arguments (line 492)
        kwargs_395074 = {}
        str_395071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 23), 'str', 'area')
        # Obtaining the member 'decode' of a type (line 492)
        decode_395072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 23), str_395071, 'decode')
        # Calling decode(args, kwargs) (line 492)
        decode_call_result_395075 = invoke(stypy.reporting.localization.Localization(__file__, 492, 23), decode_395072, *[str_395073], **kwargs_395074)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 492, 12), list_395069, decode_call_result_395075)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 16), list_395062, list_395069)
        
        # Assigning a type to the variable 'rules' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'rules', list_395062)
        
        # Getting the type of 'rules' (line 494)
        rules_395076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 20), 'rules')
        # Testing the type of a for loop iterable (line 494)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 494, 8), rules_395076)
        # Getting the type of the for loop variable (line 494)
        for_loop_var_395077 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 494, 8), rules_395076)
        # Assigning a type to the variable 'rule' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'rule', for_loop_var_395077)
        # SSA begins for a for statement (line 494)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_(...): (line 496)
        # Processing the call arguments (line 496)
        
        # Call to isinstance(...): (line 496)
        # Processing the call arguments (line 496)
        
        # Call to spilu(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'A' (line 496)
        A_395081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 37), 'A', False)
        # Processing the call keyword arguments (line 496)
        # Getting the type of 'rule' (line 496)
        rule_395082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 50), 'rule', False)
        keyword_395083 = rule_395082
        kwargs_395084 = {'drop_rule': keyword_395083}
        # Getting the type of 'spilu' (line 496)
        spilu_395080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 31), 'spilu', False)
        # Calling spilu(args, kwargs) (line 496)
        spilu_call_result_395085 = invoke(stypy.reporting.localization.Localization(__file__, 496, 31), spilu_395080, *[A_395081], **kwargs_395084)
        
        # Getting the type of 'SuperLU' (line 496)
        SuperLU_395086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 57), 'SuperLU', False)
        # Processing the call keyword arguments (line 496)
        kwargs_395087 = {}
        # Getting the type of 'isinstance' (line 496)
        isinstance_395079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 496)
        isinstance_call_result_395088 = invoke(stypy.reporting.localization.Localization(__file__, 496, 20), isinstance_395079, *[spilu_call_result_395085, SuperLU_395086], **kwargs_395087)
        
        # Processing the call keyword arguments (line 496)
        kwargs_395089 = {}
        # Getting the type of 'assert_' (line 496)
        assert__395078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 496)
        assert__call_result_395090 = invoke(stypy.reporting.localization.Localization(__file__, 496, 12), assert__395078, *[isinstance_call_result_395088], **kwargs_395089)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_spilu_drop_rule(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spilu_drop_rule' in the type store
        # Getting the type of 'stypy_return_type' (line 484)
        stypy_return_type_395091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395091)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spilu_drop_rule'
        return stypy_return_type_395091


    @norecursion
    def test_splu_nnz0(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_splu_nnz0'
        module_type_store = module_type_store.open_function_context('test_splu_nnz0', 498, 4, False)
        # Assigning a type to the variable 'self' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu.test_splu_nnz0.__dict__.__setitem__('stypy_localization', localization)
        TestSplu.test_splu_nnz0.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu.test_splu_nnz0.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu.test_splu_nnz0.__dict__.__setitem__('stypy_function_name', 'TestSplu.test_splu_nnz0')
        TestSplu.test_splu_nnz0.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu.test_splu_nnz0.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu.test_splu_nnz0.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu.test_splu_nnz0.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu.test_splu_nnz0.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu.test_splu_nnz0.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu.test_splu_nnz0.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.test_splu_nnz0', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_splu_nnz0', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_splu_nnz0(...)' code ##################

        
        # Assigning a Call to a Name (line 499):
        
        # Call to csc_matrix(...): (line 499)
        # Processing the call arguments (line 499)
        
        # Obtaining an instance of the builtin type 'tuple' (line 499)
        tuple_395093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 499)
        # Adding element type (line 499)
        int_395094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 24), tuple_395093, int_395094)
        # Adding element type (line 499)
        int_395095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 24), tuple_395093, int_395095)
        
        # Processing the call keyword arguments (line 499)
        str_395096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 36), 'str', 'd')
        keyword_395097 = str_395096
        kwargs_395098 = {'dtype': keyword_395097}
        # Getting the type of 'csc_matrix' (line 499)
        csc_matrix_395092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 499)
        csc_matrix_call_result_395099 = invoke(stypy.reporting.localization.Localization(__file__, 499, 12), csc_matrix_395092, *[tuple_395093], **kwargs_395098)
        
        # Assigning a type to the variable 'A' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'A', csc_matrix_call_result_395099)
        
        # Call to assert_raises(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of 'RuntimeError' (line 500)
        RuntimeError_395101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 22), 'RuntimeError', False)
        # Getting the type of 'splu' (line 500)
        splu_395102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 36), 'splu', False)
        # Getting the type of 'A' (line 500)
        A_395103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 42), 'A', False)
        # Processing the call keyword arguments (line 500)
        kwargs_395104 = {}
        # Getting the type of 'assert_raises' (line 500)
        assert_raises_395100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 500)
        assert_raises_call_result_395105 = invoke(stypy.reporting.localization.Localization(__file__, 500, 8), assert_raises_395100, *[RuntimeError_395101, splu_395102, A_395103], **kwargs_395104)
        
        
        # ################# End of 'test_splu_nnz0(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_splu_nnz0' in the type store
        # Getting the type of 'stypy_return_type' (line 498)
        stypy_return_type_395106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395106)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_splu_nnz0'
        return stypy_return_type_395106


    @norecursion
    def test_spilu_nnz0(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spilu_nnz0'
        module_type_store = module_type_store.open_function_context('test_spilu_nnz0', 502, 4, False)
        # Assigning a type to the variable 'self' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu.test_spilu_nnz0.__dict__.__setitem__('stypy_localization', localization)
        TestSplu.test_spilu_nnz0.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu.test_spilu_nnz0.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu.test_spilu_nnz0.__dict__.__setitem__('stypy_function_name', 'TestSplu.test_spilu_nnz0')
        TestSplu.test_spilu_nnz0.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu.test_spilu_nnz0.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu.test_spilu_nnz0.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu.test_spilu_nnz0.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu.test_spilu_nnz0.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu.test_spilu_nnz0.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu.test_spilu_nnz0.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.test_spilu_nnz0', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spilu_nnz0', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spilu_nnz0(...)' code ##################

        
        # Assigning a Call to a Name (line 503):
        
        # Call to csc_matrix(...): (line 503)
        # Processing the call arguments (line 503)
        
        # Obtaining an instance of the builtin type 'tuple' (line 503)
        tuple_395108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 503)
        # Adding element type (line 503)
        int_395109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 24), tuple_395108, int_395109)
        # Adding element type (line 503)
        int_395110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 24), tuple_395108, int_395110)
        
        # Processing the call keyword arguments (line 503)
        str_395111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 36), 'str', 'd')
        keyword_395112 = str_395111
        kwargs_395113 = {'dtype': keyword_395112}
        # Getting the type of 'csc_matrix' (line 503)
        csc_matrix_395107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 503)
        csc_matrix_call_result_395114 = invoke(stypy.reporting.localization.Localization(__file__, 503, 12), csc_matrix_395107, *[tuple_395108], **kwargs_395113)
        
        # Assigning a type to the variable 'A' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'A', csc_matrix_call_result_395114)
        
        # Call to assert_raises(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'RuntimeError' (line 504)
        RuntimeError_395116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 22), 'RuntimeError', False)
        # Getting the type of 'spilu' (line 504)
        spilu_395117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 36), 'spilu', False)
        # Getting the type of 'A' (line 504)
        A_395118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 43), 'A', False)
        # Processing the call keyword arguments (line 504)
        kwargs_395119 = {}
        # Getting the type of 'assert_raises' (line 504)
        assert_raises_395115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 504)
        assert_raises_call_result_395120 = invoke(stypy.reporting.localization.Localization(__file__, 504, 8), assert_raises_395115, *[RuntimeError_395116, spilu_395117, A_395118], **kwargs_395119)
        
        
        # ################# End of 'test_spilu_nnz0(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spilu_nnz0' in the type store
        # Getting the type of 'stypy_return_type' (line 502)
        stypy_return_type_395121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395121)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spilu_nnz0'
        return stypy_return_type_395121


    @norecursion
    def test_splu_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_splu_basic'
        module_type_store = module_type_store.open_function_context('test_splu_basic', 506, 4, False)
        # Assigning a type to the variable 'self' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu.test_splu_basic.__dict__.__setitem__('stypy_localization', localization)
        TestSplu.test_splu_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu.test_splu_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu.test_splu_basic.__dict__.__setitem__('stypy_function_name', 'TestSplu.test_splu_basic')
        TestSplu.test_splu_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu.test_splu_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu.test_splu_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu.test_splu_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu.test_splu_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu.test_splu_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu.test_splu_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.test_splu_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_splu_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_splu_basic(...)' code ##################

        
        # Assigning a Num to a Name (line 508):
        int_395122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 12), 'int')
        # Assigning a type to the variable 'n' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'n', int_395122)
        
        # Assigning a Call to a Name (line 509):
        
        # Call to RandomState(...): (line 509)
        # Processing the call arguments (line 509)
        int_395125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 33), 'int')
        # Processing the call keyword arguments (line 509)
        kwargs_395126 = {}
        # Getting the type of 'random' (line 509)
        random_395123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 14), 'random', False)
        # Obtaining the member 'RandomState' of a type (line 509)
        RandomState_395124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 14), random_395123, 'RandomState')
        # Calling RandomState(args, kwargs) (line 509)
        RandomState_call_result_395127 = invoke(stypy.reporting.localization.Localization(__file__, 509, 14), RandomState_395124, *[int_395125], **kwargs_395126)
        
        # Assigning a type to the variable 'rng' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'rng', RandomState_call_result_395127)
        
        # Assigning a Call to a Name (line 510):
        
        # Call to rand(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'n' (line 510)
        n_395130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 21), 'n', False)
        # Getting the type of 'n' (line 510)
        n_395131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 24), 'n', False)
        # Processing the call keyword arguments (line 510)
        kwargs_395132 = {}
        # Getting the type of 'rng' (line 510)
        rng_395128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'rng', False)
        # Obtaining the member 'rand' of a type (line 510)
        rand_395129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 12), rng_395128, 'rand')
        # Calling rand(args, kwargs) (line 510)
        rand_call_result_395133 = invoke(stypy.reporting.localization.Localization(__file__, 510, 12), rand_395129, *[n_395130, n_395131], **kwargs_395132)
        
        # Assigning a type to the variable 'a' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'a', rand_call_result_395133)
        
        # Assigning a Num to a Subscript (line 511):
        int_395134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 22), 'int')
        # Getting the type of 'a' (line 511)
        a_395135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'a')
        
        # Getting the type of 'a' (line 511)
        a_395136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 10), 'a')
        float_395137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 14), 'float')
        # Applying the binary operator '<' (line 511)
        result_lt_395138 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 10), '<', a_395136, float_395137)
        
        # Storing an element on a container (line 511)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 8), a_395135, (result_lt_395138, int_395134))
        
        # Assigning a Num to a Subscript (line 513):
        int_395139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 18), 'int')
        # Getting the type of 'a' (line 513)
        a_395140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'a')
        slice_395141 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 513, 8), None, None, None)
        int_395142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 13), 'int')
        # Storing an element on a container (line 513)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 8), a_395140, ((slice_395141, int_395142), int_395139))
        
        # Assigning a Call to a Name (line 514):
        
        # Call to csc_matrix(...): (line 514)
        # Processing the call arguments (line 514)
        # Getting the type of 'a' (line 514)
        a_395144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 24), 'a', False)
        # Processing the call keyword arguments (line 514)
        kwargs_395145 = {}
        # Getting the type of 'csc_matrix' (line 514)
        csc_matrix_395143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 13), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 514)
        csc_matrix_call_result_395146 = invoke(stypy.reporting.localization.Localization(__file__, 514, 13), csc_matrix_395143, *[a_395144], **kwargs_395145)
        
        # Assigning a type to the variable 'a_' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'a_', csc_matrix_call_result_395146)
        
        # Call to assert_raises(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'RuntimeError' (line 516)
        RuntimeError_395148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 22), 'RuntimeError', False)
        # Getting the type of 'splu' (line 516)
        splu_395149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 36), 'splu', False)
        # Getting the type of 'a_' (line 516)
        a__395150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 42), 'a_', False)
        # Processing the call keyword arguments (line 516)
        kwargs_395151 = {}
        # Getting the type of 'assert_raises' (line 516)
        assert_raises_395147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 516)
        assert_raises_call_result_395152 = invoke(stypy.reporting.localization.Localization(__file__, 516, 8), assert_raises_395147, *[RuntimeError_395148, splu_395149, a__395150], **kwargs_395151)
        
        
        # Getting the type of 'a' (line 519)
        a_395153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'a')
        int_395154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 13), 'int')
        
        # Call to eye(...): (line 519)
        # Processing the call arguments (line 519)
        # Getting the type of 'n' (line 519)
        n_395156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 19), 'n', False)
        # Processing the call keyword arguments (line 519)
        kwargs_395157 = {}
        # Getting the type of 'eye' (line 519)
        eye_395155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 15), 'eye', False)
        # Calling eye(args, kwargs) (line 519)
        eye_call_result_395158 = invoke(stypy.reporting.localization.Localization(__file__, 519, 15), eye_395155, *[n_395156], **kwargs_395157)
        
        # Applying the binary operator '*' (line 519)
        result_mul_395159 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 13), '*', int_395154, eye_call_result_395158)
        
        # Applying the binary operator '+=' (line 519)
        result_iadd_395160 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 8), '+=', a_395153, result_mul_395159)
        # Assigning a type to the variable 'a' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'a', result_iadd_395160)
        
        
        # Assigning a Call to a Name (line 520):
        
        # Call to csc_matrix(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 'a' (line 520)
        a_395162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 24), 'a', False)
        # Processing the call keyword arguments (line 520)
        kwargs_395163 = {}
        # Getting the type of 'csc_matrix' (line 520)
        csc_matrix_395161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 13), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 520)
        csc_matrix_call_result_395164 = invoke(stypy.reporting.localization.Localization(__file__, 520, 13), csc_matrix_395161, *[a_395162], **kwargs_395163)
        
        # Assigning a type to the variable 'a_' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'a_', csc_matrix_call_result_395164)
        
        # Assigning a Call to a Name (line 521):
        
        # Call to splu(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'a_' (line 521)
        a__395166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 18), 'a_', False)
        # Processing the call keyword arguments (line 521)
        kwargs_395167 = {}
        # Getting the type of 'splu' (line 521)
        splu_395165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 13), 'splu', False)
        # Calling splu(args, kwargs) (line 521)
        splu_call_result_395168 = invoke(stypy.reporting.localization.Localization(__file__, 521, 13), splu_395165, *[a__395166], **kwargs_395167)
        
        # Assigning a type to the variable 'lu' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'lu', splu_call_result_395168)
        
        # Assigning a Call to a Name (line 522):
        
        # Call to ones(...): (line 522)
        # Processing the call arguments (line 522)
        # Getting the type of 'n' (line 522)
        n_395170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 17), 'n', False)
        # Processing the call keyword arguments (line 522)
        kwargs_395171 = {}
        # Getting the type of 'ones' (line 522)
        ones_395169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 522)
        ones_call_result_395172 = invoke(stypy.reporting.localization.Localization(__file__, 522, 12), ones_395169, *[n_395170], **kwargs_395171)
        
        # Assigning a type to the variable 'b' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'b', ones_call_result_395172)
        
        # Assigning a Call to a Name (line 523):
        
        # Call to solve(...): (line 523)
        # Processing the call arguments (line 523)
        # Getting the type of 'b' (line 523)
        b_395175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 21), 'b', False)
        # Processing the call keyword arguments (line 523)
        kwargs_395176 = {}
        # Getting the type of 'lu' (line 523)
        lu_395173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'lu', False)
        # Obtaining the member 'solve' of a type (line 523)
        solve_395174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 12), lu_395173, 'solve')
        # Calling solve(args, kwargs) (line 523)
        solve_call_result_395177 = invoke(stypy.reporting.localization.Localization(__file__, 523, 12), solve_395174, *[b_395175], **kwargs_395176)
        
        # Assigning a type to the variable 'x' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'x', solve_call_result_395177)
        
        # Call to assert_almost_equal(...): (line 524)
        # Processing the call arguments (line 524)
        
        # Call to dot(...): (line 524)
        # Processing the call arguments (line 524)
        # Getting the type of 'a' (line 524)
        a_395180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 32), 'a', False)
        # Getting the type of 'x' (line 524)
        x_395181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 35), 'x', False)
        # Processing the call keyword arguments (line 524)
        kwargs_395182 = {}
        # Getting the type of 'dot' (line 524)
        dot_395179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 28), 'dot', False)
        # Calling dot(args, kwargs) (line 524)
        dot_call_result_395183 = invoke(stypy.reporting.localization.Localization(__file__, 524, 28), dot_395179, *[a_395180, x_395181], **kwargs_395182)
        
        # Getting the type of 'b' (line 524)
        b_395184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 39), 'b', False)
        # Processing the call keyword arguments (line 524)
        kwargs_395185 = {}
        # Getting the type of 'assert_almost_equal' (line 524)
        assert_almost_equal_395178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 524)
        assert_almost_equal_call_result_395186 = invoke(stypy.reporting.localization.Localization(__file__, 524, 8), assert_almost_equal_395178, *[dot_call_result_395183, b_395184], **kwargs_395185)
        
        
        # ################# End of 'test_splu_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_splu_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 506)
        stypy_return_type_395187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395187)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_splu_basic'
        return stypy_return_type_395187


    @norecursion
    def test_splu_perm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_splu_perm'
        module_type_store = module_type_store.open_function_context('test_splu_perm', 526, 4, False)
        # Assigning a type to the variable 'self' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu.test_splu_perm.__dict__.__setitem__('stypy_localization', localization)
        TestSplu.test_splu_perm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu.test_splu_perm.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu.test_splu_perm.__dict__.__setitem__('stypy_function_name', 'TestSplu.test_splu_perm')
        TestSplu.test_splu_perm.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu.test_splu_perm.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu.test_splu_perm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu.test_splu_perm.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu.test_splu_perm.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu.test_splu_perm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu.test_splu_perm.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.test_splu_perm', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_splu_perm', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_splu_perm(...)' code ##################

        
        # Assigning a Num to a Name (line 528):
        int_395188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 12), 'int')
        # Assigning a type to the variable 'n' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'n', int_395188)
        
        # Assigning a Call to a Name (line 529):
        
        # Call to random(...): (line 529)
        # Processing the call arguments (line 529)
        
        # Obtaining an instance of the builtin type 'tuple' (line 529)
        tuple_395191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 529)
        # Adding element type (line 529)
        # Getting the type of 'n' (line 529)
        n_395192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 27), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 27), tuple_395191, n_395192)
        # Adding element type (line 529)
        # Getting the type of 'n' (line 529)
        n_395193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 30), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 27), tuple_395191, n_395193)
        
        # Processing the call keyword arguments (line 529)
        kwargs_395194 = {}
        # Getting the type of 'random' (line 529)
        random_395189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'random', False)
        # Obtaining the member 'random' of a type (line 529)
        random_395190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 12), random_395189, 'random')
        # Calling random(args, kwargs) (line 529)
        random_call_result_395195 = invoke(stypy.reporting.localization.Localization(__file__, 529, 12), random_395190, *[tuple_395191], **kwargs_395194)
        
        # Assigning a type to the variable 'a' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'a', random_call_result_395195)
        
        # Assigning a Num to a Subscript (line 530):
        int_395196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 22), 'int')
        # Getting the type of 'a' (line 530)
        a_395197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'a')
        
        # Getting the type of 'a' (line 530)
        a_395198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 10), 'a')
        float_395199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 14), 'float')
        # Applying the binary operator '<' (line 530)
        result_lt_395200 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 10), '<', a_395198, float_395199)
        
        # Storing an element on a container (line 530)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 530, 8), a_395197, (result_lt_395200, int_395196))
        
        # Getting the type of 'a' (line 532)
        a_395201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'a')
        int_395202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 13), 'int')
        
        # Call to eye(...): (line 532)
        # Processing the call arguments (line 532)
        # Getting the type of 'n' (line 532)
        n_395204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 19), 'n', False)
        # Processing the call keyword arguments (line 532)
        kwargs_395205 = {}
        # Getting the type of 'eye' (line 532)
        eye_395203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 15), 'eye', False)
        # Calling eye(args, kwargs) (line 532)
        eye_call_result_395206 = invoke(stypy.reporting.localization.Localization(__file__, 532, 15), eye_395203, *[n_395204], **kwargs_395205)
        
        # Applying the binary operator '*' (line 532)
        result_mul_395207 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 13), '*', int_395202, eye_call_result_395206)
        
        # Applying the binary operator '+=' (line 532)
        result_iadd_395208 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 8), '+=', a_395201, result_mul_395207)
        # Assigning a type to the variable 'a' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'a', result_iadd_395208)
        
        
        # Assigning a Call to a Name (line 533):
        
        # Call to csc_matrix(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'a' (line 533)
        a_395210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 24), 'a', False)
        # Processing the call keyword arguments (line 533)
        kwargs_395211 = {}
        # Getting the type of 'csc_matrix' (line 533)
        csc_matrix_395209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 13), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 533)
        csc_matrix_call_result_395212 = invoke(stypy.reporting.localization.Localization(__file__, 533, 13), csc_matrix_395209, *[a_395210], **kwargs_395211)
        
        # Assigning a type to the variable 'a_' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'a_', csc_matrix_call_result_395212)
        
        # Assigning a Call to a Name (line 534):
        
        # Call to splu(...): (line 534)
        # Processing the call arguments (line 534)
        # Getting the type of 'a_' (line 534)
        a__395214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 18), 'a_', False)
        # Processing the call keyword arguments (line 534)
        kwargs_395215 = {}
        # Getting the type of 'splu' (line 534)
        splu_395213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 13), 'splu', False)
        # Calling splu(args, kwargs) (line 534)
        splu_call_result_395216 = invoke(stypy.reporting.localization.Localization(__file__, 534, 13), splu_395213, *[a__395214], **kwargs_395215)
        
        # Assigning a type to the variable 'lu' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'lu', splu_call_result_395216)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 536)
        tuple_395217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 536)
        # Adding element type (line 536)
        # Getting the type of 'lu' (line 536)
        lu_395218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 21), 'lu')
        # Obtaining the member 'perm_r' of a type (line 536)
        perm_r_395219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 21), lu_395218, 'perm_r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 21), tuple_395217, perm_r_395219)
        # Adding element type (line 536)
        # Getting the type of 'lu' (line 536)
        lu_395220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 32), 'lu')
        # Obtaining the member 'perm_c' of a type (line 536)
        perm_c_395221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 32), lu_395220, 'perm_c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 21), tuple_395217, perm_c_395221)
        
        # Testing the type of a for loop iterable (line 536)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 536, 8), tuple_395217)
        # Getting the type of the for loop variable (line 536)
        for_loop_var_395222 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 536, 8), tuple_395217)
        # Assigning a type to the variable 'perm' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'perm', for_loop_var_395222)
        # SSA begins for a for statement (line 536)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_(...): (line 537)
        # Processing the call arguments (line 537)
        
        # Call to all(...): (line 537)
        # Processing the call arguments (line 537)
        
        # Getting the type of 'perm' (line 537)
        perm_395225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 24), 'perm', False)
        int_395226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 31), 'int')
        # Applying the binary operator '>' (line 537)
        result_gt_395227 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 24), '>', perm_395225, int_395226)
        
        # Processing the call keyword arguments (line 537)
        kwargs_395228 = {}
        # Getting the type of 'all' (line 537)
        all_395224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 20), 'all', False)
        # Calling all(args, kwargs) (line 537)
        all_call_result_395229 = invoke(stypy.reporting.localization.Localization(__file__, 537, 20), all_395224, *[result_gt_395227], **kwargs_395228)
        
        # Processing the call keyword arguments (line 537)
        kwargs_395230 = {}
        # Getting the type of 'assert_' (line 537)
        assert__395223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 537)
        assert__call_result_395231 = invoke(stypy.reporting.localization.Localization(__file__, 537, 12), assert__395223, *[all_call_result_395229], **kwargs_395230)
        
        
        # Call to assert_(...): (line 538)
        # Processing the call arguments (line 538)
        
        # Call to all(...): (line 538)
        # Processing the call arguments (line 538)
        
        # Getting the type of 'perm' (line 538)
        perm_395234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 24), 'perm', False)
        # Getting the type of 'n' (line 538)
        n_395235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 31), 'n', False)
        # Applying the binary operator '<' (line 538)
        result_lt_395236 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 24), '<', perm_395234, n_395235)
        
        # Processing the call keyword arguments (line 538)
        kwargs_395237 = {}
        # Getting the type of 'all' (line 538)
        all_395233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 20), 'all', False)
        # Calling all(args, kwargs) (line 538)
        all_call_result_395238 = invoke(stypy.reporting.localization.Localization(__file__, 538, 20), all_395233, *[result_lt_395236], **kwargs_395237)
        
        # Processing the call keyword arguments (line 538)
        kwargs_395239 = {}
        # Getting the type of 'assert_' (line 538)
        assert__395232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 538)
        assert__call_result_395240 = invoke(stypy.reporting.localization.Localization(__file__, 538, 12), assert__395232, *[all_call_result_395238], **kwargs_395239)
        
        
        # Call to assert_equal(...): (line 539)
        # Processing the call arguments (line 539)
        
        # Call to len(...): (line 539)
        # Processing the call arguments (line 539)
        
        # Call to unique(...): (line 539)
        # Processing the call arguments (line 539)
        # Getting the type of 'perm' (line 539)
        perm_395244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 36), 'perm', False)
        # Processing the call keyword arguments (line 539)
        kwargs_395245 = {}
        # Getting the type of 'unique' (line 539)
        unique_395243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 29), 'unique', False)
        # Calling unique(args, kwargs) (line 539)
        unique_call_result_395246 = invoke(stypy.reporting.localization.Localization(__file__, 539, 29), unique_395243, *[perm_395244], **kwargs_395245)
        
        # Processing the call keyword arguments (line 539)
        kwargs_395247 = {}
        # Getting the type of 'len' (line 539)
        len_395242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 25), 'len', False)
        # Calling len(args, kwargs) (line 539)
        len_call_result_395248 = invoke(stypy.reporting.localization.Localization(__file__, 539, 25), len_395242, *[unique_call_result_395246], **kwargs_395247)
        
        
        # Call to len(...): (line 539)
        # Processing the call arguments (line 539)
        # Getting the type of 'perm' (line 539)
        perm_395250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 48), 'perm', False)
        # Processing the call keyword arguments (line 539)
        kwargs_395251 = {}
        # Getting the type of 'len' (line 539)
        len_395249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 44), 'len', False)
        # Calling len(args, kwargs) (line 539)
        len_call_result_395252 = invoke(stypy.reporting.localization.Localization(__file__, 539, 44), len_395249, *[perm_395250], **kwargs_395251)
        
        # Processing the call keyword arguments (line 539)
        kwargs_395253 = {}
        # Getting the type of 'assert_equal' (line 539)
        assert_equal_395241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 539)
        assert_equal_call_result_395254 = invoke(stypy.reporting.localization.Localization(__file__, 539, 12), assert_equal_395241, *[len_call_result_395248, len_call_result_395252], **kwargs_395253)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 544):
        # Getting the type of 'a' (line 544)
        a_395255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'a')
        # Getting the type of 'a' (line 544)
        a_395256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'a')
        # Obtaining the member 'T' of a type (line 544)
        T_395257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 16), a_395256, 'T')
        # Applying the binary operator '+' (line 544)
        result_add_395258 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 12), '+', a_395255, T_395257)
        
        # Assigning a type to the variable 'a' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'a', result_add_395258)
        
        # Assigning a Call to a Name (line 545):
        
        # Call to csc_matrix(...): (line 545)
        # Processing the call arguments (line 545)
        # Getting the type of 'a' (line 545)
        a_395260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 24), 'a', False)
        # Processing the call keyword arguments (line 545)
        kwargs_395261 = {}
        # Getting the type of 'csc_matrix' (line 545)
        csc_matrix_395259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 13), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 545)
        csc_matrix_call_result_395262 = invoke(stypy.reporting.localization.Localization(__file__, 545, 13), csc_matrix_395259, *[a_395260], **kwargs_395261)
        
        # Assigning a type to the variable 'a_' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'a_', csc_matrix_call_result_395262)
        
        # Assigning a Call to a Name (line 546):
        
        # Call to splu(...): (line 546)
        # Processing the call arguments (line 546)
        # Getting the type of 'a_' (line 546)
        a__395264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 18), 'a_', False)
        # Processing the call keyword arguments (line 546)
        kwargs_395265 = {}
        # Getting the type of 'splu' (line 546)
        splu_395263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 13), 'splu', False)
        # Calling splu(args, kwargs) (line 546)
        splu_call_result_395266 = invoke(stypy.reporting.localization.Localization(__file__, 546, 13), splu_395263, *[a__395264], **kwargs_395265)
        
        # Assigning a type to the variable 'lu' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'lu', splu_call_result_395266)
        
        # Call to assert_array_equal(...): (line 547)
        # Processing the call arguments (line 547)
        # Getting the type of 'lu' (line 547)
        lu_395268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 27), 'lu', False)
        # Obtaining the member 'perm_r' of a type (line 547)
        perm_r_395269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 27), lu_395268, 'perm_r')
        # Getting the type of 'lu' (line 547)
        lu_395270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 38), 'lu', False)
        # Obtaining the member 'perm_c' of a type (line 547)
        perm_c_395271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 38), lu_395270, 'perm_c')
        # Processing the call keyword arguments (line 547)
        kwargs_395272 = {}
        # Getting the type of 'assert_array_equal' (line 547)
        assert_array_equal_395267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 547)
        assert_array_equal_call_result_395273 = invoke(stypy.reporting.localization.Localization(__file__, 547, 8), assert_array_equal_395267, *[perm_r_395269, perm_c_395271], **kwargs_395272)
        
        
        # ################# End of 'test_splu_perm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_splu_perm' in the type store
        # Getting the type of 'stypy_return_type' (line 526)
        stypy_return_type_395274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395274)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_splu_perm'
        return stypy_return_type_395274


    @norecursion
    def test_lu_refcount(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_lu_refcount'
        module_type_store = module_type_store.open_function_context('test_lu_refcount', 549, 4, False)
        # Assigning a type to the variable 'self' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu.test_lu_refcount.__dict__.__setitem__('stypy_localization', localization)
        TestSplu.test_lu_refcount.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu.test_lu_refcount.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu.test_lu_refcount.__dict__.__setitem__('stypy_function_name', 'TestSplu.test_lu_refcount')
        TestSplu.test_lu_refcount.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu.test_lu_refcount.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu.test_lu_refcount.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu.test_lu_refcount.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu.test_lu_refcount.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu.test_lu_refcount.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu.test_lu_refcount.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.test_lu_refcount', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_lu_refcount', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_lu_refcount(...)' code ##################

        
        # Assigning a Num to a Name (line 551):
        int_395275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 12), 'int')
        # Assigning a type to the variable 'n' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'n', int_395275)
        
        # Assigning a Call to a Name (line 552):
        
        # Call to random(...): (line 552)
        # Processing the call arguments (line 552)
        
        # Obtaining an instance of the builtin type 'tuple' (line 552)
        tuple_395278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 552)
        # Adding element type (line 552)
        # Getting the type of 'n' (line 552)
        n_395279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 27), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 27), tuple_395278, n_395279)
        # Adding element type (line 552)
        # Getting the type of 'n' (line 552)
        n_395280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 30), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 27), tuple_395278, n_395280)
        
        # Processing the call keyword arguments (line 552)
        kwargs_395281 = {}
        # Getting the type of 'random' (line 552)
        random_395276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'random', False)
        # Obtaining the member 'random' of a type (line 552)
        random_395277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 12), random_395276, 'random')
        # Calling random(args, kwargs) (line 552)
        random_call_result_395282 = invoke(stypy.reporting.localization.Localization(__file__, 552, 12), random_395277, *[tuple_395278], **kwargs_395281)
        
        # Assigning a type to the variable 'a' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'a', random_call_result_395282)
        
        # Assigning a Num to a Subscript (line 553):
        int_395283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 22), 'int')
        # Getting the type of 'a' (line 553)
        a_395284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'a')
        
        # Getting the type of 'a' (line 553)
        a_395285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 10), 'a')
        float_395286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 14), 'float')
        # Applying the binary operator '<' (line 553)
        result_lt_395287 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 10), '<', a_395285, float_395286)
        
        # Storing an element on a container (line 553)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 8), a_395284, (result_lt_395287, int_395283))
        
        # Getting the type of 'a' (line 555)
        a_395288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'a')
        int_395289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 13), 'int')
        
        # Call to eye(...): (line 555)
        # Processing the call arguments (line 555)
        # Getting the type of 'n' (line 555)
        n_395291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 19), 'n', False)
        # Processing the call keyword arguments (line 555)
        kwargs_395292 = {}
        # Getting the type of 'eye' (line 555)
        eye_395290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 15), 'eye', False)
        # Calling eye(args, kwargs) (line 555)
        eye_call_result_395293 = invoke(stypy.reporting.localization.Localization(__file__, 555, 15), eye_395290, *[n_395291], **kwargs_395292)
        
        # Applying the binary operator '*' (line 555)
        result_mul_395294 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 13), '*', int_395289, eye_call_result_395293)
        
        # Applying the binary operator '+=' (line 555)
        result_iadd_395295 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 8), '+=', a_395288, result_mul_395294)
        # Assigning a type to the variable 'a' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'a', result_iadd_395295)
        
        
        # Assigning a Call to a Name (line 556):
        
        # Call to csc_matrix(...): (line 556)
        # Processing the call arguments (line 556)
        # Getting the type of 'a' (line 556)
        a_395297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 24), 'a', False)
        # Processing the call keyword arguments (line 556)
        kwargs_395298 = {}
        # Getting the type of 'csc_matrix' (line 556)
        csc_matrix_395296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 13), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 556)
        csc_matrix_call_result_395299 = invoke(stypy.reporting.localization.Localization(__file__, 556, 13), csc_matrix_395296, *[a_395297], **kwargs_395298)
        
        # Assigning a type to the variable 'a_' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'a_', csc_matrix_call_result_395299)
        
        # Assigning a Call to a Name (line 557):
        
        # Call to splu(...): (line 557)
        # Processing the call arguments (line 557)
        # Getting the type of 'a_' (line 557)
        a__395301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 18), 'a_', False)
        # Processing the call keyword arguments (line 557)
        kwargs_395302 = {}
        # Getting the type of 'splu' (line 557)
        splu_395300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 13), 'splu', False)
        # Calling splu(args, kwargs) (line 557)
        splu_call_result_395303 = invoke(stypy.reporting.localization.Localization(__file__, 557, 13), splu_395300, *[a__395301], **kwargs_395302)
        
        # Assigning a type to the variable 'lu' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'lu', splu_call_result_395303)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 560, 8))
        
        # 'import sys' statement (line 560)
        import sys

        import_module(stypy.reporting.localization.Localization(__file__, 560, 8), 'sys', sys, module_type_store)
        
        
        # Assigning a Call to a Name (line 561):
        
        # Call to getrefcount(...): (line 561)
        # Processing the call arguments (line 561)
        # Getting the type of 'lu' (line 561)
        lu_395306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 29), 'lu', False)
        # Processing the call keyword arguments (line 561)
        kwargs_395307 = {}
        # Getting the type of 'sys' (line 561)
        sys_395304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 13), 'sys', False)
        # Obtaining the member 'getrefcount' of a type (line 561)
        getrefcount_395305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 13), sys_395304, 'getrefcount')
        # Calling getrefcount(args, kwargs) (line 561)
        getrefcount_call_result_395308 = invoke(stypy.reporting.localization.Localization(__file__, 561, 13), getrefcount_395305, *[lu_395306], **kwargs_395307)
        
        # Assigning a type to the variable 'rc' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'rc', getrefcount_call_result_395308)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 562)
        tuple_395309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 562)
        # Adding element type (line 562)
        str_395310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 21), 'str', 'perm_r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 21), tuple_395309, str_395310)
        # Adding element type (line 562)
        str_395311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 31), 'str', 'perm_c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 21), tuple_395309, str_395311)
        
        # Testing the type of a for loop iterable (line 562)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 562, 8), tuple_395309)
        # Getting the type of the for loop variable (line 562)
        for_loop_var_395312 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 562, 8), tuple_395309)
        # Assigning a type to the variable 'attr' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'attr', for_loop_var_395312)
        # SSA begins for a for statement (line 562)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 563):
        
        # Call to getattr(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'lu' (line 563)
        lu_395314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 27), 'lu', False)
        # Getting the type of 'attr' (line 563)
        attr_395315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 31), 'attr', False)
        # Processing the call keyword arguments (line 563)
        kwargs_395316 = {}
        # Getting the type of 'getattr' (line 563)
        getattr_395313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 563)
        getattr_call_result_395317 = invoke(stypy.reporting.localization.Localization(__file__, 563, 19), getattr_395313, *[lu_395314, attr_395315], **kwargs_395316)
        
        # Assigning a type to the variable 'perm' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'perm', getattr_call_result_395317)
        
        # Call to assert_equal(...): (line 564)
        # Processing the call arguments (line 564)
        
        # Call to getrefcount(...): (line 564)
        # Processing the call arguments (line 564)
        # Getting the type of 'lu' (line 564)
        lu_395321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 41), 'lu', False)
        # Processing the call keyword arguments (line 564)
        kwargs_395322 = {}
        # Getting the type of 'sys' (line 564)
        sys_395319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 25), 'sys', False)
        # Obtaining the member 'getrefcount' of a type (line 564)
        getrefcount_395320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 25), sys_395319, 'getrefcount')
        # Calling getrefcount(args, kwargs) (line 564)
        getrefcount_call_result_395323 = invoke(stypy.reporting.localization.Localization(__file__, 564, 25), getrefcount_395320, *[lu_395321], **kwargs_395322)
        
        # Getting the type of 'rc' (line 564)
        rc_395324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 46), 'rc', False)
        int_395325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 51), 'int')
        # Applying the binary operator '+' (line 564)
        result_add_395326 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 46), '+', rc_395324, int_395325)
        
        # Processing the call keyword arguments (line 564)
        kwargs_395327 = {}
        # Getting the type of 'assert_equal' (line 564)
        assert_equal_395318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 564)
        assert_equal_call_result_395328 = invoke(stypy.reporting.localization.Localization(__file__, 564, 12), assert_equal_395318, *[getrefcount_call_result_395323, result_add_395326], **kwargs_395327)
        
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 565, 12), module_type_store, 'perm')
        
        # Call to assert_equal(...): (line 566)
        # Processing the call arguments (line 566)
        
        # Call to getrefcount(...): (line 566)
        # Processing the call arguments (line 566)
        # Getting the type of 'lu' (line 566)
        lu_395332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 41), 'lu', False)
        # Processing the call keyword arguments (line 566)
        kwargs_395333 = {}
        # Getting the type of 'sys' (line 566)
        sys_395330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 25), 'sys', False)
        # Obtaining the member 'getrefcount' of a type (line 566)
        getrefcount_395331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 25), sys_395330, 'getrefcount')
        # Calling getrefcount(args, kwargs) (line 566)
        getrefcount_call_result_395334 = invoke(stypy.reporting.localization.Localization(__file__, 566, 25), getrefcount_395331, *[lu_395332], **kwargs_395333)
        
        # Getting the type of 'rc' (line 566)
        rc_395335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 46), 'rc', False)
        # Processing the call keyword arguments (line 566)
        kwargs_395336 = {}
        # Getting the type of 'assert_equal' (line 566)
        assert_equal_395329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 566)
        assert_equal_call_result_395337 = invoke(stypy.reporting.localization.Localization(__file__, 566, 12), assert_equal_395329, *[getrefcount_call_result_395334, rc_395335], **kwargs_395336)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_lu_refcount(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_lu_refcount' in the type store
        # Getting the type of 'stypy_return_type' (line 549)
        stypy_return_type_395338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395338)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_lu_refcount'
        return stypy_return_type_395338


    @norecursion
    def test_bad_inputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bad_inputs'
        module_type_store = module_type_store.open_function_context('test_bad_inputs', 568, 4, False)
        # Assigning a type to the variable 'self' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu.test_bad_inputs.__dict__.__setitem__('stypy_localization', localization)
        TestSplu.test_bad_inputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu.test_bad_inputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu.test_bad_inputs.__dict__.__setitem__('stypy_function_name', 'TestSplu.test_bad_inputs')
        TestSplu.test_bad_inputs.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu.test_bad_inputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu.test_bad_inputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu.test_bad_inputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu.test_bad_inputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu.test_bad_inputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu.test_bad_inputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.test_bad_inputs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bad_inputs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bad_inputs(...)' code ##################

        
        # Assigning a Call to a Name (line 569):
        
        # Call to tocsc(...): (line 569)
        # Processing the call keyword arguments (line 569)
        kwargs_395342 = {}
        # Getting the type of 'self' (line 569)
        self_395339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'self', False)
        # Obtaining the member 'A' of a type (line 569)
        A_395340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 12), self_395339, 'A')
        # Obtaining the member 'tocsc' of a type (line 569)
        tocsc_395341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 12), A_395340, 'tocsc')
        # Calling tocsc(args, kwargs) (line 569)
        tocsc_call_result_395343 = invoke(stypy.reporting.localization.Localization(__file__, 569, 12), tocsc_395341, *[], **kwargs_395342)
        
        # Assigning a type to the variable 'A' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'A', tocsc_call_result_395343)
        
        # Call to assert_raises(...): (line 571)
        # Processing the call arguments (line 571)
        # Getting the type of 'ValueError' (line 571)
        ValueError_395345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 22), 'ValueError', False)
        # Getting the type of 'splu' (line 571)
        splu_395346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 34), 'splu', False)
        
        # Obtaining the type of the subscript
        slice_395347 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 571, 40), None, None, None)
        int_395348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 45), 'int')
        slice_395349 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 571, 40), None, int_395348, None)
        # Getting the type of 'A' (line 571)
        A_395350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 40), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 571)
        getitem___395351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 40), A_395350, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 571)
        subscript_call_result_395352 = invoke(stypy.reporting.localization.Localization(__file__, 571, 40), getitem___395351, (slice_395347, slice_395349))
        
        # Processing the call keyword arguments (line 571)
        kwargs_395353 = {}
        # Getting the type of 'assert_raises' (line 571)
        assert_raises_395344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 571)
        assert_raises_call_result_395354 = invoke(stypy.reporting.localization.Localization(__file__, 571, 8), assert_raises_395344, *[ValueError_395345, splu_395346, subscript_call_result_395352], **kwargs_395353)
        
        
        # Call to assert_raises(...): (line 572)
        # Processing the call arguments (line 572)
        # Getting the type of 'ValueError' (line 572)
        ValueError_395356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 22), 'ValueError', False)
        # Getting the type of 'spilu' (line 572)
        spilu_395357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 34), 'spilu', False)
        
        # Obtaining the type of the subscript
        slice_395358 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 572, 41), None, None, None)
        int_395359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 46), 'int')
        slice_395360 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 572, 41), None, int_395359, None)
        # Getting the type of 'A' (line 572)
        A_395361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 41), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 572)
        getitem___395362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 41), A_395361, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 572)
        subscript_call_result_395363 = invoke(stypy.reporting.localization.Localization(__file__, 572, 41), getitem___395362, (slice_395358, slice_395360))
        
        # Processing the call keyword arguments (line 572)
        kwargs_395364 = {}
        # Getting the type of 'assert_raises' (line 572)
        assert_raises_395355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 572)
        assert_raises_call_result_395365 = invoke(stypy.reporting.localization.Localization(__file__, 572, 8), assert_raises_395355, *[ValueError_395356, spilu_395357, subscript_call_result_395363], **kwargs_395364)
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 574)
        list_395366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 574)
        # Adding element type (line 574)
        
        # Call to splu(...): (line 574)
        # Processing the call arguments (line 574)
        # Getting the type of 'A' (line 574)
        A_395368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 24), 'A', False)
        # Processing the call keyword arguments (line 574)
        kwargs_395369 = {}
        # Getting the type of 'splu' (line 574)
        splu_395367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 19), 'splu', False)
        # Calling splu(args, kwargs) (line 574)
        splu_call_result_395370 = invoke(stypy.reporting.localization.Localization(__file__, 574, 19), splu_395367, *[A_395368], **kwargs_395369)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 18), list_395366, splu_call_result_395370)
        # Adding element type (line 574)
        
        # Call to spilu(...): (line 574)
        # Processing the call arguments (line 574)
        # Getting the type of 'A' (line 574)
        A_395372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 34), 'A', False)
        # Processing the call keyword arguments (line 574)
        kwargs_395373 = {}
        # Getting the type of 'spilu' (line 574)
        spilu_395371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 28), 'spilu', False)
        # Calling spilu(args, kwargs) (line 574)
        spilu_call_result_395374 = invoke(stypy.reporting.localization.Localization(__file__, 574, 28), spilu_395371, *[A_395372], **kwargs_395373)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 18), list_395366, spilu_call_result_395374)
        
        # Testing the type of a for loop iterable (line 574)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 574, 8), list_395366)
        # Getting the type of the for loop variable (line 574)
        for_loop_var_395375 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 574, 8), list_395366)
        # Assigning a type to the variable 'lu' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'lu', for_loop_var_395375)
        # SSA begins for a for statement (line 574)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 575):
        
        # Call to rand(...): (line 575)
        # Processing the call arguments (line 575)
        int_395378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 28), 'int')
        # Processing the call keyword arguments (line 575)
        kwargs_395379 = {}
        # Getting the type of 'random' (line 575)
        random_395376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'random', False)
        # Obtaining the member 'rand' of a type (line 575)
        rand_395377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 16), random_395376, 'rand')
        # Calling rand(args, kwargs) (line 575)
        rand_call_result_395380 = invoke(stypy.reporting.localization.Localization(__file__, 575, 16), rand_395377, *[int_395378], **kwargs_395379)
        
        # Assigning a type to the variable 'b' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'b', rand_call_result_395380)
        
        # Assigning a Call to a Name (line 576):
        
        # Call to rand(...): (line 576)
        # Processing the call arguments (line 576)
        int_395383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 28), 'int')
        int_395384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 32), 'int')
        # Processing the call keyword arguments (line 576)
        kwargs_395385 = {}
        # Getting the type of 'random' (line 576)
        random_395381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 16), 'random', False)
        # Obtaining the member 'rand' of a type (line 576)
        rand_395382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 16), random_395381, 'rand')
        # Calling rand(args, kwargs) (line 576)
        rand_call_result_395386 = invoke(stypy.reporting.localization.Localization(__file__, 576, 16), rand_395382, *[int_395383, int_395384], **kwargs_395385)
        
        # Assigning a type to the variable 'B' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'B', rand_call_result_395386)
        
        # Assigning a Call to a Name (line 577):
        
        # Call to rand(...): (line 577)
        # Processing the call arguments (line 577)
        # Getting the type of 'self' (line 577)
        self_395389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 29), 'self', False)
        # Obtaining the member 'n' of a type (line 577)
        n_395390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 29), self_395389, 'n')
        int_395391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 37), 'int')
        int_395392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 40), 'int')
        # Processing the call keyword arguments (line 577)
        kwargs_395393 = {}
        # Getting the type of 'random' (line 577)
        random_395387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 17), 'random', False)
        # Obtaining the member 'rand' of a type (line 577)
        rand_395388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 17), random_395387, 'rand')
        # Calling rand(args, kwargs) (line 577)
        rand_call_result_395394 = invoke(stypy.reporting.localization.Localization(__file__, 577, 17), rand_395388, *[n_395390, int_395391, int_395392], **kwargs_395393)
        
        # Assigning a type to the variable 'BB' (line 577)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 12), 'BB', rand_call_result_395394)
        
        # Call to assert_raises(...): (line 578)
        # Processing the call arguments (line 578)
        # Getting the type of 'ValueError' (line 578)
        ValueError_395396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 26), 'ValueError', False)
        # Getting the type of 'lu' (line 578)
        lu_395397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 38), 'lu', False)
        # Obtaining the member 'solve' of a type (line 578)
        solve_395398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 38), lu_395397, 'solve')
        # Getting the type of 'b' (line 578)
        b_395399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 48), 'b', False)
        # Processing the call keyword arguments (line 578)
        kwargs_395400 = {}
        # Getting the type of 'assert_raises' (line 578)
        assert_raises_395395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 578)
        assert_raises_call_result_395401 = invoke(stypy.reporting.localization.Localization(__file__, 578, 12), assert_raises_395395, *[ValueError_395396, solve_395398, b_395399], **kwargs_395400)
        
        
        # Call to assert_raises(...): (line 579)
        # Processing the call arguments (line 579)
        # Getting the type of 'ValueError' (line 579)
        ValueError_395403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 26), 'ValueError', False)
        # Getting the type of 'lu' (line 579)
        lu_395404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 38), 'lu', False)
        # Obtaining the member 'solve' of a type (line 579)
        solve_395405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 38), lu_395404, 'solve')
        # Getting the type of 'B' (line 579)
        B_395406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 48), 'B', False)
        # Processing the call keyword arguments (line 579)
        kwargs_395407 = {}
        # Getting the type of 'assert_raises' (line 579)
        assert_raises_395402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 579)
        assert_raises_call_result_395408 = invoke(stypy.reporting.localization.Localization(__file__, 579, 12), assert_raises_395402, *[ValueError_395403, solve_395405, B_395406], **kwargs_395407)
        
        
        # Call to assert_raises(...): (line 580)
        # Processing the call arguments (line 580)
        # Getting the type of 'ValueError' (line 580)
        ValueError_395410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 26), 'ValueError', False)
        # Getting the type of 'lu' (line 580)
        lu_395411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 38), 'lu', False)
        # Obtaining the member 'solve' of a type (line 580)
        solve_395412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 38), lu_395411, 'solve')
        # Getting the type of 'BB' (line 580)
        BB_395413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 48), 'BB', False)
        # Processing the call keyword arguments (line 580)
        kwargs_395414 = {}
        # Getting the type of 'assert_raises' (line 580)
        assert_raises_395409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 580)
        assert_raises_call_result_395415 = invoke(stypy.reporting.localization.Localization(__file__, 580, 12), assert_raises_395409, *[ValueError_395410, solve_395412, BB_395413], **kwargs_395414)
        
        
        # Call to assert_raises(...): (line 581)
        # Processing the call arguments (line 581)
        # Getting the type of 'TypeError' (line 581)
        TypeError_395417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 26), 'TypeError', False)
        # Getting the type of 'lu' (line 581)
        lu_395418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 37), 'lu', False)
        # Obtaining the member 'solve' of a type (line 581)
        solve_395419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 37), lu_395418, 'solve')
        
        # Call to astype(...): (line 582)
        # Processing the call arguments (line 582)
        # Getting the type of 'np' (line 582)
        np_395422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 35), 'np', False)
        # Obtaining the member 'complex64' of a type (line 582)
        complex64_395423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 35), np_395422, 'complex64')
        # Processing the call keyword arguments (line 582)
        kwargs_395424 = {}
        # Getting the type of 'b' (line 582)
        b_395420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 26), 'b', False)
        # Obtaining the member 'astype' of a type (line 582)
        astype_395421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 26), b_395420, 'astype')
        # Calling astype(args, kwargs) (line 582)
        astype_call_result_395425 = invoke(stypy.reporting.localization.Localization(__file__, 582, 26), astype_395421, *[complex64_395423], **kwargs_395424)
        
        # Processing the call keyword arguments (line 581)
        kwargs_395426 = {}
        # Getting the type of 'assert_raises' (line 581)
        assert_raises_395416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 581)
        assert_raises_call_result_395427 = invoke(stypy.reporting.localization.Localization(__file__, 581, 12), assert_raises_395416, *[TypeError_395417, solve_395419, astype_call_result_395425], **kwargs_395426)
        
        
        # Call to assert_raises(...): (line 583)
        # Processing the call arguments (line 583)
        # Getting the type of 'TypeError' (line 583)
        TypeError_395429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 26), 'TypeError', False)
        # Getting the type of 'lu' (line 583)
        lu_395430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 37), 'lu', False)
        # Obtaining the member 'solve' of a type (line 583)
        solve_395431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 37), lu_395430, 'solve')
        
        # Call to astype(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 'np' (line 584)
        np_395434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 35), 'np', False)
        # Obtaining the member 'complex128' of a type (line 584)
        complex128_395435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 35), np_395434, 'complex128')
        # Processing the call keyword arguments (line 584)
        kwargs_395436 = {}
        # Getting the type of 'b' (line 584)
        b_395432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 26), 'b', False)
        # Obtaining the member 'astype' of a type (line 584)
        astype_395433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 26), b_395432, 'astype')
        # Calling astype(args, kwargs) (line 584)
        astype_call_result_395437 = invoke(stypy.reporting.localization.Localization(__file__, 584, 26), astype_395433, *[complex128_395435], **kwargs_395436)
        
        # Processing the call keyword arguments (line 583)
        kwargs_395438 = {}
        # Getting the type of 'assert_raises' (line 583)
        assert_raises_395428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 583)
        assert_raises_call_result_395439 = invoke(stypy.reporting.localization.Localization(__file__, 583, 12), assert_raises_395428, *[TypeError_395429, solve_395431, astype_call_result_395437], **kwargs_395438)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_bad_inputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bad_inputs' in the type store
        # Getting the type of 'stypy_return_type' (line 568)
        stypy_return_type_395440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395440)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bad_inputs'
        return stypy_return_type_395440


    @norecursion
    def test_superlu_dlamch_i386_nan(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_superlu_dlamch_i386_nan'
        module_type_store = module_type_store.open_function_context('test_superlu_dlamch_i386_nan', 586, 4, False)
        # Assigning a type to the variable 'self' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu.test_superlu_dlamch_i386_nan.__dict__.__setitem__('stypy_localization', localization)
        TestSplu.test_superlu_dlamch_i386_nan.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu.test_superlu_dlamch_i386_nan.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu.test_superlu_dlamch_i386_nan.__dict__.__setitem__('stypy_function_name', 'TestSplu.test_superlu_dlamch_i386_nan')
        TestSplu.test_superlu_dlamch_i386_nan.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu.test_superlu_dlamch_i386_nan.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu.test_superlu_dlamch_i386_nan.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu.test_superlu_dlamch_i386_nan.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu.test_superlu_dlamch_i386_nan.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu.test_superlu_dlamch_i386_nan.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu.test_superlu_dlamch_i386_nan.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.test_superlu_dlamch_i386_nan', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_superlu_dlamch_i386_nan', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_superlu_dlamch_i386_nan(...)' code ##################

        
        # Assigning a Num to a Name (line 594):
        int_395441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 12), 'int')
        # Assigning a type to the variable 'n' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'n', int_395441)
        
        # Assigning a BinOp to a Name (line 595):
        
        # Call to arange(...): (line 595)
        # Processing the call arguments (line 595)
        # Getting the type of 'n' (line 595)
        n_395444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 22), 'n', False)
        # Processing the call keyword arguments (line 595)
        kwargs_395445 = {}
        # Getting the type of 'np' (line 595)
        np_395442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 595)
        arange_395443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 12), np_395442, 'arange')
        # Calling arange(args, kwargs) (line 595)
        arange_call_result_395446 = invoke(stypy.reporting.localization.Localization(__file__, 595, 12), arange_395443, *[n_395444], **kwargs_395445)
        
        int_395447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 27), 'int')
        # Applying the binary operator '+' (line 595)
        result_add_395448 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 12), '+', arange_call_result_395446, int_395447)
        
        # Assigning a type to the variable 'd' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'd', result_add_395448)
        
        # Assigning a Call to a Name (line 596):
        
        # Call to spdiags(...): (line 596)
        # Processing the call arguments (line 596)
        
        # Obtaining an instance of the builtin type 'tuple' (line 596)
        tuple_395450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 596)
        # Adding element type (line 596)
        # Getting the type of 'd' (line 596)
        d_395451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 21), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 21), tuple_395450, d_395451)
        # Adding element type (line 596)
        int_395452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 24), 'int')
        # Getting the type of 'd' (line 596)
        d_395453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 26), 'd', False)
        # Applying the binary operator '*' (line 596)
        result_mul_395454 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 24), '*', int_395452, d_395453)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 21), tuple_395450, result_mul_395454)
        # Adding element type (line 596)
        
        # Obtaining the type of the subscript
        int_395455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 33), 'int')
        slice_395456 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 596, 29), None, None, int_395455)
        # Getting the type of 'd' (line 596)
        d_395457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 29), 'd', False)
        # Obtaining the member '__getitem__' of a type (line 596)
        getitem___395458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 29), d_395457, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 596)
        subscript_call_result_395459 = invoke(stypy.reporting.localization.Localization(__file__, 596, 29), getitem___395458, slice_395456)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 21), tuple_395450, subscript_call_result_395459)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 596)
        tuple_395460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 596)
        # Adding element type (line 596)
        int_395461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 40), tuple_395460, int_395461)
        # Adding element type (line 596)
        int_395462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 40), tuple_395460, int_395462)
        # Adding element type (line 596)
        int_395463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 40), tuple_395460, int_395463)
        
        # Getting the type of 'n' (line 596)
        n_395464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 51), 'n', False)
        # Getting the type of 'n' (line 596)
        n_395465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 54), 'n', False)
        # Processing the call keyword arguments (line 596)
        kwargs_395466 = {}
        # Getting the type of 'spdiags' (line 596)
        spdiags_395449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 12), 'spdiags', False)
        # Calling spdiags(args, kwargs) (line 596)
        spdiags_call_result_395467 = invoke(stypy.reporting.localization.Localization(__file__, 596, 12), spdiags_395449, *[tuple_395450, tuple_395460, n_395464, n_395465], **kwargs_395466)
        
        # Assigning a type to the variable 'A' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'A', spdiags_call_result_395467)
        
        # Assigning a Call to a Name (line 597):
        
        # Call to astype(...): (line 597)
        # Processing the call arguments (line 597)
        # Getting the type of 'np' (line 597)
        np_395470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 21), 'np', False)
        # Obtaining the member 'float32' of a type (line 597)
        float32_395471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 21), np_395470, 'float32')
        # Processing the call keyword arguments (line 597)
        kwargs_395472 = {}
        # Getting the type of 'A' (line 597)
        A_395468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'A', False)
        # Obtaining the member 'astype' of a type (line 597)
        astype_395469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 12), A_395468, 'astype')
        # Calling astype(args, kwargs) (line 597)
        astype_call_result_395473 = invoke(stypy.reporting.localization.Localization(__file__, 597, 12), astype_395469, *[float32_395471], **kwargs_395472)
        
        # Assigning a type to the variable 'A' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'A', astype_call_result_395473)
        
        # Call to spilu(...): (line 598)
        # Processing the call arguments (line 598)
        # Getting the type of 'A' (line 598)
        A_395475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 14), 'A', False)
        # Processing the call keyword arguments (line 598)
        kwargs_395476 = {}
        # Getting the type of 'spilu' (line 598)
        spilu_395474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'spilu', False)
        # Calling spilu(args, kwargs) (line 598)
        spilu_call_result_395477 = invoke(stypy.reporting.localization.Localization(__file__, 598, 8), spilu_395474, *[A_395475], **kwargs_395476)
        
        
        # Assigning a BinOp to a Name (line 599):
        # Getting the type of 'A' (line 599)
        A_395478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 12), 'A')
        complex_395479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 16), 'complex')
        # Getting the type of 'A' (line 599)
        A_395480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 19), 'A')
        # Applying the binary operator '*' (line 599)
        result_mul_395481 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 16), '*', complex_395479, A_395480)
        
        # Applying the binary operator '+' (line 599)
        result_add_395482 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 12), '+', A_395478, result_mul_395481)
        
        # Assigning a type to the variable 'A' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'A', result_add_395482)
        
        # Assigning a Attribute to a Name (line 600):
        # Getting the type of 'A' (line 600)
        A_395483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'A')
        # Obtaining the member 'A' of a type (line 600)
        A_395484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 12), A_395483, 'A')
        # Assigning a type to the variable 'B' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'B', A_395484)
        
        # Call to assert_(...): (line 601)
        # Processing the call arguments (line 601)
        
        
        # Call to any(...): (line 601)
        # Processing the call keyword arguments (line 601)
        kwargs_395492 = {}
        
        # Call to isnan(...): (line 601)
        # Processing the call arguments (line 601)
        # Getting the type of 'B' (line 601)
        B_395488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 29), 'B', False)
        # Processing the call keyword arguments (line 601)
        kwargs_395489 = {}
        # Getting the type of 'np' (line 601)
        np_395486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 20), 'np', False)
        # Obtaining the member 'isnan' of a type (line 601)
        isnan_395487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 20), np_395486, 'isnan')
        # Calling isnan(args, kwargs) (line 601)
        isnan_call_result_395490 = invoke(stypy.reporting.localization.Localization(__file__, 601, 20), isnan_395487, *[B_395488], **kwargs_395489)
        
        # Obtaining the member 'any' of a type (line 601)
        any_395491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 20), isnan_call_result_395490, 'any')
        # Calling any(args, kwargs) (line 601)
        any_call_result_395493 = invoke(stypy.reporting.localization.Localization(__file__, 601, 20), any_395491, *[], **kwargs_395492)
        
        # Applying the 'not' unary operator (line 601)
        result_not__395494 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 16), 'not', any_call_result_395493)
        
        # Processing the call keyword arguments (line 601)
        kwargs_395495 = {}
        # Getting the type of 'assert_' (line 601)
        assert__395485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 601)
        assert__call_result_395496 = invoke(stypy.reporting.localization.Localization(__file__, 601, 8), assert__395485, *[result_not__395494], **kwargs_395495)
        
        
        # ################# End of 'test_superlu_dlamch_i386_nan(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_superlu_dlamch_i386_nan' in the type store
        # Getting the type of 'stypy_return_type' (line 586)
        stypy_return_type_395497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395497)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_superlu_dlamch_i386_nan'
        return stypy_return_type_395497


    @norecursion
    def test_lu_attr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_lu_attr'
        module_type_store = module_type_store.open_function_context('test_lu_attr', 603, 4, False)
        # Assigning a type to the variable 'self' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu.test_lu_attr.__dict__.__setitem__('stypy_localization', localization)
        TestSplu.test_lu_attr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu.test_lu_attr.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu.test_lu_attr.__dict__.__setitem__('stypy_function_name', 'TestSplu.test_lu_attr')
        TestSplu.test_lu_attr.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu.test_lu_attr.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu.test_lu_attr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu.test_lu_attr.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu.test_lu_attr.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu.test_lu_attr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu.test_lu_attr.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.test_lu_attr', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_lu_attr', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_lu_attr(...)' code ##################


        @norecursion
        def check(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'False' (line 606)
            False_395498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 35), 'False')
            defaults = [False_395498]
            # Create a new context for function 'check'
            module_type_store = module_type_store.open_function_context('check', 606, 8, False)
            
            # Passed parameters checking function
            check.stypy_localization = localization
            check.stypy_type_of_self = None
            check.stypy_type_store = module_type_store
            check.stypy_function_name = 'check'
            check.stypy_param_names_list = ['dtype', 'complex_2']
            check.stypy_varargs_param_name = None
            check.stypy_kwargs_param_name = None
            check.stypy_call_defaults = defaults
            check.stypy_call_varargs = varargs
            check.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'check', ['dtype', 'complex_2'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'check', localization, ['dtype', 'complex_2'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'check(...)' code ##################

            
            # Assigning a Call to a Name (line 607):
            
            # Call to astype(...): (line 607)
            # Processing the call arguments (line 607)
            # Getting the type of 'dtype' (line 607)
            dtype_395502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 30), 'dtype', False)
            # Processing the call keyword arguments (line 607)
            kwargs_395503 = {}
            # Getting the type of 'self' (line 607)
            self_395499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 16), 'self', False)
            # Obtaining the member 'A' of a type (line 607)
            A_395500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 16), self_395499, 'A')
            # Obtaining the member 'astype' of a type (line 607)
            astype_395501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 16), A_395500, 'astype')
            # Calling astype(args, kwargs) (line 607)
            astype_call_result_395504 = invoke(stypy.reporting.localization.Localization(__file__, 607, 16), astype_395501, *[dtype_395502], **kwargs_395503)
            
            # Assigning a type to the variable 'A' (line 607)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'A', astype_call_result_395504)
            
            # Getting the type of 'complex_2' (line 609)
            complex_2_395505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 15), 'complex_2')
            # Testing the type of an if condition (line 609)
            if_condition_395506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 609, 12), complex_2_395505)
            # Assigning a type to the variable 'if_condition_395506' (line 609)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'if_condition_395506', if_condition_395506)
            # SSA begins for if statement (line 609)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 610):
            # Getting the type of 'A' (line 610)
            A_395507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 20), 'A')
            complex_395508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 24), 'complex')
            # Getting the type of 'A' (line 610)
            A_395509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 27), 'A')
            # Obtaining the member 'T' of a type (line 610)
            T_395510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 27), A_395509, 'T')
            # Applying the binary operator '*' (line 610)
            result_mul_395511 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 24), '*', complex_395508, T_395510)
            
            # Applying the binary operator '+' (line 610)
            result_add_395512 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 20), '+', A_395507, result_mul_395511)
            
            # Assigning a type to the variable 'A' (line 610)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 16), 'A', result_add_395512)
            # SSA join for if statement (line 609)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Subscript to a Name (line 612):
            
            # Obtaining the type of the subscript
            int_395513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 24), 'int')
            # Getting the type of 'A' (line 612)
            A_395514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 16), 'A')
            # Obtaining the member 'shape' of a type (line 612)
            shape_395515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 16), A_395514, 'shape')
            # Obtaining the member '__getitem__' of a type (line 612)
            getitem___395516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 16), shape_395515, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 612)
            subscript_call_result_395517 = invoke(stypy.reporting.localization.Localization(__file__, 612, 16), getitem___395516, int_395513)
            
            # Assigning a type to the variable 'n' (line 612)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'n', subscript_call_result_395517)
            
            # Assigning a Call to a Name (line 613):
            
            # Call to splu(...): (line 613)
            # Processing the call arguments (line 613)
            # Getting the type of 'A' (line 613)
            A_395519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 22), 'A', False)
            # Processing the call keyword arguments (line 613)
            kwargs_395520 = {}
            # Getting the type of 'splu' (line 613)
            splu_395518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 17), 'splu', False)
            # Calling splu(args, kwargs) (line 613)
            splu_call_result_395521 = invoke(stypy.reporting.localization.Localization(__file__, 613, 17), splu_395518, *[A_395519], **kwargs_395520)
            
            # Assigning a type to the variable 'lu' (line 613)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'lu', splu_call_result_395521)
            
            # Assigning a Call to a Name (line 617):
            
            # Call to zeros(...): (line 617)
            # Processing the call arguments (line 617)
            
            # Obtaining an instance of the builtin type 'tuple' (line 617)
            tuple_395524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 617)
            # Adding element type (line 617)
            # Getting the type of 'n' (line 617)
            n_395525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 27), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 27), tuple_395524, n_395525)
            # Adding element type (line 617)
            # Getting the type of 'n' (line 617)
            n_395526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 30), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 27), tuple_395524, n_395526)
            
            # Processing the call keyword arguments (line 617)
            kwargs_395527 = {}
            # Getting the type of 'np' (line 617)
            np_395522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 17), 'np', False)
            # Obtaining the member 'zeros' of a type (line 617)
            zeros_395523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 17), np_395522, 'zeros')
            # Calling zeros(args, kwargs) (line 617)
            zeros_call_result_395528 = invoke(stypy.reporting.localization.Localization(__file__, 617, 17), zeros_395523, *[tuple_395524], **kwargs_395527)
            
            # Assigning a type to the variable 'Pc' (line 617)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'Pc', zeros_call_result_395528)
            
            # Assigning a Num to a Subscript (line 618):
            int_395529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 42), 'int')
            # Getting the type of 'Pc' (line 618)
            Pc_395530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 12), 'Pc')
            
            # Obtaining an instance of the builtin type 'tuple' (line 618)
            tuple_395531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 15), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 618)
            # Adding element type (line 618)
            
            # Call to arange(...): (line 618)
            # Processing the call arguments (line 618)
            # Getting the type of 'n' (line 618)
            n_395534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 25), 'n', False)
            # Processing the call keyword arguments (line 618)
            kwargs_395535 = {}
            # Getting the type of 'np' (line 618)
            np_395532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 15), 'np', False)
            # Obtaining the member 'arange' of a type (line 618)
            arange_395533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 15), np_395532, 'arange')
            # Calling arange(args, kwargs) (line 618)
            arange_call_result_395536 = invoke(stypy.reporting.localization.Localization(__file__, 618, 15), arange_395533, *[n_395534], **kwargs_395535)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 15), tuple_395531, arange_call_result_395536)
            # Adding element type (line 618)
            # Getting the type of 'lu' (line 618)
            lu_395537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 29), 'lu')
            # Obtaining the member 'perm_c' of a type (line 618)
            perm_c_395538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 29), lu_395537, 'perm_c')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 15), tuple_395531, perm_c_395538)
            
            # Storing an element on a container (line 618)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 12), Pc_395530, (tuple_395531, int_395529))
            
            # Assigning a Call to a Name (line 620):
            
            # Call to zeros(...): (line 620)
            # Processing the call arguments (line 620)
            
            # Obtaining an instance of the builtin type 'tuple' (line 620)
            tuple_395541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 620)
            # Adding element type (line 620)
            # Getting the type of 'n' (line 620)
            n_395542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 27), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 27), tuple_395541, n_395542)
            # Adding element type (line 620)
            # Getting the type of 'n' (line 620)
            n_395543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 30), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 27), tuple_395541, n_395543)
            
            # Processing the call keyword arguments (line 620)
            kwargs_395544 = {}
            # Getting the type of 'np' (line 620)
            np_395539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 17), 'np', False)
            # Obtaining the member 'zeros' of a type (line 620)
            zeros_395540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 17), np_395539, 'zeros')
            # Calling zeros(args, kwargs) (line 620)
            zeros_call_result_395545 = invoke(stypy.reporting.localization.Localization(__file__, 620, 17), zeros_395540, *[tuple_395541], **kwargs_395544)
            
            # Assigning a type to the variable 'Pr' (line 620)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 12), 'Pr', zeros_call_result_395545)
            
            # Assigning a Num to a Subscript (line 621):
            int_395546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 42), 'int')
            # Getting the type of 'Pr' (line 621)
            Pr_395547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 12), 'Pr')
            
            # Obtaining an instance of the builtin type 'tuple' (line 621)
            tuple_395548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 15), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 621)
            # Adding element type (line 621)
            # Getting the type of 'lu' (line 621)
            lu_395549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 15), 'lu')
            # Obtaining the member 'perm_r' of a type (line 621)
            perm_r_395550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 15), lu_395549, 'perm_r')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 15), tuple_395548, perm_r_395550)
            # Adding element type (line 621)
            
            # Call to arange(...): (line 621)
            # Processing the call arguments (line 621)
            # Getting the type of 'n' (line 621)
            n_395553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 36), 'n', False)
            # Processing the call keyword arguments (line 621)
            kwargs_395554 = {}
            # Getting the type of 'np' (line 621)
            np_395551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 26), 'np', False)
            # Obtaining the member 'arange' of a type (line 621)
            arange_395552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 26), np_395551, 'arange')
            # Calling arange(args, kwargs) (line 621)
            arange_call_result_395555 = invoke(stypy.reporting.localization.Localization(__file__, 621, 26), arange_395552, *[n_395553], **kwargs_395554)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 15), tuple_395548, arange_call_result_395555)
            
            # Storing an element on a container (line 621)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 12), Pr_395547, (tuple_395548, int_395546))
            
            # Assigning a Call to a Name (line 623):
            
            # Call to toarray(...): (line 623)
            # Processing the call keyword arguments (line 623)
            kwargs_395558 = {}
            # Getting the type of 'A' (line 623)
            A_395556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 17), 'A', False)
            # Obtaining the member 'toarray' of a type (line 623)
            toarray_395557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 17), A_395556, 'toarray')
            # Calling toarray(args, kwargs) (line 623)
            toarray_call_result_395559 = invoke(stypy.reporting.localization.Localization(__file__, 623, 17), toarray_395557, *[], **kwargs_395558)
            
            # Assigning a type to the variable 'Ad' (line 623)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'Ad', toarray_call_result_395559)
            
            # Assigning a Call to a Name (line 624):
            
            # Call to dot(...): (line 624)
            # Processing the call arguments (line 624)
            # Getting the type of 'Pc' (line 624)
            Pc_395566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 33), 'Pc', False)
            # Processing the call keyword arguments (line 624)
            kwargs_395567 = {}
            
            # Call to dot(...): (line 624)
            # Processing the call arguments (line 624)
            # Getting the type of 'Ad' (line 624)
            Ad_395562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 25), 'Ad', False)
            # Processing the call keyword arguments (line 624)
            kwargs_395563 = {}
            # Getting the type of 'Pr' (line 624)
            Pr_395560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 18), 'Pr', False)
            # Obtaining the member 'dot' of a type (line 624)
            dot_395561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 18), Pr_395560, 'dot')
            # Calling dot(args, kwargs) (line 624)
            dot_call_result_395564 = invoke(stypy.reporting.localization.Localization(__file__, 624, 18), dot_395561, *[Ad_395562], **kwargs_395563)
            
            # Obtaining the member 'dot' of a type (line 624)
            dot_395565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 18), dot_call_result_395564, 'dot')
            # Calling dot(args, kwargs) (line 624)
            dot_call_result_395568 = invoke(stypy.reporting.localization.Localization(__file__, 624, 18), dot_395565, *[Pc_395566], **kwargs_395567)
            
            # Assigning a type to the variable 'lhs' (line 624)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 12), 'lhs', dot_call_result_395568)
            
            # Assigning a Call to a Name (line 625):
            
            # Call to toarray(...): (line 625)
            # Processing the call keyword arguments (line 625)
            kwargs_395575 = {}
            # Getting the type of 'lu' (line 625)
            lu_395569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 19), 'lu', False)
            # Obtaining the member 'L' of a type (line 625)
            L_395570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 19), lu_395569, 'L')
            # Getting the type of 'lu' (line 625)
            lu_395571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 26), 'lu', False)
            # Obtaining the member 'U' of a type (line 625)
            U_395572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 26), lu_395571, 'U')
            # Applying the binary operator '*' (line 625)
            result_mul_395573 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 19), '*', L_395570, U_395572)
            
            # Obtaining the member 'toarray' of a type (line 625)
            toarray_395574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 19), result_mul_395573, 'toarray')
            # Calling toarray(args, kwargs) (line 625)
            toarray_call_result_395576 = invoke(stypy.reporting.localization.Localization(__file__, 625, 19), toarray_395574, *[], **kwargs_395575)
            
            # Assigning a type to the variable 'rhs' (line 625)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'rhs', toarray_call_result_395576)
            
            # Assigning a Attribute to a Name (line 627):
            
            # Call to finfo(...): (line 627)
            # Processing the call arguments (line 627)
            # Getting the type of 'dtype' (line 627)
            dtype_395579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 27), 'dtype', False)
            # Processing the call keyword arguments (line 627)
            kwargs_395580 = {}
            # Getting the type of 'np' (line 627)
            np_395577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 18), 'np', False)
            # Obtaining the member 'finfo' of a type (line 627)
            finfo_395578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 18), np_395577, 'finfo')
            # Calling finfo(args, kwargs) (line 627)
            finfo_call_result_395581 = invoke(stypy.reporting.localization.Localization(__file__, 627, 18), finfo_395578, *[dtype_395579], **kwargs_395580)
            
            # Obtaining the member 'eps' of a type (line 627)
            eps_395582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 18), finfo_call_result_395581, 'eps')
            # Assigning a type to the variable 'eps' (line 627)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 12), 'eps', eps_395582)
            
            # Call to assert_allclose(...): (line 629)
            # Processing the call arguments (line 629)
            # Getting the type of 'lhs' (line 629)
            lhs_395584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 28), 'lhs', False)
            # Getting the type of 'rhs' (line 629)
            rhs_395585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 33), 'rhs', False)
            # Processing the call keyword arguments (line 629)
            int_395586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 43), 'int')
            # Getting the type of 'eps' (line 629)
            eps_395587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 47), 'eps', False)
            # Applying the binary operator '*' (line 629)
            result_mul_395588 = python_operator(stypy.reporting.localization.Localization(__file__, 629, 43), '*', int_395586, eps_395587)
            
            keyword_395589 = result_mul_395588
            kwargs_395590 = {'atol': keyword_395589}
            # Getting the type of 'assert_allclose' (line 629)
            assert_allclose_395583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 12), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 629)
            assert_allclose_call_result_395591 = invoke(stypy.reporting.localization.Localization(__file__, 629, 12), assert_allclose_395583, *[lhs_395584, rhs_395585], **kwargs_395590)
            
            
            # ################# End of 'check(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'check' in the type store
            # Getting the type of 'stypy_return_type' (line 606)
            stypy_return_type_395592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_395592)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'check'
            return stypy_return_type_395592

        # Assigning a type to the variable 'check' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'check', check)
        
        # Call to check(...): (line 631)
        # Processing the call arguments (line 631)
        # Getting the type of 'np' (line 631)
        np_395594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 14), 'np', False)
        # Obtaining the member 'float32' of a type (line 631)
        float32_395595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 14), np_395594, 'float32')
        # Processing the call keyword arguments (line 631)
        kwargs_395596 = {}
        # Getting the type of 'check' (line 631)
        check_395593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'check', False)
        # Calling check(args, kwargs) (line 631)
        check_call_result_395597 = invoke(stypy.reporting.localization.Localization(__file__, 631, 8), check_395593, *[float32_395595], **kwargs_395596)
        
        
        # Call to check(...): (line 632)
        # Processing the call arguments (line 632)
        # Getting the type of 'np' (line 632)
        np_395599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 14), 'np', False)
        # Obtaining the member 'float64' of a type (line 632)
        float64_395600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 14), np_395599, 'float64')
        # Processing the call keyword arguments (line 632)
        kwargs_395601 = {}
        # Getting the type of 'check' (line 632)
        check_395598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'check', False)
        # Calling check(args, kwargs) (line 632)
        check_call_result_395602 = invoke(stypy.reporting.localization.Localization(__file__, 632, 8), check_395598, *[float64_395600], **kwargs_395601)
        
        
        # Call to check(...): (line 633)
        # Processing the call arguments (line 633)
        # Getting the type of 'np' (line 633)
        np_395604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 14), 'np', False)
        # Obtaining the member 'complex64' of a type (line 633)
        complex64_395605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 14), np_395604, 'complex64')
        # Processing the call keyword arguments (line 633)
        kwargs_395606 = {}
        # Getting the type of 'check' (line 633)
        check_395603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'check', False)
        # Calling check(args, kwargs) (line 633)
        check_call_result_395607 = invoke(stypy.reporting.localization.Localization(__file__, 633, 8), check_395603, *[complex64_395605], **kwargs_395606)
        
        
        # Call to check(...): (line 634)
        # Processing the call arguments (line 634)
        # Getting the type of 'np' (line 634)
        np_395609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 14), 'np', False)
        # Obtaining the member 'complex128' of a type (line 634)
        complex128_395610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 14), np_395609, 'complex128')
        # Processing the call keyword arguments (line 634)
        kwargs_395611 = {}
        # Getting the type of 'check' (line 634)
        check_395608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'check', False)
        # Calling check(args, kwargs) (line 634)
        check_call_result_395612 = invoke(stypy.reporting.localization.Localization(__file__, 634, 8), check_395608, *[complex128_395610], **kwargs_395611)
        
        
        # Call to check(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'np' (line 635)
        np_395614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 14), 'np', False)
        # Obtaining the member 'complex64' of a type (line 635)
        complex64_395615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 14), np_395614, 'complex64')
        # Getting the type of 'True' (line 635)
        True_395616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 28), 'True', False)
        # Processing the call keyword arguments (line 635)
        kwargs_395617 = {}
        # Getting the type of 'check' (line 635)
        check_395613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 8), 'check', False)
        # Calling check(args, kwargs) (line 635)
        check_call_result_395618 = invoke(stypy.reporting.localization.Localization(__file__, 635, 8), check_395613, *[complex64_395615, True_395616], **kwargs_395617)
        
        
        # Call to check(...): (line 636)
        # Processing the call arguments (line 636)
        # Getting the type of 'np' (line 636)
        np_395620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 14), 'np', False)
        # Obtaining the member 'complex128' of a type (line 636)
        complex128_395621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 14), np_395620, 'complex128')
        # Getting the type of 'True' (line 636)
        True_395622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 29), 'True', False)
        # Processing the call keyword arguments (line 636)
        kwargs_395623 = {}
        # Getting the type of 'check' (line 636)
        check_395619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 8), 'check', False)
        # Calling check(args, kwargs) (line 636)
        check_call_result_395624 = invoke(stypy.reporting.localization.Localization(__file__, 636, 8), check_395619, *[complex128_395621, True_395622], **kwargs_395623)
        
        
        # ################# End of 'test_lu_attr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_lu_attr' in the type store
        # Getting the type of 'stypy_return_type' (line 603)
        stypy_return_type_395625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395625)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_lu_attr'
        return stypy_return_type_395625


    @norecursion
    def test_threads_parallel(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_threads_parallel'
        module_type_store = module_type_store.open_function_context('test_threads_parallel', 638, 4, False)
        # Assigning a type to the variable 'self' (line 639)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplu.test_threads_parallel.__dict__.__setitem__('stypy_localization', localization)
        TestSplu.test_threads_parallel.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplu.test_threads_parallel.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplu.test_threads_parallel.__dict__.__setitem__('stypy_function_name', 'TestSplu.test_threads_parallel')
        TestSplu.test_threads_parallel.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplu.test_threads_parallel.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplu.test_threads_parallel.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplu.test_threads_parallel.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplu.test_threads_parallel.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplu.test_threads_parallel.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplu.test_threads_parallel.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.test_threads_parallel', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_threads_parallel', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_threads_parallel(...)' code ##################

        
        # Assigning a List to a Name (line 640):
        
        # Obtaining an instance of the builtin type 'list' (line 640)
        list_395626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 640)
        
        # Assigning a type to the variable 'oks' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'oks', list_395626)

        @norecursion
        def worker(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'worker'
            module_type_store = module_type_store.open_function_context('worker', 642, 8, False)
            
            # Passed parameters checking function
            worker.stypy_localization = localization
            worker.stypy_type_of_self = None
            worker.stypy_type_store = module_type_store
            worker.stypy_function_name = 'worker'
            worker.stypy_param_names_list = []
            worker.stypy_varargs_param_name = None
            worker.stypy_kwargs_param_name = None
            worker.stypy_call_defaults = defaults
            worker.stypy_call_varargs = varargs
            worker.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'worker', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'worker', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'worker(...)' code ##################

            
            
            # SSA begins for try-except statement (line 643)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to test_splu_basic(...): (line 644)
            # Processing the call keyword arguments (line 644)
            kwargs_395629 = {}
            # Getting the type of 'self' (line 644)
            self_395627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 16), 'self', False)
            # Obtaining the member 'test_splu_basic' of a type (line 644)
            test_splu_basic_395628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 16), self_395627, 'test_splu_basic')
            # Calling test_splu_basic(args, kwargs) (line 644)
            test_splu_basic_call_result_395630 = invoke(stypy.reporting.localization.Localization(__file__, 644, 16), test_splu_basic_395628, *[], **kwargs_395629)
            
            
            # Call to _internal_test_splu_smoketest(...): (line 645)
            # Processing the call keyword arguments (line 645)
            kwargs_395633 = {}
            # Getting the type of 'self' (line 645)
            self_395631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 16), 'self', False)
            # Obtaining the member '_internal_test_splu_smoketest' of a type (line 645)
            _internal_test_splu_smoketest_395632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 16), self_395631, '_internal_test_splu_smoketest')
            # Calling _internal_test_splu_smoketest(args, kwargs) (line 645)
            _internal_test_splu_smoketest_call_result_395634 = invoke(stypy.reporting.localization.Localization(__file__, 645, 16), _internal_test_splu_smoketest_395632, *[], **kwargs_395633)
            
            
            # Call to _internal_test_spilu_smoketest(...): (line 646)
            # Processing the call keyword arguments (line 646)
            kwargs_395637 = {}
            # Getting the type of 'self' (line 646)
            self_395635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 16), 'self', False)
            # Obtaining the member '_internal_test_spilu_smoketest' of a type (line 646)
            _internal_test_spilu_smoketest_395636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 16), self_395635, '_internal_test_spilu_smoketest')
            # Calling _internal_test_spilu_smoketest(args, kwargs) (line 646)
            _internal_test_spilu_smoketest_call_result_395638 = invoke(stypy.reporting.localization.Localization(__file__, 646, 16), _internal_test_spilu_smoketest_395636, *[], **kwargs_395637)
            
            
            # Call to append(...): (line 647)
            # Processing the call arguments (line 647)
            # Getting the type of 'True' (line 647)
            True_395641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 27), 'True', False)
            # Processing the call keyword arguments (line 647)
            kwargs_395642 = {}
            # Getting the type of 'oks' (line 647)
            oks_395639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 16), 'oks', False)
            # Obtaining the member 'append' of a type (line 647)
            append_395640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 16), oks_395639, 'append')
            # Calling append(args, kwargs) (line 647)
            append_call_result_395643 = invoke(stypy.reporting.localization.Localization(__file__, 647, 16), append_395640, *[True_395641], **kwargs_395642)
            
            # SSA branch for the except part of a try statement (line 643)
            # SSA branch for the except '<any exception>' branch of a try statement (line 643)
            module_type_store.open_ssa_branch('except')
            pass
            # SSA join for try-except statement (line 643)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'worker(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'worker' in the type store
            # Getting the type of 'stypy_return_type' (line 642)
            stypy_return_type_395644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_395644)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'worker'
            return stypy_return_type_395644

        # Assigning a type to the variable 'worker' (line 642)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'worker', worker)
        
        # Assigning a ListComp to a Name (line 651):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 652)
        # Processing the call arguments (line 652)
        int_395652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 34), 'int')
        # Processing the call keyword arguments (line 652)
        kwargs_395653 = {}
        # Getting the type of 'range' (line 652)
        range_395651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 28), 'range', False)
        # Calling range(args, kwargs) (line 652)
        range_call_result_395654 = invoke(stypy.reporting.localization.Localization(__file__, 652, 28), range_395651, *[int_395652], **kwargs_395653)
        
        comprehension_395655 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 651, 19), range_call_result_395654)
        # Assigning a type to the variable 'k' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 19), 'k', comprehension_395655)
        
        # Call to Thread(...): (line 651)
        # Processing the call keyword arguments (line 651)
        # Getting the type of 'worker' (line 651)
        worker_395647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 43), 'worker', False)
        keyword_395648 = worker_395647
        kwargs_395649 = {'target': keyword_395648}
        # Getting the type of 'threading' (line 651)
        threading_395645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 19), 'threading', False)
        # Obtaining the member 'Thread' of a type (line 651)
        Thread_395646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 19), threading_395645, 'Thread')
        # Calling Thread(args, kwargs) (line 651)
        Thread_call_result_395650 = invoke(stypy.reporting.localization.Localization(__file__, 651, 19), Thread_395646, *[], **kwargs_395649)
        
        list_395656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 651, 19), list_395656, Thread_call_result_395650)
        # Assigning a type to the variable 'threads' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 8), 'threads', list_395656)
        
        # Getting the type of 'threads' (line 653)
        threads_395657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 17), 'threads')
        # Testing the type of a for loop iterable (line 653)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 653, 8), threads_395657)
        # Getting the type of the for loop variable (line 653)
        for_loop_var_395658 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 653, 8), threads_395657)
        # Assigning a type to the variable 't' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 't', for_loop_var_395658)
        # SSA begins for a for statement (line 653)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to start(...): (line 654)
        # Processing the call keyword arguments (line 654)
        kwargs_395661 = {}
        # Getting the type of 't' (line 654)
        t_395659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 't', False)
        # Obtaining the member 'start' of a type (line 654)
        start_395660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 12), t_395659, 'start')
        # Calling start(args, kwargs) (line 654)
        start_call_result_395662 = invoke(stypy.reporting.localization.Localization(__file__, 654, 12), start_395660, *[], **kwargs_395661)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'threads' (line 655)
        threads_395663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 17), 'threads')
        # Testing the type of a for loop iterable (line 655)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 655, 8), threads_395663)
        # Getting the type of the for loop variable (line 655)
        for_loop_var_395664 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 655, 8), threads_395663)
        # Assigning a type to the variable 't' (line 655)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 't', for_loop_var_395664)
        # SSA begins for a for statement (line 655)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to join(...): (line 656)
        # Processing the call keyword arguments (line 656)
        kwargs_395667 = {}
        # Getting the type of 't' (line 656)
        t_395665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 12), 't', False)
        # Obtaining the member 'join' of a type (line 656)
        join_395666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 12), t_395665, 'join')
        # Calling join(args, kwargs) (line 656)
        join_call_result_395668 = invoke(stypy.reporting.localization.Localization(__file__, 656, 12), join_395666, *[], **kwargs_395667)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 658)
        # Processing the call arguments (line 658)
        
        # Call to len(...): (line 658)
        # Processing the call arguments (line 658)
        # Getting the type of 'oks' (line 658)
        oks_395671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 25), 'oks', False)
        # Processing the call keyword arguments (line 658)
        kwargs_395672 = {}
        # Getting the type of 'len' (line 658)
        len_395670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 21), 'len', False)
        # Calling len(args, kwargs) (line 658)
        len_call_result_395673 = invoke(stypy.reporting.localization.Localization(__file__, 658, 21), len_395670, *[oks_395671], **kwargs_395672)
        
        int_395674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 31), 'int')
        # Processing the call keyword arguments (line 658)
        kwargs_395675 = {}
        # Getting the type of 'assert_equal' (line 658)
        assert_equal_395669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 658)
        assert_equal_call_result_395676 = invoke(stypy.reporting.localization.Localization(__file__, 658, 8), assert_equal_395669, *[len_call_result_395673, int_395674], **kwargs_395675)
        
        
        # ################# End of 'test_threads_parallel(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_threads_parallel' in the type store
        # Getting the type of 'stypy_return_type' (line 638)
        stypy_return_type_395677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395677)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_threads_parallel'
        return stypy_return_type_395677


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 405, 0, False)
        # Assigning a type to the variable 'self' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplu.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSplu' (line 405)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 0), 'TestSplu', TestSplu)
# Declaration of the 'TestSpsolveTriangular' class

class TestSpsolveTriangular(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 662, 4, False)
        # Assigning a type to the variable 'self' (line 663)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSpsolveTriangular.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestSpsolveTriangular.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSpsolveTriangular.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSpsolveTriangular.setup_method.__dict__.__setitem__('stypy_function_name', 'TestSpsolveTriangular.setup_method')
        TestSpsolveTriangular.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestSpsolveTriangular.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSpsolveTriangular.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSpsolveTriangular.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSpsolveTriangular.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSpsolveTriangular.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSpsolveTriangular.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSpsolveTriangular.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to use_solver(...): (line 663)
        # Processing the call keyword arguments (line 663)
        # Getting the type of 'False' (line 663)
        False_395679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 30), 'False', False)
        keyword_395680 = False_395679
        kwargs_395681 = {'useUmfpack': keyword_395680}
        # Getting the type of 'use_solver' (line 663)
        use_solver_395678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'use_solver', False)
        # Calling use_solver(args, kwargs) (line 663)
        use_solver_call_result_395682 = invoke(stypy.reporting.localization.Localization(__file__, 663, 8), use_solver_395678, *[], **kwargs_395681)
        
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 662)
        stypy_return_type_395683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395683)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_395683


    @norecursion
    def test_singular(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_singular'
        module_type_store = module_type_store.open_function_context('test_singular', 665, 4, False)
        # Assigning a type to the variable 'self' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSpsolveTriangular.test_singular.__dict__.__setitem__('stypy_localization', localization)
        TestSpsolveTriangular.test_singular.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSpsolveTriangular.test_singular.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSpsolveTriangular.test_singular.__dict__.__setitem__('stypy_function_name', 'TestSpsolveTriangular.test_singular')
        TestSpsolveTriangular.test_singular.__dict__.__setitem__('stypy_param_names_list', [])
        TestSpsolveTriangular.test_singular.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSpsolveTriangular.test_singular.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSpsolveTriangular.test_singular.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSpsolveTriangular.test_singular.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSpsolveTriangular.test_singular.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSpsolveTriangular.test_singular.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSpsolveTriangular.test_singular', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_singular', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_singular(...)' code ##################

        
        # Assigning a Num to a Name (line 666):
        int_395684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 12), 'int')
        # Assigning a type to the variable 'n' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'n', int_395684)
        
        # Assigning a Call to a Name (line 667):
        
        # Call to csr_matrix(...): (line 667)
        # Processing the call arguments (line 667)
        
        # Obtaining an instance of the builtin type 'tuple' (line 667)
        tuple_395686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 667)
        # Adding element type (line 667)
        # Getting the type of 'n' (line 667)
        n_395687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 24), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 24), tuple_395686, n_395687)
        # Adding element type (line 667)
        # Getting the type of 'n' (line 667)
        n_395688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 27), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 24), tuple_395686, n_395688)
        
        # Processing the call keyword arguments (line 667)
        kwargs_395689 = {}
        # Getting the type of 'csr_matrix' (line 667)
        csr_matrix_395685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 667)
        csr_matrix_call_result_395690 = invoke(stypy.reporting.localization.Localization(__file__, 667, 12), csr_matrix_395685, *[tuple_395686], **kwargs_395689)
        
        # Assigning a type to the variable 'A' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'A', csr_matrix_call_result_395690)
        
        # Assigning a Call to a Name (line 668):
        
        # Call to arange(...): (line 668)
        # Processing the call arguments (line 668)
        # Getting the type of 'n' (line 668)
        n_395693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 22), 'n', False)
        # Processing the call keyword arguments (line 668)
        kwargs_395694 = {}
        # Getting the type of 'np' (line 668)
        np_395691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 668)
        arange_395692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 12), np_395691, 'arange')
        # Calling arange(args, kwargs) (line 668)
        arange_call_result_395695 = invoke(stypy.reporting.localization.Localization(__file__, 668, 12), arange_395692, *[n_395693], **kwargs_395694)
        
        # Assigning a type to the variable 'b' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'b', arange_call_result_395695)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 669)
        tuple_395696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 669)
        # Adding element type (line 669)
        # Getting the type of 'True' (line 669)
        True_395697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 22), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 669, 22), tuple_395696, True_395697)
        # Adding element type (line 669)
        # Getting the type of 'False' (line 669)
        False_395698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 28), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 669, 22), tuple_395696, False_395698)
        
        # Testing the type of a for loop iterable (line 669)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 669, 8), tuple_395696)
        # Getting the type of the for loop variable (line 669)
        for_loop_var_395699 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 669, 8), tuple_395696)
        # Assigning a type to the variable 'lower' (line 669)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 8), 'lower', for_loop_var_395699)
        # SSA begins for a for statement (line 669)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_raises(...): (line 670)
        # Processing the call arguments (line 670)
        # Getting the type of 'scipy' (line 670)
        scipy_395701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 26), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 670)
        linalg_395702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 26), scipy_395701, 'linalg')
        # Obtaining the member 'LinAlgError' of a type (line 670)
        LinAlgError_395703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 26), linalg_395702, 'LinAlgError')
        # Getting the type of 'spsolve_triangular' (line 670)
        spsolve_triangular_395704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 52), 'spsolve_triangular', False)
        # Getting the type of 'A' (line 670)
        A_395705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 72), 'A', False)
        # Getting the type of 'b' (line 670)
        b_395706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 75), 'b', False)
        # Processing the call keyword arguments (line 670)
        # Getting the type of 'lower' (line 670)
        lower_395707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 84), 'lower', False)
        keyword_395708 = lower_395707
        kwargs_395709 = {'lower': keyword_395708}
        # Getting the type of 'assert_raises' (line 670)
        assert_raises_395700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 670)
        assert_raises_call_result_395710 = invoke(stypy.reporting.localization.Localization(__file__, 670, 12), assert_raises_395700, *[LinAlgError_395703, spsolve_triangular_395704, A_395705, b_395706], **kwargs_395709)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_singular(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_singular' in the type store
        # Getting the type of 'stypy_return_type' (line 665)
        stypy_return_type_395711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395711)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_singular'
        return stypy_return_type_395711


    @norecursion
    def test_bad_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bad_shape'
        module_type_store = module_type_store.open_function_context('test_bad_shape', 672, 4, False)
        # Assigning a type to the variable 'self' (line 673)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSpsolveTriangular.test_bad_shape.__dict__.__setitem__('stypy_localization', localization)
        TestSpsolveTriangular.test_bad_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSpsolveTriangular.test_bad_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSpsolveTriangular.test_bad_shape.__dict__.__setitem__('stypy_function_name', 'TestSpsolveTriangular.test_bad_shape')
        TestSpsolveTriangular.test_bad_shape.__dict__.__setitem__('stypy_param_names_list', [])
        TestSpsolveTriangular.test_bad_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSpsolveTriangular.test_bad_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSpsolveTriangular.test_bad_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSpsolveTriangular.test_bad_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSpsolveTriangular.test_bad_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSpsolveTriangular.test_bad_shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSpsolveTriangular.test_bad_shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bad_shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bad_shape(...)' code ##################

        
        # Assigning a Call to a Name (line 675):
        
        # Call to zeros(...): (line 675)
        # Processing the call arguments (line 675)
        
        # Obtaining an instance of the builtin type 'tuple' (line 675)
        tuple_395714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 675)
        # Adding element type (line 675)
        int_395715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 675, 22), tuple_395714, int_395715)
        # Adding element type (line 675)
        int_395716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 675, 22), tuple_395714, int_395716)
        
        # Processing the call keyword arguments (line 675)
        kwargs_395717 = {}
        # Getting the type of 'np' (line 675)
        np_395712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 675)
        zeros_395713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 12), np_395712, 'zeros')
        # Calling zeros(args, kwargs) (line 675)
        zeros_call_result_395718 = invoke(stypy.reporting.localization.Localization(__file__, 675, 12), zeros_395713, *[tuple_395714], **kwargs_395717)
        
        # Assigning a type to the variable 'A' (line 675)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 8), 'A', zeros_call_result_395718)
        
        # Assigning a Call to a Name (line 676):
        
        # Call to ones(...): (line 676)
        # Processing the call arguments (line 676)
        
        # Obtaining an instance of the builtin type 'tuple' (line 676)
        tuple_395720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 676)
        # Adding element type (line 676)
        int_395721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 18), tuple_395720, int_395721)
        # Adding element type (line 676)
        int_395722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 18), tuple_395720, int_395722)
        
        # Processing the call keyword arguments (line 676)
        kwargs_395723 = {}
        # Getting the type of 'ones' (line 676)
        ones_395719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 676)
        ones_call_result_395724 = invoke(stypy.reporting.localization.Localization(__file__, 676, 12), ones_395719, *[tuple_395720], **kwargs_395723)
        
        # Assigning a type to the variable 'b' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'b', ones_call_result_395724)
        
        # Call to assert_raises(...): (line 677)
        # Processing the call arguments (line 677)
        # Getting the type of 'ValueError' (line 677)
        ValueError_395726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 22), 'ValueError', False)
        # Getting the type of 'spsolve_triangular' (line 677)
        spsolve_triangular_395727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 34), 'spsolve_triangular', False)
        # Getting the type of 'A' (line 677)
        A_395728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 54), 'A', False)
        # Getting the type of 'b' (line 677)
        b_395729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 57), 'b', False)
        # Processing the call keyword arguments (line 677)
        kwargs_395730 = {}
        # Getting the type of 'assert_raises' (line 677)
        assert_raises_395725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 677)
        assert_raises_call_result_395731 = invoke(stypy.reporting.localization.Localization(__file__, 677, 8), assert_raises_395725, *[ValueError_395726, spsolve_triangular_395727, A_395728, b_395729], **kwargs_395730)
        
        
        # Assigning a Call to a Name (line 679):
        
        # Call to csr_matrix(...): (line 679)
        # Processing the call arguments (line 679)
        
        # Call to eye(...): (line 679)
        # Processing the call arguments (line 679)
        int_395734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 28), 'int')
        # Processing the call keyword arguments (line 679)
        kwargs_395735 = {}
        # Getting the type of 'eye' (line 679)
        eye_395733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 24), 'eye', False)
        # Calling eye(args, kwargs) (line 679)
        eye_call_result_395736 = invoke(stypy.reporting.localization.Localization(__file__, 679, 24), eye_395733, *[int_395734], **kwargs_395735)
        
        # Processing the call keyword arguments (line 679)
        kwargs_395737 = {}
        # Getting the type of 'csr_matrix' (line 679)
        csr_matrix_395732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 13), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 679)
        csr_matrix_call_result_395738 = invoke(stypy.reporting.localization.Localization(__file__, 679, 13), csr_matrix_395732, *[eye_call_result_395736], **kwargs_395737)
        
        # Assigning a type to the variable 'A2' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 8), 'A2', csr_matrix_call_result_395738)
        
        # Assigning a Call to a Name (line 680):
        
        # Call to array(...): (line 680)
        # Processing the call arguments (line 680)
        
        # Obtaining an instance of the builtin type 'list' (line 680)
        list_395740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 680)
        # Adding element type (line 680)
        float_395741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 19), list_395740, float_395741)
        # Adding element type (line 680)
        float_395742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 19), list_395740, float_395742)
        
        # Processing the call keyword arguments (line 680)
        kwargs_395743 = {}
        # Getting the type of 'array' (line 680)
        array_395739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 13), 'array', False)
        # Calling array(args, kwargs) (line 680)
        array_call_result_395744 = invoke(stypy.reporting.localization.Localization(__file__, 680, 13), array_395739, *[list_395740], **kwargs_395743)
        
        # Assigning a type to the variable 'b2' (line 680)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 8), 'b2', array_call_result_395744)
        
        # Call to assert_raises(...): (line 681)
        # Processing the call arguments (line 681)
        # Getting the type of 'ValueError' (line 681)
        ValueError_395746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 22), 'ValueError', False)
        # Getting the type of 'spsolve_triangular' (line 681)
        spsolve_triangular_395747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 34), 'spsolve_triangular', False)
        # Getting the type of 'A2' (line 681)
        A2_395748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 54), 'A2', False)
        # Getting the type of 'b2' (line 681)
        b2_395749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 58), 'b2', False)
        # Processing the call keyword arguments (line 681)
        kwargs_395750 = {}
        # Getting the type of 'assert_raises' (line 681)
        assert_raises_395745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 681)
        assert_raises_call_result_395751 = invoke(stypy.reporting.localization.Localization(__file__, 681, 8), assert_raises_395745, *[ValueError_395746, spsolve_triangular_395747, A2_395748, b2_395749], **kwargs_395750)
        
        
        # ################# End of 'test_bad_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bad_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 672)
        stypy_return_type_395752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395752)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bad_shape'
        return stypy_return_type_395752


    @norecursion
    def test_input_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_input_types'
        module_type_store = module_type_store.open_function_context('test_input_types', 683, 4, False)
        # Assigning a type to the variable 'self' (line 684)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSpsolveTriangular.test_input_types.__dict__.__setitem__('stypy_localization', localization)
        TestSpsolveTriangular.test_input_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSpsolveTriangular.test_input_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSpsolveTriangular.test_input_types.__dict__.__setitem__('stypy_function_name', 'TestSpsolveTriangular.test_input_types')
        TestSpsolveTriangular.test_input_types.__dict__.__setitem__('stypy_param_names_list', [])
        TestSpsolveTriangular.test_input_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSpsolveTriangular.test_input_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSpsolveTriangular.test_input_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSpsolveTriangular.test_input_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSpsolveTriangular.test_input_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSpsolveTriangular.test_input_types.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSpsolveTriangular.test_input_types', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_input_types', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_input_types(...)' code ##################

        
        # Assigning a Call to a Name (line 685):
        
        # Call to array(...): (line 685)
        # Processing the call arguments (line 685)
        
        # Obtaining an instance of the builtin type 'list' (line 685)
        list_395754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 685)
        # Adding element type (line 685)
        
        # Obtaining an instance of the builtin type 'list' (line 685)
        list_395755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 685)
        # Adding element type (line 685)
        float_395756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 685, 19), list_395755, float_395756)
        # Adding element type (line 685)
        float_395757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 685, 19), list_395755, float_395757)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 685, 18), list_395754, list_395755)
        # Adding element type (line 685)
        
        # Obtaining an instance of the builtin type 'list' (line 685)
        list_395758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 685)
        # Adding element type (line 685)
        float_395759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 685, 29), list_395758, float_395759)
        # Adding element type (line 685)
        float_395760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 685, 29), list_395758, float_395760)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 685, 18), list_395754, list_395758)
        
        # Processing the call keyword arguments (line 685)
        kwargs_395761 = {}
        # Getting the type of 'array' (line 685)
        array_395753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 12), 'array', False)
        # Calling array(args, kwargs) (line 685)
        array_call_result_395762 = invoke(stypy.reporting.localization.Localization(__file__, 685, 12), array_395753, *[list_395754], **kwargs_395761)
        
        # Assigning a type to the variable 'A' (line 685)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 8), 'A', array_call_result_395762)
        
        # Assigning a Call to a Name (line 686):
        
        # Call to array(...): (line 686)
        # Processing the call arguments (line 686)
        
        # Obtaining an instance of the builtin type 'list' (line 686)
        list_395764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 686)
        # Adding element type (line 686)
        
        # Obtaining an instance of the builtin type 'list' (line 686)
        list_395765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 686)
        # Adding element type (line 686)
        float_395766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 19), list_395765, float_395766)
        # Adding element type (line 686)
        float_395767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 19), list_395765, float_395767)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 18), list_395764, list_395765)
        # Adding element type (line 686)
        
        # Obtaining an instance of the builtin type 'list' (line 686)
        list_395768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 686)
        # Adding element type (line 686)
        float_395769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 29), list_395768, float_395769)
        # Adding element type (line 686)
        float_395770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 29), list_395768, float_395770)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 18), list_395764, list_395768)
        
        # Processing the call keyword arguments (line 686)
        kwargs_395771 = {}
        # Getting the type of 'array' (line 686)
        array_395763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 12), 'array', False)
        # Calling array(args, kwargs) (line 686)
        array_call_result_395772 = invoke(stypy.reporting.localization.Localization(__file__, 686, 12), array_395763, *[list_395764], **kwargs_395771)
        
        # Assigning a type to the variable 'b' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'b', array_call_result_395772)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 687)
        tuple_395773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 687)
        # Adding element type (line 687)
        # Getting the type of 'array' (line 687)
        array_395774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 28), 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 28), tuple_395773, array_395774)
        # Adding element type (line 687)
        # Getting the type of 'csc_matrix' (line 687)
        csc_matrix_395775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 35), 'csc_matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 28), tuple_395773, csc_matrix_395775)
        # Adding element type (line 687)
        # Getting the type of 'csr_matrix' (line 687)
        csr_matrix_395776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 47), 'csr_matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 28), tuple_395773, csr_matrix_395776)
        
        # Testing the type of a for loop iterable (line 687)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 687, 8), tuple_395773)
        # Getting the type of the for loop variable (line 687)
        for_loop_var_395777 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 687, 8), tuple_395773)
        # Assigning a type to the variable 'matrix_type' (line 687)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 8), 'matrix_type', for_loop_var_395777)
        # SSA begins for a for statement (line 687)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 688):
        
        # Call to spsolve_triangular(...): (line 688)
        # Processing the call arguments (line 688)
        
        # Call to matrix_type(...): (line 688)
        # Processing the call arguments (line 688)
        # Getting the type of 'A' (line 688)
        A_395780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 47), 'A', False)
        # Processing the call keyword arguments (line 688)
        kwargs_395781 = {}
        # Getting the type of 'matrix_type' (line 688)
        matrix_type_395779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 35), 'matrix_type', False)
        # Calling matrix_type(args, kwargs) (line 688)
        matrix_type_call_result_395782 = invoke(stypy.reporting.localization.Localization(__file__, 688, 35), matrix_type_395779, *[A_395780], **kwargs_395781)
        
        # Getting the type of 'b' (line 688)
        b_395783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 51), 'b', False)
        # Processing the call keyword arguments (line 688)
        # Getting the type of 'True' (line 688)
        True_395784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 60), 'True', False)
        keyword_395785 = True_395784
        kwargs_395786 = {'lower': keyword_395785}
        # Getting the type of 'spsolve_triangular' (line 688)
        spsolve_triangular_395778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 16), 'spsolve_triangular', False)
        # Calling spsolve_triangular(args, kwargs) (line 688)
        spsolve_triangular_call_result_395787 = invoke(stypy.reporting.localization.Localization(__file__, 688, 16), spsolve_triangular_395778, *[matrix_type_call_result_395782, b_395783], **kwargs_395786)
        
        # Assigning a type to the variable 'x' (line 688)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 12), 'x', spsolve_triangular_call_result_395787)
        
        # Call to assert_array_almost_equal(...): (line 689)
        # Processing the call arguments (line 689)
        
        # Call to dot(...): (line 689)
        # Processing the call arguments (line 689)
        # Getting the type of 'x' (line 689)
        x_395791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 44), 'x', False)
        # Processing the call keyword arguments (line 689)
        kwargs_395792 = {}
        # Getting the type of 'A' (line 689)
        A_395789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 38), 'A', False)
        # Obtaining the member 'dot' of a type (line 689)
        dot_395790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 38), A_395789, 'dot')
        # Calling dot(args, kwargs) (line 689)
        dot_call_result_395793 = invoke(stypy.reporting.localization.Localization(__file__, 689, 38), dot_395790, *[x_395791], **kwargs_395792)
        
        # Getting the type of 'b' (line 689)
        b_395794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 48), 'b', False)
        # Processing the call keyword arguments (line 689)
        kwargs_395795 = {}
        # Getting the type of 'assert_array_almost_equal' (line 689)
        assert_array_almost_equal_395788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 689)
        assert_array_almost_equal_call_result_395796 = invoke(stypy.reporting.localization.Localization(__file__, 689, 12), assert_array_almost_equal_395788, *[dot_call_result_395793, b_395794], **kwargs_395795)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_input_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_input_types' in the type store
        # Getting the type of 'stypy_return_type' (line 683)
        stypy_return_type_395797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395797)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_input_types'
        return stypy_return_type_395797


    @norecursion
    def test_random(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random'
        module_type_store = module_type_store.open_function_context('test_random', 691, 4, False)
        # Assigning a type to the variable 'self' (line 692)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSpsolveTriangular.test_random.__dict__.__setitem__('stypy_localization', localization)
        TestSpsolveTriangular.test_random.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSpsolveTriangular.test_random.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSpsolveTriangular.test_random.__dict__.__setitem__('stypy_function_name', 'TestSpsolveTriangular.test_random')
        TestSpsolveTriangular.test_random.__dict__.__setitem__('stypy_param_names_list', [])
        TestSpsolveTriangular.test_random.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSpsolveTriangular.test_random.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSpsolveTriangular.test_random.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSpsolveTriangular.test_random.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSpsolveTriangular.test_random.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSpsolveTriangular.test_random.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSpsolveTriangular.test_random', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random(...)' code ##################


        @norecursion
        def random_triangle_matrix(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'True' (line 693)
            True_395798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 44), 'True')
            defaults = [True_395798]
            # Create a new context for function 'random_triangle_matrix'
            module_type_store = module_type_store.open_function_context('random_triangle_matrix', 693, 8, False)
            
            # Passed parameters checking function
            random_triangle_matrix.stypy_localization = localization
            random_triangle_matrix.stypy_type_of_self = None
            random_triangle_matrix.stypy_type_store = module_type_store
            random_triangle_matrix.stypy_function_name = 'random_triangle_matrix'
            random_triangle_matrix.stypy_param_names_list = ['n', 'lower']
            random_triangle_matrix.stypy_varargs_param_name = None
            random_triangle_matrix.stypy_kwargs_param_name = None
            random_triangle_matrix.stypy_call_defaults = defaults
            random_triangle_matrix.stypy_call_varargs = varargs
            random_triangle_matrix.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'random_triangle_matrix', ['n', 'lower'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'random_triangle_matrix', localization, ['n', 'lower'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'random_triangle_matrix(...)' code ##################

            
            # Assigning a Call to a Name (line 694):
            
            # Call to random(...): (line 694)
            # Processing the call arguments (line 694)
            # Getting the type of 'n' (line 694)
            n_395802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 36), 'n', False)
            # Getting the type of 'n' (line 694)
            n_395803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 39), 'n', False)
            # Processing the call keyword arguments (line 694)
            float_395804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 50), 'float')
            keyword_395805 = float_395804
            str_395806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 62), 'str', 'coo')
            keyword_395807 = str_395806
            kwargs_395808 = {'format': keyword_395807, 'density': keyword_395805}
            # Getting the type of 'scipy' (line 694)
            scipy_395799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), 'scipy', False)
            # Obtaining the member 'sparse' of a type (line 694)
            sparse_395800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 16), scipy_395799, 'sparse')
            # Obtaining the member 'random' of a type (line 694)
            random_395801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 16), sparse_395800, 'random')
            # Calling random(args, kwargs) (line 694)
            random_call_result_395809 = invoke(stypy.reporting.localization.Localization(__file__, 694, 16), random_395801, *[n_395802, n_395803], **kwargs_395808)
            
            # Assigning a type to the variable 'A' (line 694)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 12), 'A', random_call_result_395809)
            
            # Getting the type of 'lower' (line 695)
            lower_395810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 15), 'lower')
            # Testing the type of an if condition (line 695)
            if_condition_395811 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 695, 12), lower_395810)
            # Assigning a type to the variable 'if_condition_395811' (line 695)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 12), 'if_condition_395811', if_condition_395811)
            # SSA begins for if statement (line 695)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 696):
            
            # Call to tril(...): (line 696)
            # Processing the call arguments (line 696)
            # Getting the type of 'A' (line 696)
            A_395815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 38), 'A', False)
            # Processing the call keyword arguments (line 696)
            kwargs_395816 = {}
            # Getting the type of 'scipy' (line 696)
            scipy_395812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 20), 'scipy', False)
            # Obtaining the member 'sparse' of a type (line 696)
            sparse_395813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 20), scipy_395812, 'sparse')
            # Obtaining the member 'tril' of a type (line 696)
            tril_395814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 20), sparse_395813, 'tril')
            # Calling tril(args, kwargs) (line 696)
            tril_call_result_395817 = invoke(stypy.reporting.localization.Localization(__file__, 696, 20), tril_395814, *[A_395815], **kwargs_395816)
            
            # Assigning a type to the variable 'A' (line 696)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 16), 'A', tril_call_result_395817)
            # SSA branch for the else part of an if statement (line 695)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 698):
            
            # Call to triu(...): (line 698)
            # Processing the call arguments (line 698)
            # Getting the type of 'A' (line 698)
            A_395821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 38), 'A', False)
            # Processing the call keyword arguments (line 698)
            kwargs_395822 = {}
            # Getting the type of 'scipy' (line 698)
            scipy_395818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 20), 'scipy', False)
            # Obtaining the member 'sparse' of a type (line 698)
            sparse_395819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 20), scipy_395818, 'sparse')
            # Obtaining the member 'triu' of a type (line 698)
            triu_395820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 20), sparse_395819, 'triu')
            # Calling triu(args, kwargs) (line 698)
            triu_call_result_395823 = invoke(stypy.reporting.localization.Localization(__file__, 698, 20), triu_395820, *[A_395821], **kwargs_395822)
            
            # Assigning a type to the variable 'A' (line 698)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 16), 'A', triu_call_result_395823)
            # SSA join for if statement (line 695)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 699):
            
            # Call to tocsr(...): (line 699)
            # Processing the call keyword arguments (line 699)
            # Getting the type of 'False' (line 699)
            False_395826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 29), 'False', False)
            keyword_395827 = False_395826
            kwargs_395828 = {'copy': keyword_395827}
            # Getting the type of 'A' (line 699)
            A_395824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 16), 'A', False)
            # Obtaining the member 'tocsr' of a type (line 699)
            tocsr_395825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 16), A_395824, 'tocsr')
            # Calling tocsr(args, kwargs) (line 699)
            tocsr_call_result_395829 = invoke(stypy.reporting.localization.Localization(__file__, 699, 16), tocsr_395825, *[], **kwargs_395828)
            
            # Assigning a type to the variable 'A' (line 699)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'A', tocsr_call_result_395829)
            
            
            # Call to range(...): (line 700)
            # Processing the call arguments (line 700)
            # Getting the type of 'n' (line 700)
            n_395831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 27), 'n', False)
            # Processing the call keyword arguments (line 700)
            kwargs_395832 = {}
            # Getting the type of 'range' (line 700)
            range_395830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 21), 'range', False)
            # Calling range(args, kwargs) (line 700)
            range_call_result_395833 = invoke(stypy.reporting.localization.Localization(__file__, 700, 21), range_395830, *[n_395831], **kwargs_395832)
            
            # Testing the type of a for loop iterable (line 700)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 700, 12), range_call_result_395833)
            # Getting the type of the for loop variable (line 700)
            for_loop_var_395834 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 700, 12), range_call_result_395833)
            # Assigning a type to the variable 'i' (line 700)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 12), 'i', for_loop_var_395834)
            # SSA begins for a for statement (line 700)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Subscript (line 701):
            
            # Call to rand(...): (line 701)
            # Processing the call keyword arguments (line 701)
            kwargs_395838 = {}
            # Getting the type of 'np' (line 701)
            np_395835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 26), 'np', False)
            # Obtaining the member 'random' of a type (line 701)
            random_395836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 26), np_395835, 'random')
            # Obtaining the member 'rand' of a type (line 701)
            rand_395837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 26), random_395836, 'rand')
            # Calling rand(args, kwargs) (line 701)
            rand_call_result_395839 = invoke(stypy.reporting.localization.Localization(__file__, 701, 26), rand_395837, *[], **kwargs_395838)
            
            int_395840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 45), 'int')
            # Applying the binary operator '+' (line 701)
            result_add_395841 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 26), '+', rand_call_result_395839, int_395840)
            
            # Getting the type of 'A' (line 701)
            A_395842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 16), 'A')
            
            # Obtaining an instance of the builtin type 'tuple' (line 701)
            tuple_395843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 18), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 701)
            # Adding element type (line 701)
            # Getting the type of 'i' (line 701)
            i_395844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 18), 'i')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 701, 18), tuple_395843, i_395844)
            # Adding element type (line 701)
            # Getting the type of 'i' (line 701)
            i_395845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 21), 'i')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 701, 18), tuple_395843, i_395845)
            
            # Storing an element on a container (line 701)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 701, 16), A_395842, (tuple_395843, result_add_395841))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'A' (line 702)
            A_395846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 19), 'A')
            # Assigning a type to the variable 'stypy_return_type' (line 702)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 12), 'stypy_return_type', A_395846)
            
            # ################# End of 'random_triangle_matrix(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'random_triangle_matrix' in the type store
            # Getting the type of 'stypy_return_type' (line 693)
            stypy_return_type_395847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_395847)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'random_triangle_matrix'
            return stypy_return_type_395847

        # Assigning a type to the variable 'random_triangle_matrix' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'random_triangle_matrix', random_triangle_matrix)
        
        # Call to seed(...): (line 704)
        # Processing the call arguments (line 704)
        int_395851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 23), 'int')
        # Processing the call keyword arguments (line 704)
        kwargs_395852 = {}
        # Getting the type of 'np' (line 704)
        np_395848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 704)
        random_395849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 8), np_395848, 'random')
        # Obtaining the member 'seed' of a type (line 704)
        seed_395850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 8), random_395849, 'seed')
        # Calling seed(args, kwargs) (line 704)
        seed_call_result_395853 = invoke(stypy.reporting.localization.Localization(__file__, 704, 8), seed_395850, *[int_395851], **kwargs_395852)
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 705)
        tuple_395854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 705)
        # Adding element type (line 705)
        # Getting the type of 'True' (line 705)
        True_395855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 22), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 22), tuple_395854, True_395855)
        # Adding element type (line 705)
        # Getting the type of 'False' (line 705)
        False_395856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 28), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 22), tuple_395854, False_395856)
        
        # Testing the type of a for loop iterable (line 705)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 705, 8), tuple_395854)
        # Getting the type of the for loop variable (line 705)
        for_loop_var_395857 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 705, 8), tuple_395854)
        # Assigning a type to the variable 'lower' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'lower', for_loop_var_395857)
        # SSA begins for a for statement (line 705)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 706)
        tuple_395858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 706)
        # Adding element type (line 706)
        int_395859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 22), tuple_395858, int_395859)
        # Adding element type (line 706)
        int_395860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 26), 'int')
        int_395861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 30), 'int')
        # Applying the binary operator '**' (line 706)
        result_pow_395862 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 26), '**', int_395860, int_395861)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 22), tuple_395858, result_pow_395862)
        # Adding element type (line 706)
        int_395863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 33), 'int')
        int_395864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 37), 'int')
        # Applying the binary operator '**' (line 706)
        result_pow_395865 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 33), '**', int_395863, int_395864)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 22), tuple_395858, result_pow_395865)
        
        # Testing the type of a for loop iterable (line 706)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 706, 12), tuple_395858)
        # Getting the type of the for loop variable (line 706)
        for_loop_var_395866 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 706, 12), tuple_395858)
        # Assigning a type to the variable 'n' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 12), 'n', for_loop_var_395866)
        # SSA begins for a for statement (line 706)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 707):
        
        # Call to random_triangle_matrix(...): (line 707)
        # Processing the call arguments (line 707)
        # Getting the type of 'n' (line 707)
        n_395868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 43), 'n', False)
        # Processing the call keyword arguments (line 707)
        # Getting the type of 'lower' (line 707)
        lower_395869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 52), 'lower', False)
        keyword_395870 = lower_395869
        kwargs_395871 = {'lower': keyword_395870}
        # Getting the type of 'random_triangle_matrix' (line 707)
        random_triangle_matrix_395867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 20), 'random_triangle_matrix', False)
        # Calling random_triangle_matrix(args, kwargs) (line 707)
        random_triangle_matrix_call_result_395872 = invoke(stypy.reporting.localization.Localization(__file__, 707, 20), random_triangle_matrix_395867, *[n_395868], **kwargs_395871)
        
        # Assigning a type to the variable 'A' (line 707)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 16), 'A', random_triangle_matrix_call_result_395872)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 708)
        tuple_395873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 708)
        # Adding element type (line 708)
        int_395874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 26), tuple_395873, int_395874)
        # Adding element type (line 708)
        int_395875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 26), tuple_395873, int_395875)
        
        # Testing the type of a for loop iterable (line 708)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 708, 16), tuple_395873)
        # Getting the type of the for loop variable (line 708)
        for_loop_var_395876 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 708, 16), tuple_395873)
        # Assigning a type to the variable 'm' (line 708)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 16), 'm', for_loop_var_395876)
        # SSA begins for a for statement (line 708)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 709)
        tuple_395877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 709)
        # Adding element type (line 709)
        
        # Call to rand(...): (line 709)
        # Processing the call arguments (line 709)
        # Getting the type of 'n' (line 709)
        n_395881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 45), 'n', False)
        # Getting the type of 'm' (line 709)
        m_395882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 48), 'm', False)
        # Processing the call keyword arguments (line 709)
        kwargs_395883 = {}
        # Getting the type of 'np' (line 709)
        np_395878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 30), 'np', False)
        # Obtaining the member 'random' of a type (line 709)
        random_395879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 30), np_395878, 'random')
        # Obtaining the member 'rand' of a type (line 709)
        rand_395880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 30), random_395879, 'rand')
        # Calling rand(args, kwargs) (line 709)
        rand_call_result_395884 = invoke(stypy.reporting.localization.Localization(__file__, 709, 30), rand_395880, *[n_395881, m_395882], **kwargs_395883)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 30), tuple_395877, rand_call_result_395884)
        # Adding element type (line 709)
        
        # Call to randint(...): (line 710)
        # Processing the call arguments (line 710)
        int_395888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 48), 'int')
        int_395889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 52), 'int')
        
        # Obtaining an instance of the builtin type 'tuple' (line 710)
        tuple_395890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 710)
        # Adding element type (line 710)
        # Getting the type of 'n' (line 710)
        n_395891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 56), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 56), tuple_395890, n_395891)
        # Adding element type (line 710)
        # Getting the type of 'm' (line 710)
        m_395892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 59), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 56), tuple_395890, m_395892)
        
        # Processing the call keyword arguments (line 710)
        kwargs_395893 = {}
        # Getting the type of 'np' (line 710)
        np_395885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 30), 'np', False)
        # Obtaining the member 'random' of a type (line 710)
        random_395886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 30), np_395885, 'random')
        # Obtaining the member 'randint' of a type (line 710)
        randint_395887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 30), random_395886, 'randint')
        # Calling randint(args, kwargs) (line 710)
        randint_call_result_395894 = invoke(stypy.reporting.localization.Localization(__file__, 710, 30), randint_395887, *[int_395888, int_395889, tuple_395890], **kwargs_395893)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 30), tuple_395877, randint_call_result_395894)
        # Adding element type (line 709)
        
        # Call to randint(...): (line 711)
        # Processing the call arguments (line 711)
        int_395898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 48), 'int')
        int_395899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 52), 'int')
        
        # Obtaining an instance of the builtin type 'tuple' (line 711)
        tuple_395900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 711)
        # Adding element type (line 711)
        # Getting the type of 'n' (line 711)
        n_395901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 56), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 56), tuple_395900, n_395901)
        # Adding element type (line 711)
        # Getting the type of 'm' (line 711)
        m_395902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 59), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 56), tuple_395900, m_395902)
        
        # Processing the call keyword arguments (line 711)
        kwargs_395903 = {}
        # Getting the type of 'np' (line 711)
        np_395895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 30), 'np', False)
        # Obtaining the member 'random' of a type (line 711)
        random_395896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 30), np_395895, 'random')
        # Obtaining the member 'randint' of a type (line 711)
        randint_395897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 30), random_395896, 'randint')
        # Calling randint(args, kwargs) (line 711)
        randint_call_result_395904 = invoke(stypy.reporting.localization.Localization(__file__, 711, 30), randint_395897, *[int_395898, int_395899, tuple_395900], **kwargs_395903)
        
        
        # Call to randint(...): (line 712)
        # Processing the call arguments (line 712)
        int_395908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 48), 'int')
        int_395909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 52), 'int')
        
        # Obtaining an instance of the builtin type 'tuple' (line 712)
        tuple_395910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 712)
        # Adding element type (line 712)
        # Getting the type of 'n' (line 712)
        n_395911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 56), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 712, 56), tuple_395910, n_395911)
        # Adding element type (line 712)
        # Getting the type of 'm' (line 712)
        m_395912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 59), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 712, 56), tuple_395910, m_395912)
        
        # Processing the call keyword arguments (line 712)
        kwargs_395913 = {}
        # Getting the type of 'np' (line 712)
        np_395905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 30), 'np', False)
        # Obtaining the member 'random' of a type (line 712)
        random_395906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 30), np_395905, 'random')
        # Obtaining the member 'randint' of a type (line 712)
        randint_395907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 30), random_395906, 'randint')
        # Calling randint(args, kwargs) (line 712)
        randint_call_result_395914 = invoke(stypy.reporting.localization.Localization(__file__, 712, 30), randint_395907, *[int_395908, int_395909, tuple_395910], **kwargs_395913)
        
        complex_395915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 65), 'complex')
        # Applying the binary operator '*' (line 712)
        result_mul_395916 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 30), '*', randint_call_result_395914, complex_395915)
        
        # Applying the binary operator '+' (line 711)
        result_add_395917 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 30), '+', randint_call_result_395904, result_mul_395916)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 30), tuple_395877, result_add_395917)
        
        # Testing the type of a for loop iterable (line 709)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 709, 20), tuple_395877)
        # Getting the type of the for loop variable (line 709)
        for_loop_var_395918 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 709, 20), tuple_395877)
        # Assigning a type to the variable 'b' (line 709)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 20), 'b', for_loop_var_395918)
        # SSA begins for a for statement (line 709)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 713):
        
        # Call to spsolve_triangular(...): (line 713)
        # Processing the call arguments (line 713)
        # Getting the type of 'A' (line 713)
        A_395920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 47), 'A', False)
        # Getting the type of 'b' (line 713)
        b_395921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 50), 'b', False)
        # Processing the call keyword arguments (line 713)
        # Getting the type of 'lower' (line 713)
        lower_395922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 59), 'lower', False)
        keyword_395923 = lower_395922
        kwargs_395924 = {'lower': keyword_395923}
        # Getting the type of 'spsolve_triangular' (line 713)
        spsolve_triangular_395919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 28), 'spsolve_triangular', False)
        # Calling spsolve_triangular(args, kwargs) (line 713)
        spsolve_triangular_call_result_395925 = invoke(stypy.reporting.localization.Localization(__file__, 713, 28), spsolve_triangular_395919, *[A_395920, b_395921], **kwargs_395924)
        
        # Assigning a type to the variable 'x' (line 713)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 24), 'x', spsolve_triangular_call_result_395925)
        
        # Call to assert_array_almost_equal(...): (line 714)
        # Processing the call arguments (line 714)
        
        # Call to dot(...): (line 714)
        # Processing the call arguments (line 714)
        # Getting the type of 'x' (line 714)
        x_395929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 56), 'x', False)
        # Processing the call keyword arguments (line 714)
        kwargs_395930 = {}
        # Getting the type of 'A' (line 714)
        A_395927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 50), 'A', False)
        # Obtaining the member 'dot' of a type (line 714)
        dot_395928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 50), A_395927, 'dot')
        # Calling dot(args, kwargs) (line 714)
        dot_call_result_395931 = invoke(stypy.reporting.localization.Localization(__file__, 714, 50), dot_395928, *[x_395929], **kwargs_395930)
        
        # Getting the type of 'b' (line 714)
        b_395932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 60), 'b', False)
        # Processing the call keyword arguments (line 714)
        kwargs_395933 = {}
        # Getting the type of 'assert_array_almost_equal' (line 714)
        assert_array_almost_equal_395926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 24), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 714)
        assert_array_almost_equal_call_result_395934 = invoke(stypy.reporting.localization.Localization(__file__, 714, 24), assert_array_almost_equal_395926, *[dot_call_result_395931, b_395932], **kwargs_395933)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_random(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random' in the type store
        # Getting the type of 'stypy_return_type' (line 691)
        stypy_return_type_395935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_395935)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random'
        return stypy_return_type_395935


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 661, 0, False)
        # Assigning a type to the variable 'self' (line 662)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSpsolveTriangular.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSpsolveTriangular' (line 661)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 0), 'TestSpsolveTriangular', TestSpsolveTriangular)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
