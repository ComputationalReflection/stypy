
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Sparse matrix functions
3: '''
4: 
5: #
6: # Authors: Travis Oliphant, March 2002
7: #          Anthony Scopatz, August 2012 (Sparse Updates)
8: #          Jake Vanderplas, August 2012 (Sparse Updates)
9: #
10: 
11: from __future__ import division, print_function, absolute_import
12: 
13: __all__ = ['expm', 'inv']
14: 
15: import math
16: 
17: import numpy as np
18: 
19: import scipy.special
20: from scipy.linalg.basic import solve, solve_triangular
21: 
22: from scipy.sparse.base import isspmatrix
23: from scipy.sparse.construct import eye as speye
24: from scipy.sparse.linalg import spsolve
25: 
26: import scipy.sparse
27: import scipy.sparse.linalg
28: from scipy.sparse.linalg.interface import LinearOperator
29: 
30: 
31: UPPER_TRIANGULAR = 'upper_triangular'
32: 
33: 
34: def inv(A):
35:     '''
36:     Compute the inverse of a sparse matrix
37: 
38:     Parameters
39:     ----------
40:     A : (M,M) ndarray or sparse matrix
41:         square matrix to be inverted
42: 
43:     Returns
44:     -------
45:     Ainv : (M,M) ndarray or sparse matrix
46:         inverse of `A`
47: 
48:     Notes
49:     -----
50:     This computes the sparse inverse of `A`.  If the inverse of `A` is expected
51:     to be non-sparse, it will likely be faster to convert `A` to dense and use
52:     scipy.linalg.inv.
53: 
54:     Examples
55:     --------
56:     >>> from scipy.sparse import csc_matrix
57:     >>> from scipy.sparse.linalg import inv
58:     >>> A = csc_matrix([[1., 0.], [1., 2.]])
59:     >>> Ainv = inv(A)
60:     >>> Ainv
61:     <2x2 sparse matrix of type '<class 'numpy.float64'>'
62:         with 3 stored elements in Compressed Sparse Column format>
63:     >>> A.dot(Ainv)
64:     <2x2 sparse matrix of type '<class 'numpy.float64'>'
65:         with 2 stored elements in Compressed Sparse Column format>
66:     >>> A.dot(Ainv).todense()
67:     matrix([[ 1.,  0.],
68:             [ 0.,  1.]])
69: 
70:     .. versionadded:: 0.12.0
71: 
72:     '''
73:     I = speye(A.shape[0], A.shape[1], dtype=A.dtype, format=A.format)
74:     Ainv = spsolve(A, I)
75:     return Ainv
76: 
77: 
78: def _onenorm_matrix_power_nnm(A, p):
79:     '''
80:     Compute the 1-norm of a non-negative integer power of a non-negative matrix.
81: 
82:     Parameters
83:     ----------
84:     A : a square ndarray or matrix or sparse matrix
85:         Input matrix with non-negative entries.
86:     p : non-negative integer
87:         The power to which the matrix is to be raised.
88: 
89:     Returns
90:     -------
91:     out : float
92:         The 1-norm of the matrix power p of A.
93: 
94:     '''
95:     # check input
96:     if int(p) != p or p < 0:
97:         raise ValueError('expected non-negative integer p')
98:     p = int(p)
99:     if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
100:         raise ValueError('expected A to be like a square matrix')
101: 
102:     # Explicitly make a column vector so that this works when A is a
103:     # numpy matrix (in addition to ndarray and sparse matrix).
104:     v = np.ones((A.shape[0], 1), dtype=float)
105:     M = A.T
106:     for i in range(p):
107:         v = M.dot(v)
108:     return max(v)
109: 
110: 
111: def _onenorm(A):
112:     # A compatibility function which should eventually disappear.
113:     # This is copypasted from expm_action.
114:     if scipy.sparse.isspmatrix(A):
115:         return max(abs(A).sum(axis=0).flat)
116:     else:
117:         return np.linalg.norm(A, 1)
118: 
119: 
120: def _ident_like(A):
121:     # A compatibility function which should eventually disappear.
122:     # This is copypasted from expm_action.
123:     if scipy.sparse.isspmatrix(A):
124:         return scipy.sparse.construct.eye(A.shape[0], A.shape[1],
125:                 dtype=A.dtype, format=A.format)
126:     else:
127:         return np.eye(A.shape[0], A.shape[1], dtype=A.dtype)
128: 
129: 
130: def _is_upper_triangular(A):
131:     # This function could possibly be of wider interest.
132:     if isspmatrix(A):
133:         lower_part = scipy.sparse.tril(A, -1)
134:         # Check structural upper triangularity,
135:         # then coincidental upper triangularity if needed.
136:         return lower_part.nnz == 0 or lower_part.count_nonzero() == 0
137:     else:
138:         return not np.tril(A, -1).any()
139: 
140: 
141: def _smart_matrix_product(A, B, alpha=None, structure=None):
142:     '''
143:     A matrix product that knows about sparse and structured matrices.
144: 
145:     Parameters
146:     ----------
147:     A : 2d ndarray
148:         First matrix.
149:     B : 2d ndarray
150:         Second matrix.
151:     alpha : float
152:         The matrix product will be scaled by this constant.
153:     structure : str, optional
154:         A string describing the structure of both matrices `A` and `B`.
155:         Only `upper_triangular` is currently supported.
156: 
157:     Returns
158:     -------
159:     M : 2d ndarray
160:         Matrix product of A and B.
161: 
162:     '''
163:     if len(A.shape) != 2:
164:         raise ValueError('expected A to be a rectangular matrix')
165:     if len(B.shape) != 2:
166:         raise ValueError('expected B to be a rectangular matrix')
167:     f = None
168:     if structure == UPPER_TRIANGULAR:
169:         if not isspmatrix(A) and not isspmatrix(B):
170:             f, = scipy.linalg.get_blas_funcs(('trmm',), (A, B))
171:     if f is not None:
172:         if alpha is None:
173:             alpha = 1.
174:         out = f(alpha, A, B)
175:     else:
176:         if alpha is None:
177:             out = A.dot(B)
178:         else:
179:             out = alpha * A.dot(B)
180:     return out
181: 
182: 
183: class MatrixPowerOperator(LinearOperator):
184: 
185:     def __init__(self, A, p, structure=None):
186:         if A.ndim != 2 or A.shape[0] != A.shape[1]:
187:             raise ValueError('expected A to be like a square matrix')
188:         if p < 0:
189:             raise ValueError('expected p to be a non-negative integer')
190:         self._A = A
191:         self._p = p
192:         self._structure = structure
193:         self.dtype = A.dtype
194:         self.ndim = A.ndim
195:         self.shape = A.shape
196: 
197:     def _matvec(self, x):
198:         for i in range(self._p):
199:             x = self._A.dot(x)
200:         return x
201: 
202:     def _rmatvec(self, x):
203:         A_T = self._A.T
204:         x = x.ravel()
205:         for i in range(self._p):
206:             x = A_T.dot(x)
207:         return x
208: 
209:     def _matmat(self, X):
210:         for i in range(self._p):
211:             X = _smart_matrix_product(self._A, X, structure=self._structure)
212:         return X
213: 
214:     @property
215:     def T(self):
216:         return MatrixPowerOperator(self._A.T, self._p)
217: 
218: 
219: class ProductOperator(LinearOperator):
220:     '''
221:     For now, this is limited to products of multiple square matrices.
222:     '''
223: 
224:     def __init__(self, *args, **kwargs):
225:         self._structure = kwargs.get('structure', None)
226:         for A in args:
227:             if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
228:                 raise ValueError(
229:                         'For now, the ProductOperator implementation is '
230:                         'limited to the product of multiple square matrices.')
231:         if args:
232:             n = args[0].shape[0]
233:             for A in args:
234:                 for d in A.shape:
235:                     if d != n:
236:                         raise ValueError(
237:                                 'The square matrices of the ProductOperator '
238:                                 'must all have the same shape.')
239:             self.shape = (n, n)
240:             self.ndim = len(self.shape)
241:         self.dtype = np.find_common_type([x.dtype for x in args], [])
242:         self._operator_sequence = args
243: 
244:     def _matvec(self, x):
245:         for A in reversed(self._operator_sequence):
246:             x = A.dot(x)
247:         return x
248: 
249:     def _rmatvec(self, x):
250:         x = x.ravel()
251:         for A in self._operator_sequence:
252:             x = A.T.dot(x)
253:         return x
254: 
255:     def _matmat(self, X):
256:         for A in reversed(self._operator_sequence):
257:             X = _smart_matrix_product(A, X, structure=self._structure)
258:         return X
259: 
260:     @property
261:     def T(self):
262:         T_args = [A.T for A in reversed(self._operator_sequence)]
263:         return ProductOperator(*T_args)
264: 
265: 
266: def _onenormest_matrix_power(A, p,
267:         t=2, itmax=5, compute_v=False, compute_w=False, structure=None):
268:     '''
269:     Efficiently estimate the 1-norm of A^p.
270: 
271:     Parameters
272:     ----------
273:     A : ndarray
274:         Matrix whose 1-norm of a power is to be computed.
275:     p : int
276:         Non-negative integer power.
277:     t : int, optional
278:         A positive parameter controlling the tradeoff between
279:         accuracy versus time and memory usage.
280:         Larger values take longer and use more memory
281:         but give more accurate output.
282:     itmax : int, optional
283:         Use at most this many iterations.
284:     compute_v : bool, optional
285:         Request a norm-maximizing linear operator input vector if True.
286:     compute_w : bool, optional
287:         Request a norm-maximizing linear operator output vector if True.
288: 
289:     Returns
290:     -------
291:     est : float
292:         An underestimate of the 1-norm of the sparse matrix.
293:     v : ndarray, optional
294:         The vector such that ||Av||_1 == est*||v||_1.
295:         It can be thought of as an input to the linear operator
296:         that gives an output with particularly large norm.
297:     w : ndarray, optional
298:         The vector Av which has relatively large 1-norm.
299:         It can be thought of as an output of the linear operator
300:         that is relatively large in norm compared to the input.
301: 
302:     '''
303:     return scipy.sparse.linalg.onenormest(
304:             MatrixPowerOperator(A, p, structure=structure))
305: 
306: 
307: def _onenormest_product(operator_seq,
308:         t=2, itmax=5, compute_v=False, compute_w=False, structure=None):
309:     '''
310:     Efficiently estimate the 1-norm of the matrix product of the args.
311: 
312:     Parameters
313:     ----------
314:     operator_seq : linear operator sequence
315:         Matrices whose 1-norm of product is to be computed.
316:     t : int, optional
317:         A positive parameter controlling the tradeoff between
318:         accuracy versus time and memory usage.
319:         Larger values take longer and use more memory
320:         but give more accurate output.
321:     itmax : int, optional
322:         Use at most this many iterations.
323:     compute_v : bool, optional
324:         Request a norm-maximizing linear operator input vector if True.
325:     compute_w : bool, optional
326:         Request a norm-maximizing linear operator output vector if True.
327:     structure : str, optional
328:         A string describing the structure of all operators.
329:         Only `upper_triangular` is currently supported.
330: 
331:     Returns
332:     -------
333:     est : float
334:         An underestimate of the 1-norm of the sparse matrix.
335:     v : ndarray, optional
336:         The vector such that ||Av||_1 == est*||v||_1.
337:         It can be thought of as an input to the linear operator
338:         that gives an output with particularly large norm.
339:     w : ndarray, optional
340:         The vector Av which has relatively large 1-norm.
341:         It can be thought of as an output of the linear operator
342:         that is relatively large in norm compared to the input.
343: 
344:     '''
345:     return scipy.sparse.linalg.onenormest(
346:             ProductOperator(*operator_seq, structure=structure))
347: 
348: 
349: class _ExpmPadeHelper(object):
350:     '''
351:     Help lazily evaluate a matrix exponential.
352: 
353:     The idea is to not do more work than we need for high expm precision,
354:     so we lazily compute matrix powers and store or precompute
355:     other properties of the matrix.
356: 
357:     '''
358:     def __init__(self, A, structure=None, use_exact_onenorm=False):
359:         '''
360:         Initialize the object.
361: 
362:         Parameters
363:         ----------
364:         A : a dense or sparse square numpy matrix or ndarray
365:             The matrix to be exponentiated.
366:         structure : str, optional
367:             A string describing the structure of matrix `A`.
368:             Only `upper_triangular` is currently supported.
369:         use_exact_onenorm : bool, optional
370:             If True then only the exact one-norm of matrix powers and products
371:             will be used. Otherwise, the one-norm of powers and products
372:             may initially be estimated.
373:         '''
374:         self.A = A
375:         self._A2 = None
376:         self._A4 = None
377:         self._A6 = None
378:         self._A8 = None
379:         self._A10 = None
380:         self._d4_exact = None
381:         self._d6_exact = None
382:         self._d8_exact = None
383:         self._d10_exact = None
384:         self._d4_approx = None
385:         self._d6_approx = None
386:         self._d8_approx = None
387:         self._d10_approx = None
388:         self.ident = _ident_like(A)
389:         self.structure = structure
390:         self.use_exact_onenorm = use_exact_onenorm
391: 
392:     @property
393:     def A2(self):
394:         if self._A2 is None:
395:             self._A2 = _smart_matrix_product(
396:                     self.A, self.A, structure=self.structure)
397:         return self._A2
398: 
399:     @property
400:     def A4(self):
401:         if self._A4 is None:
402:             self._A4 = _smart_matrix_product(
403:                     self.A2, self.A2, structure=self.structure)
404:         return self._A4
405: 
406:     @property
407:     def A6(self):
408:         if self._A6 is None:
409:             self._A6 = _smart_matrix_product(
410:                     self.A4, self.A2, structure=self.structure)
411:         return self._A6
412: 
413:     @property
414:     def A8(self):
415:         if self._A8 is None:
416:             self._A8 = _smart_matrix_product(
417:                     self.A6, self.A2, structure=self.structure)
418:         return self._A8
419: 
420:     @property
421:     def A10(self):
422:         if self._A10 is None:
423:             self._A10 = _smart_matrix_product(
424:                     self.A4, self.A6, structure=self.structure)
425:         return self._A10
426: 
427:     @property
428:     def d4_tight(self):
429:         if self._d4_exact is None:
430:             self._d4_exact = _onenorm(self.A4)**(1/4.)
431:         return self._d4_exact
432: 
433:     @property
434:     def d6_tight(self):
435:         if self._d6_exact is None:
436:             self._d6_exact = _onenorm(self.A6)**(1/6.)
437:         return self._d6_exact
438: 
439:     @property
440:     def d8_tight(self):
441:         if self._d8_exact is None:
442:             self._d8_exact = _onenorm(self.A8)**(1/8.)
443:         return self._d8_exact
444: 
445:     @property
446:     def d10_tight(self):
447:         if self._d10_exact is None:
448:             self._d10_exact = _onenorm(self.A10)**(1/10.)
449:         return self._d10_exact
450: 
451:     @property
452:     def d4_loose(self):
453:         if self.use_exact_onenorm:
454:             return self.d4_tight
455:         if self._d4_exact is not None:
456:             return self._d4_exact
457:         else:
458:             if self._d4_approx is None:
459:                 self._d4_approx = _onenormest_matrix_power(self.A2, 2,
460:                         structure=self.structure)**(1/4.)
461:             return self._d4_approx
462: 
463:     @property
464:     def d6_loose(self):
465:         if self.use_exact_onenorm:
466:             return self.d6_tight
467:         if self._d6_exact is not None:
468:             return self._d6_exact
469:         else:
470:             if self._d6_approx is None:
471:                 self._d6_approx = _onenormest_matrix_power(self.A2, 3,
472:                         structure=self.structure)**(1/6.)
473:             return self._d6_approx
474: 
475:     @property
476:     def d8_loose(self):
477:         if self.use_exact_onenorm:
478:             return self.d8_tight
479:         if self._d8_exact is not None:
480:             return self._d8_exact
481:         else:
482:             if self._d8_approx is None:
483:                 self._d8_approx = _onenormest_matrix_power(self.A4, 2,
484:                         structure=self.structure)**(1/8.)
485:             return self._d8_approx
486: 
487:     @property
488:     def d10_loose(self):
489:         if self.use_exact_onenorm:
490:             return self.d10_tight
491:         if self._d10_exact is not None:
492:             return self._d10_exact
493:         else:
494:             if self._d10_approx is None:
495:                 self._d10_approx = _onenormest_product((self.A4, self.A6),
496:                         structure=self.structure)**(1/10.)
497:             return self._d10_approx
498: 
499:     def pade3(self):
500:         b = (120., 60., 12., 1.)
501:         U = _smart_matrix_product(self.A,
502:                 b[3]*self.A2 + b[1]*self.ident,
503:                 structure=self.structure)
504:         V = b[2]*self.A2 + b[0]*self.ident
505:         return U, V
506: 
507:     def pade5(self):
508:         b = (30240., 15120., 3360., 420., 30., 1.)
509:         U = _smart_matrix_product(self.A,
510:                 b[5]*self.A4 + b[3]*self.A2 + b[1]*self.ident,
511:                 structure=self.structure)
512:         V = b[4]*self.A4 + b[2]*self.A2 + b[0]*self.ident
513:         return U, V
514: 
515:     def pade7(self):
516:         b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
517:         U = _smart_matrix_product(self.A,
518:                 b[7]*self.A6 + b[5]*self.A4 + b[3]*self.A2 + b[1]*self.ident,
519:                 structure=self.structure)
520:         V = b[6]*self.A6 + b[4]*self.A4 + b[2]*self.A2 + b[0]*self.ident
521:         return U, V
522: 
523:     def pade9(self):
524:         b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
525:                 2162160., 110880., 3960., 90., 1.)
526:         U = _smart_matrix_product(self.A,
527:                 (b[9]*self.A8 + b[7]*self.A6 + b[5]*self.A4 +
528:                     b[3]*self.A2 + b[1]*self.ident),
529:                 structure=self.structure)
530:         V = (b[8]*self.A8 + b[6]*self.A6 + b[4]*self.A4 +
531:                 b[2]*self.A2 + b[0]*self.ident)
532:         return U, V
533: 
534:     def pade13_scaled(self, s):
535:         b = (64764752532480000., 32382376266240000., 7771770303897600.,
536:                 1187353796428800., 129060195264000., 10559470521600.,
537:                 670442572800., 33522128640., 1323241920., 40840800., 960960.,
538:                 16380., 182., 1.)
539:         B = self.A * 2**-s
540:         B2 = self.A2 * 2**(-2*s)
541:         B4 = self.A4 * 2**(-4*s)
542:         B6 = self.A6 * 2**(-6*s)
543:         U2 = _smart_matrix_product(B6,
544:                 b[13]*B6 + b[11]*B4 + b[9]*B2,
545:                 structure=self.structure)
546:         U = _smart_matrix_product(B,
547:                 (U2 + b[7]*B6 + b[5]*B4 +
548:                     b[3]*B2 + b[1]*self.ident),
549:                 structure=self.structure)
550:         V2 = _smart_matrix_product(B6,
551:                 b[12]*B6 + b[10]*B4 + b[8]*B2,
552:                 structure=self.structure)
553:         V = V2 + b[6]*B6 + b[4]*B4 + b[2]*B2 + b[0]*self.ident
554:         return U, V
555: 
556: 
557: def expm(A):
558:     '''
559:     Compute the matrix exponential using Pade approximation.
560: 
561:     Parameters
562:     ----------
563:     A : (M,M) array_like or sparse matrix
564:         2D Array or Matrix (sparse or dense) to be exponentiated
565: 
566:     Returns
567:     -------
568:     expA : (M,M) ndarray
569:         Matrix exponential of `A`
570: 
571:     Notes
572:     -----
573:     This is algorithm (6.1) which is a simplification of algorithm (5.1).
574: 
575:     .. versionadded:: 0.12.0
576: 
577:     References
578:     ----------
579:     .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
580:            "A New Scaling and Squaring Algorithm for the Matrix Exponential."
581:            SIAM Journal on Matrix Analysis and Applications.
582:            31 (3). pp. 970-989. ISSN 1095-7162
583: 
584:     Examples
585:     --------
586:     >>> from scipy.sparse import csc_matrix
587:     >>> from scipy.sparse.linalg import expm
588:     >>> A = csc_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
589:     >>> A.todense()
590:     matrix([[1, 0, 0],
591:             [0, 2, 0],
592:             [0, 0, 3]], dtype=int64)
593:     >>> Aexp = expm(A)
594:     >>> Aexp
595:     <3x3 sparse matrix of type '<class 'numpy.float64'>'
596:         with 3 stored elements in Compressed Sparse Column format>
597:     >>> Aexp.todense()
598:     matrix([[  2.71828183,   0.        ,   0.        ],
599:             [  0.        ,   7.3890561 ,   0.        ],
600:             [  0.        ,   0.        ,  20.08553692]])
601:     '''
602:     return _expm(A, use_exact_onenorm='auto')
603: 
604: 
605: def _expm(A, use_exact_onenorm):
606:     # Core of expm, separated to allow testing exact and approximate
607:     # algorithms.
608: 
609:     # Avoid indiscriminate asarray() to allow sparse or other strange arrays.
610:     if isinstance(A, (list, tuple)):
611:         A = np.asarray(A)
612:     if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
613:         raise ValueError('expected a square matrix')
614: 
615:     # Trivial case
616:     if A.shape == (1, 1):
617:         out = [[np.exp(A[0, 0])]]
618: 
619:         # Avoid indiscriminate casting to ndarray to
620:         # allow for sparse or other strange arrays
621:         if isspmatrix(A):
622:             return A.__class__(out)
623: 
624:         return np.array(out)
625: 
626:     # Detect upper triangularity.
627:     structure = UPPER_TRIANGULAR if _is_upper_triangular(A) else None
628: 
629:     if use_exact_onenorm == "auto":
630:         # Hardcode a matrix order threshold for exact vs. estimated one-norms.
631:         use_exact_onenorm = A.shape[0] < 200
632: 
633:     # Track functions of A to help compute the matrix exponential.
634:     h = _ExpmPadeHelper(
635:             A, structure=structure, use_exact_onenorm=use_exact_onenorm)
636: 
637:     # Try Pade order 3.
638:     eta_1 = max(h.d4_loose, h.d6_loose)
639:     if eta_1 < 1.495585217958292e-002 and _ell(h.A, 3) == 0:
640:         U, V = h.pade3()
641:         return _solve_P_Q(U, V, structure=structure)
642: 
643:     # Try Pade order 5.
644:     eta_2 = max(h.d4_tight, h.d6_loose)
645:     if eta_2 < 2.539398330063230e-001 and _ell(h.A, 5) == 0:
646:         U, V = h.pade5()
647:         return _solve_P_Q(U, V, structure=structure)
648: 
649:     # Try Pade orders 7 and 9.
650:     eta_3 = max(h.d6_tight, h.d8_loose)
651:     if eta_3 < 9.504178996162932e-001 and _ell(h.A, 7) == 0:
652:         U, V = h.pade7()
653:         return _solve_P_Q(U, V, structure=structure)
654:     if eta_3 < 2.097847961257068e+000 and _ell(h.A, 9) == 0:
655:         U, V = h.pade9()
656:         return _solve_P_Q(U, V, structure=structure)
657: 
658:     # Use Pade order 13.
659:     eta_4 = max(h.d8_loose, h.d10_loose)
660:     eta_5 = min(eta_3, eta_4)
661:     theta_13 = 4.25
662:     s = max(int(np.ceil(np.log2(eta_5 / theta_13))), 0)
663:     s = s + _ell(2**-s * h.A, 13)
664:     U, V = h.pade13_scaled(s)
665:     X = _solve_P_Q(U, V, structure=structure)
666:     if structure == UPPER_TRIANGULAR:
667:         # Invoke Code Fragment 2.1.
668:         X = _fragment_2_1(X, h.A, s)
669:     else:
670:         # X = r_13(A)^(2^s) by repeated squaring.
671:         for i in range(s):
672:             X = X.dot(X)
673:     return X
674: 
675: 
676: def _solve_P_Q(U, V, structure=None):
677:     '''
678:     A helper function for expm_2009.
679: 
680:     Parameters
681:     ----------
682:     U : ndarray
683:         Pade numerator.
684:     V : ndarray
685:         Pade denominator.
686:     structure : str, optional
687:         A string describing the structure of both matrices `U` and `V`.
688:         Only `upper_triangular` is currently supported.
689: 
690:     Notes
691:     -----
692:     The `structure` argument is inspired by similar args
693:     for theano and cvxopt functions.
694: 
695:     '''
696:     P = U + V
697:     Q = -U + V
698:     if isspmatrix(U):
699:         return spsolve(Q, P)
700:     elif structure is None:
701:         return solve(Q, P)
702:     elif structure == UPPER_TRIANGULAR:
703:         return solve_triangular(Q, P)
704:     else:
705:         raise ValueError('unsupported matrix structure: ' + str(structure))
706: 
707: 
708: def _sinch(x):
709:     '''
710:     Stably evaluate sinch.
711: 
712:     Notes
713:     -----
714:     The strategy of falling back to a sixth order Taylor expansion
715:     was suggested by the Spallation Neutron Source docs
716:     which was found on the internet by google search.
717:     http://www.ornl.gov/~t6p/resources/xal/javadoc/gov/sns/tools/math/ElementaryFunction.html
718:     The details of the cutoff point and the Horner-like evaluation
719:     was picked without reference to anything in particular.
720: 
721:     Note that sinch is not currently implemented in scipy.special,
722:     whereas the "engineer's" definition of sinc is implemented.
723:     The implementation of sinc involves a scaling factor of pi
724:     that distinguishes it from the "mathematician's" version of sinc.
725: 
726:     '''
727: 
728:     # If x is small then use sixth order Taylor expansion.
729:     # How small is small? I am using the point where the relative error
730:     # of the approximation is less than 1e-14.
731:     # If x is large then directly evaluate sinh(x) / x.
732:     x2 = x*x
733:     if abs(x) < 0.0135:
734:         return 1 + (x2/6.)*(1 + (x2/20.)*(1 + (x2/42.)))
735:     else:
736:         return np.sinh(x) / x
737: 
738: 
739: def _eq_10_42(lam_1, lam_2, t_12):
740:     '''
741:     Equation (10.42) of Functions of Matrices: Theory and Computation.
742: 
743:     Notes
744:     -----
745:     This is a helper function for _fragment_2_1 of expm_2009.
746:     Equation (10.42) is on page 251 in the section on Schur algorithms.
747:     In particular, section 10.4.3 explains the Schur-Parlett algorithm.
748:     expm([[lam_1, t_12], [0, lam_1])
749:     =
750:     [[exp(lam_1), t_12*exp((lam_1 + lam_2)/2)*sinch((lam_1 - lam_2)/2)],
751:     [0, exp(lam_2)]
752:     '''
753: 
754:     # The plain formula t_12 * (exp(lam_2) - exp(lam_2)) / (lam_2 - lam_1)
755:     # apparently suffers from cancellation, according to Higham's textbook.
756:     # A nice implementation of sinch, defined as sinh(x)/x,
757:     # will apparently work around the cancellation.
758:     a = 0.5 * (lam_1 + lam_2)
759:     b = 0.5 * (lam_1 - lam_2)
760:     return t_12 * np.exp(a) * _sinch(b)
761: 
762: 
763: def _fragment_2_1(X, T, s):
764:     '''
765:     A helper function for expm_2009.
766: 
767:     Notes
768:     -----
769:     The argument X is modified in-place, but this modification is not the same
770:     as the returned value of the function.
771:     This function also takes pains to do things in ways that are compatible
772:     with sparse matrices, for example by avoiding fancy indexing
773:     and by using methods of the matrices whenever possible instead of
774:     using functions of the numpy or scipy libraries themselves.
775: 
776:     '''
777:     # Form X = r_m(2^-s T)
778:     # Replace diag(X) by exp(2^-s diag(T)).
779:     n = X.shape[0]
780:     diag_T = np.ravel(T.diagonal().copy())
781: 
782:     # Replace diag(X) by exp(2^-s diag(T)).
783:     scale = 2 ** -s
784:     exp_diag = np.exp(scale * diag_T)
785:     for k in range(n):
786:         X[k, k] = exp_diag[k]
787: 
788:     for i in range(s-1, -1, -1):
789:         X = X.dot(X)
790: 
791:         # Replace diag(X) by exp(2^-i diag(T)).
792:         scale = 2 ** -i
793:         exp_diag = np.exp(scale * diag_T)
794:         for k in range(n):
795:             X[k, k] = exp_diag[k]
796: 
797:         # Replace (first) superdiagonal of X by explicit formula
798:         # for superdiagonal of exp(2^-i T) from Eq (10.42) of
799:         # the author's 2008 textbook
800:         # Functions of Matrices: Theory and Computation.
801:         for k in range(n-1):
802:             lam_1 = scale * diag_T[k]
803:             lam_2 = scale * diag_T[k+1]
804:             t_12 = scale * T[k, k+1]
805:             value = _eq_10_42(lam_1, lam_2, t_12)
806:             X[k, k+1] = value
807: 
808:     # Return the updated X matrix.
809:     return X
810: 
811: 
812: def _ell(A, m):
813:     '''
814:     A helper function for expm_2009.
815: 
816:     Parameters
817:     ----------
818:     A : linear operator
819:         A linear operator whose norm of power we care about.
820:     m : int
821:         The power of the linear operator
822: 
823:     Returns
824:     -------
825:     value : int
826:         A value related to a bound.
827: 
828:     '''
829:     if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
830:         raise ValueError('expected A to be like a square matrix')
831: 
832:     p = 2*m + 1
833: 
834:     # The c_i are explained in (2.2) and (2.6) of the 2005 expm paper.
835:     # They are coefficients of terms of a generating function series expansion.
836:     choose_2p_p = scipy.special.comb(2*p, p, exact=True)
837:     abs_c_recip = float(choose_2p_p * math.factorial(2*p + 1))
838: 
839:     # This is explained after Eq. (1.2) of the 2009 expm paper.
840:     # It is the "unit roundoff" of IEEE double precision arithmetic.
841:     u = 2**-53
842: 
843:     # Compute the one-norm of matrix power p of abs(A).
844:     A_abs_onenorm = _onenorm_matrix_power_nnm(abs(A), p)
845: 
846:     # Treat zero norm as a special case.
847:     if not A_abs_onenorm:
848:         return 0
849: 
850:     alpha = A_abs_onenorm / (_onenorm(A) * abs_c_recip)
851:     log2_alpha_div_u = np.log2(alpha/u)
852:     value = int(np.ceil(log2_alpha_div_u / (2 * m)))
853:     return max(value, 0)
854: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_386413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nSparse matrix functions\n')

# Assigning a List to a Name (line 13):

# Assigning a List to a Name (line 13):
__all__ = ['expm', 'inv']
module_type_store.set_exportable_members(['expm', 'inv'])

# Obtaining an instance of the builtin type 'list' (line 13)
list_386414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
str_386415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'expm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_386414, str_386415)
# Adding element type (line 13)
str_386416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'str', 'inv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_386414, str_386416)

# Assigning a type to the variable '__all__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__all__', list_386414)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import math' statement (line 15)
import math

import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import numpy' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_386417 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy')

if (type(import_386417) is not StypyTypeError):

    if (import_386417 != 'pyd_module'):
        __import__(import_386417)
        sys_modules_386418 = sys.modules[import_386417]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'np', sys_modules_386418.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy', import_386417)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import scipy.special' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_386419 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.special')

if (type(import_386419) is not StypyTypeError):

    if (import_386419 != 'pyd_module'):
        __import__(import_386419)
        sys_modules_386420 = sys.modules[import_386419]
        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.special', sys_modules_386420.module_type_store, module_type_store)
    else:
        import scipy.special

        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.special', scipy.special, module_type_store)

else:
    # Assigning a type to the variable 'scipy.special' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.special', import_386419)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy.linalg.basic import solve, solve_triangular' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_386421 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg.basic')

if (type(import_386421) is not StypyTypeError):

    if (import_386421 != 'pyd_module'):
        __import__(import_386421)
        sys_modules_386422 = sys.modules[import_386421]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg.basic', sys_modules_386422.module_type_store, module_type_store, ['solve', 'solve_triangular'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_386422, sys_modules_386422.module_type_store, module_type_store)
    else:
        from scipy.linalg.basic import solve, solve_triangular

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg.basic', None, module_type_store, ['solve', 'solve_triangular'], [solve, solve_triangular])

else:
    # Assigning a type to the variable 'scipy.linalg.basic' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.linalg.basic', import_386421)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from scipy.sparse.base import isspmatrix' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_386423 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.sparse.base')

if (type(import_386423) is not StypyTypeError):

    if (import_386423 != 'pyd_module'):
        __import__(import_386423)
        sys_modules_386424 = sys.modules[import_386423]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.sparse.base', sys_modules_386424.module_type_store, module_type_store, ['isspmatrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_386424, sys_modules_386424.module_type_store, module_type_store)
    else:
        from scipy.sparse.base import isspmatrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.sparse.base', None, module_type_store, ['isspmatrix'], [isspmatrix])

else:
    # Assigning a type to the variable 'scipy.sparse.base' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.sparse.base', import_386423)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from scipy.sparse.construct import speye' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_386425 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy.sparse.construct')

if (type(import_386425) is not StypyTypeError):

    if (import_386425 != 'pyd_module'):
        __import__(import_386425)
        sys_modules_386426 = sys.modules[import_386425]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy.sparse.construct', sys_modules_386426.module_type_store, module_type_store, ['eye'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_386426, sys_modules_386426.module_type_store, module_type_store)
    else:
        from scipy.sparse.construct import eye as speye

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy.sparse.construct', None, module_type_store, ['eye'], [speye])

else:
    # Assigning a type to the variable 'scipy.sparse.construct' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy.sparse.construct', import_386425)

# Adding an alias
module_type_store.add_alias('speye', 'eye')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from scipy.sparse.linalg import spsolve' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_386427 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.sparse.linalg')

if (type(import_386427) is not StypyTypeError):

    if (import_386427 != 'pyd_module'):
        __import__(import_386427)
        sys_modules_386428 = sys.modules[import_386427]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.sparse.linalg', sys_modules_386428.module_type_store, module_type_store, ['spsolve'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_386428, sys_modules_386428.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import spsolve

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.sparse.linalg', None, module_type_store, ['spsolve'], [spsolve])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.sparse.linalg', import_386427)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'import scipy.sparse' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_386429 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse')

if (type(import_386429) is not StypyTypeError):

    if (import_386429 != 'pyd_module'):
        __import__(import_386429)
        sys_modules_386430 = sys.modules[import_386429]
        import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse', sys_modules_386430.module_type_store, module_type_store)
    else:
        import scipy.sparse

        import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse', scipy.sparse, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse', import_386429)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'import scipy.sparse.linalg' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_386431 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.sparse.linalg')

if (type(import_386431) is not StypyTypeError):

    if (import_386431 != 'pyd_module'):
        __import__(import_386431)
        sys_modules_386432 = sys.modules[import_386431]
        import_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.sparse.linalg', sys_modules_386432.module_type_store, module_type_store)
    else:
        import scipy.sparse.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.sparse.linalg', scipy.sparse.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.sparse.linalg', import_386431)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from scipy.sparse.linalg.interface import LinearOperator' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_386433 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.sparse.linalg.interface')

if (type(import_386433) is not StypyTypeError):

    if (import_386433 != 'pyd_module'):
        __import__(import_386433)
        sys_modules_386434 = sys.modules[import_386433]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.sparse.linalg.interface', sys_modules_386434.module_type_store, module_type_store, ['LinearOperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_386434, sys_modules_386434.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.interface import LinearOperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.sparse.linalg.interface', None, module_type_store, ['LinearOperator'], [LinearOperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.interface' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.sparse.linalg.interface', import_386433)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')


# Assigning a Str to a Name (line 31):

# Assigning a Str to a Name (line 31):
str_386435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'str', 'upper_triangular')
# Assigning a type to the variable 'UPPER_TRIANGULAR' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'UPPER_TRIANGULAR', str_386435)

@norecursion
def inv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'inv'
    module_type_store = module_type_store.open_function_context('inv', 34, 0, False)
    
    # Passed parameters checking function
    inv.stypy_localization = localization
    inv.stypy_type_of_self = None
    inv.stypy_type_store = module_type_store
    inv.stypy_function_name = 'inv'
    inv.stypy_param_names_list = ['A']
    inv.stypy_varargs_param_name = None
    inv.stypy_kwargs_param_name = None
    inv.stypy_call_defaults = defaults
    inv.stypy_call_varargs = varargs
    inv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'inv', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'inv', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'inv(...)' code ##################

    str_386436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, (-1)), 'str', "\n    Compute the inverse of a sparse matrix\n\n    Parameters\n    ----------\n    A : (M,M) ndarray or sparse matrix\n        square matrix to be inverted\n\n    Returns\n    -------\n    Ainv : (M,M) ndarray or sparse matrix\n        inverse of `A`\n\n    Notes\n    -----\n    This computes the sparse inverse of `A`.  If the inverse of `A` is expected\n    to be non-sparse, it will likely be faster to convert `A` to dense and use\n    scipy.linalg.inv.\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import inv\n    >>> A = csc_matrix([[1., 0.], [1., 2.]])\n    >>> Ainv = inv(A)\n    >>> Ainv\n    <2x2 sparse matrix of type '<class 'numpy.float64'>'\n        with 3 stored elements in Compressed Sparse Column format>\n    >>> A.dot(Ainv)\n    <2x2 sparse matrix of type '<class 'numpy.float64'>'\n        with 2 stored elements in Compressed Sparse Column format>\n    >>> A.dot(Ainv).todense()\n    matrix([[ 1.,  0.],\n            [ 0.,  1.]])\n\n    .. versionadded:: 0.12.0\n\n    ")
    
    # Assigning a Call to a Name (line 73):
    
    # Assigning a Call to a Name (line 73):
    
    # Call to speye(...): (line 73)
    # Processing the call arguments (line 73)
    
    # Obtaining the type of the subscript
    int_386438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 22), 'int')
    # Getting the type of 'A' (line 73)
    A_386439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 14), 'A', False)
    # Obtaining the member 'shape' of a type (line 73)
    shape_386440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 14), A_386439, 'shape')
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___386441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 14), shape_386440, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
    subscript_call_result_386442 = invoke(stypy.reporting.localization.Localization(__file__, 73, 14), getitem___386441, int_386438)
    
    
    # Obtaining the type of the subscript
    int_386443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 34), 'int')
    # Getting the type of 'A' (line 73)
    A_386444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'A', False)
    # Obtaining the member 'shape' of a type (line 73)
    shape_386445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 26), A_386444, 'shape')
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___386446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 26), shape_386445, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
    subscript_call_result_386447 = invoke(stypy.reporting.localization.Localization(__file__, 73, 26), getitem___386446, int_386443)
    
    # Processing the call keyword arguments (line 73)
    # Getting the type of 'A' (line 73)
    A_386448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'A', False)
    # Obtaining the member 'dtype' of a type (line 73)
    dtype_386449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 44), A_386448, 'dtype')
    keyword_386450 = dtype_386449
    # Getting the type of 'A' (line 73)
    A_386451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 60), 'A', False)
    # Obtaining the member 'format' of a type (line 73)
    format_386452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 60), A_386451, 'format')
    keyword_386453 = format_386452
    kwargs_386454 = {'dtype': keyword_386450, 'format': keyword_386453}
    # Getting the type of 'speye' (line 73)
    speye_386437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'speye', False)
    # Calling speye(args, kwargs) (line 73)
    speye_call_result_386455 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), speye_386437, *[subscript_call_result_386442, subscript_call_result_386447], **kwargs_386454)
    
    # Assigning a type to the variable 'I' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'I', speye_call_result_386455)
    
    # Assigning a Call to a Name (line 74):
    
    # Assigning a Call to a Name (line 74):
    
    # Call to spsolve(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'A' (line 74)
    A_386457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'A', False)
    # Getting the type of 'I' (line 74)
    I_386458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 22), 'I', False)
    # Processing the call keyword arguments (line 74)
    kwargs_386459 = {}
    # Getting the type of 'spsolve' (line 74)
    spsolve_386456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'spsolve', False)
    # Calling spsolve(args, kwargs) (line 74)
    spsolve_call_result_386460 = invoke(stypy.reporting.localization.Localization(__file__, 74, 11), spsolve_386456, *[A_386457, I_386458], **kwargs_386459)
    
    # Assigning a type to the variable 'Ainv' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'Ainv', spsolve_call_result_386460)
    # Getting the type of 'Ainv' (line 75)
    Ainv_386461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'Ainv')
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type', Ainv_386461)
    
    # ################# End of 'inv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'inv' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_386462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_386462)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'inv'
    return stypy_return_type_386462

# Assigning a type to the variable 'inv' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'inv', inv)

@norecursion
def _onenorm_matrix_power_nnm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_onenorm_matrix_power_nnm'
    module_type_store = module_type_store.open_function_context('_onenorm_matrix_power_nnm', 78, 0, False)
    
    # Passed parameters checking function
    _onenorm_matrix_power_nnm.stypy_localization = localization
    _onenorm_matrix_power_nnm.stypy_type_of_self = None
    _onenorm_matrix_power_nnm.stypy_type_store = module_type_store
    _onenorm_matrix_power_nnm.stypy_function_name = '_onenorm_matrix_power_nnm'
    _onenorm_matrix_power_nnm.stypy_param_names_list = ['A', 'p']
    _onenorm_matrix_power_nnm.stypy_varargs_param_name = None
    _onenorm_matrix_power_nnm.stypy_kwargs_param_name = None
    _onenorm_matrix_power_nnm.stypy_call_defaults = defaults
    _onenorm_matrix_power_nnm.stypy_call_varargs = varargs
    _onenorm_matrix_power_nnm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_onenorm_matrix_power_nnm', ['A', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_onenorm_matrix_power_nnm', localization, ['A', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_onenorm_matrix_power_nnm(...)' code ##################

    str_386463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, (-1)), 'str', '\n    Compute the 1-norm of a non-negative integer power of a non-negative matrix.\n\n    Parameters\n    ----------\n    A : a square ndarray or matrix or sparse matrix\n        Input matrix with non-negative entries.\n    p : non-negative integer\n        The power to which the matrix is to be raised.\n\n    Returns\n    -------\n    out : float\n        The 1-norm of the matrix power p of A.\n\n    ')
    
    
    # Evaluating a boolean operation
    
    
    # Call to int(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'p' (line 96)
    p_386465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'p', False)
    # Processing the call keyword arguments (line 96)
    kwargs_386466 = {}
    # Getting the type of 'int' (line 96)
    int_386464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 7), 'int', False)
    # Calling int(args, kwargs) (line 96)
    int_call_result_386467 = invoke(stypy.reporting.localization.Localization(__file__, 96, 7), int_386464, *[p_386465], **kwargs_386466)
    
    # Getting the type of 'p' (line 96)
    p_386468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'p')
    # Applying the binary operator '!=' (line 96)
    result_ne_386469 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 7), '!=', int_call_result_386467, p_386468)
    
    
    # Getting the type of 'p' (line 96)
    p_386470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'p')
    int_386471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 26), 'int')
    # Applying the binary operator '<' (line 96)
    result_lt_386472 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 22), '<', p_386470, int_386471)
    
    # Applying the binary operator 'or' (line 96)
    result_or_keyword_386473 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 7), 'or', result_ne_386469, result_lt_386472)
    
    # Testing the type of an if condition (line 96)
    if_condition_386474 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), result_or_keyword_386473)
    # Assigning a type to the variable 'if_condition_386474' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'if_condition_386474', if_condition_386474)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 97)
    # Processing the call arguments (line 97)
    str_386476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 25), 'str', 'expected non-negative integer p')
    # Processing the call keyword arguments (line 97)
    kwargs_386477 = {}
    # Getting the type of 'ValueError' (line 97)
    ValueError_386475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 97)
    ValueError_call_result_386478 = invoke(stypy.reporting.localization.Localization(__file__, 97, 14), ValueError_386475, *[str_386476], **kwargs_386477)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 97, 8), ValueError_call_result_386478, 'raise parameter', BaseException)
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 98):
    
    # Assigning a Call to a Name (line 98):
    
    # Call to int(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'p' (line 98)
    p_386480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'p', False)
    # Processing the call keyword arguments (line 98)
    kwargs_386481 = {}
    # Getting the type of 'int' (line 98)
    int_386479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'int', False)
    # Calling int(args, kwargs) (line 98)
    int_call_result_386482 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), int_386479, *[p_386480], **kwargs_386481)
    
    # Assigning a type to the variable 'p' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'p', int_call_result_386482)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'A' (line 99)
    A_386484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 99)
    shape_386485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 11), A_386484, 'shape')
    # Processing the call keyword arguments (line 99)
    kwargs_386486 = {}
    # Getting the type of 'len' (line 99)
    len_386483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 7), 'len', False)
    # Calling len(args, kwargs) (line 99)
    len_call_result_386487 = invoke(stypy.reporting.localization.Localization(__file__, 99, 7), len_386483, *[shape_386485], **kwargs_386486)
    
    int_386488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 23), 'int')
    # Applying the binary operator '!=' (line 99)
    result_ne_386489 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), '!=', len_call_result_386487, int_386488)
    
    
    
    # Obtaining the type of the subscript
    int_386490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 36), 'int')
    # Getting the type of 'A' (line 99)
    A_386491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 28), 'A')
    # Obtaining the member 'shape' of a type (line 99)
    shape_386492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 28), A_386491, 'shape')
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___386493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 28), shape_386492, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_386494 = invoke(stypy.reporting.localization.Localization(__file__, 99, 28), getitem___386493, int_386490)
    
    
    # Obtaining the type of the subscript
    int_386495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 50), 'int')
    # Getting the type of 'A' (line 99)
    A_386496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 42), 'A')
    # Obtaining the member 'shape' of a type (line 99)
    shape_386497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 42), A_386496, 'shape')
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___386498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 42), shape_386497, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_386499 = invoke(stypy.reporting.localization.Localization(__file__, 99, 42), getitem___386498, int_386495)
    
    # Applying the binary operator '!=' (line 99)
    result_ne_386500 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 28), '!=', subscript_call_result_386494, subscript_call_result_386499)
    
    # Applying the binary operator 'or' (line 99)
    result_or_keyword_386501 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), 'or', result_ne_386489, result_ne_386500)
    
    # Testing the type of an if condition (line 99)
    if_condition_386502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 4), result_or_keyword_386501)
    # Assigning a type to the variable 'if_condition_386502' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'if_condition_386502', if_condition_386502)
    # SSA begins for if statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 100)
    # Processing the call arguments (line 100)
    str_386504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 25), 'str', 'expected A to be like a square matrix')
    # Processing the call keyword arguments (line 100)
    kwargs_386505 = {}
    # Getting the type of 'ValueError' (line 100)
    ValueError_386503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 100)
    ValueError_call_result_386506 = invoke(stypy.reporting.localization.Localization(__file__, 100, 14), ValueError_386503, *[str_386504], **kwargs_386505)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 100, 8), ValueError_call_result_386506, 'raise parameter', BaseException)
    # SSA join for if statement (line 99)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 104):
    
    # Assigning a Call to a Name (line 104):
    
    # Call to ones(...): (line 104)
    # Processing the call arguments (line 104)
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_386509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    
    # Obtaining the type of the subscript
    int_386510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 25), 'int')
    # Getting the type of 'A' (line 104)
    A_386511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'A', False)
    # Obtaining the member 'shape' of a type (line 104)
    shape_386512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 17), A_386511, 'shape')
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___386513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 17), shape_386512, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_386514 = invoke(stypy.reporting.localization.Localization(__file__, 104, 17), getitem___386513, int_386510)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 17), tuple_386509, subscript_call_result_386514)
    # Adding element type (line 104)
    int_386515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 17), tuple_386509, int_386515)
    
    # Processing the call keyword arguments (line 104)
    # Getting the type of 'float' (line 104)
    float_386516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'float', False)
    keyword_386517 = float_386516
    kwargs_386518 = {'dtype': keyword_386517}
    # Getting the type of 'np' (line 104)
    np_386507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 104)
    ones_386508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), np_386507, 'ones')
    # Calling ones(args, kwargs) (line 104)
    ones_call_result_386519 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), ones_386508, *[tuple_386509], **kwargs_386518)
    
    # Assigning a type to the variable 'v' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'v', ones_call_result_386519)
    
    # Assigning a Attribute to a Name (line 105):
    
    # Assigning a Attribute to a Name (line 105):
    # Getting the type of 'A' (line 105)
    A_386520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'A')
    # Obtaining the member 'T' of a type (line 105)
    T_386521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), A_386520, 'T')
    # Assigning a type to the variable 'M' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'M', T_386521)
    
    
    # Call to range(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'p' (line 106)
    p_386523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'p', False)
    # Processing the call keyword arguments (line 106)
    kwargs_386524 = {}
    # Getting the type of 'range' (line 106)
    range_386522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'range', False)
    # Calling range(args, kwargs) (line 106)
    range_call_result_386525 = invoke(stypy.reporting.localization.Localization(__file__, 106, 13), range_386522, *[p_386523], **kwargs_386524)
    
    # Testing the type of a for loop iterable (line 106)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 106, 4), range_call_result_386525)
    # Getting the type of the for loop variable (line 106)
    for_loop_var_386526 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 106, 4), range_call_result_386525)
    # Assigning a type to the variable 'i' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'i', for_loop_var_386526)
    # SSA begins for a for statement (line 106)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 107):
    
    # Assigning a Call to a Name (line 107):
    
    # Call to dot(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'v' (line 107)
    v_386529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 18), 'v', False)
    # Processing the call keyword arguments (line 107)
    kwargs_386530 = {}
    # Getting the type of 'M' (line 107)
    M_386527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'M', False)
    # Obtaining the member 'dot' of a type (line 107)
    dot_386528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), M_386527, 'dot')
    # Calling dot(args, kwargs) (line 107)
    dot_call_result_386531 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), dot_386528, *[v_386529], **kwargs_386530)
    
    # Assigning a type to the variable 'v' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'v', dot_call_result_386531)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to max(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'v' (line 108)
    v_386533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'v', False)
    # Processing the call keyword arguments (line 108)
    kwargs_386534 = {}
    # Getting the type of 'max' (line 108)
    max_386532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'max', False)
    # Calling max(args, kwargs) (line 108)
    max_call_result_386535 = invoke(stypy.reporting.localization.Localization(__file__, 108, 11), max_386532, *[v_386533], **kwargs_386534)
    
    # Assigning a type to the variable 'stypy_return_type' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type', max_call_result_386535)
    
    # ################# End of '_onenorm_matrix_power_nnm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_onenorm_matrix_power_nnm' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_386536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_386536)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_onenorm_matrix_power_nnm'
    return stypy_return_type_386536

# Assigning a type to the variable '_onenorm_matrix_power_nnm' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), '_onenorm_matrix_power_nnm', _onenorm_matrix_power_nnm)

@norecursion
def _onenorm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_onenorm'
    module_type_store = module_type_store.open_function_context('_onenorm', 111, 0, False)
    
    # Passed parameters checking function
    _onenorm.stypy_localization = localization
    _onenorm.stypy_type_of_self = None
    _onenorm.stypy_type_store = module_type_store
    _onenorm.stypy_function_name = '_onenorm'
    _onenorm.stypy_param_names_list = ['A']
    _onenorm.stypy_varargs_param_name = None
    _onenorm.stypy_kwargs_param_name = None
    _onenorm.stypy_call_defaults = defaults
    _onenorm.stypy_call_varargs = varargs
    _onenorm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_onenorm', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_onenorm', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_onenorm(...)' code ##################

    
    
    # Call to isspmatrix(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'A' (line 114)
    A_386540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 31), 'A', False)
    # Processing the call keyword arguments (line 114)
    kwargs_386541 = {}
    # Getting the type of 'scipy' (line 114)
    scipy_386537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 7), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 114)
    sparse_386538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 7), scipy_386537, 'sparse')
    # Obtaining the member 'isspmatrix' of a type (line 114)
    isspmatrix_386539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 7), sparse_386538, 'isspmatrix')
    # Calling isspmatrix(args, kwargs) (line 114)
    isspmatrix_call_result_386542 = invoke(stypy.reporting.localization.Localization(__file__, 114, 7), isspmatrix_386539, *[A_386540], **kwargs_386541)
    
    # Testing the type of an if condition (line 114)
    if_condition_386543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 4), isspmatrix_call_result_386542)
    # Assigning a type to the variable 'if_condition_386543' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'if_condition_386543', if_condition_386543)
    # SSA begins for if statement (line 114)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to max(...): (line 115)
    # Processing the call arguments (line 115)
    
    # Call to sum(...): (line 115)
    # Processing the call keyword arguments (line 115)
    int_386550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 35), 'int')
    keyword_386551 = int_386550
    kwargs_386552 = {'axis': keyword_386551}
    
    # Call to abs(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'A' (line 115)
    A_386546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 23), 'A', False)
    # Processing the call keyword arguments (line 115)
    kwargs_386547 = {}
    # Getting the type of 'abs' (line 115)
    abs_386545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'abs', False)
    # Calling abs(args, kwargs) (line 115)
    abs_call_result_386548 = invoke(stypy.reporting.localization.Localization(__file__, 115, 19), abs_386545, *[A_386546], **kwargs_386547)
    
    # Obtaining the member 'sum' of a type (line 115)
    sum_386549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 19), abs_call_result_386548, 'sum')
    # Calling sum(args, kwargs) (line 115)
    sum_call_result_386553 = invoke(stypy.reporting.localization.Localization(__file__, 115, 19), sum_386549, *[], **kwargs_386552)
    
    # Obtaining the member 'flat' of a type (line 115)
    flat_386554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 19), sum_call_result_386553, 'flat')
    # Processing the call keyword arguments (line 115)
    kwargs_386555 = {}
    # Getting the type of 'max' (line 115)
    max_386544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'max', False)
    # Calling max(args, kwargs) (line 115)
    max_call_result_386556 = invoke(stypy.reporting.localization.Localization(__file__, 115, 15), max_386544, *[flat_386554], **kwargs_386555)
    
    # Assigning a type to the variable 'stypy_return_type' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'stypy_return_type', max_call_result_386556)
    # SSA branch for the else part of an if statement (line 114)
    module_type_store.open_ssa_branch('else')
    
    # Call to norm(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'A' (line 117)
    A_386560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 30), 'A', False)
    int_386561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 33), 'int')
    # Processing the call keyword arguments (line 117)
    kwargs_386562 = {}
    # Getting the type of 'np' (line 117)
    np_386557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'np', False)
    # Obtaining the member 'linalg' of a type (line 117)
    linalg_386558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), np_386557, 'linalg')
    # Obtaining the member 'norm' of a type (line 117)
    norm_386559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), linalg_386558, 'norm')
    # Calling norm(args, kwargs) (line 117)
    norm_call_result_386563 = invoke(stypy.reporting.localization.Localization(__file__, 117, 15), norm_386559, *[A_386560, int_386561], **kwargs_386562)
    
    # Assigning a type to the variable 'stypy_return_type' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', norm_call_result_386563)
    # SSA join for if statement (line 114)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_onenorm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_onenorm' in the type store
    # Getting the type of 'stypy_return_type' (line 111)
    stypy_return_type_386564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_386564)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_onenorm'
    return stypy_return_type_386564

# Assigning a type to the variable '_onenorm' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), '_onenorm', _onenorm)

@norecursion
def _ident_like(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_ident_like'
    module_type_store = module_type_store.open_function_context('_ident_like', 120, 0, False)
    
    # Passed parameters checking function
    _ident_like.stypy_localization = localization
    _ident_like.stypy_type_of_self = None
    _ident_like.stypy_type_store = module_type_store
    _ident_like.stypy_function_name = '_ident_like'
    _ident_like.stypy_param_names_list = ['A']
    _ident_like.stypy_varargs_param_name = None
    _ident_like.stypy_kwargs_param_name = None
    _ident_like.stypy_call_defaults = defaults
    _ident_like.stypy_call_varargs = varargs
    _ident_like.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_ident_like', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_ident_like', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_ident_like(...)' code ##################

    
    
    # Call to isspmatrix(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'A' (line 123)
    A_386568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 31), 'A', False)
    # Processing the call keyword arguments (line 123)
    kwargs_386569 = {}
    # Getting the type of 'scipy' (line 123)
    scipy_386565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 7), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 123)
    sparse_386566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 7), scipy_386565, 'sparse')
    # Obtaining the member 'isspmatrix' of a type (line 123)
    isspmatrix_386567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 7), sparse_386566, 'isspmatrix')
    # Calling isspmatrix(args, kwargs) (line 123)
    isspmatrix_call_result_386570 = invoke(stypy.reporting.localization.Localization(__file__, 123, 7), isspmatrix_386567, *[A_386568], **kwargs_386569)
    
    # Testing the type of an if condition (line 123)
    if_condition_386571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 4), isspmatrix_call_result_386570)
    # Assigning a type to the variable 'if_condition_386571' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'if_condition_386571', if_condition_386571)
    # SSA begins for if statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to eye(...): (line 124)
    # Processing the call arguments (line 124)
    
    # Obtaining the type of the subscript
    int_386576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 50), 'int')
    # Getting the type of 'A' (line 124)
    A_386577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 42), 'A', False)
    # Obtaining the member 'shape' of a type (line 124)
    shape_386578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 42), A_386577, 'shape')
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___386579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 42), shape_386578, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_386580 = invoke(stypy.reporting.localization.Localization(__file__, 124, 42), getitem___386579, int_386576)
    
    
    # Obtaining the type of the subscript
    int_386581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 62), 'int')
    # Getting the type of 'A' (line 124)
    A_386582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 54), 'A', False)
    # Obtaining the member 'shape' of a type (line 124)
    shape_386583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 54), A_386582, 'shape')
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___386584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 54), shape_386583, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_386585 = invoke(stypy.reporting.localization.Localization(__file__, 124, 54), getitem___386584, int_386581)
    
    # Processing the call keyword arguments (line 124)
    # Getting the type of 'A' (line 125)
    A_386586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 22), 'A', False)
    # Obtaining the member 'dtype' of a type (line 125)
    dtype_386587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 22), A_386586, 'dtype')
    keyword_386588 = dtype_386587
    # Getting the type of 'A' (line 125)
    A_386589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 38), 'A', False)
    # Obtaining the member 'format' of a type (line 125)
    format_386590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 38), A_386589, 'format')
    keyword_386591 = format_386590
    kwargs_386592 = {'dtype': keyword_386588, 'format': keyword_386591}
    # Getting the type of 'scipy' (line 124)
    scipy_386572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 124)
    sparse_386573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 15), scipy_386572, 'sparse')
    # Obtaining the member 'construct' of a type (line 124)
    construct_386574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 15), sparse_386573, 'construct')
    # Obtaining the member 'eye' of a type (line 124)
    eye_386575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 15), construct_386574, 'eye')
    # Calling eye(args, kwargs) (line 124)
    eye_call_result_386593 = invoke(stypy.reporting.localization.Localization(__file__, 124, 15), eye_386575, *[subscript_call_result_386580, subscript_call_result_386585], **kwargs_386592)
    
    # Assigning a type to the variable 'stypy_return_type' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'stypy_return_type', eye_call_result_386593)
    # SSA branch for the else part of an if statement (line 123)
    module_type_store.open_ssa_branch('else')
    
    # Call to eye(...): (line 127)
    # Processing the call arguments (line 127)
    
    # Obtaining the type of the subscript
    int_386596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 30), 'int')
    # Getting the type of 'A' (line 127)
    A_386597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 22), 'A', False)
    # Obtaining the member 'shape' of a type (line 127)
    shape_386598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 22), A_386597, 'shape')
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___386599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 22), shape_386598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_386600 = invoke(stypy.reporting.localization.Localization(__file__, 127, 22), getitem___386599, int_386596)
    
    
    # Obtaining the type of the subscript
    int_386601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 42), 'int')
    # Getting the type of 'A' (line 127)
    A_386602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 34), 'A', False)
    # Obtaining the member 'shape' of a type (line 127)
    shape_386603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 34), A_386602, 'shape')
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___386604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 34), shape_386603, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_386605 = invoke(stypy.reporting.localization.Localization(__file__, 127, 34), getitem___386604, int_386601)
    
    # Processing the call keyword arguments (line 127)
    # Getting the type of 'A' (line 127)
    A_386606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 52), 'A', False)
    # Obtaining the member 'dtype' of a type (line 127)
    dtype_386607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 52), A_386606, 'dtype')
    keyword_386608 = dtype_386607
    kwargs_386609 = {'dtype': keyword_386608}
    # Getting the type of 'np' (line 127)
    np_386594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'np', False)
    # Obtaining the member 'eye' of a type (line 127)
    eye_386595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 15), np_386594, 'eye')
    # Calling eye(args, kwargs) (line 127)
    eye_call_result_386610 = invoke(stypy.reporting.localization.Localization(__file__, 127, 15), eye_386595, *[subscript_call_result_386600, subscript_call_result_386605], **kwargs_386609)
    
    # Assigning a type to the variable 'stypy_return_type' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'stypy_return_type', eye_call_result_386610)
    # SSA join for if statement (line 123)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_ident_like(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_ident_like' in the type store
    # Getting the type of 'stypy_return_type' (line 120)
    stypy_return_type_386611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_386611)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_ident_like'
    return stypy_return_type_386611

# Assigning a type to the variable '_ident_like' (line 120)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), '_ident_like', _ident_like)

@norecursion
def _is_upper_triangular(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_is_upper_triangular'
    module_type_store = module_type_store.open_function_context('_is_upper_triangular', 130, 0, False)
    
    # Passed parameters checking function
    _is_upper_triangular.stypy_localization = localization
    _is_upper_triangular.stypy_type_of_self = None
    _is_upper_triangular.stypy_type_store = module_type_store
    _is_upper_triangular.stypy_function_name = '_is_upper_triangular'
    _is_upper_triangular.stypy_param_names_list = ['A']
    _is_upper_triangular.stypy_varargs_param_name = None
    _is_upper_triangular.stypy_kwargs_param_name = None
    _is_upper_triangular.stypy_call_defaults = defaults
    _is_upper_triangular.stypy_call_varargs = varargs
    _is_upper_triangular.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_is_upper_triangular', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_is_upper_triangular', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_is_upper_triangular(...)' code ##################

    
    
    # Call to isspmatrix(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'A' (line 132)
    A_386613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 18), 'A', False)
    # Processing the call keyword arguments (line 132)
    kwargs_386614 = {}
    # Getting the type of 'isspmatrix' (line 132)
    isspmatrix_386612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 7), 'isspmatrix', False)
    # Calling isspmatrix(args, kwargs) (line 132)
    isspmatrix_call_result_386615 = invoke(stypy.reporting.localization.Localization(__file__, 132, 7), isspmatrix_386612, *[A_386613], **kwargs_386614)
    
    # Testing the type of an if condition (line 132)
    if_condition_386616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 4), isspmatrix_call_result_386615)
    # Assigning a type to the variable 'if_condition_386616' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'if_condition_386616', if_condition_386616)
    # SSA begins for if statement (line 132)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 133):
    
    # Assigning a Call to a Name (line 133):
    
    # Call to tril(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'A' (line 133)
    A_386620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 39), 'A', False)
    int_386621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 42), 'int')
    # Processing the call keyword arguments (line 133)
    kwargs_386622 = {}
    # Getting the type of 'scipy' (line 133)
    scipy_386617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 21), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 133)
    sparse_386618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 21), scipy_386617, 'sparse')
    # Obtaining the member 'tril' of a type (line 133)
    tril_386619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 21), sparse_386618, 'tril')
    # Calling tril(args, kwargs) (line 133)
    tril_call_result_386623 = invoke(stypy.reporting.localization.Localization(__file__, 133, 21), tril_386619, *[A_386620, int_386621], **kwargs_386622)
    
    # Assigning a type to the variable 'lower_part' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'lower_part', tril_call_result_386623)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'lower_part' (line 136)
    lower_part_386624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'lower_part')
    # Obtaining the member 'nnz' of a type (line 136)
    nnz_386625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 15), lower_part_386624, 'nnz')
    int_386626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 33), 'int')
    # Applying the binary operator '==' (line 136)
    result_eq_386627 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 15), '==', nnz_386625, int_386626)
    
    
    
    # Call to count_nonzero(...): (line 136)
    # Processing the call keyword arguments (line 136)
    kwargs_386630 = {}
    # Getting the type of 'lower_part' (line 136)
    lower_part_386628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 38), 'lower_part', False)
    # Obtaining the member 'count_nonzero' of a type (line 136)
    count_nonzero_386629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 38), lower_part_386628, 'count_nonzero')
    # Calling count_nonzero(args, kwargs) (line 136)
    count_nonzero_call_result_386631 = invoke(stypy.reporting.localization.Localization(__file__, 136, 38), count_nonzero_386629, *[], **kwargs_386630)
    
    int_386632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 68), 'int')
    # Applying the binary operator '==' (line 136)
    result_eq_386633 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 38), '==', count_nonzero_call_result_386631, int_386632)
    
    # Applying the binary operator 'or' (line 136)
    result_or_keyword_386634 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 15), 'or', result_eq_386627, result_eq_386633)
    
    # Assigning a type to the variable 'stypy_return_type' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'stypy_return_type', result_or_keyword_386634)
    # SSA branch for the else part of an if statement (line 132)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to any(...): (line 138)
    # Processing the call keyword arguments (line 138)
    kwargs_386642 = {}
    
    # Call to tril(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'A' (line 138)
    A_386637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 27), 'A', False)
    int_386638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 30), 'int')
    # Processing the call keyword arguments (line 138)
    kwargs_386639 = {}
    # Getting the type of 'np' (line 138)
    np_386635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'np', False)
    # Obtaining the member 'tril' of a type (line 138)
    tril_386636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 19), np_386635, 'tril')
    # Calling tril(args, kwargs) (line 138)
    tril_call_result_386640 = invoke(stypy.reporting.localization.Localization(__file__, 138, 19), tril_386636, *[A_386637, int_386638], **kwargs_386639)
    
    # Obtaining the member 'any' of a type (line 138)
    any_386641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 19), tril_call_result_386640, 'any')
    # Calling any(args, kwargs) (line 138)
    any_call_result_386643 = invoke(stypy.reporting.localization.Localization(__file__, 138, 19), any_386641, *[], **kwargs_386642)
    
    # Applying the 'not' unary operator (line 138)
    result_not__386644 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 15), 'not', any_call_result_386643)
    
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type', result_not__386644)
    # SSA join for if statement (line 132)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_is_upper_triangular(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_is_upper_triangular' in the type store
    # Getting the type of 'stypy_return_type' (line 130)
    stypy_return_type_386645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_386645)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_is_upper_triangular'
    return stypy_return_type_386645

# Assigning a type to the variable '_is_upper_triangular' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), '_is_upper_triangular', _is_upper_triangular)

@norecursion
def _smart_matrix_product(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 141)
    None_386646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 38), 'None')
    # Getting the type of 'None' (line 141)
    None_386647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 54), 'None')
    defaults = [None_386646, None_386647]
    # Create a new context for function '_smart_matrix_product'
    module_type_store = module_type_store.open_function_context('_smart_matrix_product', 141, 0, False)
    
    # Passed parameters checking function
    _smart_matrix_product.stypy_localization = localization
    _smart_matrix_product.stypy_type_of_self = None
    _smart_matrix_product.stypy_type_store = module_type_store
    _smart_matrix_product.stypy_function_name = '_smart_matrix_product'
    _smart_matrix_product.stypy_param_names_list = ['A', 'B', 'alpha', 'structure']
    _smart_matrix_product.stypy_varargs_param_name = None
    _smart_matrix_product.stypy_kwargs_param_name = None
    _smart_matrix_product.stypy_call_defaults = defaults
    _smart_matrix_product.stypy_call_varargs = varargs
    _smart_matrix_product.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_smart_matrix_product', ['A', 'B', 'alpha', 'structure'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_smart_matrix_product', localization, ['A', 'B', 'alpha', 'structure'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_smart_matrix_product(...)' code ##################

    str_386648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, (-1)), 'str', '\n    A matrix product that knows about sparse and structured matrices.\n\n    Parameters\n    ----------\n    A : 2d ndarray\n        First matrix.\n    B : 2d ndarray\n        Second matrix.\n    alpha : float\n        The matrix product will be scaled by this constant.\n    structure : str, optional\n        A string describing the structure of both matrices `A` and `B`.\n        Only `upper_triangular` is currently supported.\n\n    Returns\n    -------\n    M : 2d ndarray\n        Matrix product of A and B.\n\n    ')
    
    
    
    # Call to len(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'A' (line 163)
    A_386650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 163)
    shape_386651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 11), A_386650, 'shape')
    # Processing the call keyword arguments (line 163)
    kwargs_386652 = {}
    # Getting the type of 'len' (line 163)
    len_386649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 7), 'len', False)
    # Calling len(args, kwargs) (line 163)
    len_call_result_386653 = invoke(stypy.reporting.localization.Localization(__file__, 163, 7), len_386649, *[shape_386651], **kwargs_386652)
    
    int_386654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 23), 'int')
    # Applying the binary operator '!=' (line 163)
    result_ne_386655 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 7), '!=', len_call_result_386653, int_386654)
    
    # Testing the type of an if condition (line 163)
    if_condition_386656 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 4), result_ne_386655)
    # Assigning a type to the variable 'if_condition_386656' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'if_condition_386656', if_condition_386656)
    # SSA begins for if statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 164)
    # Processing the call arguments (line 164)
    str_386658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 25), 'str', 'expected A to be a rectangular matrix')
    # Processing the call keyword arguments (line 164)
    kwargs_386659 = {}
    # Getting the type of 'ValueError' (line 164)
    ValueError_386657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 164)
    ValueError_call_result_386660 = invoke(stypy.reporting.localization.Localization(__file__, 164, 14), ValueError_386657, *[str_386658], **kwargs_386659)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 164, 8), ValueError_call_result_386660, 'raise parameter', BaseException)
    # SSA join for if statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'B' (line 165)
    B_386662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'B', False)
    # Obtaining the member 'shape' of a type (line 165)
    shape_386663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 11), B_386662, 'shape')
    # Processing the call keyword arguments (line 165)
    kwargs_386664 = {}
    # Getting the type of 'len' (line 165)
    len_386661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 7), 'len', False)
    # Calling len(args, kwargs) (line 165)
    len_call_result_386665 = invoke(stypy.reporting.localization.Localization(__file__, 165, 7), len_386661, *[shape_386663], **kwargs_386664)
    
    int_386666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 23), 'int')
    # Applying the binary operator '!=' (line 165)
    result_ne_386667 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 7), '!=', len_call_result_386665, int_386666)
    
    # Testing the type of an if condition (line 165)
    if_condition_386668 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 4), result_ne_386667)
    # Assigning a type to the variable 'if_condition_386668' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'if_condition_386668', if_condition_386668)
    # SSA begins for if statement (line 165)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 166)
    # Processing the call arguments (line 166)
    str_386670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 25), 'str', 'expected B to be a rectangular matrix')
    # Processing the call keyword arguments (line 166)
    kwargs_386671 = {}
    # Getting the type of 'ValueError' (line 166)
    ValueError_386669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 166)
    ValueError_call_result_386672 = invoke(stypy.reporting.localization.Localization(__file__, 166, 14), ValueError_386669, *[str_386670], **kwargs_386671)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 166, 8), ValueError_call_result_386672, 'raise parameter', BaseException)
    # SSA join for if statement (line 165)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 167):
    
    # Assigning a Name to a Name (line 167):
    # Getting the type of 'None' (line 167)
    None_386673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'None')
    # Assigning a type to the variable 'f' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'f', None_386673)
    
    
    # Getting the type of 'structure' (line 168)
    structure_386674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 7), 'structure')
    # Getting the type of 'UPPER_TRIANGULAR' (line 168)
    UPPER_TRIANGULAR_386675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'UPPER_TRIANGULAR')
    # Applying the binary operator '==' (line 168)
    result_eq_386676 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 7), '==', structure_386674, UPPER_TRIANGULAR_386675)
    
    # Testing the type of an if condition (line 168)
    if_condition_386677 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 4), result_eq_386676)
    # Assigning a type to the variable 'if_condition_386677' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'if_condition_386677', if_condition_386677)
    # SSA begins for if statement (line 168)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    
    # Call to isspmatrix(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'A' (line 169)
    A_386679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 26), 'A', False)
    # Processing the call keyword arguments (line 169)
    kwargs_386680 = {}
    # Getting the type of 'isspmatrix' (line 169)
    isspmatrix_386678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'isspmatrix', False)
    # Calling isspmatrix(args, kwargs) (line 169)
    isspmatrix_call_result_386681 = invoke(stypy.reporting.localization.Localization(__file__, 169, 15), isspmatrix_386678, *[A_386679], **kwargs_386680)
    
    # Applying the 'not' unary operator (line 169)
    result_not__386682 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 11), 'not', isspmatrix_call_result_386681)
    
    
    
    # Call to isspmatrix(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'B' (line 169)
    B_386684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 48), 'B', False)
    # Processing the call keyword arguments (line 169)
    kwargs_386685 = {}
    # Getting the type of 'isspmatrix' (line 169)
    isspmatrix_386683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 37), 'isspmatrix', False)
    # Calling isspmatrix(args, kwargs) (line 169)
    isspmatrix_call_result_386686 = invoke(stypy.reporting.localization.Localization(__file__, 169, 37), isspmatrix_386683, *[B_386684], **kwargs_386685)
    
    # Applying the 'not' unary operator (line 169)
    result_not__386687 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 33), 'not', isspmatrix_call_result_386686)
    
    # Applying the binary operator 'and' (line 169)
    result_and_keyword_386688 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 11), 'and', result_not__386682, result_not__386687)
    
    # Testing the type of an if condition (line 169)
    if_condition_386689 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 8), result_and_keyword_386688)
    # Assigning a type to the variable 'if_condition_386689' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'if_condition_386689', if_condition_386689)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 170):
    
    # Assigning a Subscript to a Name (line 170):
    
    # Obtaining the type of the subscript
    int_386690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 12), 'int')
    
    # Call to get_blas_funcs(...): (line 170)
    # Processing the call arguments (line 170)
    
    # Obtaining an instance of the builtin type 'tuple' (line 170)
    tuple_386694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 170)
    # Adding element type (line 170)
    str_386695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 46), 'str', 'trmm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 46), tuple_386694, str_386695)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 170)
    tuple_386696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 170)
    # Adding element type (line 170)
    # Getting the type of 'A' (line 170)
    A_386697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 57), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 57), tuple_386696, A_386697)
    # Adding element type (line 170)
    # Getting the type of 'B' (line 170)
    B_386698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 60), 'B', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 57), tuple_386696, B_386698)
    
    # Processing the call keyword arguments (line 170)
    kwargs_386699 = {}
    # Getting the type of 'scipy' (line 170)
    scipy_386691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 17), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 170)
    linalg_386692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 17), scipy_386691, 'linalg')
    # Obtaining the member 'get_blas_funcs' of a type (line 170)
    get_blas_funcs_386693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 17), linalg_386692, 'get_blas_funcs')
    # Calling get_blas_funcs(args, kwargs) (line 170)
    get_blas_funcs_call_result_386700 = invoke(stypy.reporting.localization.Localization(__file__, 170, 17), get_blas_funcs_386693, *[tuple_386694, tuple_386696], **kwargs_386699)
    
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___386701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 12), get_blas_funcs_call_result_386700, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_386702 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), getitem___386701, int_386690)
    
    # Assigning a type to the variable 'tuple_var_assignment_386402' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'tuple_var_assignment_386402', subscript_call_result_386702)
    
    # Assigning a Name to a Name (line 170):
    # Getting the type of 'tuple_var_assignment_386402' (line 170)
    tuple_var_assignment_386402_386703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'tuple_var_assignment_386402')
    # Assigning a type to the variable 'f' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'f', tuple_var_assignment_386402_386703)
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 168)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 171)
    # Getting the type of 'f' (line 171)
    f_386704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'f')
    # Getting the type of 'None' (line 171)
    None_386705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'None')
    
    (may_be_386706, more_types_in_union_386707) = may_not_be_none(f_386704, None_386705)

    if may_be_386706:

        if more_types_in_union_386707:
            # Runtime conditional SSA (line 171)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 172)
        # Getting the type of 'alpha' (line 172)
        alpha_386708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'alpha')
        # Getting the type of 'None' (line 172)
        None_386709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'None')
        
        (may_be_386710, more_types_in_union_386711) = may_be_none(alpha_386708, None_386709)

        if may_be_386710:

            if more_types_in_union_386711:
                # Runtime conditional SSA (line 172)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 173):
            
            # Assigning a Num to a Name (line 173):
            float_386712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 20), 'float')
            # Assigning a type to the variable 'alpha' (line 173)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'alpha', float_386712)

            if more_types_in_union_386711:
                # SSA join for if statement (line 172)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 174):
        
        # Assigning a Call to a Name (line 174):
        
        # Call to f(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'alpha' (line 174)
        alpha_386714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'alpha', False)
        # Getting the type of 'A' (line 174)
        A_386715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 23), 'A', False)
        # Getting the type of 'B' (line 174)
        B_386716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 26), 'B', False)
        # Processing the call keyword arguments (line 174)
        kwargs_386717 = {}
        # Getting the type of 'f' (line 174)
        f_386713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 14), 'f', False)
        # Calling f(args, kwargs) (line 174)
        f_call_result_386718 = invoke(stypy.reporting.localization.Localization(__file__, 174, 14), f_386713, *[alpha_386714, A_386715, B_386716], **kwargs_386717)
        
        # Assigning a type to the variable 'out' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'out', f_call_result_386718)

        if more_types_in_union_386707:
            # Runtime conditional SSA for else branch (line 171)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_386706) or more_types_in_union_386707):
        
        # Type idiom detected: calculating its left and rigth part (line 176)
        # Getting the type of 'alpha' (line 176)
        alpha_386719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'alpha')
        # Getting the type of 'None' (line 176)
        None_386720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 20), 'None')
        
        (may_be_386721, more_types_in_union_386722) = may_be_none(alpha_386719, None_386720)

        if may_be_386721:

            if more_types_in_union_386722:
                # Runtime conditional SSA (line 176)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 177):
            
            # Assigning a Call to a Name (line 177):
            
            # Call to dot(...): (line 177)
            # Processing the call arguments (line 177)
            # Getting the type of 'B' (line 177)
            B_386725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'B', False)
            # Processing the call keyword arguments (line 177)
            kwargs_386726 = {}
            # Getting the type of 'A' (line 177)
            A_386723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 18), 'A', False)
            # Obtaining the member 'dot' of a type (line 177)
            dot_386724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 18), A_386723, 'dot')
            # Calling dot(args, kwargs) (line 177)
            dot_call_result_386727 = invoke(stypy.reporting.localization.Localization(__file__, 177, 18), dot_386724, *[B_386725], **kwargs_386726)
            
            # Assigning a type to the variable 'out' (line 177)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'out', dot_call_result_386727)

            if more_types_in_union_386722:
                # Runtime conditional SSA for else branch (line 176)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_386721) or more_types_in_union_386722):
            
            # Assigning a BinOp to a Name (line 179):
            
            # Assigning a BinOp to a Name (line 179):
            # Getting the type of 'alpha' (line 179)
            alpha_386728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 18), 'alpha')
            
            # Call to dot(...): (line 179)
            # Processing the call arguments (line 179)
            # Getting the type of 'B' (line 179)
            B_386731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 32), 'B', False)
            # Processing the call keyword arguments (line 179)
            kwargs_386732 = {}
            # Getting the type of 'A' (line 179)
            A_386729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 26), 'A', False)
            # Obtaining the member 'dot' of a type (line 179)
            dot_386730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 26), A_386729, 'dot')
            # Calling dot(args, kwargs) (line 179)
            dot_call_result_386733 = invoke(stypy.reporting.localization.Localization(__file__, 179, 26), dot_386730, *[B_386731], **kwargs_386732)
            
            # Applying the binary operator '*' (line 179)
            result_mul_386734 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 18), '*', alpha_386728, dot_call_result_386733)
            
            # Assigning a type to the variable 'out' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'out', result_mul_386734)

            if (may_be_386721 and more_types_in_union_386722):
                # SSA join for if statement (line 176)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_386706 and more_types_in_union_386707):
            # SSA join for if statement (line 171)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'out' (line 180)
    out_386735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type', out_386735)
    
    # ################# End of '_smart_matrix_product(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_smart_matrix_product' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_386736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_386736)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_smart_matrix_product'
    return stypy_return_type_386736

# Assigning a type to the variable '_smart_matrix_product' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), '_smart_matrix_product', _smart_matrix_product)
# Declaration of the 'MatrixPowerOperator' class
# Getting the type of 'LinearOperator' (line 183)
LinearOperator_386737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 26), 'LinearOperator')

class MatrixPowerOperator(LinearOperator_386737, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 185)
        None_386738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 39), 'None')
        defaults = [None_386738]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixPowerOperator.__init__', ['A', 'p', 'structure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['A', 'p', 'structure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'A' (line 186)
        A_386739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 11), 'A')
        # Obtaining the member 'ndim' of a type (line 186)
        ndim_386740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 11), A_386739, 'ndim')
        int_386741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 21), 'int')
        # Applying the binary operator '!=' (line 186)
        result_ne_386742 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 11), '!=', ndim_386740, int_386741)
        
        
        
        # Obtaining the type of the subscript
        int_386743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 34), 'int')
        # Getting the type of 'A' (line 186)
        A_386744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 26), 'A')
        # Obtaining the member 'shape' of a type (line 186)
        shape_386745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 26), A_386744, 'shape')
        # Obtaining the member '__getitem__' of a type (line 186)
        getitem___386746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 26), shape_386745, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 186)
        subscript_call_result_386747 = invoke(stypy.reporting.localization.Localization(__file__, 186, 26), getitem___386746, int_386743)
        
        
        # Obtaining the type of the subscript
        int_386748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 48), 'int')
        # Getting the type of 'A' (line 186)
        A_386749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 40), 'A')
        # Obtaining the member 'shape' of a type (line 186)
        shape_386750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 40), A_386749, 'shape')
        # Obtaining the member '__getitem__' of a type (line 186)
        getitem___386751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 40), shape_386750, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 186)
        subscript_call_result_386752 = invoke(stypy.reporting.localization.Localization(__file__, 186, 40), getitem___386751, int_386748)
        
        # Applying the binary operator '!=' (line 186)
        result_ne_386753 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 26), '!=', subscript_call_result_386747, subscript_call_result_386752)
        
        # Applying the binary operator 'or' (line 186)
        result_or_keyword_386754 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 11), 'or', result_ne_386742, result_ne_386753)
        
        # Testing the type of an if condition (line 186)
        if_condition_386755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 8), result_or_keyword_386754)
        # Assigning a type to the variable 'if_condition_386755' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'if_condition_386755', if_condition_386755)
        # SSA begins for if statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 187)
        # Processing the call arguments (line 187)
        str_386757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 29), 'str', 'expected A to be like a square matrix')
        # Processing the call keyword arguments (line 187)
        kwargs_386758 = {}
        # Getting the type of 'ValueError' (line 187)
        ValueError_386756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 187)
        ValueError_call_result_386759 = invoke(stypy.reporting.localization.Localization(__file__, 187, 18), ValueError_386756, *[str_386757], **kwargs_386758)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 187, 12), ValueError_call_result_386759, 'raise parameter', BaseException)
        # SSA join for if statement (line 186)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'p' (line 188)
        p_386760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'p')
        int_386761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 15), 'int')
        # Applying the binary operator '<' (line 188)
        result_lt_386762 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 11), '<', p_386760, int_386761)
        
        # Testing the type of an if condition (line 188)
        if_condition_386763 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 8), result_lt_386762)
        # Assigning a type to the variable 'if_condition_386763' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'if_condition_386763', if_condition_386763)
        # SSA begins for if statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 189)
        # Processing the call arguments (line 189)
        str_386765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 29), 'str', 'expected p to be a non-negative integer')
        # Processing the call keyword arguments (line 189)
        kwargs_386766 = {}
        # Getting the type of 'ValueError' (line 189)
        ValueError_386764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 189)
        ValueError_call_result_386767 = invoke(stypy.reporting.localization.Localization(__file__, 189, 18), ValueError_386764, *[str_386765], **kwargs_386766)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 189, 12), ValueError_call_result_386767, 'raise parameter', BaseException)
        # SSA join for if statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 190):
        
        # Assigning a Name to a Attribute (line 190):
        # Getting the type of 'A' (line 190)
        A_386768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 18), 'A')
        # Getting the type of 'self' (line 190)
        self_386769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'self')
        # Setting the type of the member '_A' of a type (line 190)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), self_386769, '_A', A_386768)
        
        # Assigning a Name to a Attribute (line 191):
        
        # Assigning a Name to a Attribute (line 191):
        # Getting the type of 'p' (line 191)
        p_386770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 18), 'p')
        # Getting the type of 'self' (line 191)
        self_386771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'self')
        # Setting the type of the member '_p' of a type (line 191)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), self_386771, '_p', p_386770)
        
        # Assigning a Name to a Attribute (line 192):
        
        # Assigning a Name to a Attribute (line 192):
        # Getting the type of 'structure' (line 192)
        structure_386772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 26), 'structure')
        # Getting the type of 'self' (line 192)
        self_386773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self')
        # Setting the type of the member '_structure' of a type (line 192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_386773, '_structure', structure_386772)
        
        # Assigning a Attribute to a Attribute (line 193):
        
        # Assigning a Attribute to a Attribute (line 193):
        # Getting the type of 'A' (line 193)
        A_386774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 21), 'A')
        # Obtaining the member 'dtype' of a type (line 193)
        dtype_386775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 21), A_386774, 'dtype')
        # Getting the type of 'self' (line 193)
        self_386776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'self')
        # Setting the type of the member 'dtype' of a type (line 193)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), self_386776, 'dtype', dtype_386775)
        
        # Assigning a Attribute to a Attribute (line 194):
        
        # Assigning a Attribute to a Attribute (line 194):
        # Getting the type of 'A' (line 194)
        A_386777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 20), 'A')
        # Obtaining the member 'ndim' of a type (line 194)
        ndim_386778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 20), A_386777, 'ndim')
        # Getting the type of 'self' (line 194)
        self_386779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'self')
        # Setting the type of the member 'ndim' of a type (line 194)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), self_386779, 'ndim', ndim_386778)
        
        # Assigning a Attribute to a Attribute (line 195):
        
        # Assigning a Attribute to a Attribute (line 195):
        # Getting the type of 'A' (line 195)
        A_386780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 21), 'A')
        # Obtaining the member 'shape' of a type (line 195)
        shape_386781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 21), A_386780, 'shape')
        # Getting the type of 'self' (line 195)
        self_386782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'self')
        # Setting the type of the member 'shape' of a type (line 195)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), self_386782, 'shape', shape_386781)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matvec'
        module_type_store = module_type_store.open_function_context('_matvec', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatrixPowerOperator._matvec.__dict__.__setitem__('stypy_localization', localization)
        MatrixPowerOperator._matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatrixPowerOperator._matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatrixPowerOperator._matvec.__dict__.__setitem__('stypy_function_name', 'MatrixPowerOperator._matvec')
        MatrixPowerOperator._matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        MatrixPowerOperator._matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatrixPowerOperator._matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatrixPowerOperator._matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatrixPowerOperator._matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatrixPowerOperator._matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatrixPowerOperator._matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixPowerOperator._matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matvec(...)' code ##################

        
        
        # Call to range(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'self' (line 198)
        self_386784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'self', False)
        # Obtaining the member '_p' of a type (line 198)
        _p_386785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 23), self_386784, '_p')
        # Processing the call keyword arguments (line 198)
        kwargs_386786 = {}
        # Getting the type of 'range' (line 198)
        range_386783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'range', False)
        # Calling range(args, kwargs) (line 198)
        range_call_result_386787 = invoke(stypy.reporting.localization.Localization(__file__, 198, 17), range_386783, *[_p_386785], **kwargs_386786)
        
        # Testing the type of a for loop iterable (line 198)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 198, 8), range_call_result_386787)
        # Getting the type of the for loop variable (line 198)
        for_loop_var_386788 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 198, 8), range_call_result_386787)
        # Assigning a type to the variable 'i' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'i', for_loop_var_386788)
        # SSA begins for a for statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 199):
        
        # Assigning a Call to a Name (line 199):
        
        # Call to dot(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'x' (line 199)
        x_386792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'x', False)
        # Processing the call keyword arguments (line 199)
        kwargs_386793 = {}
        # Getting the type of 'self' (line 199)
        self_386789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'self', False)
        # Obtaining the member '_A' of a type (line 199)
        _A_386790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), self_386789, '_A')
        # Obtaining the member 'dot' of a type (line 199)
        dot_386791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), _A_386790, 'dot')
        # Calling dot(args, kwargs) (line 199)
        dot_call_result_386794 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), dot_386791, *[x_386792], **kwargs_386793)
        
        # Assigning a type to the variable 'x' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'x', dot_call_result_386794)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'x' (line 200)
        x_386795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'stypy_return_type', x_386795)
        
        # ################# End of '_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_386796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386796)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matvec'
        return stypy_return_type_386796


    @norecursion
    def _rmatvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rmatvec'
        module_type_store = module_type_store.open_function_context('_rmatvec', 202, 4, False)
        # Assigning a type to the variable 'self' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatrixPowerOperator._rmatvec.__dict__.__setitem__('stypy_localization', localization)
        MatrixPowerOperator._rmatvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatrixPowerOperator._rmatvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatrixPowerOperator._rmatvec.__dict__.__setitem__('stypy_function_name', 'MatrixPowerOperator._rmatvec')
        MatrixPowerOperator._rmatvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        MatrixPowerOperator._rmatvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatrixPowerOperator._rmatvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatrixPowerOperator._rmatvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatrixPowerOperator._rmatvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatrixPowerOperator._rmatvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatrixPowerOperator._rmatvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixPowerOperator._rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rmatvec(...)' code ##################

        
        # Assigning a Attribute to a Name (line 203):
        
        # Assigning a Attribute to a Name (line 203):
        # Getting the type of 'self' (line 203)
        self_386797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 14), 'self')
        # Obtaining the member '_A' of a type (line 203)
        _A_386798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 14), self_386797, '_A')
        # Obtaining the member 'T' of a type (line 203)
        T_386799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 14), _A_386798, 'T')
        # Assigning a type to the variable 'A_T' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'A_T', T_386799)
        
        # Assigning a Call to a Name (line 204):
        
        # Assigning a Call to a Name (line 204):
        
        # Call to ravel(...): (line 204)
        # Processing the call keyword arguments (line 204)
        kwargs_386802 = {}
        # Getting the type of 'x' (line 204)
        x_386800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'x', False)
        # Obtaining the member 'ravel' of a type (line 204)
        ravel_386801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), x_386800, 'ravel')
        # Calling ravel(args, kwargs) (line 204)
        ravel_call_result_386803 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), ravel_386801, *[], **kwargs_386802)
        
        # Assigning a type to the variable 'x' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'x', ravel_call_result_386803)
        
        
        # Call to range(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'self' (line 205)
        self_386805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 23), 'self', False)
        # Obtaining the member '_p' of a type (line 205)
        _p_386806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 23), self_386805, '_p')
        # Processing the call keyword arguments (line 205)
        kwargs_386807 = {}
        # Getting the type of 'range' (line 205)
        range_386804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 'range', False)
        # Calling range(args, kwargs) (line 205)
        range_call_result_386808 = invoke(stypy.reporting.localization.Localization(__file__, 205, 17), range_386804, *[_p_386806], **kwargs_386807)
        
        # Testing the type of a for loop iterable (line 205)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 205, 8), range_call_result_386808)
        # Getting the type of the for loop variable (line 205)
        for_loop_var_386809 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 205, 8), range_call_result_386808)
        # Assigning a type to the variable 'i' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'i', for_loop_var_386809)
        # SSA begins for a for statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 206):
        
        # Assigning a Call to a Name (line 206):
        
        # Call to dot(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'x' (line 206)
        x_386812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'x', False)
        # Processing the call keyword arguments (line 206)
        kwargs_386813 = {}
        # Getting the type of 'A_T' (line 206)
        A_T_386810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'A_T', False)
        # Obtaining the member 'dot' of a type (line 206)
        dot_386811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 16), A_T_386810, 'dot')
        # Calling dot(args, kwargs) (line 206)
        dot_call_result_386814 = invoke(stypy.reporting.localization.Localization(__file__, 206, 16), dot_386811, *[x_386812], **kwargs_386813)
        
        # Assigning a type to the variable 'x' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'x', dot_call_result_386814)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'x' (line 207)
        x_386815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', x_386815)
        
        # ################# End of '_rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 202)
        stypy_return_type_386816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386816)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rmatvec'
        return stypy_return_type_386816


    @norecursion
    def _matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matmat'
        module_type_store = module_type_store.open_function_context('_matmat', 209, 4, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatrixPowerOperator._matmat.__dict__.__setitem__('stypy_localization', localization)
        MatrixPowerOperator._matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatrixPowerOperator._matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatrixPowerOperator._matmat.__dict__.__setitem__('stypy_function_name', 'MatrixPowerOperator._matmat')
        MatrixPowerOperator._matmat.__dict__.__setitem__('stypy_param_names_list', ['X'])
        MatrixPowerOperator._matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatrixPowerOperator._matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatrixPowerOperator._matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatrixPowerOperator._matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatrixPowerOperator._matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatrixPowerOperator._matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixPowerOperator._matmat', ['X'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matmat', localization, ['X'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matmat(...)' code ##################

        
        
        # Call to range(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'self' (line 210)
        self_386818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 23), 'self', False)
        # Obtaining the member '_p' of a type (line 210)
        _p_386819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 23), self_386818, '_p')
        # Processing the call keyword arguments (line 210)
        kwargs_386820 = {}
        # Getting the type of 'range' (line 210)
        range_386817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 17), 'range', False)
        # Calling range(args, kwargs) (line 210)
        range_call_result_386821 = invoke(stypy.reporting.localization.Localization(__file__, 210, 17), range_386817, *[_p_386819], **kwargs_386820)
        
        # Testing the type of a for loop iterable (line 210)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 210, 8), range_call_result_386821)
        # Getting the type of the for loop variable (line 210)
        for_loop_var_386822 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 210, 8), range_call_result_386821)
        # Assigning a type to the variable 'i' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'i', for_loop_var_386822)
        # SSA begins for a for statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 211):
        
        # Assigning a Call to a Name (line 211):
        
        # Call to _smart_matrix_product(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'self' (line 211)
        self_386824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 38), 'self', False)
        # Obtaining the member '_A' of a type (line 211)
        _A_386825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 38), self_386824, '_A')
        # Getting the type of 'X' (line 211)
        X_386826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 47), 'X', False)
        # Processing the call keyword arguments (line 211)
        # Getting the type of 'self' (line 211)
        self_386827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 60), 'self', False)
        # Obtaining the member '_structure' of a type (line 211)
        _structure_386828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 60), self_386827, '_structure')
        keyword_386829 = _structure_386828
        kwargs_386830 = {'structure': keyword_386829}
        # Getting the type of '_smart_matrix_product' (line 211)
        _smart_matrix_product_386823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), '_smart_matrix_product', False)
        # Calling _smart_matrix_product(args, kwargs) (line 211)
        _smart_matrix_product_call_result_386831 = invoke(stypy.reporting.localization.Localization(__file__, 211, 16), _smart_matrix_product_386823, *[_A_386825, X_386826], **kwargs_386830)
        
        # Assigning a type to the variable 'X' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'X', _smart_matrix_product_call_result_386831)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'X' (line 212)
        X_386832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'X')
        # Assigning a type to the variable 'stypy_return_type' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'stypy_return_type', X_386832)
        
        # ################# End of '_matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_386833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386833)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matmat'
        return stypy_return_type_386833


    @norecursion
    def T(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'T'
        module_type_store = module_type_store.open_function_context('T', 214, 4, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatrixPowerOperator.T.__dict__.__setitem__('stypy_localization', localization)
        MatrixPowerOperator.T.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatrixPowerOperator.T.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatrixPowerOperator.T.__dict__.__setitem__('stypy_function_name', 'MatrixPowerOperator.T')
        MatrixPowerOperator.T.__dict__.__setitem__('stypy_param_names_list', [])
        MatrixPowerOperator.T.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatrixPowerOperator.T.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatrixPowerOperator.T.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatrixPowerOperator.T.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatrixPowerOperator.T.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatrixPowerOperator.T.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixPowerOperator.T', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'T', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'T(...)' code ##################

        
        # Call to MatrixPowerOperator(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'self' (line 216)
        self_386835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 35), 'self', False)
        # Obtaining the member '_A' of a type (line 216)
        _A_386836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 35), self_386835, '_A')
        # Obtaining the member 'T' of a type (line 216)
        T_386837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 35), _A_386836, 'T')
        # Getting the type of 'self' (line 216)
        self_386838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 46), 'self', False)
        # Obtaining the member '_p' of a type (line 216)
        _p_386839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 46), self_386838, '_p')
        # Processing the call keyword arguments (line 216)
        kwargs_386840 = {}
        # Getting the type of 'MatrixPowerOperator' (line 216)
        MatrixPowerOperator_386834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'MatrixPowerOperator', False)
        # Calling MatrixPowerOperator(args, kwargs) (line 216)
        MatrixPowerOperator_call_result_386841 = invoke(stypy.reporting.localization.Localization(__file__, 216, 15), MatrixPowerOperator_386834, *[T_386837, _p_386839], **kwargs_386840)
        
        # Assigning a type to the variable 'stypy_return_type' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'stypy_return_type', MatrixPowerOperator_call_result_386841)
        
        # ################# End of 'T(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'T' in the type store
        # Getting the type of 'stypy_return_type' (line 214)
        stypy_return_type_386842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386842)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'T'
        return stypy_return_type_386842


# Assigning a type to the variable 'MatrixPowerOperator' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'MatrixPowerOperator', MatrixPowerOperator)
# Declaration of the 'ProductOperator' class
# Getting the type of 'LinearOperator' (line 219)
LinearOperator_386843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 22), 'LinearOperator')

class ProductOperator(LinearOperator_386843, ):
    str_386844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, (-1)), 'str', '\n    For now, this is limited to products of multiple square matrices.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 224, 4, False)
        # Assigning a type to the variable 'self' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ProductOperator.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 225):
        
        # Assigning a Call to a Attribute (line 225):
        
        # Call to get(...): (line 225)
        # Processing the call arguments (line 225)
        str_386847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 37), 'str', 'structure')
        # Getting the type of 'None' (line 225)
        None_386848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 50), 'None', False)
        # Processing the call keyword arguments (line 225)
        kwargs_386849 = {}
        # Getting the type of 'kwargs' (line 225)
        kwargs_386845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 26), 'kwargs', False)
        # Obtaining the member 'get' of a type (line 225)
        get_386846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 26), kwargs_386845, 'get')
        # Calling get(args, kwargs) (line 225)
        get_call_result_386850 = invoke(stypy.reporting.localization.Localization(__file__, 225, 26), get_386846, *[str_386847, None_386848], **kwargs_386849)
        
        # Getting the type of 'self' (line 225)
        self_386851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'self')
        # Setting the type of the member '_structure' of a type (line 225)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), self_386851, '_structure', get_call_result_386850)
        
        # Getting the type of 'args' (line 226)
        args_386852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), 'args')
        # Testing the type of a for loop iterable (line 226)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 226, 8), args_386852)
        # Getting the type of the for loop variable (line 226)
        for_loop_var_386853 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 226, 8), args_386852)
        # Assigning a type to the variable 'A' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'A', for_loop_var_386853)
        # SSA begins for a for statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'A' (line 227)
        A_386855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 19), 'A', False)
        # Obtaining the member 'shape' of a type (line 227)
        shape_386856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 19), A_386855, 'shape')
        # Processing the call keyword arguments (line 227)
        kwargs_386857 = {}
        # Getting the type of 'len' (line 227)
        len_386854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 15), 'len', False)
        # Calling len(args, kwargs) (line 227)
        len_call_result_386858 = invoke(stypy.reporting.localization.Localization(__file__, 227, 15), len_386854, *[shape_386856], **kwargs_386857)
        
        int_386859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 31), 'int')
        # Applying the binary operator '!=' (line 227)
        result_ne_386860 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 15), '!=', len_call_result_386858, int_386859)
        
        
        
        # Obtaining the type of the subscript
        int_386861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 44), 'int')
        # Getting the type of 'A' (line 227)
        A_386862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 36), 'A')
        # Obtaining the member 'shape' of a type (line 227)
        shape_386863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 36), A_386862, 'shape')
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___386864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 36), shape_386863, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_386865 = invoke(stypy.reporting.localization.Localization(__file__, 227, 36), getitem___386864, int_386861)
        
        
        # Obtaining the type of the subscript
        int_386866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 58), 'int')
        # Getting the type of 'A' (line 227)
        A_386867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 50), 'A')
        # Obtaining the member 'shape' of a type (line 227)
        shape_386868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 50), A_386867, 'shape')
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___386869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 50), shape_386868, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_386870 = invoke(stypy.reporting.localization.Localization(__file__, 227, 50), getitem___386869, int_386866)
        
        # Applying the binary operator '!=' (line 227)
        result_ne_386871 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 36), '!=', subscript_call_result_386865, subscript_call_result_386870)
        
        # Applying the binary operator 'or' (line 227)
        result_or_keyword_386872 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 15), 'or', result_ne_386860, result_ne_386871)
        
        # Testing the type of an if condition (line 227)
        if_condition_386873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 12), result_or_keyword_386872)
        # Assigning a type to the variable 'if_condition_386873' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'if_condition_386873', if_condition_386873)
        # SSA begins for if statement (line 227)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 228)
        # Processing the call arguments (line 228)
        str_386875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 24), 'str', 'For now, the ProductOperator implementation is limited to the product of multiple square matrices.')
        # Processing the call keyword arguments (line 228)
        kwargs_386876 = {}
        # Getting the type of 'ValueError' (line 228)
        ValueError_386874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 228)
        ValueError_call_result_386877 = invoke(stypy.reporting.localization.Localization(__file__, 228, 22), ValueError_386874, *[str_386875], **kwargs_386876)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 228, 16), ValueError_call_result_386877, 'raise parameter', BaseException)
        # SSA join for if statement (line 227)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'args' (line 231)
        args_386878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 11), 'args')
        # Testing the type of an if condition (line 231)
        if_condition_386879 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 8), args_386878)
        # Assigning a type to the variable 'if_condition_386879' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'if_condition_386879', if_condition_386879)
        # SSA begins for if statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 232):
        
        # Assigning a Subscript to a Name (line 232):
        
        # Obtaining the type of the subscript
        int_386880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 30), 'int')
        
        # Obtaining the type of the subscript
        int_386881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 21), 'int')
        # Getting the type of 'args' (line 232)
        args_386882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'args')
        # Obtaining the member '__getitem__' of a type (line 232)
        getitem___386883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 16), args_386882, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 232)
        subscript_call_result_386884 = invoke(stypy.reporting.localization.Localization(__file__, 232, 16), getitem___386883, int_386881)
        
        # Obtaining the member 'shape' of a type (line 232)
        shape_386885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 16), subscript_call_result_386884, 'shape')
        # Obtaining the member '__getitem__' of a type (line 232)
        getitem___386886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 16), shape_386885, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 232)
        subscript_call_result_386887 = invoke(stypy.reporting.localization.Localization(__file__, 232, 16), getitem___386886, int_386880)
        
        # Assigning a type to the variable 'n' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'n', subscript_call_result_386887)
        
        # Getting the type of 'args' (line 233)
        args_386888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 21), 'args')
        # Testing the type of a for loop iterable (line 233)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 233, 12), args_386888)
        # Getting the type of the for loop variable (line 233)
        for_loop_var_386889 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 233, 12), args_386888)
        # Assigning a type to the variable 'A' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'A', for_loop_var_386889)
        # SSA begins for a for statement (line 233)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'A' (line 234)
        A_386890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 25), 'A')
        # Obtaining the member 'shape' of a type (line 234)
        shape_386891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 25), A_386890, 'shape')
        # Testing the type of a for loop iterable (line 234)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 234, 16), shape_386891)
        # Getting the type of the for loop variable (line 234)
        for_loop_var_386892 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 234, 16), shape_386891)
        # Assigning a type to the variable 'd' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'd', for_loop_var_386892)
        # SSA begins for a for statement (line 234)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'd' (line 235)
        d_386893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 23), 'd')
        # Getting the type of 'n' (line 235)
        n_386894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 28), 'n')
        # Applying the binary operator '!=' (line 235)
        result_ne_386895 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 23), '!=', d_386893, n_386894)
        
        # Testing the type of an if condition (line 235)
        if_condition_386896 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 20), result_ne_386895)
        # Assigning a type to the variable 'if_condition_386896' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'if_condition_386896', if_condition_386896)
        # SSA begins for if statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 236)
        # Processing the call arguments (line 236)
        str_386898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 32), 'str', 'The square matrices of the ProductOperator must all have the same shape.')
        # Processing the call keyword arguments (line 236)
        kwargs_386899 = {}
        # Getting the type of 'ValueError' (line 236)
        ValueError_386897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 30), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 236)
        ValueError_call_result_386900 = invoke(stypy.reporting.localization.Localization(__file__, 236, 30), ValueError_386897, *[str_386898], **kwargs_386899)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 236, 24), ValueError_call_result_386900, 'raise parameter', BaseException)
        # SSA join for if statement (line 235)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Attribute (line 239):
        
        # Assigning a Tuple to a Attribute (line 239):
        
        # Obtaining an instance of the builtin type 'tuple' (line 239)
        tuple_386901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 239)
        # Adding element type (line 239)
        # Getting the type of 'n' (line 239)
        n_386902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 26), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 26), tuple_386901, n_386902)
        # Adding element type (line 239)
        # Getting the type of 'n' (line 239)
        n_386903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 29), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 26), tuple_386901, n_386903)
        
        # Getting the type of 'self' (line 239)
        self_386904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'self')
        # Setting the type of the member 'shape' of a type (line 239)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), self_386904, 'shape', tuple_386901)
        
        # Assigning a Call to a Attribute (line 240):
        
        # Assigning a Call to a Attribute (line 240):
        
        # Call to len(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'self' (line 240)
        self_386906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 28), 'self', False)
        # Obtaining the member 'shape' of a type (line 240)
        shape_386907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 28), self_386906, 'shape')
        # Processing the call keyword arguments (line 240)
        kwargs_386908 = {}
        # Getting the type of 'len' (line 240)
        len_386905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 24), 'len', False)
        # Calling len(args, kwargs) (line 240)
        len_call_result_386909 = invoke(stypy.reporting.localization.Localization(__file__, 240, 24), len_386905, *[shape_386907], **kwargs_386908)
        
        # Getting the type of 'self' (line 240)
        self_386910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'self')
        # Setting the type of the member 'ndim' of a type (line 240)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), self_386910, 'ndim', len_call_result_386909)
        # SSA join for if statement (line 231)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 241):
        
        # Assigning a Call to a Attribute (line 241):
        
        # Call to find_common_type(...): (line 241)
        # Processing the call arguments (line 241)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'args' (line 241)
        args_386915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 59), 'args', False)
        comprehension_386916 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 42), args_386915)
        # Assigning a type to the variable 'x' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 42), 'x', comprehension_386916)
        # Getting the type of 'x' (line 241)
        x_386913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 42), 'x', False)
        # Obtaining the member 'dtype' of a type (line 241)
        dtype_386914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 42), x_386913, 'dtype')
        list_386917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 42), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 42), list_386917, dtype_386914)
        
        # Obtaining an instance of the builtin type 'list' (line 241)
        list_386918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 241)
        
        # Processing the call keyword arguments (line 241)
        kwargs_386919 = {}
        # Getting the type of 'np' (line 241)
        np_386911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'np', False)
        # Obtaining the member 'find_common_type' of a type (line 241)
        find_common_type_386912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 21), np_386911, 'find_common_type')
        # Calling find_common_type(args, kwargs) (line 241)
        find_common_type_call_result_386920 = invoke(stypy.reporting.localization.Localization(__file__, 241, 21), find_common_type_386912, *[list_386917, list_386918], **kwargs_386919)
        
        # Getting the type of 'self' (line 241)
        self_386921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'self')
        # Setting the type of the member 'dtype' of a type (line 241)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), self_386921, 'dtype', find_common_type_call_result_386920)
        
        # Assigning a Name to a Attribute (line 242):
        
        # Assigning a Name to a Attribute (line 242):
        # Getting the type of 'args' (line 242)
        args_386922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 34), 'args')
        # Getting the type of 'self' (line 242)
        self_386923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'self')
        # Setting the type of the member '_operator_sequence' of a type (line 242)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), self_386923, '_operator_sequence', args_386922)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matvec'
        module_type_store = module_type_store.open_function_context('_matvec', 244, 4, False)
        # Assigning a type to the variable 'self' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ProductOperator._matvec.__dict__.__setitem__('stypy_localization', localization)
        ProductOperator._matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ProductOperator._matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        ProductOperator._matvec.__dict__.__setitem__('stypy_function_name', 'ProductOperator._matvec')
        ProductOperator._matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        ProductOperator._matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        ProductOperator._matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ProductOperator._matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        ProductOperator._matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        ProductOperator._matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ProductOperator._matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ProductOperator._matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matvec(...)' code ##################

        
        
        # Call to reversed(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'self' (line 245)
        self_386925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 26), 'self', False)
        # Obtaining the member '_operator_sequence' of a type (line 245)
        _operator_sequence_386926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 26), self_386925, '_operator_sequence')
        # Processing the call keyword arguments (line 245)
        kwargs_386927 = {}
        # Getting the type of 'reversed' (line 245)
        reversed_386924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 17), 'reversed', False)
        # Calling reversed(args, kwargs) (line 245)
        reversed_call_result_386928 = invoke(stypy.reporting.localization.Localization(__file__, 245, 17), reversed_386924, *[_operator_sequence_386926], **kwargs_386927)
        
        # Testing the type of a for loop iterable (line 245)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 245, 8), reversed_call_result_386928)
        # Getting the type of the for loop variable (line 245)
        for_loop_var_386929 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 245, 8), reversed_call_result_386928)
        # Assigning a type to the variable 'A' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'A', for_loop_var_386929)
        # SSA begins for a for statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Call to dot(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'x' (line 246)
        x_386932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 22), 'x', False)
        # Processing the call keyword arguments (line 246)
        kwargs_386933 = {}
        # Getting the type of 'A' (line 246)
        A_386930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'A', False)
        # Obtaining the member 'dot' of a type (line 246)
        dot_386931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), A_386930, 'dot')
        # Calling dot(args, kwargs) (line 246)
        dot_call_result_386934 = invoke(stypy.reporting.localization.Localization(__file__, 246, 16), dot_386931, *[x_386932], **kwargs_386933)
        
        # Assigning a type to the variable 'x' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'x', dot_call_result_386934)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'x' (line 247)
        x_386935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'stypy_return_type', x_386935)
        
        # ################# End of '_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 244)
        stypy_return_type_386936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386936)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matvec'
        return stypy_return_type_386936


    @norecursion
    def _rmatvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rmatvec'
        module_type_store = module_type_store.open_function_context('_rmatvec', 249, 4, False)
        # Assigning a type to the variable 'self' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ProductOperator._rmatvec.__dict__.__setitem__('stypy_localization', localization)
        ProductOperator._rmatvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ProductOperator._rmatvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        ProductOperator._rmatvec.__dict__.__setitem__('stypy_function_name', 'ProductOperator._rmatvec')
        ProductOperator._rmatvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        ProductOperator._rmatvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        ProductOperator._rmatvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ProductOperator._rmatvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        ProductOperator._rmatvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        ProductOperator._rmatvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ProductOperator._rmatvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ProductOperator._rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rmatvec(...)' code ##################

        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to ravel(...): (line 250)
        # Processing the call keyword arguments (line 250)
        kwargs_386939 = {}
        # Getting the type of 'x' (line 250)
        x_386937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'x', False)
        # Obtaining the member 'ravel' of a type (line 250)
        ravel_386938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 12), x_386937, 'ravel')
        # Calling ravel(args, kwargs) (line 250)
        ravel_call_result_386940 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), ravel_386938, *[], **kwargs_386939)
        
        # Assigning a type to the variable 'x' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'x', ravel_call_result_386940)
        
        # Getting the type of 'self' (line 251)
        self_386941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 17), 'self')
        # Obtaining the member '_operator_sequence' of a type (line 251)
        _operator_sequence_386942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 17), self_386941, '_operator_sequence')
        # Testing the type of a for loop iterable (line 251)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 251, 8), _operator_sequence_386942)
        # Getting the type of the for loop variable (line 251)
        for_loop_var_386943 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 251, 8), _operator_sequence_386942)
        # Assigning a type to the variable 'A' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'A', for_loop_var_386943)
        # SSA begins for a for statement (line 251)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 252):
        
        # Assigning a Call to a Name (line 252):
        
        # Call to dot(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'x' (line 252)
        x_386947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 24), 'x', False)
        # Processing the call keyword arguments (line 252)
        kwargs_386948 = {}
        # Getting the type of 'A' (line 252)
        A_386944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'A', False)
        # Obtaining the member 'T' of a type (line 252)
        T_386945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), A_386944, 'T')
        # Obtaining the member 'dot' of a type (line 252)
        dot_386946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), T_386945, 'dot')
        # Calling dot(args, kwargs) (line 252)
        dot_call_result_386949 = invoke(stypy.reporting.localization.Localization(__file__, 252, 16), dot_386946, *[x_386947], **kwargs_386948)
        
        # Assigning a type to the variable 'x' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'x', dot_call_result_386949)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'x' (line 253)
        x_386950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'stypy_return_type', x_386950)
        
        # ################# End of '_rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 249)
        stypy_return_type_386951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386951)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rmatvec'
        return stypy_return_type_386951


    @norecursion
    def _matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matmat'
        module_type_store = module_type_store.open_function_context('_matmat', 255, 4, False)
        # Assigning a type to the variable 'self' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ProductOperator._matmat.__dict__.__setitem__('stypy_localization', localization)
        ProductOperator._matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ProductOperator._matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        ProductOperator._matmat.__dict__.__setitem__('stypy_function_name', 'ProductOperator._matmat')
        ProductOperator._matmat.__dict__.__setitem__('stypy_param_names_list', ['X'])
        ProductOperator._matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        ProductOperator._matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ProductOperator._matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        ProductOperator._matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        ProductOperator._matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ProductOperator._matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ProductOperator._matmat', ['X'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matmat', localization, ['X'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matmat(...)' code ##################

        
        
        # Call to reversed(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'self' (line 256)
        self_386953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 26), 'self', False)
        # Obtaining the member '_operator_sequence' of a type (line 256)
        _operator_sequence_386954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 26), self_386953, '_operator_sequence')
        # Processing the call keyword arguments (line 256)
        kwargs_386955 = {}
        # Getting the type of 'reversed' (line 256)
        reversed_386952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 17), 'reversed', False)
        # Calling reversed(args, kwargs) (line 256)
        reversed_call_result_386956 = invoke(stypy.reporting.localization.Localization(__file__, 256, 17), reversed_386952, *[_operator_sequence_386954], **kwargs_386955)
        
        # Testing the type of a for loop iterable (line 256)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 256, 8), reversed_call_result_386956)
        # Getting the type of the for loop variable (line 256)
        for_loop_var_386957 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 256, 8), reversed_call_result_386956)
        # Assigning a type to the variable 'A' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'A', for_loop_var_386957)
        # SSA begins for a for statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Call to _smart_matrix_product(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'A' (line 257)
        A_386959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 38), 'A', False)
        # Getting the type of 'X' (line 257)
        X_386960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 41), 'X', False)
        # Processing the call keyword arguments (line 257)
        # Getting the type of 'self' (line 257)
        self_386961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 54), 'self', False)
        # Obtaining the member '_structure' of a type (line 257)
        _structure_386962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 54), self_386961, '_structure')
        keyword_386963 = _structure_386962
        kwargs_386964 = {'structure': keyword_386963}
        # Getting the type of '_smart_matrix_product' (line 257)
        _smart_matrix_product_386958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), '_smart_matrix_product', False)
        # Calling _smart_matrix_product(args, kwargs) (line 257)
        _smart_matrix_product_call_result_386965 = invoke(stypy.reporting.localization.Localization(__file__, 257, 16), _smart_matrix_product_386958, *[A_386959, X_386960], **kwargs_386964)
        
        # Assigning a type to the variable 'X' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'X', _smart_matrix_product_call_result_386965)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'X' (line 258)
        X_386966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'X')
        # Assigning a type to the variable 'stypy_return_type' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type', X_386966)
        
        # ################# End of '_matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 255)
        stypy_return_type_386967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386967)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matmat'
        return stypy_return_type_386967


    @norecursion
    def T(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'T'
        module_type_store = module_type_store.open_function_context('T', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ProductOperator.T.__dict__.__setitem__('stypy_localization', localization)
        ProductOperator.T.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ProductOperator.T.__dict__.__setitem__('stypy_type_store', module_type_store)
        ProductOperator.T.__dict__.__setitem__('stypy_function_name', 'ProductOperator.T')
        ProductOperator.T.__dict__.__setitem__('stypy_param_names_list', [])
        ProductOperator.T.__dict__.__setitem__('stypy_varargs_param_name', None)
        ProductOperator.T.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ProductOperator.T.__dict__.__setitem__('stypy_call_defaults', defaults)
        ProductOperator.T.__dict__.__setitem__('stypy_call_varargs', varargs)
        ProductOperator.T.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ProductOperator.T.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ProductOperator.T', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'T', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'T(...)' code ##################

        
        # Assigning a ListComp to a Name (line 262):
        
        # Assigning a ListComp to a Name (line 262):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to reversed(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'self' (line 262)
        self_386971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 40), 'self', False)
        # Obtaining the member '_operator_sequence' of a type (line 262)
        _operator_sequence_386972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 40), self_386971, '_operator_sequence')
        # Processing the call keyword arguments (line 262)
        kwargs_386973 = {}
        # Getting the type of 'reversed' (line 262)
        reversed_386970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 31), 'reversed', False)
        # Calling reversed(args, kwargs) (line 262)
        reversed_call_result_386974 = invoke(stypy.reporting.localization.Localization(__file__, 262, 31), reversed_386970, *[_operator_sequence_386972], **kwargs_386973)
        
        comprehension_386975 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 18), reversed_call_result_386974)
        # Assigning a type to the variable 'A' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 18), 'A', comprehension_386975)
        # Getting the type of 'A' (line 262)
        A_386968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 18), 'A')
        # Obtaining the member 'T' of a type (line 262)
        T_386969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 18), A_386968, 'T')
        list_386976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 18), list_386976, T_386969)
        # Assigning a type to the variable 'T_args' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'T_args', list_386976)
        
        # Call to ProductOperator(...): (line 263)
        # Getting the type of 'T_args' (line 263)
        T_args_386978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 32), 'T_args', False)
        # Processing the call keyword arguments (line 263)
        kwargs_386979 = {}
        # Getting the type of 'ProductOperator' (line 263)
        ProductOperator_386977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'ProductOperator', False)
        # Calling ProductOperator(args, kwargs) (line 263)
        ProductOperator_call_result_386980 = invoke(stypy.reporting.localization.Localization(__file__, 263, 15), ProductOperator_386977, *[T_args_386978], **kwargs_386979)
        
        # Assigning a type to the variable 'stypy_return_type' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'stypy_return_type', ProductOperator_call_result_386980)
        
        # ################# End of 'T(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'T' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_386981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386981)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'T'
        return stypy_return_type_386981


# Assigning a type to the variable 'ProductOperator' (line 219)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'ProductOperator', ProductOperator)

@norecursion
def _onenormest_matrix_power(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_386982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 10), 'int')
    int_386983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 19), 'int')
    # Getting the type of 'False' (line 267)
    False_386984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 32), 'False')
    # Getting the type of 'False' (line 267)
    False_386985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 49), 'False')
    # Getting the type of 'None' (line 267)
    None_386986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 66), 'None')
    defaults = [int_386982, int_386983, False_386984, False_386985, None_386986]
    # Create a new context for function '_onenormest_matrix_power'
    module_type_store = module_type_store.open_function_context('_onenormest_matrix_power', 266, 0, False)
    
    # Passed parameters checking function
    _onenormest_matrix_power.stypy_localization = localization
    _onenormest_matrix_power.stypy_type_of_self = None
    _onenormest_matrix_power.stypy_type_store = module_type_store
    _onenormest_matrix_power.stypy_function_name = '_onenormest_matrix_power'
    _onenormest_matrix_power.stypy_param_names_list = ['A', 'p', 't', 'itmax', 'compute_v', 'compute_w', 'structure']
    _onenormest_matrix_power.stypy_varargs_param_name = None
    _onenormest_matrix_power.stypy_kwargs_param_name = None
    _onenormest_matrix_power.stypy_call_defaults = defaults
    _onenormest_matrix_power.stypy_call_varargs = varargs
    _onenormest_matrix_power.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_onenormest_matrix_power', ['A', 'p', 't', 'itmax', 'compute_v', 'compute_w', 'structure'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_onenormest_matrix_power', localization, ['A', 'p', 't', 'itmax', 'compute_v', 'compute_w', 'structure'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_onenormest_matrix_power(...)' code ##################

    str_386987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, (-1)), 'str', '\n    Efficiently estimate the 1-norm of A^p.\n\n    Parameters\n    ----------\n    A : ndarray\n        Matrix whose 1-norm of a power is to be computed.\n    p : int\n        Non-negative integer power.\n    t : int, optional\n        A positive parameter controlling the tradeoff between\n        accuracy versus time and memory usage.\n        Larger values take longer and use more memory\n        but give more accurate output.\n    itmax : int, optional\n        Use at most this many iterations.\n    compute_v : bool, optional\n        Request a norm-maximizing linear operator input vector if True.\n    compute_w : bool, optional\n        Request a norm-maximizing linear operator output vector if True.\n\n    Returns\n    -------\n    est : float\n        An underestimate of the 1-norm of the sparse matrix.\n    v : ndarray, optional\n        The vector such that ||Av||_1 == est*||v||_1.\n        It can be thought of as an input to the linear operator\n        that gives an output with particularly large norm.\n    w : ndarray, optional\n        The vector Av which has relatively large 1-norm.\n        It can be thought of as an output of the linear operator\n        that is relatively large in norm compared to the input.\n\n    ')
    
    # Call to onenormest(...): (line 303)
    # Processing the call arguments (line 303)
    
    # Call to MatrixPowerOperator(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'A' (line 304)
    A_386993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 32), 'A', False)
    # Getting the type of 'p' (line 304)
    p_386994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 35), 'p', False)
    # Processing the call keyword arguments (line 304)
    # Getting the type of 'structure' (line 304)
    structure_386995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 48), 'structure', False)
    keyword_386996 = structure_386995
    kwargs_386997 = {'structure': keyword_386996}
    # Getting the type of 'MatrixPowerOperator' (line 304)
    MatrixPowerOperator_386992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'MatrixPowerOperator', False)
    # Calling MatrixPowerOperator(args, kwargs) (line 304)
    MatrixPowerOperator_call_result_386998 = invoke(stypy.reporting.localization.Localization(__file__, 304, 12), MatrixPowerOperator_386992, *[A_386993, p_386994], **kwargs_386997)
    
    # Processing the call keyword arguments (line 303)
    kwargs_386999 = {}
    # Getting the type of 'scipy' (line 303)
    scipy_386988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 11), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 303)
    sparse_386989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 11), scipy_386988, 'sparse')
    # Obtaining the member 'linalg' of a type (line 303)
    linalg_386990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 11), sparse_386989, 'linalg')
    # Obtaining the member 'onenormest' of a type (line 303)
    onenormest_386991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 11), linalg_386990, 'onenormest')
    # Calling onenormest(args, kwargs) (line 303)
    onenormest_call_result_387000 = invoke(stypy.reporting.localization.Localization(__file__, 303, 11), onenormest_386991, *[MatrixPowerOperator_call_result_386998], **kwargs_386999)
    
    # Assigning a type to the variable 'stypy_return_type' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type', onenormest_call_result_387000)
    
    # ################# End of '_onenormest_matrix_power(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_onenormest_matrix_power' in the type store
    # Getting the type of 'stypy_return_type' (line 266)
    stypy_return_type_387001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_387001)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_onenormest_matrix_power'
    return stypy_return_type_387001

# Assigning a type to the variable '_onenormest_matrix_power' (line 266)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 0), '_onenormest_matrix_power', _onenormest_matrix_power)

@norecursion
def _onenormest_product(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_387002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 10), 'int')
    int_387003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 19), 'int')
    # Getting the type of 'False' (line 308)
    False_387004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 32), 'False')
    # Getting the type of 'False' (line 308)
    False_387005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 49), 'False')
    # Getting the type of 'None' (line 308)
    None_387006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 66), 'None')
    defaults = [int_387002, int_387003, False_387004, False_387005, None_387006]
    # Create a new context for function '_onenormest_product'
    module_type_store = module_type_store.open_function_context('_onenormest_product', 307, 0, False)
    
    # Passed parameters checking function
    _onenormest_product.stypy_localization = localization
    _onenormest_product.stypy_type_of_self = None
    _onenormest_product.stypy_type_store = module_type_store
    _onenormest_product.stypy_function_name = '_onenormest_product'
    _onenormest_product.stypy_param_names_list = ['operator_seq', 't', 'itmax', 'compute_v', 'compute_w', 'structure']
    _onenormest_product.stypy_varargs_param_name = None
    _onenormest_product.stypy_kwargs_param_name = None
    _onenormest_product.stypy_call_defaults = defaults
    _onenormest_product.stypy_call_varargs = varargs
    _onenormest_product.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_onenormest_product', ['operator_seq', 't', 'itmax', 'compute_v', 'compute_w', 'structure'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_onenormest_product', localization, ['operator_seq', 't', 'itmax', 'compute_v', 'compute_w', 'structure'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_onenormest_product(...)' code ##################

    str_387007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, (-1)), 'str', '\n    Efficiently estimate the 1-norm of the matrix product of the args.\n\n    Parameters\n    ----------\n    operator_seq : linear operator sequence\n        Matrices whose 1-norm of product is to be computed.\n    t : int, optional\n        A positive parameter controlling the tradeoff between\n        accuracy versus time and memory usage.\n        Larger values take longer and use more memory\n        but give more accurate output.\n    itmax : int, optional\n        Use at most this many iterations.\n    compute_v : bool, optional\n        Request a norm-maximizing linear operator input vector if True.\n    compute_w : bool, optional\n        Request a norm-maximizing linear operator output vector if True.\n    structure : str, optional\n        A string describing the structure of all operators.\n        Only `upper_triangular` is currently supported.\n\n    Returns\n    -------\n    est : float\n        An underestimate of the 1-norm of the sparse matrix.\n    v : ndarray, optional\n        The vector such that ||Av||_1 == est*||v||_1.\n        It can be thought of as an input to the linear operator\n        that gives an output with particularly large norm.\n    w : ndarray, optional\n        The vector Av which has relatively large 1-norm.\n        It can be thought of as an output of the linear operator\n        that is relatively large in norm compared to the input.\n\n    ')
    
    # Call to onenormest(...): (line 345)
    # Processing the call arguments (line 345)
    
    # Call to ProductOperator(...): (line 346)
    # Getting the type of 'operator_seq' (line 346)
    operator_seq_387013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 29), 'operator_seq', False)
    # Processing the call keyword arguments (line 346)
    # Getting the type of 'structure' (line 346)
    structure_387014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 53), 'structure', False)
    keyword_387015 = structure_387014
    kwargs_387016 = {'structure': keyword_387015}
    # Getting the type of 'ProductOperator' (line 346)
    ProductOperator_387012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'ProductOperator', False)
    # Calling ProductOperator(args, kwargs) (line 346)
    ProductOperator_call_result_387017 = invoke(stypy.reporting.localization.Localization(__file__, 346, 12), ProductOperator_387012, *[operator_seq_387013], **kwargs_387016)
    
    # Processing the call keyword arguments (line 345)
    kwargs_387018 = {}
    # Getting the type of 'scipy' (line 345)
    scipy_387008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 11), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 345)
    sparse_387009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 11), scipy_387008, 'sparse')
    # Obtaining the member 'linalg' of a type (line 345)
    linalg_387010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 11), sparse_387009, 'linalg')
    # Obtaining the member 'onenormest' of a type (line 345)
    onenormest_387011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 11), linalg_387010, 'onenormest')
    # Calling onenormest(args, kwargs) (line 345)
    onenormest_call_result_387019 = invoke(stypy.reporting.localization.Localization(__file__, 345, 11), onenormest_387011, *[ProductOperator_call_result_387017], **kwargs_387018)
    
    # Assigning a type to the variable 'stypy_return_type' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'stypy_return_type', onenormest_call_result_387019)
    
    # ################# End of '_onenormest_product(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_onenormest_product' in the type store
    # Getting the type of 'stypy_return_type' (line 307)
    stypy_return_type_387020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_387020)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_onenormest_product'
    return stypy_return_type_387020

# Assigning a type to the variable '_onenormest_product' (line 307)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), '_onenormest_product', _onenormest_product)
# Declaration of the '_ExpmPadeHelper' class

class _ExpmPadeHelper(object, ):
    str_387021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, (-1)), 'str', '\n    Help lazily evaluate a matrix exponential.\n\n    The idea is to not do more work than we need for high expm precision,\n    so we lazily compute matrix powers and store or precompute\n    other properties of the matrix.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 358)
        None_387022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 36), 'None')
        # Getting the type of 'False' (line 358)
        False_387023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 60), 'False')
        defaults = [None_387022, False_387023]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 358, 4, False)
        # Assigning a type to the variable 'self' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.__init__', ['A', 'structure', 'use_exact_onenorm'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['A', 'structure', 'use_exact_onenorm'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_387024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, (-1)), 'str', '\n        Initialize the object.\n\n        Parameters\n        ----------\n        A : a dense or sparse square numpy matrix or ndarray\n            The matrix to be exponentiated.\n        structure : str, optional\n            A string describing the structure of matrix `A`.\n            Only `upper_triangular` is currently supported.\n        use_exact_onenorm : bool, optional\n            If True then only the exact one-norm of matrix powers and products\n            will be used. Otherwise, the one-norm of powers and products\n            may initially be estimated.\n        ')
        
        # Assigning a Name to a Attribute (line 374):
        
        # Assigning a Name to a Attribute (line 374):
        # Getting the type of 'A' (line 374)
        A_387025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 17), 'A')
        # Getting the type of 'self' (line 374)
        self_387026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'self')
        # Setting the type of the member 'A' of a type (line 374)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), self_387026, 'A', A_387025)
        
        # Assigning a Name to a Attribute (line 375):
        
        # Assigning a Name to a Attribute (line 375):
        # Getting the type of 'None' (line 375)
        None_387027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 19), 'None')
        # Getting the type of 'self' (line 375)
        self_387028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'self')
        # Setting the type of the member '_A2' of a type (line 375)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 8), self_387028, '_A2', None_387027)
        
        # Assigning a Name to a Attribute (line 376):
        
        # Assigning a Name to a Attribute (line 376):
        # Getting the type of 'None' (line 376)
        None_387029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 'None')
        # Getting the type of 'self' (line 376)
        self_387030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'self')
        # Setting the type of the member '_A4' of a type (line 376)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 8), self_387030, '_A4', None_387029)
        
        # Assigning a Name to a Attribute (line 377):
        
        # Assigning a Name to a Attribute (line 377):
        # Getting the type of 'None' (line 377)
        None_387031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), 'None')
        # Getting the type of 'self' (line 377)
        self_387032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'self')
        # Setting the type of the member '_A6' of a type (line 377)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 8), self_387032, '_A6', None_387031)
        
        # Assigning a Name to a Attribute (line 378):
        
        # Assigning a Name to a Attribute (line 378):
        # Getting the type of 'None' (line 378)
        None_387033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 19), 'None')
        # Getting the type of 'self' (line 378)
        self_387034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'self')
        # Setting the type of the member '_A8' of a type (line 378)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), self_387034, '_A8', None_387033)
        
        # Assigning a Name to a Attribute (line 379):
        
        # Assigning a Name to a Attribute (line 379):
        # Getting the type of 'None' (line 379)
        None_387035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 20), 'None')
        # Getting the type of 'self' (line 379)
        self_387036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'self')
        # Setting the type of the member '_A10' of a type (line 379)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), self_387036, '_A10', None_387035)
        
        # Assigning a Name to a Attribute (line 380):
        
        # Assigning a Name to a Attribute (line 380):
        # Getting the type of 'None' (line 380)
        None_387037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 25), 'None')
        # Getting the type of 'self' (line 380)
        self_387038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'self')
        # Setting the type of the member '_d4_exact' of a type (line 380)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), self_387038, '_d4_exact', None_387037)
        
        # Assigning a Name to a Attribute (line 381):
        
        # Assigning a Name to a Attribute (line 381):
        # Getting the type of 'None' (line 381)
        None_387039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 25), 'None')
        # Getting the type of 'self' (line 381)
        self_387040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'self')
        # Setting the type of the member '_d6_exact' of a type (line 381)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 8), self_387040, '_d6_exact', None_387039)
        
        # Assigning a Name to a Attribute (line 382):
        
        # Assigning a Name to a Attribute (line 382):
        # Getting the type of 'None' (line 382)
        None_387041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 25), 'None')
        # Getting the type of 'self' (line 382)
        self_387042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'self')
        # Setting the type of the member '_d8_exact' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 8), self_387042, '_d8_exact', None_387041)
        
        # Assigning a Name to a Attribute (line 383):
        
        # Assigning a Name to a Attribute (line 383):
        # Getting the type of 'None' (line 383)
        None_387043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 26), 'None')
        # Getting the type of 'self' (line 383)
        self_387044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self')
        # Setting the type of the member '_d10_exact' of a type (line 383)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_387044, '_d10_exact', None_387043)
        
        # Assigning a Name to a Attribute (line 384):
        
        # Assigning a Name to a Attribute (line 384):
        # Getting the type of 'None' (line 384)
        None_387045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 26), 'None')
        # Getting the type of 'self' (line 384)
        self_387046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'self')
        # Setting the type of the member '_d4_approx' of a type (line 384)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), self_387046, '_d4_approx', None_387045)
        
        # Assigning a Name to a Attribute (line 385):
        
        # Assigning a Name to a Attribute (line 385):
        # Getting the type of 'None' (line 385)
        None_387047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 26), 'None')
        # Getting the type of 'self' (line 385)
        self_387048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'self')
        # Setting the type of the member '_d6_approx' of a type (line 385)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 8), self_387048, '_d6_approx', None_387047)
        
        # Assigning a Name to a Attribute (line 386):
        
        # Assigning a Name to a Attribute (line 386):
        # Getting the type of 'None' (line 386)
        None_387049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 26), 'None')
        # Getting the type of 'self' (line 386)
        self_387050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'self')
        # Setting the type of the member '_d8_approx' of a type (line 386)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 8), self_387050, '_d8_approx', None_387049)
        
        # Assigning a Name to a Attribute (line 387):
        
        # Assigning a Name to a Attribute (line 387):
        # Getting the type of 'None' (line 387)
        None_387051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 27), 'None')
        # Getting the type of 'self' (line 387)
        self_387052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'self')
        # Setting the type of the member '_d10_approx' of a type (line 387)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), self_387052, '_d10_approx', None_387051)
        
        # Assigning a Call to a Attribute (line 388):
        
        # Assigning a Call to a Attribute (line 388):
        
        # Call to _ident_like(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'A' (line 388)
        A_387054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 33), 'A', False)
        # Processing the call keyword arguments (line 388)
        kwargs_387055 = {}
        # Getting the type of '_ident_like' (line 388)
        _ident_like_387053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 21), '_ident_like', False)
        # Calling _ident_like(args, kwargs) (line 388)
        _ident_like_call_result_387056 = invoke(stypy.reporting.localization.Localization(__file__, 388, 21), _ident_like_387053, *[A_387054], **kwargs_387055)
        
        # Getting the type of 'self' (line 388)
        self_387057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'self')
        # Setting the type of the member 'ident' of a type (line 388)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), self_387057, 'ident', _ident_like_call_result_387056)
        
        # Assigning a Name to a Attribute (line 389):
        
        # Assigning a Name to a Attribute (line 389):
        # Getting the type of 'structure' (line 389)
        structure_387058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 25), 'structure')
        # Getting the type of 'self' (line 389)
        self_387059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'self')
        # Setting the type of the member 'structure' of a type (line 389)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), self_387059, 'structure', structure_387058)
        
        # Assigning a Name to a Attribute (line 390):
        
        # Assigning a Name to a Attribute (line 390):
        # Getting the type of 'use_exact_onenorm' (line 390)
        use_exact_onenorm_387060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 33), 'use_exact_onenorm')
        # Getting the type of 'self' (line 390)
        self_387061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'self')
        # Setting the type of the member 'use_exact_onenorm' of a type (line 390)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), self_387061, 'use_exact_onenorm', use_exact_onenorm_387060)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def A2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'A2'
        module_type_store = module_type_store.open_function_context('A2', 392, 4, False)
        # Assigning a type to the variable 'self' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.A2.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.A2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.A2.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.A2.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.A2')
        _ExpmPadeHelper.A2.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.A2.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.A2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.A2.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.A2.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.A2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.A2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.A2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'A2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'A2(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 394)
        # Getting the type of 'self' (line 394)
        self_387062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 11), 'self')
        # Obtaining the member '_A2' of a type (line 394)
        _A2_387063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 11), self_387062, '_A2')
        # Getting the type of 'None' (line 394)
        None_387064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 23), 'None')
        
        (may_be_387065, more_types_in_union_387066) = may_be_none(_A2_387063, None_387064)

        if may_be_387065:

            if more_types_in_union_387066:
                # Runtime conditional SSA (line 394)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 395):
            
            # Assigning a Call to a Attribute (line 395):
            
            # Call to _smart_matrix_product(...): (line 395)
            # Processing the call arguments (line 395)
            # Getting the type of 'self' (line 396)
            self_387068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 20), 'self', False)
            # Obtaining the member 'A' of a type (line 396)
            A_387069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 20), self_387068, 'A')
            # Getting the type of 'self' (line 396)
            self_387070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 28), 'self', False)
            # Obtaining the member 'A' of a type (line 396)
            A_387071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 28), self_387070, 'A')
            # Processing the call keyword arguments (line 395)
            # Getting the type of 'self' (line 396)
            self_387072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 46), 'self', False)
            # Obtaining the member 'structure' of a type (line 396)
            structure_387073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 46), self_387072, 'structure')
            keyword_387074 = structure_387073
            kwargs_387075 = {'structure': keyword_387074}
            # Getting the type of '_smart_matrix_product' (line 395)
            _smart_matrix_product_387067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 23), '_smart_matrix_product', False)
            # Calling _smart_matrix_product(args, kwargs) (line 395)
            _smart_matrix_product_call_result_387076 = invoke(stypy.reporting.localization.Localization(__file__, 395, 23), _smart_matrix_product_387067, *[A_387069, A_387071], **kwargs_387075)
            
            # Getting the type of 'self' (line 395)
            self_387077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'self')
            # Setting the type of the member '_A2' of a type (line 395)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), self_387077, '_A2', _smart_matrix_product_call_result_387076)

            if more_types_in_union_387066:
                # SSA join for if statement (line 394)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 397)
        self_387078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 15), 'self')
        # Obtaining the member '_A2' of a type (line 397)
        _A2_387079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 15), self_387078, '_A2')
        # Assigning a type to the variable 'stypy_return_type' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'stypy_return_type', _A2_387079)
        
        # ################# End of 'A2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'A2' in the type store
        # Getting the type of 'stypy_return_type' (line 392)
        stypy_return_type_387080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387080)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'A2'
        return stypy_return_type_387080


    @norecursion
    def A4(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'A4'
        module_type_store = module_type_store.open_function_context('A4', 399, 4, False)
        # Assigning a type to the variable 'self' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.A4.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.A4.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.A4.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.A4.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.A4')
        _ExpmPadeHelper.A4.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.A4.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.A4.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.A4.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.A4.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.A4.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.A4.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.A4', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'A4', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'A4(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 401)
        # Getting the type of 'self' (line 401)
        self_387081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 11), 'self')
        # Obtaining the member '_A4' of a type (line 401)
        _A4_387082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 11), self_387081, '_A4')
        # Getting the type of 'None' (line 401)
        None_387083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 23), 'None')
        
        (may_be_387084, more_types_in_union_387085) = may_be_none(_A4_387082, None_387083)

        if may_be_387084:

            if more_types_in_union_387085:
                # Runtime conditional SSA (line 401)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 402):
            
            # Assigning a Call to a Attribute (line 402):
            
            # Call to _smart_matrix_product(...): (line 402)
            # Processing the call arguments (line 402)
            # Getting the type of 'self' (line 403)
            self_387087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 20), 'self', False)
            # Obtaining the member 'A2' of a type (line 403)
            A2_387088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 20), self_387087, 'A2')
            # Getting the type of 'self' (line 403)
            self_387089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 29), 'self', False)
            # Obtaining the member 'A2' of a type (line 403)
            A2_387090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 29), self_387089, 'A2')
            # Processing the call keyword arguments (line 402)
            # Getting the type of 'self' (line 403)
            self_387091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 48), 'self', False)
            # Obtaining the member 'structure' of a type (line 403)
            structure_387092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 48), self_387091, 'structure')
            keyword_387093 = structure_387092
            kwargs_387094 = {'structure': keyword_387093}
            # Getting the type of '_smart_matrix_product' (line 402)
            _smart_matrix_product_387086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 23), '_smart_matrix_product', False)
            # Calling _smart_matrix_product(args, kwargs) (line 402)
            _smart_matrix_product_call_result_387095 = invoke(stypy.reporting.localization.Localization(__file__, 402, 23), _smart_matrix_product_387086, *[A2_387088, A2_387090], **kwargs_387094)
            
            # Getting the type of 'self' (line 402)
            self_387096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'self')
            # Setting the type of the member '_A4' of a type (line 402)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 12), self_387096, '_A4', _smart_matrix_product_call_result_387095)

            if more_types_in_union_387085:
                # SSA join for if statement (line 401)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 404)
        self_387097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'self')
        # Obtaining the member '_A4' of a type (line 404)
        _A4_387098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 15), self_387097, '_A4')
        # Assigning a type to the variable 'stypy_return_type' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'stypy_return_type', _A4_387098)
        
        # ################# End of 'A4(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'A4' in the type store
        # Getting the type of 'stypy_return_type' (line 399)
        stypy_return_type_387099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387099)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'A4'
        return stypy_return_type_387099


    @norecursion
    def A6(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'A6'
        module_type_store = module_type_store.open_function_context('A6', 406, 4, False)
        # Assigning a type to the variable 'self' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.A6.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.A6.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.A6.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.A6.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.A6')
        _ExpmPadeHelper.A6.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.A6.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.A6.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.A6.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.A6.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.A6.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.A6.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.A6', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'A6', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'A6(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 408)
        # Getting the type of 'self' (line 408)
        self_387100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 11), 'self')
        # Obtaining the member '_A6' of a type (line 408)
        _A6_387101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 11), self_387100, '_A6')
        # Getting the type of 'None' (line 408)
        None_387102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 23), 'None')
        
        (may_be_387103, more_types_in_union_387104) = may_be_none(_A6_387101, None_387102)

        if may_be_387103:

            if more_types_in_union_387104:
                # Runtime conditional SSA (line 408)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 409):
            
            # Assigning a Call to a Attribute (line 409):
            
            # Call to _smart_matrix_product(...): (line 409)
            # Processing the call arguments (line 409)
            # Getting the type of 'self' (line 410)
            self_387106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 20), 'self', False)
            # Obtaining the member 'A4' of a type (line 410)
            A4_387107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 20), self_387106, 'A4')
            # Getting the type of 'self' (line 410)
            self_387108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 29), 'self', False)
            # Obtaining the member 'A2' of a type (line 410)
            A2_387109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 29), self_387108, 'A2')
            # Processing the call keyword arguments (line 409)
            # Getting the type of 'self' (line 410)
            self_387110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 48), 'self', False)
            # Obtaining the member 'structure' of a type (line 410)
            structure_387111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 48), self_387110, 'structure')
            keyword_387112 = structure_387111
            kwargs_387113 = {'structure': keyword_387112}
            # Getting the type of '_smart_matrix_product' (line 409)
            _smart_matrix_product_387105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 23), '_smart_matrix_product', False)
            # Calling _smart_matrix_product(args, kwargs) (line 409)
            _smart_matrix_product_call_result_387114 = invoke(stypy.reporting.localization.Localization(__file__, 409, 23), _smart_matrix_product_387105, *[A4_387107, A2_387109], **kwargs_387113)
            
            # Getting the type of 'self' (line 409)
            self_387115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'self')
            # Setting the type of the member '_A6' of a type (line 409)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 12), self_387115, '_A6', _smart_matrix_product_call_result_387114)

            if more_types_in_union_387104:
                # SSA join for if statement (line 408)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 411)
        self_387116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 15), 'self')
        # Obtaining the member '_A6' of a type (line 411)
        _A6_387117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 15), self_387116, '_A6')
        # Assigning a type to the variable 'stypy_return_type' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'stypy_return_type', _A6_387117)
        
        # ################# End of 'A6(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'A6' in the type store
        # Getting the type of 'stypy_return_type' (line 406)
        stypy_return_type_387118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387118)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'A6'
        return stypy_return_type_387118


    @norecursion
    def A8(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'A8'
        module_type_store = module_type_store.open_function_context('A8', 413, 4, False)
        # Assigning a type to the variable 'self' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.A8.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.A8.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.A8.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.A8.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.A8')
        _ExpmPadeHelper.A8.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.A8.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.A8.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.A8.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.A8.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.A8.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.A8.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.A8', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'A8', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'A8(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 415)
        # Getting the type of 'self' (line 415)
        self_387119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 11), 'self')
        # Obtaining the member '_A8' of a type (line 415)
        _A8_387120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 11), self_387119, '_A8')
        # Getting the type of 'None' (line 415)
        None_387121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 23), 'None')
        
        (may_be_387122, more_types_in_union_387123) = may_be_none(_A8_387120, None_387121)

        if may_be_387122:

            if more_types_in_union_387123:
                # Runtime conditional SSA (line 415)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 416):
            
            # Assigning a Call to a Attribute (line 416):
            
            # Call to _smart_matrix_product(...): (line 416)
            # Processing the call arguments (line 416)
            # Getting the type of 'self' (line 417)
            self_387125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 20), 'self', False)
            # Obtaining the member 'A6' of a type (line 417)
            A6_387126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 20), self_387125, 'A6')
            # Getting the type of 'self' (line 417)
            self_387127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 29), 'self', False)
            # Obtaining the member 'A2' of a type (line 417)
            A2_387128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 29), self_387127, 'A2')
            # Processing the call keyword arguments (line 416)
            # Getting the type of 'self' (line 417)
            self_387129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 48), 'self', False)
            # Obtaining the member 'structure' of a type (line 417)
            structure_387130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 48), self_387129, 'structure')
            keyword_387131 = structure_387130
            kwargs_387132 = {'structure': keyword_387131}
            # Getting the type of '_smart_matrix_product' (line 416)
            _smart_matrix_product_387124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 23), '_smart_matrix_product', False)
            # Calling _smart_matrix_product(args, kwargs) (line 416)
            _smart_matrix_product_call_result_387133 = invoke(stypy.reporting.localization.Localization(__file__, 416, 23), _smart_matrix_product_387124, *[A6_387126, A2_387128], **kwargs_387132)
            
            # Getting the type of 'self' (line 416)
            self_387134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'self')
            # Setting the type of the member '_A8' of a type (line 416)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 12), self_387134, '_A8', _smart_matrix_product_call_result_387133)

            if more_types_in_union_387123:
                # SSA join for if statement (line 415)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 418)
        self_387135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'self')
        # Obtaining the member '_A8' of a type (line 418)
        _A8_387136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 15), self_387135, '_A8')
        # Assigning a type to the variable 'stypy_return_type' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'stypy_return_type', _A8_387136)
        
        # ################# End of 'A8(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'A8' in the type store
        # Getting the type of 'stypy_return_type' (line 413)
        stypy_return_type_387137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387137)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'A8'
        return stypy_return_type_387137


    @norecursion
    def A10(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'A10'
        module_type_store = module_type_store.open_function_context('A10', 420, 4, False)
        # Assigning a type to the variable 'self' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.A10.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.A10.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.A10.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.A10.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.A10')
        _ExpmPadeHelper.A10.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.A10.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.A10.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.A10.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.A10.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.A10.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.A10.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.A10', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'A10', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'A10(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 422)
        # Getting the type of 'self' (line 422)
        self_387138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 11), 'self')
        # Obtaining the member '_A10' of a type (line 422)
        _A10_387139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 11), self_387138, '_A10')
        # Getting the type of 'None' (line 422)
        None_387140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 24), 'None')
        
        (may_be_387141, more_types_in_union_387142) = may_be_none(_A10_387139, None_387140)

        if may_be_387141:

            if more_types_in_union_387142:
                # Runtime conditional SSA (line 422)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 423):
            
            # Assigning a Call to a Attribute (line 423):
            
            # Call to _smart_matrix_product(...): (line 423)
            # Processing the call arguments (line 423)
            # Getting the type of 'self' (line 424)
            self_387144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 20), 'self', False)
            # Obtaining the member 'A4' of a type (line 424)
            A4_387145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 20), self_387144, 'A4')
            # Getting the type of 'self' (line 424)
            self_387146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 29), 'self', False)
            # Obtaining the member 'A6' of a type (line 424)
            A6_387147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 29), self_387146, 'A6')
            # Processing the call keyword arguments (line 423)
            # Getting the type of 'self' (line 424)
            self_387148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 48), 'self', False)
            # Obtaining the member 'structure' of a type (line 424)
            structure_387149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 48), self_387148, 'structure')
            keyword_387150 = structure_387149
            kwargs_387151 = {'structure': keyword_387150}
            # Getting the type of '_smart_matrix_product' (line 423)
            _smart_matrix_product_387143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 24), '_smart_matrix_product', False)
            # Calling _smart_matrix_product(args, kwargs) (line 423)
            _smart_matrix_product_call_result_387152 = invoke(stypy.reporting.localization.Localization(__file__, 423, 24), _smart_matrix_product_387143, *[A4_387145, A6_387147], **kwargs_387151)
            
            # Getting the type of 'self' (line 423)
            self_387153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'self')
            # Setting the type of the member '_A10' of a type (line 423)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 12), self_387153, '_A10', _smart_matrix_product_call_result_387152)

            if more_types_in_union_387142:
                # SSA join for if statement (line 422)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 425)
        self_387154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 15), 'self')
        # Obtaining the member '_A10' of a type (line 425)
        _A10_387155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 15), self_387154, '_A10')
        # Assigning a type to the variable 'stypy_return_type' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'stypy_return_type', _A10_387155)
        
        # ################# End of 'A10(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'A10' in the type store
        # Getting the type of 'stypy_return_type' (line 420)
        stypy_return_type_387156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387156)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'A10'
        return stypy_return_type_387156


    @norecursion
    def d4_tight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'd4_tight'
        module_type_store = module_type_store.open_function_context('d4_tight', 427, 4, False)
        # Assigning a type to the variable 'self' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.d4_tight.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.d4_tight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.d4_tight.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.d4_tight.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.d4_tight')
        _ExpmPadeHelper.d4_tight.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.d4_tight.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.d4_tight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.d4_tight.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.d4_tight.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.d4_tight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.d4_tight.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.d4_tight', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'd4_tight', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'd4_tight(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 429)
        # Getting the type of 'self' (line 429)
        self_387157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 11), 'self')
        # Obtaining the member '_d4_exact' of a type (line 429)
        _d4_exact_387158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 11), self_387157, '_d4_exact')
        # Getting the type of 'None' (line 429)
        None_387159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 29), 'None')
        
        (may_be_387160, more_types_in_union_387161) = may_be_none(_d4_exact_387158, None_387159)

        if may_be_387160:

            if more_types_in_union_387161:
                # Runtime conditional SSA (line 429)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Attribute (line 430):
            
            # Assigning a BinOp to a Attribute (line 430):
            
            # Call to _onenorm(...): (line 430)
            # Processing the call arguments (line 430)
            # Getting the type of 'self' (line 430)
            self_387163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 38), 'self', False)
            # Obtaining the member 'A4' of a type (line 430)
            A4_387164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 38), self_387163, 'A4')
            # Processing the call keyword arguments (line 430)
            kwargs_387165 = {}
            # Getting the type of '_onenorm' (line 430)
            _onenorm_387162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 29), '_onenorm', False)
            # Calling _onenorm(args, kwargs) (line 430)
            _onenorm_call_result_387166 = invoke(stypy.reporting.localization.Localization(__file__, 430, 29), _onenorm_387162, *[A4_387164], **kwargs_387165)
            
            int_387167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 49), 'int')
            float_387168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 51), 'float')
            # Applying the binary operator 'div' (line 430)
            result_div_387169 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 49), 'div', int_387167, float_387168)
            
            # Applying the binary operator '**' (line 430)
            result_pow_387170 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 29), '**', _onenorm_call_result_387166, result_div_387169)
            
            # Getting the type of 'self' (line 430)
            self_387171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'self')
            # Setting the type of the member '_d4_exact' of a type (line 430)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), self_387171, '_d4_exact', result_pow_387170)

            if more_types_in_union_387161:
                # SSA join for if statement (line 429)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 431)
        self_387172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 15), 'self')
        # Obtaining the member '_d4_exact' of a type (line 431)
        _d4_exact_387173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 15), self_387172, '_d4_exact')
        # Assigning a type to the variable 'stypy_return_type' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'stypy_return_type', _d4_exact_387173)
        
        # ################# End of 'd4_tight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'd4_tight' in the type store
        # Getting the type of 'stypy_return_type' (line 427)
        stypy_return_type_387174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387174)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'd4_tight'
        return stypy_return_type_387174


    @norecursion
    def d6_tight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'd6_tight'
        module_type_store = module_type_store.open_function_context('d6_tight', 433, 4, False)
        # Assigning a type to the variable 'self' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.d6_tight.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.d6_tight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.d6_tight.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.d6_tight.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.d6_tight')
        _ExpmPadeHelper.d6_tight.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.d6_tight.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.d6_tight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.d6_tight.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.d6_tight.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.d6_tight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.d6_tight.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.d6_tight', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'd6_tight', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'd6_tight(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 435)
        # Getting the type of 'self' (line 435)
        self_387175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 11), 'self')
        # Obtaining the member '_d6_exact' of a type (line 435)
        _d6_exact_387176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 11), self_387175, '_d6_exact')
        # Getting the type of 'None' (line 435)
        None_387177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 29), 'None')
        
        (may_be_387178, more_types_in_union_387179) = may_be_none(_d6_exact_387176, None_387177)

        if may_be_387178:

            if more_types_in_union_387179:
                # Runtime conditional SSA (line 435)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Attribute (line 436):
            
            # Assigning a BinOp to a Attribute (line 436):
            
            # Call to _onenorm(...): (line 436)
            # Processing the call arguments (line 436)
            # Getting the type of 'self' (line 436)
            self_387181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 38), 'self', False)
            # Obtaining the member 'A6' of a type (line 436)
            A6_387182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 38), self_387181, 'A6')
            # Processing the call keyword arguments (line 436)
            kwargs_387183 = {}
            # Getting the type of '_onenorm' (line 436)
            _onenorm_387180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 29), '_onenorm', False)
            # Calling _onenorm(args, kwargs) (line 436)
            _onenorm_call_result_387184 = invoke(stypy.reporting.localization.Localization(__file__, 436, 29), _onenorm_387180, *[A6_387182], **kwargs_387183)
            
            int_387185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 49), 'int')
            float_387186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 51), 'float')
            # Applying the binary operator 'div' (line 436)
            result_div_387187 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 49), 'div', int_387185, float_387186)
            
            # Applying the binary operator '**' (line 436)
            result_pow_387188 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 29), '**', _onenorm_call_result_387184, result_div_387187)
            
            # Getting the type of 'self' (line 436)
            self_387189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'self')
            # Setting the type of the member '_d6_exact' of a type (line 436)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 12), self_387189, '_d6_exact', result_pow_387188)

            if more_types_in_union_387179:
                # SSA join for if statement (line 435)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 437)
        self_387190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'self')
        # Obtaining the member '_d6_exact' of a type (line 437)
        _d6_exact_387191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 15), self_387190, '_d6_exact')
        # Assigning a type to the variable 'stypy_return_type' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'stypy_return_type', _d6_exact_387191)
        
        # ################# End of 'd6_tight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'd6_tight' in the type store
        # Getting the type of 'stypy_return_type' (line 433)
        stypy_return_type_387192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387192)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'd6_tight'
        return stypy_return_type_387192


    @norecursion
    def d8_tight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'd8_tight'
        module_type_store = module_type_store.open_function_context('d8_tight', 439, 4, False)
        # Assigning a type to the variable 'self' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.d8_tight.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.d8_tight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.d8_tight.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.d8_tight.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.d8_tight')
        _ExpmPadeHelper.d8_tight.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.d8_tight.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.d8_tight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.d8_tight.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.d8_tight.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.d8_tight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.d8_tight.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.d8_tight', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'd8_tight', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'd8_tight(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 441)
        # Getting the type of 'self' (line 441)
        self_387193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 11), 'self')
        # Obtaining the member '_d8_exact' of a type (line 441)
        _d8_exact_387194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 11), self_387193, '_d8_exact')
        # Getting the type of 'None' (line 441)
        None_387195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 29), 'None')
        
        (may_be_387196, more_types_in_union_387197) = may_be_none(_d8_exact_387194, None_387195)

        if may_be_387196:

            if more_types_in_union_387197:
                # Runtime conditional SSA (line 441)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Attribute (line 442):
            
            # Assigning a BinOp to a Attribute (line 442):
            
            # Call to _onenorm(...): (line 442)
            # Processing the call arguments (line 442)
            # Getting the type of 'self' (line 442)
            self_387199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 38), 'self', False)
            # Obtaining the member 'A8' of a type (line 442)
            A8_387200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 38), self_387199, 'A8')
            # Processing the call keyword arguments (line 442)
            kwargs_387201 = {}
            # Getting the type of '_onenorm' (line 442)
            _onenorm_387198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 29), '_onenorm', False)
            # Calling _onenorm(args, kwargs) (line 442)
            _onenorm_call_result_387202 = invoke(stypy.reporting.localization.Localization(__file__, 442, 29), _onenorm_387198, *[A8_387200], **kwargs_387201)
            
            int_387203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 49), 'int')
            float_387204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 51), 'float')
            # Applying the binary operator 'div' (line 442)
            result_div_387205 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 49), 'div', int_387203, float_387204)
            
            # Applying the binary operator '**' (line 442)
            result_pow_387206 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 29), '**', _onenorm_call_result_387202, result_div_387205)
            
            # Getting the type of 'self' (line 442)
            self_387207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'self')
            # Setting the type of the member '_d8_exact' of a type (line 442)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 12), self_387207, '_d8_exact', result_pow_387206)

            if more_types_in_union_387197:
                # SSA join for if statement (line 441)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 443)
        self_387208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 15), 'self')
        # Obtaining the member '_d8_exact' of a type (line 443)
        _d8_exact_387209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 15), self_387208, '_d8_exact')
        # Assigning a type to the variable 'stypy_return_type' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'stypy_return_type', _d8_exact_387209)
        
        # ################# End of 'd8_tight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'd8_tight' in the type store
        # Getting the type of 'stypy_return_type' (line 439)
        stypy_return_type_387210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387210)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'd8_tight'
        return stypy_return_type_387210


    @norecursion
    def d10_tight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'd10_tight'
        module_type_store = module_type_store.open_function_context('d10_tight', 445, 4, False)
        # Assigning a type to the variable 'self' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.d10_tight.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.d10_tight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.d10_tight.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.d10_tight.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.d10_tight')
        _ExpmPadeHelper.d10_tight.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.d10_tight.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.d10_tight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.d10_tight.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.d10_tight.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.d10_tight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.d10_tight.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.d10_tight', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'd10_tight', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'd10_tight(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 447)
        # Getting the type of 'self' (line 447)
        self_387211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 11), 'self')
        # Obtaining the member '_d10_exact' of a type (line 447)
        _d10_exact_387212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 11), self_387211, '_d10_exact')
        # Getting the type of 'None' (line 447)
        None_387213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 30), 'None')
        
        (may_be_387214, more_types_in_union_387215) = may_be_none(_d10_exact_387212, None_387213)

        if may_be_387214:

            if more_types_in_union_387215:
                # Runtime conditional SSA (line 447)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Attribute (line 448):
            
            # Assigning a BinOp to a Attribute (line 448):
            
            # Call to _onenorm(...): (line 448)
            # Processing the call arguments (line 448)
            # Getting the type of 'self' (line 448)
            self_387217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 39), 'self', False)
            # Obtaining the member 'A10' of a type (line 448)
            A10_387218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 39), self_387217, 'A10')
            # Processing the call keyword arguments (line 448)
            kwargs_387219 = {}
            # Getting the type of '_onenorm' (line 448)
            _onenorm_387216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 30), '_onenorm', False)
            # Calling _onenorm(args, kwargs) (line 448)
            _onenorm_call_result_387220 = invoke(stypy.reporting.localization.Localization(__file__, 448, 30), _onenorm_387216, *[A10_387218], **kwargs_387219)
            
            int_387221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 51), 'int')
            float_387222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 53), 'float')
            # Applying the binary operator 'div' (line 448)
            result_div_387223 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 51), 'div', int_387221, float_387222)
            
            # Applying the binary operator '**' (line 448)
            result_pow_387224 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 30), '**', _onenorm_call_result_387220, result_div_387223)
            
            # Getting the type of 'self' (line 448)
            self_387225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'self')
            # Setting the type of the member '_d10_exact' of a type (line 448)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 12), self_387225, '_d10_exact', result_pow_387224)

            if more_types_in_union_387215:
                # SSA join for if statement (line 447)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 449)
        self_387226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 15), 'self')
        # Obtaining the member '_d10_exact' of a type (line 449)
        _d10_exact_387227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 15), self_387226, '_d10_exact')
        # Assigning a type to the variable 'stypy_return_type' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'stypy_return_type', _d10_exact_387227)
        
        # ################# End of 'd10_tight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'd10_tight' in the type store
        # Getting the type of 'stypy_return_type' (line 445)
        stypy_return_type_387228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387228)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'd10_tight'
        return stypy_return_type_387228


    @norecursion
    def d4_loose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'd4_loose'
        module_type_store = module_type_store.open_function_context('d4_loose', 451, 4, False)
        # Assigning a type to the variable 'self' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.d4_loose.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.d4_loose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.d4_loose.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.d4_loose.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.d4_loose')
        _ExpmPadeHelper.d4_loose.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.d4_loose.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.d4_loose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.d4_loose.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.d4_loose.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.d4_loose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.d4_loose.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.d4_loose', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'd4_loose', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'd4_loose(...)' code ##################

        
        # Getting the type of 'self' (line 453)
        self_387229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 11), 'self')
        # Obtaining the member 'use_exact_onenorm' of a type (line 453)
        use_exact_onenorm_387230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 11), self_387229, 'use_exact_onenorm')
        # Testing the type of an if condition (line 453)
        if_condition_387231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 8), use_exact_onenorm_387230)
        # Assigning a type to the variable 'if_condition_387231' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'if_condition_387231', if_condition_387231)
        # SSA begins for if statement (line 453)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 454)
        self_387232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 19), 'self')
        # Obtaining the member 'd4_tight' of a type (line 454)
        d4_tight_387233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 19), self_387232, 'd4_tight')
        # Assigning a type to the variable 'stypy_return_type' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'stypy_return_type', d4_tight_387233)
        # SSA join for if statement (line 453)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 455)
        self_387234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 11), 'self')
        # Obtaining the member '_d4_exact' of a type (line 455)
        _d4_exact_387235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 11), self_387234, '_d4_exact')
        # Getting the type of 'None' (line 455)
        None_387236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 33), 'None')
        # Applying the binary operator 'isnot' (line 455)
        result_is_not_387237 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 11), 'isnot', _d4_exact_387235, None_387236)
        
        # Testing the type of an if condition (line 455)
        if_condition_387238 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 8), result_is_not_387237)
        # Assigning a type to the variable 'if_condition_387238' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'if_condition_387238', if_condition_387238)
        # SSA begins for if statement (line 455)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 456)
        self_387239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 19), 'self')
        # Obtaining the member '_d4_exact' of a type (line 456)
        _d4_exact_387240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 19), self_387239, '_d4_exact')
        # Assigning a type to the variable 'stypy_return_type' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'stypy_return_type', _d4_exact_387240)
        # SSA branch for the else part of an if statement (line 455)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 458)
        # Getting the type of 'self' (line 458)
        self_387241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 15), 'self')
        # Obtaining the member '_d4_approx' of a type (line 458)
        _d4_approx_387242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 15), self_387241, '_d4_approx')
        # Getting the type of 'None' (line 458)
        None_387243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 34), 'None')
        
        (may_be_387244, more_types_in_union_387245) = may_be_none(_d4_approx_387242, None_387243)

        if may_be_387244:

            if more_types_in_union_387245:
                # Runtime conditional SSA (line 458)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Attribute (line 459):
            
            # Assigning a BinOp to a Attribute (line 459):
            
            # Call to _onenormest_matrix_power(...): (line 459)
            # Processing the call arguments (line 459)
            # Getting the type of 'self' (line 459)
            self_387247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 59), 'self', False)
            # Obtaining the member 'A2' of a type (line 459)
            A2_387248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 59), self_387247, 'A2')
            int_387249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 68), 'int')
            # Processing the call keyword arguments (line 459)
            # Getting the type of 'self' (line 460)
            self_387250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'self', False)
            # Obtaining the member 'structure' of a type (line 460)
            structure_387251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 34), self_387250, 'structure')
            keyword_387252 = structure_387251
            kwargs_387253 = {'structure': keyword_387252}
            # Getting the type of '_onenormest_matrix_power' (line 459)
            _onenormest_matrix_power_387246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 34), '_onenormest_matrix_power', False)
            # Calling _onenormest_matrix_power(args, kwargs) (line 459)
            _onenormest_matrix_power_call_result_387254 = invoke(stypy.reporting.localization.Localization(__file__, 459, 34), _onenormest_matrix_power_387246, *[A2_387248, int_387249], **kwargs_387253)
            
            int_387255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 52), 'int')
            float_387256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 54), 'float')
            # Applying the binary operator 'div' (line 460)
            result_div_387257 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 52), 'div', int_387255, float_387256)
            
            # Applying the binary operator '**' (line 459)
            result_pow_387258 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 34), '**', _onenormest_matrix_power_call_result_387254, result_div_387257)
            
            # Getting the type of 'self' (line 459)
            self_387259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 16), 'self')
            # Setting the type of the member '_d4_approx' of a type (line 459)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 16), self_387259, '_d4_approx', result_pow_387258)

            if more_types_in_union_387245:
                # SSA join for if statement (line 458)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 461)
        self_387260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 19), 'self')
        # Obtaining the member '_d4_approx' of a type (line 461)
        _d4_approx_387261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 19), self_387260, '_d4_approx')
        # Assigning a type to the variable 'stypy_return_type' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'stypy_return_type', _d4_approx_387261)
        # SSA join for if statement (line 455)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'd4_loose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'd4_loose' in the type store
        # Getting the type of 'stypy_return_type' (line 451)
        stypy_return_type_387262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387262)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'd4_loose'
        return stypy_return_type_387262


    @norecursion
    def d6_loose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'd6_loose'
        module_type_store = module_type_store.open_function_context('d6_loose', 463, 4, False)
        # Assigning a type to the variable 'self' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.d6_loose.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.d6_loose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.d6_loose.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.d6_loose.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.d6_loose')
        _ExpmPadeHelper.d6_loose.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.d6_loose.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.d6_loose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.d6_loose.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.d6_loose.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.d6_loose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.d6_loose.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.d6_loose', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'd6_loose', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'd6_loose(...)' code ##################

        
        # Getting the type of 'self' (line 465)
        self_387263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 11), 'self')
        # Obtaining the member 'use_exact_onenorm' of a type (line 465)
        use_exact_onenorm_387264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 11), self_387263, 'use_exact_onenorm')
        # Testing the type of an if condition (line 465)
        if_condition_387265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 465, 8), use_exact_onenorm_387264)
        # Assigning a type to the variable 'if_condition_387265' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'if_condition_387265', if_condition_387265)
        # SSA begins for if statement (line 465)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 466)
        self_387266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 19), 'self')
        # Obtaining the member 'd6_tight' of a type (line 466)
        d6_tight_387267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 19), self_387266, 'd6_tight')
        # Assigning a type to the variable 'stypy_return_type' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'stypy_return_type', d6_tight_387267)
        # SSA join for if statement (line 465)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 467)
        self_387268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 11), 'self')
        # Obtaining the member '_d6_exact' of a type (line 467)
        _d6_exact_387269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 11), self_387268, '_d6_exact')
        # Getting the type of 'None' (line 467)
        None_387270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 33), 'None')
        # Applying the binary operator 'isnot' (line 467)
        result_is_not_387271 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 11), 'isnot', _d6_exact_387269, None_387270)
        
        # Testing the type of an if condition (line 467)
        if_condition_387272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 467, 8), result_is_not_387271)
        # Assigning a type to the variable 'if_condition_387272' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'if_condition_387272', if_condition_387272)
        # SSA begins for if statement (line 467)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 468)
        self_387273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 19), 'self')
        # Obtaining the member '_d6_exact' of a type (line 468)
        _d6_exact_387274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 19), self_387273, '_d6_exact')
        # Assigning a type to the variable 'stypy_return_type' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'stypy_return_type', _d6_exact_387274)
        # SSA branch for the else part of an if statement (line 467)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 470)
        # Getting the type of 'self' (line 470)
        self_387275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 15), 'self')
        # Obtaining the member '_d6_approx' of a type (line 470)
        _d6_approx_387276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 15), self_387275, '_d6_approx')
        # Getting the type of 'None' (line 470)
        None_387277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 34), 'None')
        
        (may_be_387278, more_types_in_union_387279) = may_be_none(_d6_approx_387276, None_387277)

        if may_be_387278:

            if more_types_in_union_387279:
                # Runtime conditional SSA (line 470)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Attribute (line 471):
            
            # Assigning a BinOp to a Attribute (line 471):
            
            # Call to _onenormest_matrix_power(...): (line 471)
            # Processing the call arguments (line 471)
            # Getting the type of 'self' (line 471)
            self_387281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 59), 'self', False)
            # Obtaining the member 'A2' of a type (line 471)
            A2_387282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 59), self_387281, 'A2')
            int_387283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 68), 'int')
            # Processing the call keyword arguments (line 471)
            # Getting the type of 'self' (line 472)
            self_387284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 34), 'self', False)
            # Obtaining the member 'structure' of a type (line 472)
            structure_387285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 34), self_387284, 'structure')
            keyword_387286 = structure_387285
            kwargs_387287 = {'structure': keyword_387286}
            # Getting the type of '_onenormest_matrix_power' (line 471)
            _onenormest_matrix_power_387280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 34), '_onenormest_matrix_power', False)
            # Calling _onenormest_matrix_power(args, kwargs) (line 471)
            _onenormest_matrix_power_call_result_387288 = invoke(stypy.reporting.localization.Localization(__file__, 471, 34), _onenormest_matrix_power_387280, *[A2_387282, int_387283], **kwargs_387287)
            
            int_387289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 52), 'int')
            float_387290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 54), 'float')
            # Applying the binary operator 'div' (line 472)
            result_div_387291 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 52), 'div', int_387289, float_387290)
            
            # Applying the binary operator '**' (line 471)
            result_pow_387292 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 34), '**', _onenormest_matrix_power_call_result_387288, result_div_387291)
            
            # Getting the type of 'self' (line 471)
            self_387293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'self')
            # Setting the type of the member '_d6_approx' of a type (line 471)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 16), self_387293, '_d6_approx', result_pow_387292)

            if more_types_in_union_387279:
                # SSA join for if statement (line 470)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 473)
        self_387294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 19), 'self')
        # Obtaining the member '_d6_approx' of a type (line 473)
        _d6_approx_387295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 19), self_387294, '_d6_approx')
        # Assigning a type to the variable 'stypy_return_type' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'stypy_return_type', _d6_approx_387295)
        # SSA join for if statement (line 467)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'd6_loose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'd6_loose' in the type store
        # Getting the type of 'stypy_return_type' (line 463)
        stypy_return_type_387296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387296)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'd6_loose'
        return stypy_return_type_387296


    @norecursion
    def d8_loose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'd8_loose'
        module_type_store = module_type_store.open_function_context('d8_loose', 475, 4, False)
        # Assigning a type to the variable 'self' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.d8_loose.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.d8_loose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.d8_loose.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.d8_loose.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.d8_loose')
        _ExpmPadeHelper.d8_loose.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.d8_loose.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.d8_loose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.d8_loose.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.d8_loose.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.d8_loose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.d8_loose.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.d8_loose', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'd8_loose', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'd8_loose(...)' code ##################

        
        # Getting the type of 'self' (line 477)
        self_387297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 11), 'self')
        # Obtaining the member 'use_exact_onenorm' of a type (line 477)
        use_exact_onenorm_387298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 11), self_387297, 'use_exact_onenorm')
        # Testing the type of an if condition (line 477)
        if_condition_387299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 477, 8), use_exact_onenorm_387298)
        # Assigning a type to the variable 'if_condition_387299' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'if_condition_387299', if_condition_387299)
        # SSA begins for if statement (line 477)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 478)
        self_387300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 19), 'self')
        # Obtaining the member 'd8_tight' of a type (line 478)
        d8_tight_387301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 19), self_387300, 'd8_tight')
        # Assigning a type to the variable 'stypy_return_type' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'stypy_return_type', d8_tight_387301)
        # SSA join for if statement (line 477)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 479)
        self_387302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 11), 'self')
        # Obtaining the member '_d8_exact' of a type (line 479)
        _d8_exact_387303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 11), self_387302, '_d8_exact')
        # Getting the type of 'None' (line 479)
        None_387304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 33), 'None')
        # Applying the binary operator 'isnot' (line 479)
        result_is_not_387305 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 11), 'isnot', _d8_exact_387303, None_387304)
        
        # Testing the type of an if condition (line 479)
        if_condition_387306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 479, 8), result_is_not_387305)
        # Assigning a type to the variable 'if_condition_387306' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'if_condition_387306', if_condition_387306)
        # SSA begins for if statement (line 479)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 480)
        self_387307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 19), 'self')
        # Obtaining the member '_d8_exact' of a type (line 480)
        _d8_exact_387308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 19), self_387307, '_d8_exact')
        # Assigning a type to the variable 'stypy_return_type' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'stypy_return_type', _d8_exact_387308)
        # SSA branch for the else part of an if statement (line 479)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 482)
        # Getting the type of 'self' (line 482)
        self_387309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 15), 'self')
        # Obtaining the member '_d8_approx' of a type (line 482)
        _d8_approx_387310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 15), self_387309, '_d8_approx')
        # Getting the type of 'None' (line 482)
        None_387311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 34), 'None')
        
        (may_be_387312, more_types_in_union_387313) = may_be_none(_d8_approx_387310, None_387311)

        if may_be_387312:

            if more_types_in_union_387313:
                # Runtime conditional SSA (line 482)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Attribute (line 483):
            
            # Assigning a BinOp to a Attribute (line 483):
            
            # Call to _onenormest_matrix_power(...): (line 483)
            # Processing the call arguments (line 483)
            # Getting the type of 'self' (line 483)
            self_387315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 59), 'self', False)
            # Obtaining the member 'A4' of a type (line 483)
            A4_387316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 59), self_387315, 'A4')
            int_387317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 68), 'int')
            # Processing the call keyword arguments (line 483)
            # Getting the type of 'self' (line 484)
            self_387318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 34), 'self', False)
            # Obtaining the member 'structure' of a type (line 484)
            structure_387319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 34), self_387318, 'structure')
            keyword_387320 = structure_387319
            kwargs_387321 = {'structure': keyword_387320}
            # Getting the type of '_onenormest_matrix_power' (line 483)
            _onenormest_matrix_power_387314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 34), '_onenormest_matrix_power', False)
            # Calling _onenormest_matrix_power(args, kwargs) (line 483)
            _onenormest_matrix_power_call_result_387322 = invoke(stypy.reporting.localization.Localization(__file__, 483, 34), _onenormest_matrix_power_387314, *[A4_387316, int_387317], **kwargs_387321)
            
            int_387323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 52), 'int')
            float_387324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 54), 'float')
            # Applying the binary operator 'div' (line 484)
            result_div_387325 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 52), 'div', int_387323, float_387324)
            
            # Applying the binary operator '**' (line 483)
            result_pow_387326 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 34), '**', _onenormest_matrix_power_call_result_387322, result_div_387325)
            
            # Getting the type of 'self' (line 483)
            self_387327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 16), 'self')
            # Setting the type of the member '_d8_approx' of a type (line 483)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 16), self_387327, '_d8_approx', result_pow_387326)

            if more_types_in_union_387313:
                # SSA join for if statement (line 482)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 485)
        self_387328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 19), 'self')
        # Obtaining the member '_d8_approx' of a type (line 485)
        _d8_approx_387329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 19), self_387328, '_d8_approx')
        # Assigning a type to the variable 'stypy_return_type' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'stypy_return_type', _d8_approx_387329)
        # SSA join for if statement (line 479)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'd8_loose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'd8_loose' in the type store
        # Getting the type of 'stypy_return_type' (line 475)
        stypy_return_type_387330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387330)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'd8_loose'
        return stypy_return_type_387330


    @norecursion
    def d10_loose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'd10_loose'
        module_type_store = module_type_store.open_function_context('d10_loose', 487, 4, False)
        # Assigning a type to the variable 'self' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.d10_loose.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.d10_loose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.d10_loose.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.d10_loose.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.d10_loose')
        _ExpmPadeHelper.d10_loose.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.d10_loose.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.d10_loose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.d10_loose.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.d10_loose.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.d10_loose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.d10_loose.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.d10_loose', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'd10_loose', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'd10_loose(...)' code ##################

        
        # Getting the type of 'self' (line 489)
        self_387331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 11), 'self')
        # Obtaining the member 'use_exact_onenorm' of a type (line 489)
        use_exact_onenorm_387332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 11), self_387331, 'use_exact_onenorm')
        # Testing the type of an if condition (line 489)
        if_condition_387333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 489, 8), use_exact_onenorm_387332)
        # Assigning a type to the variable 'if_condition_387333' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'if_condition_387333', if_condition_387333)
        # SSA begins for if statement (line 489)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 490)
        self_387334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 19), 'self')
        # Obtaining the member 'd10_tight' of a type (line 490)
        d10_tight_387335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 19), self_387334, 'd10_tight')
        # Assigning a type to the variable 'stypy_return_type' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'stypy_return_type', d10_tight_387335)
        # SSA join for if statement (line 489)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 491)
        self_387336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 11), 'self')
        # Obtaining the member '_d10_exact' of a type (line 491)
        _d10_exact_387337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 11), self_387336, '_d10_exact')
        # Getting the type of 'None' (line 491)
        None_387338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 34), 'None')
        # Applying the binary operator 'isnot' (line 491)
        result_is_not_387339 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 11), 'isnot', _d10_exact_387337, None_387338)
        
        # Testing the type of an if condition (line 491)
        if_condition_387340 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 491, 8), result_is_not_387339)
        # Assigning a type to the variable 'if_condition_387340' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'if_condition_387340', if_condition_387340)
        # SSA begins for if statement (line 491)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 492)
        self_387341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 19), 'self')
        # Obtaining the member '_d10_exact' of a type (line 492)
        _d10_exact_387342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 19), self_387341, '_d10_exact')
        # Assigning a type to the variable 'stypy_return_type' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'stypy_return_type', _d10_exact_387342)
        # SSA branch for the else part of an if statement (line 491)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 494)
        # Getting the type of 'self' (line 494)
        self_387343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 15), 'self')
        # Obtaining the member '_d10_approx' of a type (line 494)
        _d10_approx_387344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 15), self_387343, '_d10_approx')
        # Getting the type of 'None' (line 494)
        None_387345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 35), 'None')
        
        (may_be_387346, more_types_in_union_387347) = may_be_none(_d10_approx_387344, None_387345)

        if may_be_387346:

            if more_types_in_union_387347:
                # Runtime conditional SSA (line 494)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Attribute (line 495):
            
            # Assigning a BinOp to a Attribute (line 495):
            
            # Call to _onenormest_product(...): (line 495)
            # Processing the call arguments (line 495)
            
            # Obtaining an instance of the builtin type 'tuple' (line 495)
            tuple_387349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 56), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 495)
            # Adding element type (line 495)
            # Getting the type of 'self' (line 495)
            self_387350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 56), 'self', False)
            # Obtaining the member 'A4' of a type (line 495)
            A4_387351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 56), self_387350, 'A4')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 56), tuple_387349, A4_387351)
            # Adding element type (line 495)
            # Getting the type of 'self' (line 495)
            self_387352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 65), 'self', False)
            # Obtaining the member 'A6' of a type (line 495)
            A6_387353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 65), self_387352, 'A6')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 56), tuple_387349, A6_387353)
            
            # Processing the call keyword arguments (line 495)
            # Getting the type of 'self' (line 496)
            self_387354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 34), 'self', False)
            # Obtaining the member 'structure' of a type (line 496)
            structure_387355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 34), self_387354, 'structure')
            keyword_387356 = structure_387355
            kwargs_387357 = {'structure': keyword_387356}
            # Getting the type of '_onenormest_product' (line 495)
            _onenormest_product_387348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 35), '_onenormest_product', False)
            # Calling _onenormest_product(args, kwargs) (line 495)
            _onenormest_product_call_result_387358 = invoke(stypy.reporting.localization.Localization(__file__, 495, 35), _onenormest_product_387348, *[tuple_387349], **kwargs_387357)
            
            int_387359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 52), 'int')
            float_387360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 54), 'float')
            # Applying the binary operator 'div' (line 496)
            result_div_387361 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 52), 'div', int_387359, float_387360)
            
            # Applying the binary operator '**' (line 495)
            result_pow_387362 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 35), '**', _onenormest_product_call_result_387358, result_div_387361)
            
            # Getting the type of 'self' (line 495)
            self_387363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 16), 'self')
            # Setting the type of the member '_d10_approx' of a type (line 495)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 16), self_387363, '_d10_approx', result_pow_387362)

            if more_types_in_union_387347:
                # SSA join for if statement (line 494)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 497)
        self_387364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 19), 'self')
        # Obtaining the member '_d10_approx' of a type (line 497)
        _d10_approx_387365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 19), self_387364, '_d10_approx')
        # Assigning a type to the variable 'stypy_return_type' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'stypy_return_type', _d10_approx_387365)
        # SSA join for if statement (line 491)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'd10_loose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'd10_loose' in the type store
        # Getting the type of 'stypy_return_type' (line 487)
        stypy_return_type_387366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387366)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'd10_loose'
        return stypy_return_type_387366


    @norecursion
    def pade3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pade3'
        module_type_store = module_type_store.open_function_context('pade3', 499, 4, False)
        # Assigning a type to the variable 'self' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.pade3.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.pade3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.pade3.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.pade3.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.pade3')
        _ExpmPadeHelper.pade3.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.pade3.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.pade3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.pade3.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.pade3.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.pade3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.pade3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.pade3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pade3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pade3(...)' code ##################

        
        # Assigning a Tuple to a Name (line 500):
        
        # Assigning a Tuple to a Name (line 500):
        
        # Obtaining an instance of the builtin type 'tuple' (line 500)
        tuple_387367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 500)
        # Adding element type (line 500)
        float_387368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 13), tuple_387367, float_387368)
        # Adding element type (line 500)
        float_387369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 13), tuple_387367, float_387369)
        # Adding element type (line 500)
        float_387370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 13), tuple_387367, float_387370)
        # Adding element type (line 500)
        float_387371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 13), tuple_387367, float_387371)
        
        # Assigning a type to the variable 'b' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'b', tuple_387367)
        
        # Assigning a Call to a Name (line 501):
        
        # Assigning a Call to a Name (line 501):
        
        # Call to _smart_matrix_product(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of 'self' (line 501)
        self_387373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 34), 'self', False)
        # Obtaining the member 'A' of a type (line 501)
        A_387374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 34), self_387373, 'A')
        
        # Obtaining the type of the subscript
        int_387375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 18), 'int')
        # Getting the type of 'b' (line 502)
        b_387376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 502)
        getitem___387377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 16), b_387376, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 502)
        subscript_call_result_387378 = invoke(stypy.reporting.localization.Localization(__file__, 502, 16), getitem___387377, int_387375)
        
        # Getting the type of 'self' (line 502)
        self_387379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 21), 'self', False)
        # Obtaining the member 'A2' of a type (line 502)
        A2_387380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 21), self_387379, 'A2')
        # Applying the binary operator '*' (line 502)
        result_mul_387381 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 16), '*', subscript_call_result_387378, A2_387380)
        
        
        # Obtaining the type of the subscript
        int_387382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 33), 'int')
        # Getting the type of 'b' (line 502)
        b_387383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 31), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 502)
        getitem___387384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 31), b_387383, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 502)
        subscript_call_result_387385 = invoke(stypy.reporting.localization.Localization(__file__, 502, 31), getitem___387384, int_387382)
        
        # Getting the type of 'self' (line 502)
        self_387386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 36), 'self', False)
        # Obtaining the member 'ident' of a type (line 502)
        ident_387387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 36), self_387386, 'ident')
        # Applying the binary operator '*' (line 502)
        result_mul_387388 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 31), '*', subscript_call_result_387385, ident_387387)
        
        # Applying the binary operator '+' (line 502)
        result_add_387389 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 16), '+', result_mul_387381, result_mul_387388)
        
        # Processing the call keyword arguments (line 501)
        # Getting the type of 'self' (line 503)
        self_387390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 26), 'self', False)
        # Obtaining the member 'structure' of a type (line 503)
        structure_387391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 26), self_387390, 'structure')
        keyword_387392 = structure_387391
        kwargs_387393 = {'structure': keyword_387392}
        # Getting the type of '_smart_matrix_product' (line 501)
        _smart_matrix_product_387372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), '_smart_matrix_product', False)
        # Calling _smart_matrix_product(args, kwargs) (line 501)
        _smart_matrix_product_call_result_387394 = invoke(stypy.reporting.localization.Localization(__file__, 501, 12), _smart_matrix_product_387372, *[A_387374, result_add_387389], **kwargs_387393)
        
        # Assigning a type to the variable 'U' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'U', _smart_matrix_product_call_result_387394)
        
        # Assigning a BinOp to a Name (line 504):
        
        # Assigning a BinOp to a Name (line 504):
        
        # Obtaining the type of the subscript
        int_387395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 14), 'int')
        # Getting the type of 'b' (line 504)
        b_387396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'b')
        # Obtaining the member '__getitem__' of a type (line 504)
        getitem___387397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 12), b_387396, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 504)
        subscript_call_result_387398 = invoke(stypy.reporting.localization.Localization(__file__, 504, 12), getitem___387397, int_387395)
        
        # Getting the type of 'self' (line 504)
        self_387399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 17), 'self')
        # Obtaining the member 'A2' of a type (line 504)
        A2_387400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 17), self_387399, 'A2')
        # Applying the binary operator '*' (line 504)
        result_mul_387401 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 12), '*', subscript_call_result_387398, A2_387400)
        
        
        # Obtaining the type of the subscript
        int_387402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 29), 'int')
        # Getting the type of 'b' (line 504)
        b_387403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 27), 'b')
        # Obtaining the member '__getitem__' of a type (line 504)
        getitem___387404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 27), b_387403, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 504)
        subscript_call_result_387405 = invoke(stypy.reporting.localization.Localization(__file__, 504, 27), getitem___387404, int_387402)
        
        # Getting the type of 'self' (line 504)
        self_387406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 32), 'self')
        # Obtaining the member 'ident' of a type (line 504)
        ident_387407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 32), self_387406, 'ident')
        # Applying the binary operator '*' (line 504)
        result_mul_387408 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 27), '*', subscript_call_result_387405, ident_387407)
        
        # Applying the binary operator '+' (line 504)
        result_add_387409 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 12), '+', result_mul_387401, result_mul_387408)
        
        # Assigning a type to the variable 'V' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'V', result_add_387409)
        
        # Obtaining an instance of the builtin type 'tuple' (line 505)
        tuple_387410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 505)
        # Adding element type (line 505)
        # Getting the type of 'U' (line 505)
        U_387411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 15), 'U')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 15), tuple_387410, U_387411)
        # Adding element type (line 505)
        # Getting the type of 'V' (line 505)
        V_387412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 18), 'V')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 15), tuple_387410, V_387412)
        
        # Assigning a type to the variable 'stypy_return_type' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'stypy_return_type', tuple_387410)
        
        # ################# End of 'pade3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pade3' in the type store
        # Getting the type of 'stypy_return_type' (line 499)
        stypy_return_type_387413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387413)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pade3'
        return stypy_return_type_387413


    @norecursion
    def pade5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pade5'
        module_type_store = module_type_store.open_function_context('pade5', 507, 4, False)
        # Assigning a type to the variable 'self' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.pade5.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.pade5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.pade5.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.pade5.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.pade5')
        _ExpmPadeHelper.pade5.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.pade5.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.pade5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.pade5.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.pade5.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.pade5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.pade5.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.pade5', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pade5', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pade5(...)' code ##################

        
        # Assigning a Tuple to a Name (line 508):
        
        # Assigning a Tuple to a Name (line 508):
        
        # Obtaining an instance of the builtin type 'tuple' (line 508)
        tuple_387414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 508)
        # Adding element type (line 508)
        float_387415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 13), tuple_387414, float_387415)
        # Adding element type (line 508)
        float_387416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 13), tuple_387414, float_387416)
        # Adding element type (line 508)
        float_387417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 13), tuple_387414, float_387417)
        # Adding element type (line 508)
        float_387418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 13), tuple_387414, float_387418)
        # Adding element type (line 508)
        float_387419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 13), tuple_387414, float_387419)
        # Adding element type (line 508)
        float_387420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 13), tuple_387414, float_387420)
        
        # Assigning a type to the variable 'b' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'b', tuple_387414)
        
        # Assigning a Call to a Name (line 509):
        
        # Assigning a Call to a Name (line 509):
        
        # Call to _smart_matrix_product(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'self' (line 509)
        self_387422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 34), 'self', False)
        # Obtaining the member 'A' of a type (line 509)
        A_387423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 34), self_387422, 'A')
        
        # Obtaining the type of the subscript
        int_387424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 18), 'int')
        # Getting the type of 'b' (line 510)
        b_387425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 510)
        getitem___387426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 16), b_387425, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 510)
        subscript_call_result_387427 = invoke(stypy.reporting.localization.Localization(__file__, 510, 16), getitem___387426, int_387424)
        
        # Getting the type of 'self' (line 510)
        self_387428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 21), 'self', False)
        # Obtaining the member 'A4' of a type (line 510)
        A4_387429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 21), self_387428, 'A4')
        # Applying the binary operator '*' (line 510)
        result_mul_387430 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 16), '*', subscript_call_result_387427, A4_387429)
        
        
        # Obtaining the type of the subscript
        int_387431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 33), 'int')
        # Getting the type of 'b' (line 510)
        b_387432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 31), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 510)
        getitem___387433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 31), b_387432, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 510)
        subscript_call_result_387434 = invoke(stypy.reporting.localization.Localization(__file__, 510, 31), getitem___387433, int_387431)
        
        # Getting the type of 'self' (line 510)
        self_387435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 36), 'self', False)
        # Obtaining the member 'A2' of a type (line 510)
        A2_387436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 36), self_387435, 'A2')
        # Applying the binary operator '*' (line 510)
        result_mul_387437 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 31), '*', subscript_call_result_387434, A2_387436)
        
        # Applying the binary operator '+' (line 510)
        result_add_387438 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 16), '+', result_mul_387430, result_mul_387437)
        
        
        # Obtaining the type of the subscript
        int_387439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 48), 'int')
        # Getting the type of 'b' (line 510)
        b_387440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 46), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 510)
        getitem___387441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 46), b_387440, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 510)
        subscript_call_result_387442 = invoke(stypy.reporting.localization.Localization(__file__, 510, 46), getitem___387441, int_387439)
        
        # Getting the type of 'self' (line 510)
        self_387443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 51), 'self', False)
        # Obtaining the member 'ident' of a type (line 510)
        ident_387444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 51), self_387443, 'ident')
        # Applying the binary operator '*' (line 510)
        result_mul_387445 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 46), '*', subscript_call_result_387442, ident_387444)
        
        # Applying the binary operator '+' (line 510)
        result_add_387446 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 44), '+', result_add_387438, result_mul_387445)
        
        # Processing the call keyword arguments (line 509)
        # Getting the type of 'self' (line 511)
        self_387447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 26), 'self', False)
        # Obtaining the member 'structure' of a type (line 511)
        structure_387448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 26), self_387447, 'structure')
        keyword_387449 = structure_387448
        kwargs_387450 = {'structure': keyword_387449}
        # Getting the type of '_smart_matrix_product' (line 509)
        _smart_matrix_product_387421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), '_smart_matrix_product', False)
        # Calling _smart_matrix_product(args, kwargs) (line 509)
        _smart_matrix_product_call_result_387451 = invoke(stypy.reporting.localization.Localization(__file__, 509, 12), _smart_matrix_product_387421, *[A_387423, result_add_387446], **kwargs_387450)
        
        # Assigning a type to the variable 'U' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'U', _smart_matrix_product_call_result_387451)
        
        # Assigning a BinOp to a Name (line 512):
        
        # Assigning a BinOp to a Name (line 512):
        
        # Obtaining the type of the subscript
        int_387452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 14), 'int')
        # Getting the type of 'b' (line 512)
        b_387453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'b')
        # Obtaining the member '__getitem__' of a type (line 512)
        getitem___387454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 12), b_387453, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 512)
        subscript_call_result_387455 = invoke(stypy.reporting.localization.Localization(__file__, 512, 12), getitem___387454, int_387452)
        
        # Getting the type of 'self' (line 512)
        self_387456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 17), 'self')
        # Obtaining the member 'A4' of a type (line 512)
        A4_387457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 17), self_387456, 'A4')
        # Applying the binary operator '*' (line 512)
        result_mul_387458 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 12), '*', subscript_call_result_387455, A4_387457)
        
        
        # Obtaining the type of the subscript
        int_387459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 29), 'int')
        # Getting the type of 'b' (line 512)
        b_387460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 27), 'b')
        # Obtaining the member '__getitem__' of a type (line 512)
        getitem___387461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 27), b_387460, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 512)
        subscript_call_result_387462 = invoke(stypy.reporting.localization.Localization(__file__, 512, 27), getitem___387461, int_387459)
        
        # Getting the type of 'self' (line 512)
        self_387463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 32), 'self')
        # Obtaining the member 'A2' of a type (line 512)
        A2_387464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 32), self_387463, 'A2')
        # Applying the binary operator '*' (line 512)
        result_mul_387465 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 27), '*', subscript_call_result_387462, A2_387464)
        
        # Applying the binary operator '+' (line 512)
        result_add_387466 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 12), '+', result_mul_387458, result_mul_387465)
        
        
        # Obtaining the type of the subscript
        int_387467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 44), 'int')
        # Getting the type of 'b' (line 512)
        b_387468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 42), 'b')
        # Obtaining the member '__getitem__' of a type (line 512)
        getitem___387469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 42), b_387468, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 512)
        subscript_call_result_387470 = invoke(stypy.reporting.localization.Localization(__file__, 512, 42), getitem___387469, int_387467)
        
        # Getting the type of 'self' (line 512)
        self_387471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 47), 'self')
        # Obtaining the member 'ident' of a type (line 512)
        ident_387472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 47), self_387471, 'ident')
        # Applying the binary operator '*' (line 512)
        result_mul_387473 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 42), '*', subscript_call_result_387470, ident_387472)
        
        # Applying the binary operator '+' (line 512)
        result_add_387474 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 40), '+', result_add_387466, result_mul_387473)
        
        # Assigning a type to the variable 'V' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'V', result_add_387474)
        
        # Obtaining an instance of the builtin type 'tuple' (line 513)
        tuple_387475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 513)
        # Adding element type (line 513)
        # Getting the type of 'U' (line 513)
        U_387476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 15), 'U')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 15), tuple_387475, U_387476)
        # Adding element type (line 513)
        # Getting the type of 'V' (line 513)
        V_387477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 18), 'V')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 15), tuple_387475, V_387477)
        
        # Assigning a type to the variable 'stypy_return_type' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'stypy_return_type', tuple_387475)
        
        # ################# End of 'pade5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pade5' in the type store
        # Getting the type of 'stypy_return_type' (line 507)
        stypy_return_type_387478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387478)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pade5'
        return stypy_return_type_387478


    @norecursion
    def pade7(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pade7'
        module_type_store = module_type_store.open_function_context('pade7', 515, 4, False)
        # Assigning a type to the variable 'self' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.pade7.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.pade7.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.pade7.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.pade7.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.pade7')
        _ExpmPadeHelper.pade7.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.pade7.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.pade7.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.pade7.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.pade7.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.pade7.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.pade7.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.pade7', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pade7', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pade7(...)' code ##################

        
        # Assigning a Tuple to a Name (line 516):
        
        # Assigning a Tuple to a Name (line 516):
        
        # Obtaining an instance of the builtin type 'tuple' (line 516)
        tuple_387479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 516)
        # Adding element type (line 516)
        float_387480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 13), tuple_387479, float_387480)
        # Adding element type (line 516)
        float_387481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 13), tuple_387479, float_387481)
        # Adding element type (line 516)
        float_387482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 13), tuple_387479, float_387482)
        # Adding element type (line 516)
        float_387483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 13), tuple_387479, float_387483)
        # Adding element type (line 516)
        float_387484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 13), tuple_387479, float_387484)
        # Adding element type (line 516)
        float_387485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 13), tuple_387479, float_387485)
        # Adding element type (line 516)
        float_387486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 68), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 13), tuple_387479, float_387486)
        # Adding element type (line 516)
        float_387487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 13), tuple_387479, float_387487)
        
        # Assigning a type to the variable 'b' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'b', tuple_387479)
        
        # Assigning a Call to a Name (line 517):
        
        # Assigning a Call to a Name (line 517):
        
        # Call to _smart_matrix_product(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 'self' (line 517)
        self_387489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 34), 'self', False)
        # Obtaining the member 'A' of a type (line 517)
        A_387490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 34), self_387489, 'A')
        
        # Obtaining the type of the subscript
        int_387491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 18), 'int')
        # Getting the type of 'b' (line 518)
        b_387492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 16), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 518)
        getitem___387493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 16), b_387492, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 518)
        subscript_call_result_387494 = invoke(stypy.reporting.localization.Localization(__file__, 518, 16), getitem___387493, int_387491)
        
        # Getting the type of 'self' (line 518)
        self_387495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 21), 'self', False)
        # Obtaining the member 'A6' of a type (line 518)
        A6_387496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 21), self_387495, 'A6')
        # Applying the binary operator '*' (line 518)
        result_mul_387497 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 16), '*', subscript_call_result_387494, A6_387496)
        
        
        # Obtaining the type of the subscript
        int_387498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 33), 'int')
        # Getting the type of 'b' (line 518)
        b_387499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 31), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 518)
        getitem___387500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 31), b_387499, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 518)
        subscript_call_result_387501 = invoke(stypy.reporting.localization.Localization(__file__, 518, 31), getitem___387500, int_387498)
        
        # Getting the type of 'self' (line 518)
        self_387502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 36), 'self', False)
        # Obtaining the member 'A4' of a type (line 518)
        A4_387503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 36), self_387502, 'A4')
        # Applying the binary operator '*' (line 518)
        result_mul_387504 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 31), '*', subscript_call_result_387501, A4_387503)
        
        # Applying the binary operator '+' (line 518)
        result_add_387505 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 16), '+', result_mul_387497, result_mul_387504)
        
        
        # Obtaining the type of the subscript
        int_387506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 48), 'int')
        # Getting the type of 'b' (line 518)
        b_387507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 46), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 518)
        getitem___387508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 46), b_387507, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 518)
        subscript_call_result_387509 = invoke(stypy.reporting.localization.Localization(__file__, 518, 46), getitem___387508, int_387506)
        
        # Getting the type of 'self' (line 518)
        self_387510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 51), 'self', False)
        # Obtaining the member 'A2' of a type (line 518)
        A2_387511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 51), self_387510, 'A2')
        # Applying the binary operator '*' (line 518)
        result_mul_387512 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 46), '*', subscript_call_result_387509, A2_387511)
        
        # Applying the binary operator '+' (line 518)
        result_add_387513 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 44), '+', result_add_387505, result_mul_387512)
        
        
        # Obtaining the type of the subscript
        int_387514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 63), 'int')
        # Getting the type of 'b' (line 518)
        b_387515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 61), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 518)
        getitem___387516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 61), b_387515, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 518)
        subscript_call_result_387517 = invoke(stypy.reporting.localization.Localization(__file__, 518, 61), getitem___387516, int_387514)
        
        # Getting the type of 'self' (line 518)
        self_387518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 66), 'self', False)
        # Obtaining the member 'ident' of a type (line 518)
        ident_387519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 66), self_387518, 'ident')
        # Applying the binary operator '*' (line 518)
        result_mul_387520 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 61), '*', subscript_call_result_387517, ident_387519)
        
        # Applying the binary operator '+' (line 518)
        result_add_387521 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 59), '+', result_add_387513, result_mul_387520)
        
        # Processing the call keyword arguments (line 517)
        # Getting the type of 'self' (line 519)
        self_387522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 26), 'self', False)
        # Obtaining the member 'structure' of a type (line 519)
        structure_387523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 26), self_387522, 'structure')
        keyword_387524 = structure_387523
        kwargs_387525 = {'structure': keyword_387524}
        # Getting the type of '_smart_matrix_product' (line 517)
        _smart_matrix_product_387488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), '_smart_matrix_product', False)
        # Calling _smart_matrix_product(args, kwargs) (line 517)
        _smart_matrix_product_call_result_387526 = invoke(stypy.reporting.localization.Localization(__file__, 517, 12), _smart_matrix_product_387488, *[A_387490, result_add_387521], **kwargs_387525)
        
        # Assigning a type to the variable 'U' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'U', _smart_matrix_product_call_result_387526)
        
        # Assigning a BinOp to a Name (line 520):
        
        # Assigning a BinOp to a Name (line 520):
        
        # Obtaining the type of the subscript
        int_387527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 14), 'int')
        # Getting the type of 'b' (line 520)
        b_387528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'b')
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___387529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), b_387528, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 520)
        subscript_call_result_387530 = invoke(stypy.reporting.localization.Localization(__file__, 520, 12), getitem___387529, int_387527)
        
        # Getting the type of 'self' (line 520)
        self_387531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 17), 'self')
        # Obtaining the member 'A6' of a type (line 520)
        A6_387532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 17), self_387531, 'A6')
        # Applying the binary operator '*' (line 520)
        result_mul_387533 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 12), '*', subscript_call_result_387530, A6_387532)
        
        
        # Obtaining the type of the subscript
        int_387534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 29), 'int')
        # Getting the type of 'b' (line 520)
        b_387535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 27), 'b')
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___387536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 27), b_387535, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 520)
        subscript_call_result_387537 = invoke(stypy.reporting.localization.Localization(__file__, 520, 27), getitem___387536, int_387534)
        
        # Getting the type of 'self' (line 520)
        self_387538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 32), 'self')
        # Obtaining the member 'A4' of a type (line 520)
        A4_387539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 32), self_387538, 'A4')
        # Applying the binary operator '*' (line 520)
        result_mul_387540 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 27), '*', subscript_call_result_387537, A4_387539)
        
        # Applying the binary operator '+' (line 520)
        result_add_387541 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 12), '+', result_mul_387533, result_mul_387540)
        
        
        # Obtaining the type of the subscript
        int_387542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 44), 'int')
        # Getting the type of 'b' (line 520)
        b_387543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 42), 'b')
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___387544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 42), b_387543, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 520)
        subscript_call_result_387545 = invoke(stypy.reporting.localization.Localization(__file__, 520, 42), getitem___387544, int_387542)
        
        # Getting the type of 'self' (line 520)
        self_387546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 47), 'self')
        # Obtaining the member 'A2' of a type (line 520)
        A2_387547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 47), self_387546, 'A2')
        # Applying the binary operator '*' (line 520)
        result_mul_387548 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 42), '*', subscript_call_result_387545, A2_387547)
        
        # Applying the binary operator '+' (line 520)
        result_add_387549 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 40), '+', result_add_387541, result_mul_387548)
        
        
        # Obtaining the type of the subscript
        int_387550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 59), 'int')
        # Getting the type of 'b' (line 520)
        b_387551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 57), 'b')
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___387552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 57), b_387551, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 520)
        subscript_call_result_387553 = invoke(stypy.reporting.localization.Localization(__file__, 520, 57), getitem___387552, int_387550)
        
        # Getting the type of 'self' (line 520)
        self_387554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 62), 'self')
        # Obtaining the member 'ident' of a type (line 520)
        ident_387555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 62), self_387554, 'ident')
        # Applying the binary operator '*' (line 520)
        result_mul_387556 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 57), '*', subscript_call_result_387553, ident_387555)
        
        # Applying the binary operator '+' (line 520)
        result_add_387557 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 55), '+', result_add_387549, result_mul_387556)
        
        # Assigning a type to the variable 'V' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'V', result_add_387557)
        
        # Obtaining an instance of the builtin type 'tuple' (line 521)
        tuple_387558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 521)
        # Adding element type (line 521)
        # Getting the type of 'U' (line 521)
        U_387559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 15), 'U')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 15), tuple_387558, U_387559)
        # Adding element type (line 521)
        # Getting the type of 'V' (line 521)
        V_387560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 18), 'V')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 15), tuple_387558, V_387560)
        
        # Assigning a type to the variable 'stypy_return_type' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'stypy_return_type', tuple_387558)
        
        # ################# End of 'pade7(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pade7' in the type store
        # Getting the type of 'stypy_return_type' (line 515)
        stypy_return_type_387561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387561)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pade7'
        return stypy_return_type_387561


    @norecursion
    def pade9(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pade9'
        module_type_store = module_type_store.open_function_context('pade9', 523, 4, False)
        # Assigning a type to the variable 'self' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.pade9.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.pade9.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.pade9.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.pade9.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.pade9')
        _ExpmPadeHelper.pade9.__dict__.__setitem__('stypy_param_names_list', [])
        _ExpmPadeHelper.pade9.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.pade9.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.pade9.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.pade9.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.pade9.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.pade9.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.pade9', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pade9', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pade9(...)' code ##################

        
        # Assigning a Tuple to a Name (line 524):
        
        # Assigning a Tuple to a Name (line 524):
        
        # Obtaining an instance of the builtin type 'tuple' (line 524)
        tuple_387562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 524)
        # Adding element type (line 524)
        float_387563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 13), tuple_387562, float_387563)
        # Adding element type (line 524)
        float_387564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 13), tuple_387562, float_387564)
        # Adding element type (line 524)
        float_387565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 13), tuple_387562, float_387565)
        # Adding element type (line 524)
        float_387566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 13), tuple_387562, float_387566)
        # Adding element type (line 524)
        float_387567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 13), tuple_387562, float_387567)
        # Adding element type (line 524)
        float_387568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 13), tuple_387562, float_387568)
        # Adding element type (line 524)
        float_387569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 13), tuple_387562, float_387569)
        # Adding element type (line 524)
        float_387570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 13), tuple_387562, float_387570)
        # Adding element type (line 524)
        float_387571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 13), tuple_387562, float_387571)
        # Adding element type (line 524)
        float_387572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 13), tuple_387562, float_387572)
        
        # Assigning a type to the variable 'b' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'b', tuple_387562)
        
        # Assigning a Call to a Name (line 526):
        
        # Assigning a Call to a Name (line 526):
        
        # Call to _smart_matrix_product(...): (line 526)
        # Processing the call arguments (line 526)
        # Getting the type of 'self' (line 526)
        self_387574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 34), 'self', False)
        # Obtaining the member 'A' of a type (line 526)
        A_387575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 34), self_387574, 'A')
        
        # Obtaining the type of the subscript
        int_387576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 19), 'int')
        # Getting the type of 'b' (line 527)
        b_387577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 17), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 527)
        getitem___387578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 17), b_387577, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 527)
        subscript_call_result_387579 = invoke(stypy.reporting.localization.Localization(__file__, 527, 17), getitem___387578, int_387576)
        
        # Getting the type of 'self' (line 527)
        self_387580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 22), 'self', False)
        # Obtaining the member 'A8' of a type (line 527)
        A8_387581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 22), self_387580, 'A8')
        # Applying the binary operator '*' (line 527)
        result_mul_387582 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 17), '*', subscript_call_result_387579, A8_387581)
        
        
        # Obtaining the type of the subscript
        int_387583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 34), 'int')
        # Getting the type of 'b' (line 527)
        b_387584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 32), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 527)
        getitem___387585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 32), b_387584, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 527)
        subscript_call_result_387586 = invoke(stypy.reporting.localization.Localization(__file__, 527, 32), getitem___387585, int_387583)
        
        # Getting the type of 'self' (line 527)
        self_387587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 37), 'self', False)
        # Obtaining the member 'A6' of a type (line 527)
        A6_387588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 37), self_387587, 'A6')
        # Applying the binary operator '*' (line 527)
        result_mul_387589 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 32), '*', subscript_call_result_387586, A6_387588)
        
        # Applying the binary operator '+' (line 527)
        result_add_387590 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 17), '+', result_mul_387582, result_mul_387589)
        
        
        # Obtaining the type of the subscript
        int_387591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 49), 'int')
        # Getting the type of 'b' (line 527)
        b_387592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 47), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 527)
        getitem___387593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 47), b_387592, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 527)
        subscript_call_result_387594 = invoke(stypy.reporting.localization.Localization(__file__, 527, 47), getitem___387593, int_387591)
        
        # Getting the type of 'self' (line 527)
        self_387595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 52), 'self', False)
        # Obtaining the member 'A4' of a type (line 527)
        A4_387596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 52), self_387595, 'A4')
        # Applying the binary operator '*' (line 527)
        result_mul_387597 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 47), '*', subscript_call_result_387594, A4_387596)
        
        # Applying the binary operator '+' (line 527)
        result_add_387598 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 45), '+', result_add_387590, result_mul_387597)
        
        
        # Obtaining the type of the subscript
        int_387599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 22), 'int')
        # Getting the type of 'b' (line 528)
        b_387600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 20), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 528)
        getitem___387601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 20), b_387600, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 528)
        subscript_call_result_387602 = invoke(stypy.reporting.localization.Localization(__file__, 528, 20), getitem___387601, int_387599)
        
        # Getting the type of 'self' (line 528)
        self_387603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 25), 'self', False)
        # Obtaining the member 'A2' of a type (line 528)
        A2_387604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 25), self_387603, 'A2')
        # Applying the binary operator '*' (line 528)
        result_mul_387605 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 20), '*', subscript_call_result_387602, A2_387604)
        
        # Applying the binary operator '+' (line 527)
        result_add_387606 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 60), '+', result_add_387598, result_mul_387605)
        
        
        # Obtaining the type of the subscript
        int_387607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 37), 'int')
        # Getting the type of 'b' (line 528)
        b_387608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 35), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 528)
        getitem___387609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 35), b_387608, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 528)
        subscript_call_result_387610 = invoke(stypy.reporting.localization.Localization(__file__, 528, 35), getitem___387609, int_387607)
        
        # Getting the type of 'self' (line 528)
        self_387611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 40), 'self', False)
        # Obtaining the member 'ident' of a type (line 528)
        ident_387612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 40), self_387611, 'ident')
        # Applying the binary operator '*' (line 528)
        result_mul_387613 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 35), '*', subscript_call_result_387610, ident_387612)
        
        # Applying the binary operator '+' (line 528)
        result_add_387614 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 33), '+', result_add_387606, result_mul_387613)
        
        # Processing the call keyword arguments (line 526)
        # Getting the type of 'self' (line 529)
        self_387615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 26), 'self', False)
        # Obtaining the member 'structure' of a type (line 529)
        structure_387616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 26), self_387615, 'structure')
        keyword_387617 = structure_387616
        kwargs_387618 = {'structure': keyword_387617}
        # Getting the type of '_smart_matrix_product' (line 526)
        _smart_matrix_product_387573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), '_smart_matrix_product', False)
        # Calling _smart_matrix_product(args, kwargs) (line 526)
        _smart_matrix_product_call_result_387619 = invoke(stypy.reporting.localization.Localization(__file__, 526, 12), _smart_matrix_product_387573, *[A_387575, result_add_387614], **kwargs_387618)
        
        # Assigning a type to the variable 'U' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'U', _smart_matrix_product_call_result_387619)
        
        # Assigning a BinOp to a Name (line 530):
        
        # Assigning a BinOp to a Name (line 530):
        
        # Obtaining the type of the subscript
        int_387620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 15), 'int')
        # Getting the type of 'b' (line 530)
        b_387621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 13), 'b')
        # Obtaining the member '__getitem__' of a type (line 530)
        getitem___387622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 13), b_387621, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 530)
        subscript_call_result_387623 = invoke(stypy.reporting.localization.Localization(__file__, 530, 13), getitem___387622, int_387620)
        
        # Getting the type of 'self' (line 530)
        self_387624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 18), 'self')
        # Obtaining the member 'A8' of a type (line 530)
        A8_387625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 18), self_387624, 'A8')
        # Applying the binary operator '*' (line 530)
        result_mul_387626 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 13), '*', subscript_call_result_387623, A8_387625)
        
        
        # Obtaining the type of the subscript
        int_387627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 30), 'int')
        # Getting the type of 'b' (line 530)
        b_387628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 28), 'b')
        # Obtaining the member '__getitem__' of a type (line 530)
        getitem___387629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 28), b_387628, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 530)
        subscript_call_result_387630 = invoke(stypy.reporting.localization.Localization(__file__, 530, 28), getitem___387629, int_387627)
        
        # Getting the type of 'self' (line 530)
        self_387631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 33), 'self')
        # Obtaining the member 'A6' of a type (line 530)
        A6_387632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 33), self_387631, 'A6')
        # Applying the binary operator '*' (line 530)
        result_mul_387633 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 28), '*', subscript_call_result_387630, A6_387632)
        
        # Applying the binary operator '+' (line 530)
        result_add_387634 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 13), '+', result_mul_387626, result_mul_387633)
        
        
        # Obtaining the type of the subscript
        int_387635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 45), 'int')
        # Getting the type of 'b' (line 530)
        b_387636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 43), 'b')
        # Obtaining the member '__getitem__' of a type (line 530)
        getitem___387637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 43), b_387636, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 530)
        subscript_call_result_387638 = invoke(stypy.reporting.localization.Localization(__file__, 530, 43), getitem___387637, int_387635)
        
        # Getting the type of 'self' (line 530)
        self_387639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 48), 'self')
        # Obtaining the member 'A4' of a type (line 530)
        A4_387640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 48), self_387639, 'A4')
        # Applying the binary operator '*' (line 530)
        result_mul_387641 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 43), '*', subscript_call_result_387638, A4_387640)
        
        # Applying the binary operator '+' (line 530)
        result_add_387642 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 41), '+', result_add_387634, result_mul_387641)
        
        
        # Obtaining the type of the subscript
        int_387643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 18), 'int')
        # Getting the type of 'b' (line 531)
        b_387644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 16), 'b')
        # Obtaining the member '__getitem__' of a type (line 531)
        getitem___387645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 16), b_387644, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 531)
        subscript_call_result_387646 = invoke(stypy.reporting.localization.Localization(__file__, 531, 16), getitem___387645, int_387643)
        
        # Getting the type of 'self' (line 531)
        self_387647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 21), 'self')
        # Obtaining the member 'A2' of a type (line 531)
        A2_387648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 21), self_387647, 'A2')
        # Applying the binary operator '*' (line 531)
        result_mul_387649 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 16), '*', subscript_call_result_387646, A2_387648)
        
        # Applying the binary operator '+' (line 530)
        result_add_387650 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 56), '+', result_add_387642, result_mul_387649)
        
        
        # Obtaining the type of the subscript
        int_387651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 33), 'int')
        # Getting the type of 'b' (line 531)
        b_387652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 31), 'b')
        # Obtaining the member '__getitem__' of a type (line 531)
        getitem___387653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 31), b_387652, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 531)
        subscript_call_result_387654 = invoke(stypy.reporting.localization.Localization(__file__, 531, 31), getitem___387653, int_387651)
        
        # Getting the type of 'self' (line 531)
        self_387655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 36), 'self')
        # Obtaining the member 'ident' of a type (line 531)
        ident_387656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 36), self_387655, 'ident')
        # Applying the binary operator '*' (line 531)
        result_mul_387657 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 31), '*', subscript_call_result_387654, ident_387656)
        
        # Applying the binary operator '+' (line 531)
        result_add_387658 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 29), '+', result_add_387650, result_mul_387657)
        
        # Assigning a type to the variable 'V' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'V', result_add_387658)
        
        # Obtaining an instance of the builtin type 'tuple' (line 532)
        tuple_387659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 532)
        # Adding element type (line 532)
        # Getting the type of 'U' (line 532)
        U_387660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 15), 'U')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 15), tuple_387659, U_387660)
        # Adding element type (line 532)
        # Getting the type of 'V' (line 532)
        V_387661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 18), 'V')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 15), tuple_387659, V_387661)
        
        # Assigning a type to the variable 'stypy_return_type' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'stypy_return_type', tuple_387659)
        
        # ################# End of 'pade9(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pade9' in the type store
        # Getting the type of 'stypy_return_type' (line 523)
        stypy_return_type_387662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387662)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pade9'
        return stypy_return_type_387662


    @norecursion
    def pade13_scaled(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pade13_scaled'
        module_type_store = module_type_store.open_function_context('pade13_scaled', 534, 4, False)
        # Assigning a type to the variable 'self' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ExpmPadeHelper.pade13_scaled.__dict__.__setitem__('stypy_localization', localization)
        _ExpmPadeHelper.pade13_scaled.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ExpmPadeHelper.pade13_scaled.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ExpmPadeHelper.pade13_scaled.__dict__.__setitem__('stypy_function_name', '_ExpmPadeHelper.pade13_scaled')
        _ExpmPadeHelper.pade13_scaled.__dict__.__setitem__('stypy_param_names_list', ['s'])
        _ExpmPadeHelper.pade13_scaled.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ExpmPadeHelper.pade13_scaled.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ExpmPadeHelper.pade13_scaled.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ExpmPadeHelper.pade13_scaled.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ExpmPadeHelper.pade13_scaled.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ExpmPadeHelper.pade13_scaled.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpmPadeHelper.pade13_scaled', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pade13_scaled', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pade13_scaled(...)' code ##################

        
        # Assigning a Tuple to a Name (line 535):
        
        # Assigning a Tuple to a Name (line 535):
        
        # Obtaining an instance of the builtin type 'tuple' (line 535)
        tuple_387663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 535)
        # Adding element type (line 535)
        float_387664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387664)
        # Adding element type (line 535)
        float_387665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387665)
        # Adding element type (line 535)
        float_387666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387666)
        # Adding element type (line 535)
        float_387667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387667)
        # Adding element type (line 535)
        float_387668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387668)
        # Adding element type (line 535)
        float_387669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387669)
        # Adding element type (line 535)
        float_387670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387670)
        # Adding element type (line 535)
        float_387671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387671)
        # Adding element type (line 535)
        float_387672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387672)
        # Adding element type (line 535)
        float_387673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387673)
        # Adding element type (line 535)
        float_387674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 69), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387674)
        # Adding element type (line 535)
        float_387675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387675)
        # Adding element type (line 535)
        float_387676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387676)
        # Adding element type (line 535)
        float_387677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), tuple_387663, float_387677)
        
        # Assigning a type to the variable 'b' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'b', tuple_387663)
        
        # Assigning a BinOp to a Name (line 539):
        
        # Assigning a BinOp to a Name (line 539):
        # Getting the type of 'self' (line 539)
        self_387678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'self')
        # Obtaining the member 'A' of a type (line 539)
        A_387679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 12), self_387678, 'A')
        int_387680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 21), 'int')
        
        # Getting the type of 's' (line 539)
        s_387681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 25), 's')
        # Applying the 'usub' unary operator (line 539)
        result___neg___387682 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 24), 'usub', s_387681)
        
        # Applying the binary operator '**' (line 539)
        result_pow_387683 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 21), '**', int_387680, result___neg___387682)
        
        # Applying the binary operator '*' (line 539)
        result_mul_387684 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 12), '*', A_387679, result_pow_387683)
        
        # Assigning a type to the variable 'B' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'B', result_mul_387684)
        
        # Assigning a BinOp to a Name (line 540):
        
        # Assigning a BinOp to a Name (line 540):
        # Getting the type of 'self' (line 540)
        self_387685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 13), 'self')
        # Obtaining the member 'A2' of a type (line 540)
        A2_387686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 13), self_387685, 'A2')
        int_387687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 23), 'int')
        int_387688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 27), 'int')
        # Getting the type of 's' (line 540)
        s_387689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 30), 's')
        # Applying the binary operator '*' (line 540)
        result_mul_387690 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 27), '*', int_387688, s_387689)
        
        # Applying the binary operator '**' (line 540)
        result_pow_387691 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 23), '**', int_387687, result_mul_387690)
        
        # Applying the binary operator '*' (line 540)
        result_mul_387692 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 13), '*', A2_387686, result_pow_387691)
        
        # Assigning a type to the variable 'B2' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'B2', result_mul_387692)
        
        # Assigning a BinOp to a Name (line 541):
        
        # Assigning a BinOp to a Name (line 541):
        # Getting the type of 'self' (line 541)
        self_387693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 13), 'self')
        # Obtaining the member 'A4' of a type (line 541)
        A4_387694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 13), self_387693, 'A4')
        int_387695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 23), 'int')
        int_387696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 27), 'int')
        # Getting the type of 's' (line 541)
        s_387697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 30), 's')
        # Applying the binary operator '*' (line 541)
        result_mul_387698 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 27), '*', int_387696, s_387697)
        
        # Applying the binary operator '**' (line 541)
        result_pow_387699 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 23), '**', int_387695, result_mul_387698)
        
        # Applying the binary operator '*' (line 541)
        result_mul_387700 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 13), '*', A4_387694, result_pow_387699)
        
        # Assigning a type to the variable 'B4' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'B4', result_mul_387700)
        
        # Assigning a BinOp to a Name (line 542):
        
        # Assigning a BinOp to a Name (line 542):
        # Getting the type of 'self' (line 542)
        self_387701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 13), 'self')
        # Obtaining the member 'A6' of a type (line 542)
        A6_387702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 13), self_387701, 'A6')
        int_387703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 23), 'int')
        int_387704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 27), 'int')
        # Getting the type of 's' (line 542)
        s_387705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 30), 's')
        # Applying the binary operator '*' (line 542)
        result_mul_387706 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 27), '*', int_387704, s_387705)
        
        # Applying the binary operator '**' (line 542)
        result_pow_387707 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 23), '**', int_387703, result_mul_387706)
        
        # Applying the binary operator '*' (line 542)
        result_mul_387708 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 13), '*', A6_387702, result_pow_387707)
        
        # Assigning a type to the variable 'B6' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'B6', result_mul_387708)
        
        # Assigning a Call to a Name (line 543):
        
        # Assigning a Call to a Name (line 543):
        
        # Call to _smart_matrix_product(...): (line 543)
        # Processing the call arguments (line 543)
        # Getting the type of 'B6' (line 543)
        B6_387710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 35), 'B6', False)
        
        # Obtaining the type of the subscript
        int_387711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 18), 'int')
        # Getting the type of 'b' (line 544)
        b_387712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 544)
        getitem___387713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 16), b_387712, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 544)
        subscript_call_result_387714 = invoke(stypy.reporting.localization.Localization(__file__, 544, 16), getitem___387713, int_387711)
        
        # Getting the type of 'B6' (line 544)
        B6_387715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 22), 'B6', False)
        # Applying the binary operator '*' (line 544)
        result_mul_387716 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 16), '*', subscript_call_result_387714, B6_387715)
        
        
        # Obtaining the type of the subscript
        int_387717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 29), 'int')
        # Getting the type of 'b' (line 544)
        b_387718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 27), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 544)
        getitem___387719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 27), b_387718, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 544)
        subscript_call_result_387720 = invoke(stypy.reporting.localization.Localization(__file__, 544, 27), getitem___387719, int_387717)
        
        # Getting the type of 'B4' (line 544)
        B4_387721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 33), 'B4', False)
        # Applying the binary operator '*' (line 544)
        result_mul_387722 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 27), '*', subscript_call_result_387720, B4_387721)
        
        # Applying the binary operator '+' (line 544)
        result_add_387723 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 16), '+', result_mul_387716, result_mul_387722)
        
        
        # Obtaining the type of the subscript
        int_387724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 40), 'int')
        # Getting the type of 'b' (line 544)
        b_387725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 38), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 544)
        getitem___387726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 38), b_387725, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 544)
        subscript_call_result_387727 = invoke(stypy.reporting.localization.Localization(__file__, 544, 38), getitem___387726, int_387724)
        
        # Getting the type of 'B2' (line 544)
        B2_387728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 43), 'B2', False)
        # Applying the binary operator '*' (line 544)
        result_mul_387729 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 38), '*', subscript_call_result_387727, B2_387728)
        
        # Applying the binary operator '+' (line 544)
        result_add_387730 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 36), '+', result_add_387723, result_mul_387729)
        
        # Processing the call keyword arguments (line 543)
        # Getting the type of 'self' (line 545)
        self_387731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 26), 'self', False)
        # Obtaining the member 'structure' of a type (line 545)
        structure_387732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 26), self_387731, 'structure')
        keyword_387733 = structure_387732
        kwargs_387734 = {'structure': keyword_387733}
        # Getting the type of '_smart_matrix_product' (line 543)
        _smart_matrix_product_387709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 13), '_smart_matrix_product', False)
        # Calling _smart_matrix_product(args, kwargs) (line 543)
        _smart_matrix_product_call_result_387735 = invoke(stypy.reporting.localization.Localization(__file__, 543, 13), _smart_matrix_product_387709, *[B6_387710, result_add_387730], **kwargs_387734)
        
        # Assigning a type to the variable 'U2' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'U2', _smart_matrix_product_call_result_387735)
        
        # Assigning a Call to a Name (line 546):
        
        # Assigning a Call to a Name (line 546):
        
        # Call to _smart_matrix_product(...): (line 546)
        # Processing the call arguments (line 546)
        # Getting the type of 'B' (line 546)
        B_387737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 34), 'B', False)
        # Getting the type of 'U2' (line 547)
        U2_387738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 17), 'U2', False)
        
        # Obtaining the type of the subscript
        int_387739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 24), 'int')
        # Getting the type of 'b' (line 547)
        b_387740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 22), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 547)
        getitem___387741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 22), b_387740, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 547)
        subscript_call_result_387742 = invoke(stypy.reporting.localization.Localization(__file__, 547, 22), getitem___387741, int_387739)
        
        # Getting the type of 'B6' (line 547)
        B6_387743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 27), 'B6', False)
        # Applying the binary operator '*' (line 547)
        result_mul_387744 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 22), '*', subscript_call_result_387742, B6_387743)
        
        # Applying the binary operator '+' (line 547)
        result_add_387745 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 17), '+', U2_387738, result_mul_387744)
        
        
        # Obtaining the type of the subscript
        int_387746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 34), 'int')
        # Getting the type of 'b' (line 547)
        b_387747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 32), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 547)
        getitem___387748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 32), b_387747, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 547)
        subscript_call_result_387749 = invoke(stypy.reporting.localization.Localization(__file__, 547, 32), getitem___387748, int_387746)
        
        # Getting the type of 'B4' (line 547)
        B4_387750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 37), 'B4', False)
        # Applying the binary operator '*' (line 547)
        result_mul_387751 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 32), '*', subscript_call_result_387749, B4_387750)
        
        # Applying the binary operator '+' (line 547)
        result_add_387752 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 30), '+', result_add_387745, result_mul_387751)
        
        
        # Obtaining the type of the subscript
        int_387753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 22), 'int')
        # Getting the type of 'b' (line 548)
        b_387754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 20), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 548)
        getitem___387755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 20), b_387754, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 548)
        subscript_call_result_387756 = invoke(stypy.reporting.localization.Localization(__file__, 548, 20), getitem___387755, int_387753)
        
        # Getting the type of 'B2' (line 548)
        B2_387757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 25), 'B2', False)
        # Applying the binary operator '*' (line 548)
        result_mul_387758 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 20), '*', subscript_call_result_387756, B2_387757)
        
        # Applying the binary operator '+' (line 547)
        result_add_387759 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 40), '+', result_add_387752, result_mul_387758)
        
        
        # Obtaining the type of the subscript
        int_387760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 32), 'int')
        # Getting the type of 'b' (line 548)
        b_387761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 30), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 548)
        getitem___387762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 30), b_387761, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 548)
        subscript_call_result_387763 = invoke(stypy.reporting.localization.Localization(__file__, 548, 30), getitem___387762, int_387760)
        
        # Getting the type of 'self' (line 548)
        self_387764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 35), 'self', False)
        # Obtaining the member 'ident' of a type (line 548)
        ident_387765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 35), self_387764, 'ident')
        # Applying the binary operator '*' (line 548)
        result_mul_387766 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 30), '*', subscript_call_result_387763, ident_387765)
        
        # Applying the binary operator '+' (line 548)
        result_add_387767 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 28), '+', result_add_387759, result_mul_387766)
        
        # Processing the call keyword arguments (line 546)
        # Getting the type of 'self' (line 549)
        self_387768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 26), 'self', False)
        # Obtaining the member 'structure' of a type (line 549)
        structure_387769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 26), self_387768, 'structure')
        keyword_387770 = structure_387769
        kwargs_387771 = {'structure': keyword_387770}
        # Getting the type of '_smart_matrix_product' (line 546)
        _smart_matrix_product_387736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), '_smart_matrix_product', False)
        # Calling _smart_matrix_product(args, kwargs) (line 546)
        _smart_matrix_product_call_result_387772 = invoke(stypy.reporting.localization.Localization(__file__, 546, 12), _smart_matrix_product_387736, *[B_387737, result_add_387767], **kwargs_387771)
        
        # Assigning a type to the variable 'U' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'U', _smart_matrix_product_call_result_387772)
        
        # Assigning a Call to a Name (line 550):
        
        # Assigning a Call to a Name (line 550):
        
        # Call to _smart_matrix_product(...): (line 550)
        # Processing the call arguments (line 550)
        # Getting the type of 'B6' (line 550)
        B6_387774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 35), 'B6', False)
        
        # Obtaining the type of the subscript
        int_387775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 18), 'int')
        # Getting the type of 'b' (line 551)
        b_387776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 16), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 551)
        getitem___387777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 16), b_387776, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 551)
        subscript_call_result_387778 = invoke(stypy.reporting.localization.Localization(__file__, 551, 16), getitem___387777, int_387775)
        
        # Getting the type of 'B6' (line 551)
        B6_387779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 22), 'B6', False)
        # Applying the binary operator '*' (line 551)
        result_mul_387780 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 16), '*', subscript_call_result_387778, B6_387779)
        
        
        # Obtaining the type of the subscript
        int_387781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 29), 'int')
        # Getting the type of 'b' (line 551)
        b_387782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 27), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 551)
        getitem___387783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 27), b_387782, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 551)
        subscript_call_result_387784 = invoke(stypy.reporting.localization.Localization(__file__, 551, 27), getitem___387783, int_387781)
        
        # Getting the type of 'B4' (line 551)
        B4_387785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 33), 'B4', False)
        # Applying the binary operator '*' (line 551)
        result_mul_387786 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 27), '*', subscript_call_result_387784, B4_387785)
        
        # Applying the binary operator '+' (line 551)
        result_add_387787 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 16), '+', result_mul_387780, result_mul_387786)
        
        
        # Obtaining the type of the subscript
        int_387788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 40), 'int')
        # Getting the type of 'b' (line 551)
        b_387789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 38), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 551)
        getitem___387790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 38), b_387789, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 551)
        subscript_call_result_387791 = invoke(stypy.reporting.localization.Localization(__file__, 551, 38), getitem___387790, int_387788)
        
        # Getting the type of 'B2' (line 551)
        B2_387792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 43), 'B2', False)
        # Applying the binary operator '*' (line 551)
        result_mul_387793 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 38), '*', subscript_call_result_387791, B2_387792)
        
        # Applying the binary operator '+' (line 551)
        result_add_387794 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 36), '+', result_add_387787, result_mul_387793)
        
        # Processing the call keyword arguments (line 550)
        # Getting the type of 'self' (line 552)
        self_387795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 26), 'self', False)
        # Obtaining the member 'structure' of a type (line 552)
        structure_387796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 26), self_387795, 'structure')
        keyword_387797 = structure_387796
        kwargs_387798 = {'structure': keyword_387797}
        # Getting the type of '_smart_matrix_product' (line 550)
        _smart_matrix_product_387773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 13), '_smart_matrix_product', False)
        # Calling _smart_matrix_product(args, kwargs) (line 550)
        _smart_matrix_product_call_result_387799 = invoke(stypy.reporting.localization.Localization(__file__, 550, 13), _smart_matrix_product_387773, *[B6_387774, result_add_387794], **kwargs_387798)
        
        # Assigning a type to the variable 'V2' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'V2', _smart_matrix_product_call_result_387799)
        
        # Assigning a BinOp to a Name (line 553):
        
        # Assigning a BinOp to a Name (line 553):
        # Getting the type of 'V2' (line 553)
        V2_387800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'V2')
        
        # Obtaining the type of the subscript
        int_387801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 19), 'int')
        # Getting the type of 'b' (line 553)
        b_387802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 17), 'b')
        # Obtaining the member '__getitem__' of a type (line 553)
        getitem___387803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 17), b_387802, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 553)
        subscript_call_result_387804 = invoke(stypy.reporting.localization.Localization(__file__, 553, 17), getitem___387803, int_387801)
        
        # Getting the type of 'B6' (line 553)
        B6_387805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 22), 'B6')
        # Applying the binary operator '*' (line 553)
        result_mul_387806 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 17), '*', subscript_call_result_387804, B6_387805)
        
        # Applying the binary operator '+' (line 553)
        result_add_387807 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 12), '+', V2_387800, result_mul_387806)
        
        
        # Obtaining the type of the subscript
        int_387808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 29), 'int')
        # Getting the type of 'b' (line 553)
        b_387809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 27), 'b')
        # Obtaining the member '__getitem__' of a type (line 553)
        getitem___387810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 27), b_387809, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 553)
        subscript_call_result_387811 = invoke(stypy.reporting.localization.Localization(__file__, 553, 27), getitem___387810, int_387808)
        
        # Getting the type of 'B4' (line 553)
        B4_387812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 32), 'B4')
        # Applying the binary operator '*' (line 553)
        result_mul_387813 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 27), '*', subscript_call_result_387811, B4_387812)
        
        # Applying the binary operator '+' (line 553)
        result_add_387814 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 25), '+', result_add_387807, result_mul_387813)
        
        
        # Obtaining the type of the subscript
        int_387815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 39), 'int')
        # Getting the type of 'b' (line 553)
        b_387816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 37), 'b')
        # Obtaining the member '__getitem__' of a type (line 553)
        getitem___387817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 37), b_387816, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 553)
        subscript_call_result_387818 = invoke(stypy.reporting.localization.Localization(__file__, 553, 37), getitem___387817, int_387815)
        
        # Getting the type of 'B2' (line 553)
        B2_387819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 42), 'B2')
        # Applying the binary operator '*' (line 553)
        result_mul_387820 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 37), '*', subscript_call_result_387818, B2_387819)
        
        # Applying the binary operator '+' (line 553)
        result_add_387821 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 35), '+', result_add_387814, result_mul_387820)
        
        
        # Obtaining the type of the subscript
        int_387822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 49), 'int')
        # Getting the type of 'b' (line 553)
        b_387823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 47), 'b')
        # Obtaining the member '__getitem__' of a type (line 553)
        getitem___387824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 47), b_387823, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 553)
        subscript_call_result_387825 = invoke(stypy.reporting.localization.Localization(__file__, 553, 47), getitem___387824, int_387822)
        
        # Getting the type of 'self' (line 553)
        self_387826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 52), 'self')
        # Obtaining the member 'ident' of a type (line 553)
        ident_387827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 52), self_387826, 'ident')
        # Applying the binary operator '*' (line 553)
        result_mul_387828 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 47), '*', subscript_call_result_387825, ident_387827)
        
        # Applying the binary operator '+' (line 553)
        result_add_387829 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 45), '+', result_add_387821, result_mul_387828)
        
        # Assigning a type to the variable 'V' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'V', result_add_387829)
        
        # Obtaining an instance of the builtin type 'tuple' (line 554)
        tuple_387830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 554)
        # Adding element type (line 554)
        # Getting the type of 'U' (line 554)
        U_387831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 15), 'U')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 15), tuple_387830, U_387831)
        # Adding element type (line 554)
        # Getting the type of 'V' (line 554)
        V_387832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 18), 'V')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 15), tuple_387830, V_387832)
        
        # Assigning a type to the variable 'stypy_return_type' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'stypy_return_type', tuple_387830)
        
        # ################# End of 'pade13_scaled(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pade13_scaled' in the type store
        # Getting the type of 'stypy_return_type' (line 534)
        stypy_return_type_387833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387833)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pade13_scaled'
        return stypy_return_type_387833


# Assigning a type to the variable '_ExpmPadeHelper' (line 349)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 0), '_ExpmPadeHelper', _ExpmPadeHelper)

@norecursion
def expm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'expm'
    module_type_store = module_type_store.open_function_context('expm', 557, 0, False)
    
    # Passed parameters checking function
    expm.stypy_localization = localization
    expm.stypy_type_of_self = None
    expm.stypy_type_store = module_type_store
    expm.stypy_function_name = 'expm'
    expm.stypy_param_names_list = ['A']
    expm.stypy_varargs_param_name = None
    expm.stypy_kwargs_param_name = None
    expm.stypy_call_defaults = defaults
    expm.stypy_call_varargs = varargs
    expm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expm', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expm', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expm(...)' code ##################

    str_387834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, (-1)), 'str', '\n    Compute the matrix exponential using Pade approximation.\n\n    Parameters\n    ----------\n    A : (M,M) array_like or sparse matrix\n        2D Array or Matrix (sparse or dense) to be exponentiated\n\n    Returns\n    -------\n    expA : (M,M) ndarray\n        Matrix exponential of `A`\n\n    Notes\n    -----\n    This is algorithm (6.1) which is a simplification of algorithm (5.1).\n\n    .. versionadded:: 0.12.0\n\n    References\n    ----------\n    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)\n           "A New Scaling and Squaring Algorithm for the Matrix Exponential."\n           SIAM Journal on Matrix Analysis and Applications.\n           31 (3). pp. 970-989. ISSN 1095-7162\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import expm\n    >>> A = csc_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])\n    >>> A.todense()\n    matrix([[1, 0, 0],\n            [0, 2, 0],\n            [0, 0, 3]], dtype=int64)\n    >>> Aexp = expm(A)\n    >>> Aexp\n    <3x3 sparse matrix of type \'<class \'numpy.float64\'>\'\n        with 3 stored elements in Compressed Sparse Column format>\n    >>> Aexp.todense()\n    matrix([[  2.71828183,   0.        ,   0.        ],\n            [  0.        ,   7.3890561 ,   0.        ],\n            [  0.        ,   0.        ,  20.08553692]])\n    ')
    
    # Call to _expm(...): (line 602)
    # Processing the call arguments (line 602)
    # Getting the type of 'A' (line 602)
    A_387836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 17), 'A', False)
    # Processing the call keyword arguments (line 602)
    str_387837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 38), 'str', 'auto')
    keyword_387838 = str_387837
    kwargs_387839 = {'use_exact_onenorm': keyword_387838}
    # Getting the type of '_expm' (line 602)
    _expm_387835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 11), '_expm', False)
    # Calling _expm(args, kwargs) (line 602)
    _expm_call_result_387840 = invoke(stypy.reporting.localization.Localization(__file__, 602, 11), _expm_387835, *[A_387836], **kwargs_387839)
    
    # Assigning a type to the variable 'stypy_return_type' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'stypy_return_type', _expm_call_result_387840)
    
    # ################# End of 'expm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expm' in the type store
    # Getting the type of 'stypy_return_type' (line 557)
    stypy_return_type_387841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_387841)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expm'
    return stypy_return_type_387841

# Assigning a type to the variable 'expm' (line 557)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 0), 'expm', expm)

@norecursion
def _expm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_expm'
    module_type_store = module_type_store.open_function_context('_expm', 605, 0, False)
    
    # Passed parameters checking function
    _expm.stypy_localization = localization
    _expm.stypy_type_of_self = None
    _expm.stypy_type_store = module_type_store
    _expm.stypy_function_name = '_expm'
    _expm.stypy_param_names_list = ['A', 'use_exact_onenorm']
    _expm.stypy_varargs_param_name = None
    _expm.stypy_kwargs_param_name = None
    _expm.stypy_call_defaults = defaults
    _expm.stypy_call_varargs = varargs
    _expm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_expm', ['A', 'use_exact_onenorm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_expm', localization, ['A', 'use_exact_onenorm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_expm(...)' code ##################

    
    
    # Call to isinstance(...): (line 610)
    # Processing the call arguments (line 610)
    # Getting the type of 'A' (line 610)
    A_387843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 18), 'A', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 610)
    tuple_387844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 610)
    # Adding element type (line 610)
    # Getting the type of 'list' (line 610)
    list_387845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 22), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 22), tuple_387844, list_387845)
    # Adding element type (line 610)
    # Getting the type of 'tuple' (line 610)
    tuple_387846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 28), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 22), tuple_387844, tuple_387846)
    
    # Processing the call keyword arguments (line 610)
    kwargs_387847 = {}
    # Getting the type of 'isinstance' (line 610)
    isinstance_387842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 610)
    isinstance_call_result_387848 = invoke(stypy.reporting.localization.Localization(__file__, 610, 7), isinstance_387842, *[A_387843, tuple_387844], **kwargs_387847)
    
    # Testing the type of an if condition (line 610)
    if_condition_387849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 610, 4), isinstance_call_result_387848)
    # Assigning a type to the variable 'if_condition_387849' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'if_condition_387849', if_condition_387849)
    # SSA begins for if statement (line 610)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 611):
    
    # Assigning a Call to a Name (line 611):
    
    # Call to asarray(...): (line 611)
    # Processing the call arguments (line 611)
    # Getting the type of 'A' (line 611)
    A_387852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 23), 'A', False)
    # Processing the call keyword arguments (line 611)
    kwargs_387853 = {}
    # Getting the type of 'np' (line 611)
    np_387850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 611)
    asarray_387851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 12), np_387850, 'asarray')
    # Calling asarray(args, kwargs) (line 611)
    asarray_call_result_387854 = invoke(stypy.reporting.localization.Localization(__file__, 611, 12), asarray_387851, *[A_387852], **kwargs_387853)
    
    # Assigning a type to the variable 'A' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'A', asarray_call_result_387854)
    # SSA join for if statement (line 610)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 612)
    # Processing the call arguments (line 612)
    # Getting the type of 'A' (line 612)
    A_387856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 612)
    shape_387857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 11), A_387856, 'shape')
    # Processing the call keyword arguments (line 612)
    kwargs_387858 = {}
    # Getting the type of 'len' (line 612)
    len_387855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 7), 'len', False)
    # Calling len(args, kwargs) (line 612)
    len_call_result_387859 = invoke(stypy.reporting.localization.Localization(__file__, 612, 7), len_387855, *[shape_387857], **kwargs_387858)
    
    int_387860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 23), 'int')
    # Applying the binary operator '!=' (line 612)
    result_ne_387861 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 7), '!=', len_call_result_387859, int_387860)
    
    
    
    # Obtaining the type of the subscript
    int_387862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 36), 'int')
    # Getting the type of 'A' (line 612)
    A_387863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 28), 'A')
    # Obtaining the member 'shape' of a type (line 612)
    shape_387864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 28), A_387863, 'shape')
    # Obtaining the member '__getitem__' of a type (line 612)
    getitem___387865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 28), shape_387864, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 612)
    subscript_call_result_387866 = invoke(stypy.reporting.localization.Localization(__file__, 612, 28), getitem___387865, int_387862)
    
    
    # Obtaining the type of the subscript
    int_387867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 50), 'int')
    # Getting the type of 'A' (line 612)
    A_387868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 42), 'A')
    # Obtaining the member 'shape' of a type (line 612)
    shape_387869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 42), A_387868, 'shape')
    # Obtaining the member '__getitem__' of a type (line 612)
    getitem___387870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 42), shape_387869, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 612)
    subscript_call_result_387871 = invoke(stypy.reporting.localization.Localization(__file__, 612, 42), getitem___387870, int_387867)
    
    # Applying the binary operator '!=' (line 612)
    result_ne_387872 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 28), '!=', subscript_call_result_387866, subscript_call_result_387871)
    
    # Applying the binary operator 'or' (line 612)
    result_or_keyword_387873 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 7), 'or', result_ne_387861, result_ne_387872)
    
    # Testing the type of an if condition (line 612)
    if_condition_387874 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 612, 4), result_or_keyword_387873)
    # Assigning a type to the variable 'if_condition_387874' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 4), 'if_condition_387874', if_condition_387874)
    # SSA begins for if statement (line 612)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 613)
    # Processing the call arguments (line 613)
    str_387876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 25), 'str', 'expected a square matrix')
    # Processing the call keyword arguments (line 613)
    kwargs_387877 = {}
    # Getting the type of 'ValueError' (line 613)
    ValueError_387875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 613)
    ValueError_call_result_387878 = invoke(stypy.reporting.localization.Localization(__file__, 613, 14), ValueError_387875, *[str_387876], **kwargs_387877)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 613, 8), ValueError_call_result_387878, 'raise parameter', BaseException)
    # SSA join for if statement (line 612)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'A' (line 616)
    A_387879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 7), 'A')
    # Obtaining the member 'shape' of a type (line 616)
    shape_387880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 7), A_387879, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 616)
    tuple_387881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 616)
    # Adding element type (line 616)
    int_387882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 19), tuple_387881, int_387882)
    # Adding element type (line 616)
    int_387883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 19), tuple_387881, int_387883)
    
    # Applying the binary operator '==' (line 616)
    result_eq_387884 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 7), '==', shape_387880, tuple_387881)
    
    # Testing the type of an if condition (line 616)
    if_condition_387885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 616, 4), result_eq_387884)
    # Assigning a type to the variable 'if_condition_387885' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 4), 'if_condition_387885', if_condition_387885)
    # SSA begins for if statement (line 616)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 617):
    
    # Assigning a List to a Name (line 617):
    
    # Obtaining an instance of the builtin type 'list' (line 617)
    list_387886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 617)
    # Adding element type (line 617)
    
    # Obtaining an instance of the builtin type 'list' (line 617)
    list_387887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 617)
    # Adding element type (line 617)
    
    # Call to exp(...): (line 617)
    # Processing the call arguments (line 617)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 617)
    tuple_387890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 617)
    # Adding element type (line 617)
    int_387891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 25), tuple_387890, int_387891)
    # Adding element type (line 617)
    int_387892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 25), tuple_387890, int_387892)
    
    # Getting the type of 'A' (line 617)
    A_387893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 23), 'A', False)
    # Obtaining the member '__getitem__' of a type (line 617)
    getitem___387894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 23), A_387893, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 617)
    subscript_call_result_387895 = invoke(stypy.reporting.localization.Localization(__file__, 617, 23), getitem___387894, tuple_387890)
    
    # Processing the call keyword arguments (line 617)
    kwargs_387896 = {}
    # Getting the type of 'np' (line 617)
    np_387888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 16), 'np', False)
    # Obtaining the member 'exp' of a type (line 617)
    exp_387889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 16), np_387888, 'exp')
    # Calling exp(args, kwargs) (line 617)
    exp_call_result_387897 = invoke(stypy.reporting.localization.Localization(__file__, 617, 16), exp_387889, *[subscript_call_result_387895], **kwargs_387896)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 15), list_387887, exp_call_result_387897)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 14), list_387886, list_387887)
    
    # Assigning a type to the variable 'out' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'out', list_387886)
    
    
    # Call to isspmatrix(...): (line 621)
    # Processing the call arguments (line 621)
    # Getting the type of 'A' (line 621)
    A_387899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 22), 'A', False)
    # Processing the call keyword arguments (line 621)
    kwargs_387900 = {}
    # Getting the type of 'isspmatrix' (line 621)
    isspmatrix_387898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 11), 'isspmatrix', False)
    # Calling isspmatrix(args, kwargs) (line 621)
    isspmatrix_call_result_387901 = invoke(stypy.reporting.localization.Localization(__file__, 621, 11), isspmatrix_387898, *[A_387899], **kwargs_387900)
    
    # Testing the type of an if condition (line 621)
    if_condition_387902 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 621, 8), isspmatrix_call_result_387901)
    # Assigning a type to the variable 'if_condition_387902' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'if_condition_387902', if_condition_387902)
    # SSA begins for if statement (line 621)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to __class__(...): (line 622)
    # Processing the call arguments (line 622)
    # Getting the type of 'out' (line 622)
    out_387905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 31), 'out', False)
    # Processing the call keyword arguments (line 622)
    kwargs_387906 = {}
    # Getting the type of 'A' (line 622)
    A_387903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 19), 'A', False)
    # Obtaining the member '__class__' of a type (line 622)
    class___387904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 19), A_387903, '__class__')
    # Calling __class__(args, kwargs) (line 622)
    class___call_result_387907 = invoke(stypy.reporting.localization.Localization(__file__, 622, 19), class___387904, *[out_387905], **kwargs_387906)
    
    # Assigning a type to the variable 'stypy_return_type' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 12), 'stypy_return_type', class___call_result_387907)
    # SSA join for if statement (line 621)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to array(...): (line 624)
    # Processing the call arguments (line 624)
    # Getting the type of 'out' (line 624)
    out_387910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 24), 'out', False)
    # Processing the call keyword arguments (line 624)
    kwargs_387911 = {}
    # Getting the type of 'np' (line 624)
    np_387908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 624)
    array_387909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 15), np_387908, 'array')
    # Calling array(args, kwargs) (line 624)
    array_call_result_387912 = invoke(stypy.reporting.localization.Localization(__file__, 624, 15), array_387909, *[out_387910], **kwargs_387911)
    
    # Assigning a type to the variable 'stypy_return_type' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'stypy_return_type', array_call_result_387912)
    # SSA join for if statement (line 616)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a IfExp to a Name (line 627):
    
    # Assigning a IfExp to a Name (line 627):
    
    
    # Call to _is_upper_triangular(...): (line 627)
    # Processing the call arguments (line 627)
    # Getting the type of 'A' (line 627)
    A_387914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 57), 'A', False)
    # Processing the call keyword arguments (line 627)
    kwargs_387915 = {}
    # Getting the type of '_is_upper_triangular' (line 627)
    _is_upper_triangular_387913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 36), '_is_upper_triangular', False)
    # Calling _is_upper_triangular(args, kwargs) (line 627)
    _is_upper_triangular_call_result_387916 = invoke(stypy.reporting.localization.Localization(__file__, 627, 36), _is_upper_triangular_387913, *[A_387914], **kwargs_387915)
    
    # Testing the type of an if expression (line 627)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 627, 16), _is_upper_triangular_call_result_387916)
    # SSA begins for if expression (line 627)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'UPPER_TRIANGULAR' (line 627)
    UPPER_TRIANGULAR_387917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 16), 'UPPER_TRIANGULAR')
    # SSA branch for the else part of an if expression (line 627)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'None' (line 627)
    None_387918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 65), 'None')
    # SSA join for if expression (line 627)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_387919 = union_type.UnionType.add(UPPER_TRIANGULAR_387917, None_387918)
    
    # Assigning a type to the variable 'structure' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'structure', if_exp_387919)
    
    
    # Getting the type of 'use_exact_onenorm' (line 629)
    use_exact_onenorm_387920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 7), 'use_exact_onenorm')
    str_387921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 28), 'str', 'auto')
    # Applying the binary operator '==' (line 629)
    result_eq_387922 = python_operator(stypy.reporting.localization.Localization(__file__, 629, 7), '==', use_exact_onenorm_387920, str_387921)
    
    # Testing the type of an if condition (line 629)
    if_condition_387923 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 629, 4), result_eq_387922)
    # Assigning a type to the variable 'if_condition_387923' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'if_condition_387923', if_condition_387923)
    # SSA begins for if statement (line 629)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Compare to a Name (line 631):
    
    # Assigning a Compare to a Name (line 631):
    
    
    # Obtaining the type of the subscript
    int_387924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 36), 'int')
    # Getting the type of 'A' (line 631)
    A_387925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 28), 'A')
    # Obtaining the member 'shape' of a type (line 631)
    shape_387926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 28), A_387925, 'shape')
    # Obtaining the member '__getitem__' of a type (line 631)
    getitem___387927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 28), shape_387926, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 631)
    subscript_call_result_387928 = invoke(stypy.reporting.localization.Localization(__file__, 631, 28), getitem___387927, int_387924)
    
    int_387929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 41), 'int')
    # Applying the binary operator '<' (line 631)
    result_lt_387930 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 28), '<', subscript_call_result_387928, int_387929)
    
    # Assigning a type to the variable 'use_exact_onenorm' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'use_exact_onenorm', result_lt_387930)
    # SSA join for if statement (line 629)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 634):
    
    # Assigning a Call to a Name (line 634):
    
    # Call to _ExpmPadeHelper(...): (line 634)
    # Processing the call arguments (line 634)
    # Getting the type of 'A' (line 635)
    A_387932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 12), 'A', False)
    # Processing the call keyword arguments (line 634)
    # Getting the type of 'structure' (line 635)
    structure_387933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 25), 'structure', False)
    keyword_387934 = structure_387933
    # Getting the type of 'use_exact_onenorm' (line 635)
    use_exact_onenorm_387935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 54), 'use_exact_onenorm', False)
    keyword_387936 = use_exact_onenorm_387935
    kwargs_387937 = {'structure': keyword_387934, 'use_exact_onenorm': keyword_387936}
    # Getting the type of '_ExpmPadeHelper' (line 634)
    _ExpmPadeHelper_387931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), '_ExpmPadeHelper', False)
    # Calling _ExpmPadeHelper(args, kwargs) (line 634)
    _ExpmPadeHelper_call_result_387938 = invoke(stypy.reporting.localization.Localization(__file__, 634, 8), _ExpmPadeHelper_387931, *[A_387932], **kwargs_387937)
    
    # Assigning a type to the variable 'h' (line 634)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 4), 'h', _ExpmPadeHelper_call_result_387938)
    
    # Assigning a Call to a Name (line 638):
    
    # Assigning a Call to a Name (line 638):
    
    # Call to max(...): (line 638)
    # Processing the call arguments (line 638)
    # Getting the type of 'h' (line 638)
    h_387940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 16), 'h', False)
    # Obtaining the member 'd4_loose' of a type (line 638)
    d4_loose_387941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 16), h_387940, 'd4_loose')
    # Getting the type of 'h' (line 638)
    h_387942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 28), 'h', False)
    # Obtaining the member 'd6_loose' of a type (line 638)
    d6_loose_387943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 28), h_387942, 'd6_loose')
    # Processing the call keyword arguments (line 638)
    kwargs_387944 = {}
    # Getting the type of 'max' (line 638)
    max_387939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 12), 'max', False)
    # Calling max(args, kwargs) (line 638)
    max_call_result_387945 = invoke(stypy.reporting.localization.Localization(__file__, 638, 12), max_387939, *[d4_loose_387941, d6_loose_387943], **kwargs_387944)
    
    # Assigning a type to the variable 'eta_1' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'eta_1', max_call_result_387945)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'eta_1' (line 639)
    eta_1_387946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 7), 'eta_1')
    float_387947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 15), 'float')
    # Applying the binary operator '<' (line 639)
    result_lt_387948 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 7), '<', eta_1_387946, float_387947)
    
    
    
    # Call to _ell(...): (line 639)
    # Processing the call arguments (line 639)
    # Getting the type of 'h' (line 639)
    h_387950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 47), 'h', False)
    # Obtaining the member 'A' of a type (line 639)
    A_387951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 47), h_387950, 'A')
    int_387952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 52), 'int')
    # Processing the call keyword arguments (line 639)
    kwargs_387953 = {}
    # Getting the type of '_ell' (line 639)
    _ell_387949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 42), '_ell', False)
    # Calling _ell(args, kwargs) (line 639)
    _ell_call_result_387954 = invoke(stypy.reporting.localization.Localization(__file__, 639, 42), _ell_387949, *[A_387951, int_387952], **kwargs_387953)
    
    int_387955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 58), 'int')
    # Applying the binary operator '==' (line 639)
    result_eq_387956 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 42), '==', _ell_call_result_387954, int_387955)
    
    # Applying the binary operator 'and' (line 639)
    result_and_keyword_387957 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 7), 'and', result_lt_387948, result_eq_387956)
    
    # Testing the type of an if condition (line 639)
    if_condition_387958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 639, 4), result_and_keyword_387957)
    # Assigning a type to the variable 'if_condition_387958' (line 639)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'if_condition_387958', if_condition_387958)
    # SSA begins for if statement (line 639)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 640):
    
    # Assigning a Subscript to a Name (line 640):
    
    # Obtaining the type of the subscript
    int_387959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 8), 'int')
    
    # Call to pade3(...): (line 640)
    # Processing the call keyword arguments (line 640)
    kwargs_387962 = {}
    # Getting the type of 'h' (line 640)
    h_387960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 15), 'h', False)
    # Obtaining the member 'pade3' of a type (line 640)
    pade3_387961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 15), h_387960, 'pade3')
    # Calling pade3(args, kwargs) (line 640)
    pade3_call_result_387963 = invoke(stypy.reporting.localization.Localization(__file__, 640, 15), pade3_387961, *[], **kwargs_387962)
    
    # Obtaining the member '__getitem__' of a type (line 640)
    getitem___387964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 8), pade3_call_result_387963, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 640)
    subscript_call_result_387965 = invoke(stypy.reporting.localization.Localization(__file__, 640, 8), getitem___387964, int_387959)
    
    # Assigning a type to the variable 'tuple_var_assignment_386403' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'tuple_var_assignment_386403', subscript_call_result_387965)
    
    # Assigning a Subscript to a Name (line 640):
    
    # Obtaining the type of the subscript
    int_387966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 8), 'int')
    
    # Call to pade3(...): (line 640)
    # Processing the call keyword arguments (line 640)
    kwargs_387969 = {}
    # Getting the type of 'h' (line 640)
    h_387967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 15), 'h', False)
    # Obtaining the member 'pade3' of a type (line 640)
    pade3_387968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 15), h_387967, 'pade3')
    # Calling pade3(args, kwargs) (line 640)
    pade3_call_result_387970 = invoke(stypy.reporting.localization.Localization(__file__, 640, 15), pade3_387968, *[], **kwargs_387969)
    
    # Obtaining the member '__getitem__' of a type (line 640)
    getitem___387971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 8), pade3_call_result_387970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 640)
    subscript_call_result_387972 = invoke(stypy.reporting.localization.Localization(__file__, 640, 8), getitem___387971, int_387966)
    
    # Assigning a type to the variable 'tuple_var_assignment_386404' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'tuple_var_assignment_386404', subscript_call_result_387972)
    
    # Assigning a Name to a Name (line 640):
    # Getting the type of 'tuple_var_assignment_386403' (line 640)
    tuple_var_assignment_386403_387973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'tuple_var_assignment_386403')
    # Assigning a type to the variable 'U' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'U', tuple_var_assignment_386403_387973)
    
    # Assigning a Name to a Name (line 640):
    # Getting the type of 'tuple_var_assignment_386404' (line 640)
    tuple_var_assignment_386404_387974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'tuple_var_assignment_386404')
    # Assigning a type to the variable 'V' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 11), 'V', tuple_var_assignment_386404_387974)
    
    # Call to _solve_P_Q(...): (line 641)
    # Processing the call arguments (line 641)
    # Getting the type of 'U' (line 641)
    U_387976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 26), 'U', False)
    # Getting the type of 'V' (line 641)
    V_387977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 29), 'V', False)
    # Processing the call keyword arguments (line 641)
    # Getting the type of 'structure' (line 641)
    structure_387978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 42), 'structure', False)
    keyword_387979 = structure_387978
    kwargs_387980 = {'structure': keyword_387979}
    # Getting the type of '_solve_P_Q' (line 641)
    _solve_P_Q_387975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 15), '_solve_P_Q', False)
    # Calling _solve_P_Q(args, kwargs) (line 641)
    _solve_P_Q_call_result_387981 = invoke(stypy.reporting.localization.Localization(__file__, 641, 15), _solve_P_Q_387975, *[U_387976, V_387977], **kwargs_387980)
    
    # Assigning a type to the variable 'stypy_return_type' (line 641)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'stypy_return_type', _solve_P_Q_call_result_387981)
    # SSA join for if statement (line 639)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 644):
    
    # Assigning a Call to a Name (line 644):
    
    # Call to max(...): (line 644)
    # Processing the call arguments (line 644)
    # Getting the type of 'h' (line 644)
    h_387983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 16), 'h', False)
    # Obtaining the member 'd4_tight' of a type (line 644)
    d4_tight_387984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 16), h_387983, 'd4_tight')
    # Getting the type of 'h' (line 644)
    h_387985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 28), 'h', False)
    # Obtaining the member 'd6_loose' of a type (line 644)
    d6_loose_387986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 28), h_387985, 'd6_loose')
    # Processing the call keyword arguments (line 644)
    kwargs_387987 = {}
    # Getting the type of 'max' (line 644)
    max_387982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 12), 'max', False)
    # Calling max(args, kwargs) (line 644)
    max_call_result_387988 = invoke(stypy.reporting.localization.Localization(__file__, 644, 12), max_387982, *[d4_tight_387984, d6_loose_387986], **kwargs_387987)
    
    # Assigning a type to the variable 'eta_2' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'eta_2', max_call_result_387988)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'eta_2' (line 645)
    eta_2_387989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 7), 'eta_2')
    float_387990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 15), 'float')
    # Applying the binary operator '<' (line 645)
    result_lt_387991 = python_operator(stypy.reporting.localization.Localization(__file__, 645, 7), '<', eta_2_387989, float_387990)
    
    
    
    # Call to _ell(...): (line 645)
    # Processing the call arguments (line 645)
    # Getting the type of 'h' (line 645)
    h_387993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 47), 'h', False)
    # Obtaining the member 'A' of a type (line 645)
    A_387994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 47), h_387993, 'A')
    int_387995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 52), 'int')
    # Processing the call keyword arguments (line 645)
    kwargs_387996 = {}
    # Getting the type of '_ell' (line 645)
    _ell_387992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 42), '_ell', False)
    # Calling _ell(args, kwargs) (line 645)
    _ell_call_result_387997 = invoke(stypy.reporting.localization.Localization(__file__, 645, 42), _ell_387992, *[A_387994, int_387995], **kwargs_387996)
    
    int_387998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 58), 'int')
    # Applying the binary operator '==' (line 645)
    result_eq_387999 = python_operator(stypy.reporting.localization.Localization(__file__, 645, 42), '==', _ell_call_result_387997, int_387998)
    
    # Applying the binary operator 'and' (line 645)
    result_and_keyword_388000 = python_operator(stypy.reporting.localization.Localization(__file__, 645, 7), 'and', result_lt_387991, result_eq_387999)
    
    # Testing the type of an if condition (line 645)
    if_condition_388001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 645, 4), result_and_keyword_388000)
    # Assigning a type to the variable 'if_condition_388001' (line 645)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 4), 'if_condition_388001', if_condition_388001)
    # SSA begins for if statement (line 645)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 646):
    
    # Assigning a Subscript to a Name (line 646):
    
    # Obtaining the type of the subscript
    int_388002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 8), 'int')
    
    # Call to pade5(...): (line 646)
    # Processing the call keyword arguments (line 646)
    kwargs_388005 = {}
    # Getting the type of 'h' (line 646)
    h_388003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 15), 'h', False)
    # Obtaining the member 'pade5' of a type (line 646)
    pade5_388004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 15), h_388003, 'pade5')
    # Calling pade5(args, kwargs) (line 646)
    pade5_call_result_388006 = invoke(stypy.reporting.localization.Localization(__file__, 646, 15), pade5_388004, *[], **kwargs_388005)
    
    # Obtaining the member '__getitem__' of a type (line 646)
    getitem___388007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 8), pade5_call_result_388006, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 646)
    subscript_call_result_388008 = invoke(stypy.reporting.localization.Localization(__file__, 646, 8), getitem___388007, int_388002)
    
    # Assigning a type to the variable 'tuple_var_assignment_386405' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'tuple_var_assignment_386405', subscript_call_result_388008)
    
    # Assigning a Subscript to a Name (line 646):
    
    # Obtaining the type of the subscript
    int_388009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 8), 'int')
    
    # Call to pade5(...): (line 646)
    # Processing the call keyword arguments (line 646)
    kwargs_388012 = {}
    # Getting the type of 'h' (line 646)
    h_388010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 15), 'h', False)
    # Obtaining the member 'pade5' of a type (line 646)
    pade5_388011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 15), h_388010, 'pade5')
    # Calling pade5(args, kwargs) (line 646)
    pade5_call_result_388013 = invoke(stypy.reporting.localization.Localization(__file__, 646, 15), pade5_388011, *[], **kwargs_388012)
    
    # Obtaining the member '__getitem__' of a type (line 646)
    getitem___388014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 8), pade5_call_result_388013, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 646)
    subscript_call_result_388015 = invoke(stypy.reporting.localization.Localization(__file__, 646, 8), getitem___388014, int_388009)
    
    # Assigning a type to the variable 'tuple_var_assignment_386406' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'tuple_var_assignment_386406', subscript_call_result_388015)
    
    # Assigning a Name to a Name (line 646):
    # Getting the type of 'tuple_var_assignment_386405' (line 646)
    tuple_var_assignment_386405_388016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'tuple_var_assignment_386405')
    # Assigning a type to the variable 'U' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'U', tuple_var_assignment_386405_388016)
    
    # Assigning a Name to a Name (line 646):
    # Getting the type of 'tuple_var_assignment_386406' (line 646)
    tuple_var_assignment_386406_388017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'tuple_var_assignment_386406')
    # Assigning a type to the variable 'V' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 11), 'V', tuple_var_assignment_386406_388017)
    
    # Call to _solve_P_Q(...): (line 647)
    # Processing the call arguments (line 647)
    # Getting the type of 'U' (line 647)
    U_388019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 26), 'U', False)
    # Getting the type of 'V' (line 647)
    V_388020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 29), 'V', False)
    # Processing the call keyword arguments (line 647)
    # Getting the type of 'structure' (line 647)
    structure_388021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 42), 'structure', False)
    keyword_388022 = structure_388021
    kwargs_388023 = {'structure': keyword_388022}
    # Getting the type of '_solve_P_Q' (line 647)
    _solve_P_Q_388018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 15), '_solve_P_Q', False)
    # Calling _solve_P_Q(args, kwargs) (line 647)
    _solve_P_Q_call_result_388024 = invoke(stypy.reporting.localization.Localization(__file__, 647, 15), _solve_P_Q_388018, *[U_388019, V_388020], **kwargs_388023)
    
    # Assigning a type to the variable 'stypy_return_type' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'stypy_return_type', _solve_P_Q_call_result_388024)
    # SSA join for if statement (line 645)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 650):
    
    # Assigning a Call to a Name (line 650):
    
    # Call to max(...): (line 650)
    # Processing the call arguments (line 650)
    # Getting the type of 'h' (line 650)
    h_388026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 16), 'h', False)
    # Obtaining the member 'd6_tight' of a type (line 650)
    d6_tight_388027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 16), h_388026, 'd6_tight')
    # Getting the type of 'h' (line 650)
    h_388028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 28), 'h', False)
    # Obtaining the member 'd8_loose' of a type (line 650)
    d8_loose_388029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 28), h_388028, 'd8_loose')
    # Processing the call keyword arguments (line 650)
    kwargs_388030 = {}
    # Getting the type of 'max' (line 650)
    max_388025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 12), 'max', False)
    # Calling max(args, kwargs) (line 650)
    max_call_result_388031 = invoke(stypy.reporting.localization.Localization(__file__, 650, 12), max_388025, *[d6_tight_388027, d8_loose_388029], **kwargs_388030)
    
    # Assigning a type to the variable 'eta_3' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'eta_3', max_call_result_388031)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'eta_3' (line 651)
    eta_3_388032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 7), 'eta_3')
    float_388033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 15), 'float')
    # Applying the binary operator '<' (line 651)
    result_lt_388034 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 7), '<', eta_3_388032, float_388033)
    
    
    
    # Call to _ell(...): (line 651)
    # Processing the call arguments (line 651)
    # Getting the type of 'h' (line 651)
    h_388036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 47), 'h', False)
    # Obtaining the member 'A' of a type (line 651)
    A_388037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 47), h_388036, 'A')
    int_388038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 52), 'int')
    # Processing the call keyword arguments (line 651)
    kwargs_388039 = {}
    # Getting the type of '_ell' (line 651)
    _ell_388035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 42), '_ell', False)
    # Calling _ell(args, kwargs) (line 651)
    _ell_call_result_388040 = invoke(stypy.reporting.localization.Localization(__file__, 651, 42), _ell_388035, *[A_388037, int_388038], **kwargs_388039)
    
    int_388041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 58), 'int')
    # Applying the binary operator '==' (line 651)
    result_eq_388042 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 42), '==', _ell_call_result_388040, int_388041)
    
    # Applying the binary operator 'and' (line 651)
    result_and_keyword_388043 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 7), 'and', result_lt_388034, result_eq_388042)
    
    # Testing the type of an if condition (line 651)
    if_condition_388044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 651, 4), result_and_keyword_388043)
    # Assigning a type to the variable 'if_condition_388044' (line 651)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 4), 'if_condition_388044', if_condition_388044)
    # SSA begins for if statement (line 651)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 652):
    
    # Assigning a Subscript to a Name (line 652):
    
    # Obtaining the type of the subscript
    int_388045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 8), 'int')
    
    # Call to pade7(...): (line 652)
    # Processing the call keyword arguments (line 652)
    kwargs_388048 = {}
    # Getting the type of 'h' (line 652)
    h_388046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 15), 'h', False)
    # Obtaining the member 'pade7' of a type (line 652)
    pade7_388047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 15), h_388046, 'pade7')
    # Calling pade7(args, kwargs) (line 652)
    pade7_call_result_388049 = invoke(stypy.reporting.localization.Localization(__file__, 652, 15), pade7_388047, *[], **kwargs_388048)
    
    # Obtaining the member '__getitem__' of a type (line 652)
    getitem___388050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 8), pade7_call_result_388049, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 652)
    subscript_call_result_388051 = invoke(stypy.reporting.localization.Localization(__file__, 652, 8), getitem___388050, int_388045)
    
    # Assigning a type to the variable 'tuple_var_assignment_386407' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'tuple_var_assignment_386407', subscript_call_result_388051)
    
    # Assigning a Subscript to a Name (line 652):
    
    # Obtaining the type of the subscript
    int_388052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 8), 'int')
    
    # Call to pade7(...): (line 652)
    # Processing the call keyword arguments (line 652)
    kwargs_388055 = {}
    # Getting the type of 'h' (line 652)
    h_388053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 15), 'h', False)
    # Obtaining the member 'pade7' of a type (line 652)
    pade7_388054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 15), h_388053, 'pade7')
    # Calling pade7(args, kwargs) (line 652)
    pade7_call_result_388056 = invoke(stypy.reporting.localization.Localization(__file__, 652, 15), pade7_388054, *[], **kwargs_388055)
    
    # Obtaining the member '__getitem__' of a type (line 652)
    getitem___388057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 8), pade7_call_result_388056, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 652)
    subscript_call_result_388058 = invoke(stypy.reporting.localization.Localization(__file__, 652, 8), getitem___388057, int_388052)
    
    # Assigning a type to the variable 'tuple_var_assignment_386408' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'tuple_var_assignment_386408', subscript_call_result_388058)
    
    # Assigning a Name to a Name (line 652):
    # Getting the type of 'tuple_var_assignment_386407' (line 652)
    tuple_var_assignment_386407_388059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'tuple_var_assignment_386407')
    # Assigning a type to the variable 'U' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'U', tuple_var_assignment_386407_388059)
    
    # Assigning a Name to a Name (line 652):
    # Getting the type of 'tuple_var_assignment_386408' (line 652)
    tuple_var_assignment_386408_388060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'tuple_var_assignment_386408')
    # Assigning a type to the variable 'V' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 11), 'V', tuple_var_assignment_386408_388060)
    
    # Call to _solve_P_Q(...): (line 653)
    # Processing the call arguments (line 653)
    # Getting the type of 'U' (line 653)
    U_388062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 26), 'U', False)
    # Getting the type of 'V' (line 653)
    V_388063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 29), 'V', False)
    # Processing the call keyword arguments (line 653)
    # Getting the type of 'structure' (line 653)
    structure_388064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 42), 'structure', False)
    keyword_388065 = structure_388064
    kwargs_388066 = {'structure': keyword_388065}
    # Getting the type of '_solve_P_Q' (line 653)
    _solve_P_Q_388061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 15), '_solve_P_Q', False)
    # Calling _solve_P_Q(args, kwargs) (line 653)
    _solve_P_Q_call_result_388067 = invoke(stypy.reporting.localization.Localization(__file__, 653, 15), _solve_P_Q_388061, *[U_388062, V_388063], **kwargs_388066)
    
    # Assigning a type to the variable 'stypy_return_type' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'stypy_return_type', _solve_P_Q_call_result_388067)
    # SSA join for if statement (line 651)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'eta_3' (line 654)
    eta_3_388068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 7), 'eta_3')
    float_388069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 15), 'float')
    # Applying the binary operator '<' (line 654)
    result_lt_388070 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 7), '<', eta_3_388068, float_388069)
    
    
    
    # Call to _ell(...): (line 654)
    # Processing the call arguments (line 654)
    # Getting the type of 'h' (line 654)
    h_388072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 47), 'h', False)
    # Obtaining the member 'A' of a type (line 654)
    A_388073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 47), h_388072, 'A')
    int_388074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 52), 'int')
    # Processing the call keyword arguments (line 654)
    kwargs_388075 = {}
    # Getting the type of '_ell' (line 654)
    _ell_388071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 42), '_ell', False)
    # Calling _ell(args, kwargs) (line 654)
    _ell_call_result_388076 = invoke(stypy.reporting.localization.Localization(__file__, 654, 42), _ell_388071, *[A_388073, int_388074], **kwargs_388075)
    
    int_388077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 58), 'int')
    # Applying the binary operator '==' (line 654)
    result_eq_388078 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 42), '==', _ell_call_result_388076, int_388077)
    
    # Applying the binary operator 'and' (line 654)
    result_and_keyword_388079 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 7), 'and', result_lt_388070, result_eq_388078)
    
    # Testing the type of an if condition (line 654)
    if_condition_388080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 654, 4), result_and_keyword_388079)
    # Assigning a type to the variable 'if_condition_388080' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 'if_condition_388080', if_condition_388080)
    # SSA begins for if statement (line 654)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 655):
    
    # Assigning a Subscript to a Name (line 655):
    
    # Obtaining the type of the subscript
    int_388081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 8), 'int')
    
    # Call to pade9(...): (line 655)
    # Processing the call keyword arguments (line 655)
    kwargs_388084 = {}
    # Getting the type of 'h' (line 655)
    h_388082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 15), 'h', False)
    # Obtaining the member 'pade9' of a type (line 655)
    pade9_388083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 15), h_388082, 'pade9')
    # Calling pade9(args, kwargs) (line 655)
    pade9_call_result_388085 = invoke(stypy.reporting.localization.Localization(__file__, 655, 15), pade9_388083, *[], **kwargs_388084)
    
    # Obtaining the member '__getitem__' of a type (line 655)
    getitem___388086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 8), pade9_call_result_388085, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 655)
    subscript_call_result_388087 = invoke(stypy.reporting.localization.Localization(__file__, 655, 8), getitem___388086, int_388081)
    
    # Assigning a type to the variable 'tuple_var_assignment_386409' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'tuple_var_assignment_386409', subscript_call_result_388087)
    
    # Assigning a Subscript to a Name (line 655):
    
    # Obtaining the type of the subscript
    int_388088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 8), 'int')
    
    # Call to pade9(...): (line 655)
    # Processing the call keyword arguments (line 655)
    kwargs_388091 = {}
    # Getting the type of 'h' (line 655)
    h_388089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 15), 'h', False)
    # Obtaining the member 'pade9' of a type (line 655)
    pade9_388090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 15), h_388089, 'pade9')
    # Calling pade9(args, kwargs) (line 655)
    pade9_call_result_388092 = invoke(stypy.reporting.localization.Localization(__file__, 655, 15), pade9_388090, *[], **kwargs_388091)
    
    # Obtaining the member '__getitem__' of a type (line 655)
    getitem___388093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 8), pade9_call_result_388092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 655)
    subscript_call_result_388094 = invoke(stypy.reporting.localization.Localization(__file__, 655, 8), getitem___388093, int_388088)
    
    # Assigning a type to the variable 'tuple_var_assignment_386410' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'tuple_var_assignment_386410', subscript_call_result_388094)
    
    # Assigning a Name to a Name (line 655):
    # Getting the type of 'tuple_var_assignment_386409' (line 655)
    tuple_var_assignment_386409_388095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'tuple_var_assignment_386409')
    # Assigning a type to the variable 'U' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'U', tuple_var_assignment_386409_388095)
    
    # Assigning a Name to a Name (line 655):
    # Getting the type of 'tuple_var_assignment_386410' (line 655)
    tuple_var_assignment_386410_388096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'tuple_var_assignment_386410')
    # Assigning a type to the variable 'V' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 11), 'V', tuple_var_assignment_386410_388096)
    
    # Call to _solve_P_Q(...): (line 656)
    # Processing the call arguments (line 656)
    # Getting the type of 'U' (line 656)
    U_388098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 26), 'U', False)
    # Getting the type of 'V' (line 656)
    V_388099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 29), 'V', False)
    # Processing the call keyword arguments (line 656)
    # Getting the type of 'structure' (line 656)
    structure_388100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 42), 'structure', False)
    keyword_388101 = structure_388100
    kwargs_388102 = {'structure': keyword_388101}
    # Getting the type of '_solve_P_Q' (line 656)
    _solve_P_Q_388097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 15), '_solve_P_Q', False)
    # Calling _solve_P_Q(args, kwargs) (line 656)
    _solve_P_Q_call_result_388103 = invoke(stypy.reporting.localization.Localization(__file__, 656, 15), _solve_P_Q_388097, *[U_388098, V_388099], **kwargs_388102)
    
    # Assigning a type to the variable 'stypy_return_type' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'stypy_return_type', _solve_P_Q_call_result_388103)
    # SSA join for if statement (line 654)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 659):
    
    # Assigning a Call to a Name (line 659):
    
    # Call to max(...): (line 659)
    # Processing the call arguments (line 659)
    # Getting the type of 'h' (line 659)
    h_388105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 16), 'h', False)
    # Obtaining the member 'd8_loose' of a type (line 659)
    d8_loose_388106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 16), h_388105, 'd8_loose')
    # Getting the type of 'h' (line 659)
    h_388107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 28), 'h', False)
    # Obtaining the member 'd10_loose' of a type (line 659)
    d10_loose_388108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 28), h_388107, 'd10_loose')
    # Processing the call keyword arguments (line 659)
    kwargs_388109 = {}
    # Getting the type of 'max' (line 659)
    max_388104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'max', False)
    # Calling max(args, kwargs) (line 659)
    max_call_result_388110 = invoke(stypy.reporting.localization.Localization(__file__, 659, 12), max_388104, *[d8_loose_388106, d10_loose_388108], **kwargs_388109)
    
    # Assigning a type to the variable 'eta_4' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'eta_4', max_call_result_388110)
    
    # Assigning a Call to a Name (line 660):
    
    # Assigning a Call to a Name (line 660):
    
    # Call to min(...): (line 660)
    # Processing the call arguments (line 660)
    # Getting the type of 'eta_3' (line 660)
    eta_3_388112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 16), 'eta_3', False)
    # Getting the type of 'eta_4' (line 660)
    eta_4_388113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 23), 'eta_4', False)
    # Processing the call keyword arguments (line 660)
    kwargs_388114 = {}
    # Getting the type of 'min' (line 660)
    min_388111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 12), 'min', False)
    # Calling min(args, kwargs) (line 660)
    min_call_result_388115 = invoke(stypy.reporting.localization.Localization(__file__, 660, 12), min_388111, *[eta_3_388112, eta_4_388113], **kwargs_388114)
    
    # Assigning a type to the variable 'eta_5' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'eta_5', min_call_result_388115)
    
    # Assigning a Num to a Name (line 661):
    
    # Assigning a Num to a Name (line 661):
    float_388116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 15), 'float')
    # Assigning a type to the variable 'theta_13' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 4), 'theta_13', float_388116)
    
    # Assigning a Call to a Name (line 662):
    
    # Assigning a Call to a Name (line 662):
    
    # Call to max(...): (line 662)
    # Processing the call arguments (line 662)
    
    # Call to int(...): (line 662)
    # Processing the call arguments (line 662)
    
    # Call to ceil(...): (line 662)
    # Processing the call arguments (line 662)
    
    # Call to log2(...): (line 662)
    # Processing the call arguments (line 662)
    # Getting the type of 'eta_5' (line 662)
    eta_5_388123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 32), 'eta_5', False)
    # Getting the type of 'theta_13' (line 662)
    theta_13_388124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 40), 'theta_13', False)
    # Applying the binary operator 'div' (line 662)
    result_div_388125 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 32), 'div', eta_5_388123, theta_13_388124)
    
    # Processing the call keyword arguments (line 662)
    kwargs_388126 = {}
    # Getting the type of 'np' (line 662)
    np_388121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 24), 'np', False)
    # Obtaining the member 'log2' of a type (line 662)
    log2_388122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 24), np_388121, 'log2')
    # Calling log2(args, kwargs) (line 662)
    log2_call_result_388127 = invoke(stypy.reporting.localization.Localization(__file__, 662, 24), log2_388122, *[result_div_388125], **kwargs_388126)
    
    # Processing the call keyword arguments (line 662)
    kwargs_388128 = {}
    # Getting the type of 'np' (line 662)
    np_388119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'np', False)
    # Obtaining the member 'ceil' of a type (line 662)
    ceil_388120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 16), np_388119, 'ceil')
    # Calling ceil(args, kwargs) (line 662)
    ceil_call_result_388129 = invoke(stypy.reporting.localization.Localization(__file__, 662, 16), ceil_388120, *[log2_call_result_388127], **kwargs_388128)
    
    # Processing the call keyword arguments (line 662)
    kwargs_388130 = {}
    # Getting the type of 'int' (line 662)
    int_388118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 12), 'int', False)
    # Calling int(args, kwargs) (line 662)
    int_call_result_388131 = invoke(stypy.reporting.localization.Localization(__file__, 662, 12), int_388118, *[ceil_call_result_388129], **kwargs_388130)
    
    int_388132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 53), 'int')
    # Processing the call keyword arguments (line 662)
    kwargs_388133 = {}
    # Getting the type of 'max' (line 662)
    max_388117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'max', False)
    # Calling max(args, kwargs) (line 662)
    max_call_result_388134 = invoke(stypy.reporting.localization.Localization(__file__, 662, 8), max_388117, *[int_call_result_388131, int_388132], **kwargs_388133)
    
    # Assigning a type to the variable 's' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 's', max_call_result_388134)
    
    # Assigning a BinOp to a Name (line 663):
    
    # Assigning a BinOp to a Name (line 663):
    # Getting the type of 's' (line 663)
    s_388135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 's')
    
    # Call to _ell(...): (line 663)
    # Processing the call arguments (line 663)
    int_388137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 17), 'int')
    
    # Getting the type of 's' (line 663)
    s_388138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 21), 's', False)
    # Applying the 'usub' unary operator (line 663)
    result___neg___388139 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 20), 'usub', s_388138)
    
    # Applying the binary operator '**' (line 663)
    result_pow_388140 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 17), '**', int_388137, result___neg___388139)
    
    # Getting the type of 'h' (line 663)
    h_388141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 25), 'h', False)
    # Obtaining the member 'A' of a type (line 663)
    A_388142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 25), h_388141, 'A')
    # Applying the binary operator '*' (line 663)
    result_mul_388143 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 17), '*', result_pow_388140, A_388142)
    
    int_388144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 30), 'int')
    # Processing the call keyword arguments (line 663)
    kwargs_388145 = {}
    # Getting the type of '_ell' (line 663)
    _ell_388136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 12), '_ell', False)
    # Calling _ell(args, kwargs) (line 663)
    _ell_call_result_388146 = invoke(stypy.reporting.localization.Localization(__file__, 663, 12), _ell_388136, *[result_mul_388143, int_388144], **kwargs_388145)
    
    # Applying the binary operator '+' (line 663)
    result_add_388147 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 8), '+', s_388135, _ell_call_result_388146)
    
    # Assigning a type to the variable 's' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 4), 's', result_add_388147)
    
    # Assigning a Call to a Tuple (line 664):
    
    # Assigning a Subscript to a Name (line 664):
    
    # Obtaining the type of the subscript
    int_388148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 4), 'int')
    
    # Call to pade13_scaled(...): (line 664)
    # Processing the call arguments (line 664)
    # Getting the type of 's' (line 664)
    s_388151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 27), 's', False)
    # Processing the call keyword arguments (line 664)
    kwargs_388152 = {}
    # Getting the type of 'h' (line 664)
    h_388149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 11), 'h', False)
    # Obtaining the member 'pade13_scaled' of a type (line 664)
    pade13_scaled_388150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 11), h_388149, 'pade13_scaled')
    # Calling pade13_scaled(args, kwargs) (line 664)
    pade13_scaled_call_result_388153 = invoke(stypy.reporting.localization.Localization(__file__, 664, 11), pade13_scaled_388150, *[s_388151], **kwargs_388152)
    
    # Obtaining the member '__getitem__' of a type (line 664)
    getitem___388154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 4), pade13_scaled_call_result_388153, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 664)
    subscript_call_result_388155 = invoke(stypy.reporting.localization.Localization(__file__, 664, 4), getitem___388154, int_388148)
    
    # Assigning a type to the variable 'tuple_var_assignment_386411' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 4), 'tuple_var_assignment_386411', subscript_call_result_388155)
    
    # Assigning a Subscript to a Name (line 664):
    
    # Obtaining the type of the subscript
    int_388156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 4), 'int')
    
    # Call to pade13_scaled(...): (line 664)
    # Processing the call arguments (line 664)
    # Getting the type of 's' (line 664)
    s_388159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 27), 's', False)
    # Processing the call keyword arguments (line 664)
    kwargs_388160 = {}
    # Getting the type of 'h' (line 664)
    h_388157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 11), 'h', False)
    # Obtaining the member 'pade13_scaled' of a type (line 664)
    pade13_scaled_388158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 11), h_388157, 'pade13_scaled')
    # Calling pade13_scaled(args, kwargs) (line 664)
    pade13_scaled_call_result_388161 = invoke(stypy.reporting.localization.Localization(__file__, 664, 11), pade13_scaled_388158, *[s_388159], **kwargs_388160)
    
    # Obtaining the member '__getitem__' of a type (line 664)
    getitem___388162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 4), pade13_scaled_call_result_388161, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 664)
    subscript_call_result_388163 = invoke(stypy.reporting.localization.Localization(__file__, 664, 4), getitem___388162, int_388156)
    
    # Assigning a type to the variable 'tuple_var_assignment_386412' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 4), 'tuple_var_assignment_386412', subscript_call_result_388163)
    
    # Assigning a Name to a Name (line 664):
    # Getting the type of 'tuple_var_assignment_386411' (line 664)
    tuple_var_assignment_386411_388164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 4), 'tuple_var_assignment_386411')
    # Assigning a type to the variable 'U' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 4), 'U', tuple_var_assignment_386411_388164)
    
    # Assigning a Name to a Name (line 664):
    # Getting the type of 'tuple_var_assignment_386412' (line 664)
    tuple_var_assignment_386412_388165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 4), 'tuple_var_assignment_386412')
    # Assigning a type to the variable 'V' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 7), 'V', tuple_var_assignment_386412_388165)
    
    # Assigning a Call to a Name (line 665):
    
    # Assigning a Call to a Name (line 665):
    
    # Call to _solve_P_Q(...): (line 665)
    # Processing the call arguments (line 665)
    # Getting the type of 'U' (line 665)
    U_388167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 19), 'U', False)
    # Getting the type of 'V' (line 665)
    V_388168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 22), 'V', False)
    # Processing the call keyword arguments (line 665)
    # Getting the type of 'structure' (line 665)
    structure_388169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 35), 'structure', False)
    keyword_388170 = structure_388169
    kwargs_388171 = {'structure': keyword_388170}
    # Getting the type of '_solve_P_Q' (line 665)
    _solve_P_Q_388166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 8), '_solve_P_Q', False)
    # Calling _solve_P_Q(args, kwargs) (line 665)
    _solve_P_Q_call_result_388172 = invoke(stypy.reporting.localization.Localization(__file__, 665, 8), _solve_P_Q_388166, *[U_388167, V_388168], **kwargs_388171)
    
    # Assigning a type to the variable 'X' (line 665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'X', _solve_P_Q_call_result_388172)
    
    
    # Getting the type of 'structure' (line 666)
    structure_388173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 7), 'structure')
    # Getting the type of 'UPPER_TRIANGULAR' (line 666)
    UPPER_TRIANGULAR_388174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 20), 'UPPER_TRIANGULAR')
    # Applying the binary operator '==' (line 666)
    result_eq_388175 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 7), '==', structure_388173, UPPER_TRIANGULAR_388174)
    
    # Testing the type of an if condition (line 666)
    if_condition_388176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 666, 4), result_eq_388175)
    # Assigning a type to the variable 'if_condition_388176' (line 666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 4), 'if_condition_388176', if_condition_388176)
    # SSA begins for if statement (line 666)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 668):
    
    # Assigning a Call to a Name (line 668):
    
    # Call to _fragment_2_1(...): (line 668)
    # Processing the call arguments (line 668)
    # Getting the type of 'X' (line 668)
    X_388178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 26), 'X', False)
    # Getting the type of 'h' (line 668)
    h_388179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 29), 'h', False)
    # Obtaining the member 'A' of a type (line 668)
    A_388180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 29), h_388179, 'A')
    # Getting the type of 's' (line 668)
    s_388181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 34), 's', False)
    # Processing the call keyword arguments (line 668)
    kwargs_388182 = {}
    # Getting the type of '_fragment_2_1' (line 668)
    _fragment_2_1_388177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 12), '_fragment_2_1', False)
    # Calling _fragment_2_1(args, kwargs) (line 668)
    _fragment_2_1_call_result_388183 = invoke(stypy.reporting.localization.Localization(__file__, 668, 12), _fragment_2_1_388177, *[X_388178, A_388180, s_388181], **kwargs_388182)
    
    # Assigning a type to the variable 'X' (line 668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'X', _fragment_2_1_call_result_388183)
    # SSA branch for the else part of an if statement (line 666)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to range(...): (line 671)
    # Processing the call arguments (line 671)
    # Getting the type of 's' (line 671)
    s_388185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 23), 's', False)
    # Processing the call keyword arguments (line 671)
    kwargs_388186 = {}
    # Getting the type of 'range' (line 671)
    range_388184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 17), 'range', False)
    # Calling range(args, kwargs) (line 671)
    range_call_result_388187 = invoke(stypy.reporting.localization.Localization(__file__, 671, 17), range_388184, *[s_388185], **kwargs_388186)
    
    # Testing the type of a for loop iterable (line 671)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 671, 8), range_call_result_388187)
    # Getting the type of the for loop variable (line 671)
    for_loop_var_388188 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 671, 8), range_call_result_388187)
    # Assigning a type to the variable 'i' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'i', for_loop_var_388188)
    # SSA begins for a for statement (line 671)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 672):
    
    # Assigning a Call to a Name (line 672):
    
    # Call to dot(...): (line 672)
    # Processing the call arguments (line 672)
    # Getting the type of 'X' (line 672)
    X_388191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 22), 'X', False)
    # Processing the call keyword arguments (line 672)
    kwargs_388192 = {}
    # Getting the type of 'X' (line 672)
    X_388189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 16), 'X', False)
    # Obtaining the member 'dot' of a type (line 672)
    dot_388190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 16), X_388189, 'dot')
    # Calling dot(args, kwargs) (line 672)
    dot_call_result_388193 = invoke(stypy.reporting.localization.Localization(__file__, 672, 16), dot_388190, *[X_388191], **kwargs_388192)
    
    # Assigning a type to the variable 'X' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'X', dot_call_result_388193)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 666)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'X' (line 673)
    X_388194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 11), 'X')
    # Assigning a type to the variable 'stypy_return_type' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 4), 'stypy_return_type', X_388194)
    
    # ################# End of '_expm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_expm' in the type store
    # Getting the type of 'stypy_return_type' (line 605)
    stypy_return_type_388195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_388195)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_expm'
    return stypy_return_type_388195

# Assigning a type to the variable '_expm' (line 605)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 0), '_expm', _expm)

@norecursion
def _solve_P_Q(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 676)
    None_388196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 31), 'None')
    defaults = [None_388196]
    # Create a new context for function '_solve_P_Q'
    module_type_store = module_type_store.open_function_context('_solve_P_Q', 676, 0, False)
    
    # Passed parameters checking function
    _solve_P_Q.stypy_localization = localization
    _solve_P_Q.stypy_type_of_self = None
    _solve_P_Q.stypy_type_store = module_type_store
    _solve_P_Q.stypy_function_name = '_solve_P_Q'
    _solve_P_Q.stypy_param_names_list = ['U', 'V', 'structure']
    _solve_P_Q.stypy_varargs_param_name = None
    _solve_P_Q.stypy_kwargs_param_name = None
    _solve_P_Q.stypy_call_defaults = defaults
    _solve_P_Q.stypy_call_varargs = varargs
    _solve_P_Q.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_solve_P_Q', ['U', 'V', 'structure'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_solve_P_Q', localization, ['U', 'V', 'structure'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_solve_P_Q(...)' code ##################

    str_388197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, (-1)), 'str', '\n    A helper function for expm_2009.\n\n    Parameters\n    ----------\n    U : ndarray\n        Pade numerator.\n    V : ndarray\n        Pade denominator.\n    structure : str, optional\n        A string describing the structure of both matrices `U` and `V`.\n        Only `upper_triangular` is currently supported.\n\n    Notes\n    -----\n    The `structure` argument is inspired by similar args\n    for theano and cvxopt functions.\n\n    ')
    
    # Assigning a BinOp to a Name (line 696):
    
    # Assigning a BinOp to a Name (line 696):
    # Getting the type of 'U' (line 696)
    U_388198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'U')
    # Getting the type of 'V' (line 696)
    V_388199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 12), 'V')
    # Applying the binary operator '+' (line 696)
    result_add_388200 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 8), '+', U_388198, V_388199)
    
    # Assigning a type to the variable 'P' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 4), 'P', result_add_388200)
    
    # Assigning a BinOp to a Name (line 697):
    
    # Assigning a BinOp to a Name (line 697):
    
    # Getting the type of 'U' (line 697)
    U_388201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 9), 'U')
    # Applying the 'usub' unary operator (line 697)
    result___neg___388202 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 8), 'usub', U_388201)
    
    # Getting the type of 'V' (line 697)
    V_388203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 13), 'V')
    # Applying the binary operator '+' (line 697)
    result_add_388204 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 8), '+', result___neg___388202, V_388203)
    
    # Assigning a type to the variable 'Q' (line 697)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 4), 'Q', result_add_388204)
    
    
    # Call to isspmatrix(...): (line 698)
    # Processing the call arguments (line 698)
    # Getting the type of 'U' (line 698)
    U_388206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 18), 'U', False)
    # Processing the call keyword arguments (line 698)
    kwargs_388207 = {}
    # Getting the type of 'isspmatrix' (line 698)
    isspmatrix_388205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 7), 'isspmatrix', False)
    # Calling isspmatrix(args, kwargs) (line 698)
    isspmatrix_call_result_388208 = invoke(stypy.reporting.localization.Localization(__file__, 698, 7), isspmatrix_388205, *[U_388206], **kwargs_388207)
    
    # Testing the type of an if condition (line 698)
    if_condition_388209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 698, 4), isspmatrix_call_result_388208)
    # Assigning a type to the variable 'if_condition_388209' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'if_condition_388209', if_condition_388209)
    # SSA begins for if statement (line 698)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to spsolve(...): (line 699)
    # Processing the call arguments (line 699)
    # Getting the type of 'Q' (line 699)
    Q_388211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 23), 'Q', False)
    # Getting the type of 'P' (line 699)
    P_388212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 26), 'P', False)
    # Processing the call keyword arguments (line 699)
    kwargs_388213 = {}
    # Getting the type of 'spsolve' (line 699)
    spsolve_388210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 15), 'spsolve', False)
    # Calling spsolve(args, kwargs) (line 699)
    spsolve_call_result_388214 = invoke(stypy.reporting.localization.Localization(__file__, 699, 15), spsolve_388210, *[Q_388211, P_388212], **kwargs_388213)
    
    # Assigning a type to the variable 'stypy_return_type' (line 699)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'stypy_return_type', spsolve_call_result_388214)
    # SSA branch for the else part of an if statement (line 698)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 700)
    # Getting the type of 'structure' (line 700)
    structure_388215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 9), 'structure')
    # Getting the type of 'None' (line 700)
    None_388216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 22), 'None')
    
    (may_be_388217, more_types_in_union_388218) = may_be_none(structure_388215, None_388216)

    if may_be_388217:

        if more_types_in_union_388218:
            # Runtime conditional SSA (line 700)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to solve(...): (line 701)
        # Processing the call arguments (line 701)
        # Getting the type of 'Q' (line 701)
        Q_388220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 21), 'Q', False)
        # Getting the type of 'P' (line 701)
        P_388221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 24), 'P', False)
        # Processing the call keyword arguments (line 701)
        kwargs_388222 = {}
        # Getting the type of 'solve' (line 701)
        solve_388219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 15), 'solve', False)
        # Calling solve(args, kwargs) (line 701)
        solve_call_result_388223 = invoke(stypy.reporting.localization.Localization(__file__, 701, 15), solve_388219, *[Q_388220, P_388221], **kwargs_388222)
        
        # Assigning a type to the variable 'stypy_return_type' (line 701)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 8), 'stypy_return_type', solve_call_result_388223)

        if more_types_in_union_388218:
            # Runtime conditional SSA for else branch (line 700)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_388217) or more_types_in_union_388218):
        
        
        # Getting the type of 'structure' (line 702)
        structure_388224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 9), 'structure')
        # Getting the type of 'UPPER_TRIANGULAR' (line 702)
        UPPER_TRIANGULAR_388225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 22), 'UPPER_TRIANGULAR')
        # Applying the binary operator '==' (line 702)
        result_eq_388226 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 9), '==', structure_388224, UPPER_TRIANGULAR_388225)
        
        # Testing the type of an if condition (line 702)
        if_condition_388227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 702, 9), result_eq_388226)
        # Assigning a type to the variable 'if_condition_388227' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 9), 'if_condition_388227', if_condition_388227)
        # SSA begins for if statement (line 702)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to solve_triangular(...): (line 703)
        # Processing the call arguments (line 703)
        # Getting the type of 'Q' (line 703)
        Q_388229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 32), 'Q', False)
        # Getting the type of 'P' (line 703)
        P_388230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 35), 'P', False)
        # Processing the call keyword arguments (line 703)
        kwargs_388231 = {}
        # Getting the type of 'solve_triangular' (line 703)
        solve_triangular_388228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 15), 'solve_triangular', False)
        # Calling solve_triangular(args, kwargs) (line 703)
        solve_triangular_call_result_388232 = invoke(stypy.reporting.localization.Localization(__file__, 703, 15), solve_triangular_388228, *[Q_388229, P_388230], **kwargs_388231)
        
        # Assigning a type to the variable 'stypy_return_type' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'stypy_return_type', solve_triangular_call_result_388232)
        # SSA branch for the else part of an if statement (line 702)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 705)
        # Processing the call arguments (line 705)
        str_388234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 25), 'str', 'unsupported matrix structure: ')
        
        # Call to str(...): (line 705)
        # Processing the call arguments (line 705)
        # Getting the type of 'structure' (line 705)
        structure_388236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 64), 'structure', False)
        # Processing the call keyword arguments (line 705)
        kwargs_388237 = {}
        # Getting the type of 'str' (line 705)
        str_388235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 60), 'str', False)
        # Calling str(args, kwargs) (line 705)
        str_call_result_388238 = invoke(stypy.reporting.localization.Localization(__file__, 705, 60), str_388235, *[structure_388236], **kwargs_388237)
        
        # Applying the binary operator '+' (line 705)
        result_add_388239 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 25), '+', str_388234, str_call_result_388238)
        
        # Processing the call keyword arguments (line 705)
        kwargs_388240 = {}
        # Getting the type of 'ValueError' (line 705)
        ValueError_388233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 705)
        ValueError_call_result_388241 = invoke(stypy.reporting.localization.Localization(__file__, 705, 14), ValueError_388233, *[result_add_388239], **kwargs_388240)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 705, 8), ValueError_call_result_388241, 'raise parameter', BaseException)
        # SSA join for if statement (line 702)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_388217 and more_types_in_union_388218):
            # SSA join for if statement (line 700)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 698)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_solve_P_Q(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_solve_P_Q' in the type store
    # Getting the type of 'stypy_return_type' (line 676)
    stypy_return_type_388242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_388242)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_solve_P_Q'
    return stypy_return_type_388242

# Assigning a type to the variable '_solve_P_Q' (line 676)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 0), '_solve_P_Q', _solve_P_Q)

@norecursion
def _sinch(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_sinch'
    module_type_store = module_type_store.open_function_context('_sinch', 708, 0, False)
    
    # Passed parameters checking function
    _sinch.stypy_localization = localization
    _sinch.stypy_type_of_self = None
    _sinch.stypy_type_store = module_type_store
    _sinch.stypy_function_name = '_sinch'
    _sinch.stypy_param_names_list = ['x']
    _sinch.stypy_varargs_param_name = None
    _sinch.stypy_kwargs_param_name = None
    _sinch.stypy_call_defaults = defaults
    _sinch.stypy_call_varargs = varargs
    _sinch.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_sinch', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_sinch', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_sinch(...)' code ##################

    str_388243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, (-1)), 'str', '\n    Stably evaluate sinch.\n\n    Notes\n    -----\n    The strategy of falling back to a sixth order Taylor expansion\n    was suggested by the Spallation Neutron Source docs\n    which was found on the internet by google search.\n    http://www.ornl.gov/~t6p/resources/xal/javadoc/gov/sns/tools/math/ElementaryFunction.html\n    The details of the cutoff point and the Horner-like evaluation\n    was picked without reference to anything in particular.\n\n    Note that sinch is not currently implemented in scipy.special,\n    whereas the "engineer\'s" definition of sinc is implemented.\n    The implementation of sinc involves a scaling factor of pi\n    that distinguishes it from the "mathematician\'s" version of sinc.\n\n    ')
    
    # Assigning a BinOp to a Name (line 732):
    
    # Assigning a BinOp to a Name (line 732):
    # Getting the type of 'x' (line 732)
    x_388244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 9), 'x')
    # Getting the type of 'x' (line 732)
    x_388245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 11), 'x')
    # Applying the binary operator '*' (line 732)
    result_mul_388246 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 9), '*', x_388244, x_388245)
    
    # Assigning a type to the variable 'x2' (line 732)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'x2', result_mul_388246)
    
    
    
    # Call to abs(...): (line 733)
    # Processing the call arguments (line 733)
    # Getting the type of 'x' (line 733)
    x_388248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 11), 'x', False)
    # Processing the call keyword arguments (line 733)
    kwargs_388249 = {}
    # Getting the type of 'abs' (line 733)
    abs_388247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 7), 'abs', False)
    # Calling abs(args, kwargs) (line 733)
    abs_call_result_388250 = invoke(stypy.reporting.localization.Localization(__file__, 733, 7), abs_388247, *[x_388248], **kwargs_388249)
    
    float_388251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 16), 'float')
    # Applying the binary operator '<' (line 733)
    result_lt_388252 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 7), '<', abs_call_result_388250, float_388251)
    
    # Testing the type of an if condition (line 733)
    if_condition_388253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 733, 4), result_lt_388252)
    # Assigning a type to the variable 'if_condition_388253' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), 'if_condition_388253', if_condition_388253)
    # SSA begins for if statement (line 733)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_388254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 15), 'int')
    # Getting the type of 'x2' (line 734)
    x2_388255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 20), 'x2')
    float_388256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 23), 'float')
    # Applying the binary operator 'div' (line 734)
    result_div_388257 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 20), 'div', x2_388255, float_388256)
    
    int_388258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 28), 'int')
    # Getting the type of 'x2' (line 734)
    x2_388259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 33), 'x2')
    float_388260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 36), 'float')
    # Applying the binary operator 'div' (line 734)
    result_div_388261 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 33), 'div', x2_388259, float_388260)
    
    int_388262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 42), 'int')
    # Getting the type of 'x2' (line 734)
    x2_388263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 47), 'x2')
    float_388264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 50), 'float')
    # Applying the binary operator 'div' (line 734)
    result_div_388265 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 47), 'div', x2_388263, float_388264)
    
    # Applying the binary operator '+' (line 734)
    result_add_388266 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 42), '+', int_388262, result_div_388265)
    
    # Applying the binary operator '*' (line 734)
    result_mul_388267 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 32), '*', result_div_388261, result_add_388266)
    
    # Applying the binary operator '+' (line 734)
    result_add_388268 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 28), '+', int_388258, result_mul_388267)
    
    # Applying the binary operator '*' (line 734)
    result_mul_388269 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 19), '*', result_div_388257, result_add_388268)
    
    # Applying the binary operator '+' (line 734)
    result_add_388270 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 15), '+', int_388254, result_mul_388269)
    
    # Assigning a type to the variable 'stypy_return_type' (line 734)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), 'stypy_return_type', result_add_388270)
    # SSA branch for the else part of an if statement (line 733)
    module_type_store.open_ssa_branch('else')
    
    # Call to sinh(...): (line 736)
    # Processing the call arguments (line 736)
    # Getting the type of 'x' (line 736)
    x_388273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 23), 'x', False)
    # Processing the call keyword arguments (line 736)
    kwargs_388274 = {}
    # Getting the type of 'np' (line 736)
    np_388271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 15), 'np', False)
    # Obtaining the member 'sinh' of a type (line 736)
    sinh_388272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 15), np_388271, 'sinh')
    # Calling sinh(args, kwargs) (line 736)
    sinh_call_result_388275 = invoke(stypy.reporting.localization.Localization(__file__, 736, 15), sinh_388272, *[x_388273], **kwargs_388274)
    
    # Getting the type of 'x' (line 736)
    x_388276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 28), 'x')
    # Applying the binary operator 'div' (line 736)
    result_div_388277 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 15), 'div', sinh_call_result_388275, x_388276)
    
    # Assigning a type to the variable 'stypy_return_type' (line 736)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 8), 'stypy_return_type', result_div_388277)
    # SSA join for if statement (line 733)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_sinch(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_sinch' in the type store
    # Getting the type of 'stypy_return_type' (line 708)
    stypy_return_type_388278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_388278)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_sinch'
    return stypy_return_type_388278

# Assigning a type to the variable '_sinch' (line 708)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 0), '_sinch', _sinch)

@norecursion
def _eq_10_42(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_eq_10_42'
    module_type_store = module_type_store.open_function_context('_eq_10_42', 739, 0, False)
    
    # Passed parameters checking function
    _eq_10_42.stypy_localization = localization
    _eq_10_42.stypy_type_of_self = None
    _eq_10_42.stypy_type_store = module_type_store
    _eq_10_42.stypy_function_name = '_eq_10_42'
    _eq_10_42.stypy_param_names_list = ['lam_1', 'lam_2', 't_12']
    _eq_10_42.stypy_varargs_param_name = None
    _eq_10_42.stypy_kwargs_param_name = None
    _eq_10_42.stypy_call_defaults = defaults
    _eq_10_42.stypy_call_varargs = varargs
    _eq_10_42.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_eq_10_42', ['lam_1', 'lam_2', 't_12'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_eq_10_42', localization, ['lam_1', 'lam_2', 't_12'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_eq_10_42(...)' code ##################

    str_388279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, (-1)), 'str', '\n    Equation (10.42) of Functions of Matrices: Theory and Computation.\n\n    Notes\n    -----\n    This is a helper function for _fragment_2_1 of expm_2009.\n    Equation (10.42) is on page 251 in the section on Schur algorithms.\n    In particular, section 10.4.3 explains the Schur-Parlett algorithm.\n    expm([[lam_1, t_12], [0, lam_1])\n    =\n    [[exp(lam_1), t_12*exp((lam_1 + lam_2)/2)*sinch((lam_1 - lam_2)/2)],\n    [0, exp(lam_2)]\n    ')
    
    # Assigning a BinOp to a Name (line 758):
    
    # Assigning a BinOp to a Name (line 758):
    float_388280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 8), 'float')
    # Getting the type of 'lam_1' (line 758)
    lam_1_388281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 15), 'lam_1')
    # Getting the type of 'lam_2' (line 758)
    lam_2_388282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 23), 'lam_2')
    # Applying the binary operator '+' (line 758)
    result_add_388283 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 15), '+', lam_1_388281, lam_2_388282)
    
    # Applying the binary operator '*' (line 758)
    result_mul_388284 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 8), '*', float_388280, result_add_388283)
    
    # Assigning a type to the variable 'a' (line 758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 4), 'a', result_mul_388284)
    
    # Assigning a BinOp to a Name (line 759):
    
    # Assigning a BinOp to a Name (line 759):
    float_388285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 8), 'float')
    # Getting the type of 'lam_1' (line 759)
    lam_1_388286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 15), 'lam_1')
    # Getting the type of 'lam_2' (line 759)
    lam_2_388287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 23), 'lam_2')
    # Applying the binary operator '-' (line 759)
    result_sub_388288 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 15), '-', lam_1_388286, lam_2_388287)
    
    # Applying the binary operator '*' (line 759)
    result_mul_388289 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 8), '*', float_388285, result_sub_388288)
    
    # Assigning a type to the variable 'b' (line 759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 4), 'b', result_mul_388289)
    # Getting the type of 't_12' (line 760)
    t_12_388290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 11), 't_12')
    
    # Call to exp(...): (line 760)
    # Processing the call arguments (line 760)
    # Getting the type of 'a' (line 760)
    a_388293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 25), 'a', False)
    # Processing the call keyword arguments (line 760)
    kwargs_388294 = {}
    # Getting the type of 'np' (line 760)
    np_388291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 18), 'np', False)
    # Obtaining the member 'exp' of a type (line 760)
    exp_388292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 18), np_388291, 'exp')
    # Calling exp(args, kwargs) (line 760)
    exp_call_result_388295 = invoke(stypy.reporting.localization.Localization(__file__, 760, 18), exp_388292, *[a_388293], **kwargs_388294)
    
    # Applying the binary operator '*' (line 760)
    result_mul_388296 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 11), '*', t_12_388290, exp_call_result_388295)
    
    
    # Call to _sinch(...): (line 760)
    # Processing the call arguments (line 760)
    # Getting the type of 'b' (line 760)
    b_388298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 37), 'b', False)
    # Processing the call keyword arguments (line 760)
    kwargs_388299 = {}
    # Getting the type of '_sinch' (line 760)
    _sinch_388297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 30), '_sinch', False)
    # Calling _sinch(args, kwargs) (line 760)
    _sinch_call_result_388300 = invoke(stypy.reporting.localization.Localization(__file__, 760, 30), _sinch_388297, *[b_388298], **kwargs_388299)
    
    # Applying the binary operator '*' (line 760)
    result_mul_388301 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 28), '*', result_mul_388296, _sinch_call_result_388300)
    
    # Assigning a type to the variable 'stypy_return_type' (line 760)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 4), 'stypy_return_type', result_mul_388301)
    
    # ################# End of '_eq_10_42(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_eq_10_42' in the type store
    # Getting the type of 'stypy_return_type' (line 739)
    stypy_return_type_388302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_388302)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_eq_10_42'
    return stypy_return_type_388302

# Assigning a type to the variable '_eq_10_42' (line 739)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 0), '_eq_10_42', _eq_10_42)

@norecursion
def _fragment_2_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fragment_2_1'
    module_type_store = module_type_store.open_function_context('_fragment_2_1', 763, 0, False)
    
    # Passed parameters checking function
    _fragment_2_1.stypy_localization = localization
    _fragment_2_1.stypy_type_of_self = None
    _fragment_2_1.stypy_type_store = module_type_store
    _fragment_2_1.stypy_function_name = '_fragment_2_1'
    _fragment_2_1.stypy_param_names_list = ['X', 'T', 's']
    _fragment_2_1.stypy_varargs_param_name = None
    _fragment_2_1.stypy_kwargs_param_name = None
    _fragment_2_1.stypy_call_defaults = defaults
    _fragment_2_1.stypy_call_varargs = varargs
    _fragment_2_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fragment_2_1', ['X', 'T', 's'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fragment_2_1', localization, ['X', 'T', 's'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fragment_2_1(...)' code ##################

    str_388303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, (-1)), 'str', '\n    A helper function for expm_2009.\n\n    Notes\n    -----\n    The argument X is modified in-place, but this modification is not the same\n    as the returned value of the function.\n    This function also takes pains to do things in ways that are compatible\n    with sparse matrices, for example by avoiding fancy indexing\n    and by using methods of the matrices whenever possible instead of\n    using functions of the numpy or scipy libraries themselves.\n\n    ')
    
    # Assigning a Subscript to a Name (line 779):
    
    # Assigning a Subscript to a Name (line 779):
    
    # Obtaining the type of the subscript
    int_388304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 16), 'int')
    # Getting the type of 'X' (line 779)
    X_388305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'X')
    # Obtaining the member 'shape' of a type (line 779)
    shape_388306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 8), X_388305, 'shape')
    # Obtaining the member '__getitem__' of a type (line 779)
    getitem___388307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 8), shape_388306, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 779)
    subscript_call_result_388308 = invoke(stypy.reporting.localization.Localization(__file__, 779, 8), getitem___388307, int_388304)
    
    # Assigning a type to the variable 'n' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'n', subscript_call_result_388308)
    
    # Assigning a Call to a Name (line 780):
    
    # Assigning a Call to a Name (line 780):
    
    # Call to ravel(...): (line 780)
    # Processing the call arguments (line 780)
    
    # Call to copy(...): (line 780)
    # Processing the call keyword arguments (line 780)
    kwargs_388316 = {}
    
    # Call to diagonal(...): (line 780)
    # Processing the call keyword arguments (line 780)
    kwargs_388313 = {}
    # Getting the type of 'T' (line 780)
    T_388311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 22), 'T', False)
    # Obtaining the member 'diagonal' of a type (line 780)
    diagonal_388312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 22), T_388311, 'diagonal')
    # Calling diagonal(args, kwargs) (line 780)
    diagonal_call_result_388314 = invoke(stypy.reporting.localization.Localization(__file__, 780, 22), diagonal_388312, *[], **kwargs_388313)
    
    # Obtaining the member 'copy' of a type (line 780)
    copy_388315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 22), diagonal_call_result_388314, 'copy')
    # Calling copy(args, kwargs) (line 780)
    copy_call_result_388317 = invoke(stypy.reporting.localization.Localization(__file__, 780, 22), copy_388315, *[], **kwargs_388316)
    
    # Processing the call keyword arguments (line 780)
    kwargs_388318 = {}
    # Getting the type of 'np' (line 780)
    np_388309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 13), 'np', False)
    # Obtaining the member 'ravel' of a type (line 780)
    ravel_388310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 13), np_388309, 'ravel')
    # Calling ravel(args, kwargs) (line 780)
    ravel_call_result_388319 = invoke(stypy.reporting.localization.Localization(__file__, 780, 13), ravel_388310, *[copy_call_result_388317], **kwargs_388318)
    
    # Assigning a type to the variable 'diag_T' (line 780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 4), 'diag_T', ravel_call_result_388319)
    
    # Assigning a BinOp to a Name (line 783):
    
    # Assigning a BinOp to a Name (line 783):
    int_388320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 12), 'int')
    
    # Getting the type of 's' (line 783)
    s_388321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 18), 's')
    # Applying the 'usub' unary operator (line 783)
    result___neg___388322 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 17), 'usub', s_388321)
    
    # Applying the binary operator '**' (line 783)
    result_pow_388323 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 12), '**', int_388320, result___neg___388322)
    
    # Assigning a type to the variable 'scale' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'scale', result_pow_388323)
    
    # Assigning a Call to a Name (line 784):
    
    # Assigning a Call to a Name (line 784):
    
    # Call to exp(...): (line 784)
    # Processing the call arguments (line 784)
    # Getting the type of 'scale' (line 784)
    scale_388326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 22), 'scale', False)
    # Getting the type of 'diag_T' (line 784)
    diag_T_388327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 30), 'diag_T', False)
    # Applying the binary operator '*' (line 784)
    result_mul_388328 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 22), '*', scale_388326, diag_T_388327)
    
    # Processing the call keyword arguments (line 784)
    kwargs_388329 = {}
    # Getting the type of 'np' (line 784)
    np_388324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 15), 'np', False)
    # Obtaining the member 'exp' of a type (line 784)
    exp_388325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 15), np_388324, 'exp')
    # Calling exp(args, kwargs) (line 784)
    exp_call_result_388330 = invoke(stypy.reporting.localization.Localization(__file__, 784, 15), exp_388325, *[result_mul_388328], **kwargs_388329)
    
    # Assigning a type to the variable 'exp_diag' (line 784)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 4), 'exp_diag', exp_call_result_388330)
    
    
    # Call to range(...): (line 785)
    # Processing the call arguments (line 785)
    # Getting the type of 'n' (line 785)
    n_388332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 19), 'n', False)
    # Processing the call keyword arguments (line 785)
    kwargs_388333 = {}
    # Getting the type of 'range' (line 785)
    range_388331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 13), 'range', False)
    # Calling range(args, kwargs) (line 785)
    range_call_result_388334 = invoke(stypy.reporting.localization.Localization(__file__, 785, 13), range_388331, *[n_388332], **kwargs_388333)
    
    # Testing the type of a for loop iterable (line 785)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 785, 4), range_call_result_388334)
    # Getting the type of the for loop variable (line 785)
    for_loop_var_388335 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 785, 4), range_call_result_388334)
    # Assigning a type to the variable 'k' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 4), 'k', for_loop_var_388335)
    # SSA begins for a for statement (line 785)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Subscript (line 786):
    
    # Assigning a Subscript to a Subscript (line 786):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 786)
    k_388336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 27), 'k')
    # Getting the type of 'exp_diag' (line 786)
    exp_diag_388337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 18), 'exp_diag')
    # Obtaining the member '__getitem__' of a type (line 786)
    getitem___388338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 18), exp_diag_388337, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 786)
    subscript_call_result_388339 = invoke(stypy.reporting.localization.Localization(__file__, 786, 18), getitem___388338, k_388336)
    
    # Getting the type of 'X' (line 786)
    X_388340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'X')
    
    # Obtaining an instance of the builtin type 'tuple' (line 786)
    tuple_388341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 786)
    # Adding element type (line 786)
    # Getting the type of 'k' (line 786)
    k_388342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 10), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 786, 10), tuple_388341, k_388342)
    # Adding element type (line 786)
    # Getting the type of 'k' (line 786)
    k_388343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 13), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 786, 10), tuple_388341, k_388343)
    
    # Storing an element on a container (line 786)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 786, 8), X_388340, (tuple_388341, subscript_call_result_388339))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 788)
    # Processing the call arguments (line 788)
    # Getting the type of 's' (line 788)
    s_388345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 19), 's', False)
    int_388346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 21), 'int')
    # Applying the binary operator '-' (line 788)
    result_sub_388347 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 19), '-', s_388345, int_388346)
    
    int_388348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 24), 'int')
    int_388349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 28), 'int')
    # Processing the call keyword arguments (line 788)
    kwargs_388350 = {}
    # Getting the type of 'range' (line 788)
    range_388344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 13), 'range', False)
    # Calling range(args, kwargs) (line 788)
    range_call_result_388351 = invoke(stypy.reporting.localization.Localization(__file__, 788, 13), range_388344, *[result_sub_388347, int_388348, int_388349], **kwargs_388350)
    
    # Testing the type of a for loop iterable (line 788)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 788, 4), range_call_result_388351)
    # Getting the type of the for loop variable (line 788)
    for_loop_var_388352 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 788, 4), range_call_result_388351)
    # Assigning a type to the variable 'i' (line 788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 4), 'i', for_loop_var_388352)
    # SSA begins for a for statement (line 788)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 789):
    
    # Assigning a Call to a Name (line 789):
    
    # Call to dot(...): (line 789)
    # Processing the call arguments (line 789)
    # Getting the type of 'X' (line 789)
    X_388355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 18), 'X', False)
    # Processing the call keyword arguments (line 789)
    kwargs_388356 = {}
    # Getting the type of 'X' (line 789)
    X_388353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 12), 'X', False)
    # Obtaining the member 'dot' of a type (line 789)
    dot_388354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 12), X_388353, 'dot')
    # Calling dot(args, kwargs) (line 789)
    dot_call_result_388357 = invoke(stypy.reporting.localization.Localization(__file__, 789, 12), dot_388354, *[X_388355], **kwargs_388356)
    
    # Assigning a type to the variable 'X' (line 789)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 8), 'X', dot_call_result_388357)
    
    # Assigning a BinOp to a Name (line 792):
    
    # Assigning a BinOp to a Name (line 792):
    int_388358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 16), 'int')
    
    # Getting the type of 'i' (line 792)
    i_388359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 22), 'i')
    # Applying the 'usub' unary operator (line 792)
    result___neg___388360 = python_operator(stypy.reporting.localization.Localization(__file__, 792, 21), 'usub', i_388359)
    
    # Applying the binary operator '**' (line 792)
    result_pow_388361 = python_operator(stypy.reporting.localization.Localization(__file__, 792, 16), '**', int_388358, result___neg___388360)
    
    # Assigning a type to the variable 'scale' (line 792)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 792, 8), 'scale', result_pow_388361)
    
    # Assigning a Call to a Name (line 793):
    
    # Assigning a Call to a Name (line 793):
    
    # Call to exp(...): (line 793)
    # Processing the call arguments (line 793)
    # Getting the type of 'scale' (line 793)
    scale_388364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 26), 'scale', False)
    # Getting the type of 'diag_T' (line 793)
    diag_T_388365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 34), 'diag_T', False)
    # Applying the binary operator '*' (line 793)
    result_mul_388366 = python_operator(stypy.reporting.localization.Localization(__file__, 793, 26), '*', scale_388364, diag_T_388365)
    
    # Processing the call keyword arguments (line 793)
    kwargs_388367 = {}
    # Getting the type of 'np' (line 793)
    np_388362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 19), 'np', False)
    # Obtaining the member 'exp' of a type (line 793)
    exp_388363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 19), np_388362, 'exp')
    # Calling exp(args, kwargs) (line 793)
    exp_call_result_388368 = invoke(stypy.reporting.localization.Localization(__file__, 793, 19), exp_388363, *[result_mul_388366], **kwargs_388367)
    
    # Assigning a type to the variable 'exp_diag' (line 793)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 8), 'exp_diag', exp_call_result_388368)
    
    
    # Call to range(...): (line 794)
    # Processing the call arguments (line 794)
    # Getting the type of 'n' (line 794)
    n_388370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 23), 'n', False)
    # Processing the call keyword arguments (line 794)
    kwargs_388371 = {}
    # Getting the type of 'range' (line 794)
    range_388369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 17), 'range', False)
    # Calling range(args, kwargs) (line 794)
    range_call_result_388372 = invoke(stypy.reporting.localization.Localization(__file__, 794, 17), range_388369, *[n_388370], **kwargs_388371)
    
    # Testing the type of a for loop iterable (line 794)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 794, 8), range_call_result_388372)
    # Getting the type of the for loop variable (line 794)
    for_loop_var_388373 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 794, 8), range_call_result_388372)
    # Assigning a type to the variable 'k' (line 794)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'k', for_loop_var_388373)
    # SSA begins for a for statement (line 794)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Subscript (line 795):
    
    # Assigning a Subscript to a Subscript (line 795):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 795)
    k_388374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 31), 'k')
    # Getting the type of 'exp_diag' (line 795)
    exp_diag_388375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 22), 'exp_diag')
    # Obtaining the member '__getitem__' of a type (line 795)
    getitem___388376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 22), exp_diag_388375, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 795)
    subscript_call_result_388377 = invoke(stypy.reporting.localization.Localization(__file__, 795, 22), getitem___388376, k_388374)
    
    # Getting the type of 'X' (line 795)
    X_388378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 12), 'X')
    
    # Obtaining an instance of the builtin type 'tuple' (line 795)
    tuple_388379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 795)
    # Adding element type (line 795)
    # Getting the type of 'k' (line 795)
    k_388380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 14), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 795, 14), tuple_388379, k_388380)
    # Adding element type (line 795)
    # Getting the type of 'k' (line 795)
    k_388381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 17), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 795, 14), tuple_388379, k_388381)
    
    # Storing an element on a container (line 795)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 795, 12), X_388378, (tuple_388379, subscript_call_result_388377))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 801)
    # Processing the call arguments (line 801)
    # Getting the type of 'n' (line 801)
    n_388383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 23), 'n', False)
    int_388384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 25), 'int')
    # Applying the binary operator '-' (line 801)
    result_sub_388385 = python_operator(stypy.reporting.localization.Localization(__file__, 801, 23), '-', n_388383, int_388384)
    
    # Processing the call keyword arguments (line 801)
    kwargs_388386 = {}
    # Getting the type of 'range' (line 801)
    range_388382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 17), 'range', False)
    # Calling range(args, kwargs) (line 801)
    range_call_result_388387 = invoke(stypy.reporting.localization.Localization(__file__, 801, 17), range_388382, *[result_sub_388385], **kwargs_388386)
    
    # Testing the type of a for loop iterable (line 801)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 801, 8), range_call_result_388387)
    # Getting the type of the for loop variable (line 801)
    for_loop_var_388388 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 801, 8), range_call_result_388387)
    # Assigning a type to the variable 'k' (line 801)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 8), 'k', for_loop_var_388388)
    # SSA begins for a for statement (line 801)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 802):
    
    # Assigning a BinOp to a Name (line 802):
    # Getting the type of 'scale' (line 802)
    scale_388389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 20), 'scale')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 802)
    k_388390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 35), 'k')
    # Getting the type of 'diag_T' (line 802)
    diag_T_388391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 28), 'diag_T')
    # Obtaining the member '__getitem__' of a type (line 802)
    getitem___388392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 802, 28), diag_T_388391, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 802)
    subscript_call_result_388393 = invoke(stypy.reporting.localization.Localization(__file__, 802, 28), getitem___388392, k_388390)
    
    # Applying the binary operator '*' (line 802)
    result_mul_388394 = python_operator(stypy.reporting.localization.Localization(__file__, 802, 20), '*', scale_388389, subscript_call_result_388393)
    
    # Assigning a type to the variable 'lam_1' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 12), 'lam_1', result_mul_388394)
    
    # Assigning a BinOp to a Name (line 803):
    
    # Assigning a BinOp to a Name (line 803):
    # Getting the type of 'scale' (line 803)
    scale_388395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 20), 'scale')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 803)
    k_388396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 35), 'k')
    int_388397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 37), 'int')
    # Applying the binary operator '+' (line 803)
    result_add_388398 = python_operator(stypy.reporting.localization.Localization(__file__, 803, 35), '+', k_388396, int_388397)
    
    # Getting the type of 'diag_T' (line 803)
    diag_T_388399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 28), 'diag_T')
    # Obtaining the member '__getitem__' of a type (line 803)
    getitem___388400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 28), diag_T_388399, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 803)
    subscript_call_result_388401 = invoke(stypy.reporting.localization.Localization(__file__, 803, 28), getitem___388400, result_add_388398)
    
    # Applying the binary operator '*' (line 803)
    result_mul_388402 = python_operator(stypy.reporting.localization.Localization(__file__, 803, 20), '*', scale_388395, subscript_call_result_388401)
    
    # Assigning a type to the variable 'lam_2' (line 803)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 12), 'lam_2', result_mul_388402)
    
    # Assigning a BinOp to a Name (line 804):
    
    # Assigning a BinOp to a Name (line 804):
    # Getting the type of 'scale' (line 804)
    scale_388403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 19), 'scale')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 804)
    tuple_388404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 804)
    # Adding element type (line 804)
    # Getting the type of 'k' (line 804)
    k_388405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 29), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 804, 29), tuple_388404, k_388405)
    # Adding element type (line 804)
    # Getting the type of 'k' (line 804)
    k_388406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 32), 'k')
    int_388407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 34), 'int')
    # Applying the binary operator '+' (line 804)
    result_add_388408 = python_operator(stypy.reporting.localization.Localization(__file__, 804, 32), '+', k_388406, int_388407)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 804, 29), tuple_388404, result_add_388408)
    
    # Getting the type of 'T' (line 804)
    T_388409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 27), 'T')
    # Obtaining the member '__getitem__' of a type (line 804)
    getitem___388410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 804, 27), T_388409, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 804)
    subscript_call_result_388411 = invoke(stypy.reporting.localization.Localization(__file__, 804, 27), getitem___388410, tuple_388404)
    
    # Applying the binary operator '*' (line 804)
    result_mul_388412 = python_operator(stypy.reporting.localization.Localization(__file__, 804, 19), '*', scale_388403, subscript_call_result_388411)
    
    # Assigning a type to the variable 't_12' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 12), 't_12', result_mul_388412)
    
    # Assigning a Call to a Name (line 805):
    
    # Assigning a Call to a Name (line 805):
    
    # Call to _eq_10_42(...): (line 805)
    # Processing the call arguments (line 805)
    # Getting the type of 'lam_1' (line 805)
    lam_1_388414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 30), 'lam_1', False)
    # Getting the type of 'lam_2' (line 805)
    lam_2_388415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 37), 'lam_2', False)
    # Getting the type of 't_12' (line 805)
    t_12_388416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 44), 't_12', False)
    # Processing the call keyword arguments (line 805)
    kwargs_388417 = {}
    # Getting the type of '_eq_10_42' (line 805)
    _eq_10_42_388413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 20), '_eq_10_42', False)
    # Calling _eq_10_42(args, kwargs) (line 805)
    _eq_10_42_call_result_388418 = invoke(stypy.reporting.localization.Localization(__file__, 805, 20), _eq_10_42_388413, *[lam_1_388414, lam_2_388415, t_12_388416], **kwargs_388417)
    
    # Assigning a type to the variable 'value' (line 805)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 12), 'value', _eq_10_42_call_result_388418)
    
    # Assigning a Name to a Subscript (line 806):
    
    # Assigning a Name to a Subscript (line 806):
    # Getting the type of 'value' (line 806)
    value_388419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 24), 'value')
    # Getting the type of 'X' (line 806)
    X_388420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 12), 'X')
    
    # Obtaining an instance of the builtin type 'tuple' (line 806)
    tuple_388421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 806)
    # Adding element type (line 806)
    # Getting the type of 'k' (line 806)
    k_388422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 14), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 14), tuple_388421, k_388422)
    # Adding element type (line 806)
    # Getting the type of 'k' (line 806)
    k_388423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 17), 'k')
    int_388424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 19), 'int')
    # Applying the binary operator '+' (line 806)
    result_add_388425 = python_operator(stypy.reporting.localization.Localization(__file__, 806, 17), '+', k_388423, int_388424)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 14), tuple_388421, result_add_388425)
    
    # Storing an element on a container (line 806)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 12), X_388420, (tuple_388421, value_388419))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'X' (line 809)
    X_388426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 11), 'X')
    # Assigning a type to the variable 'stypy_return_type' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'stypy_return_type', X_388426)
    
    # ################# End of '_fragment_2_1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fragment_2_1' in the type store
    # Getting the type of 'stypy_return_type' (line 763)
    stypy_return_type_388427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_388427)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fragment_2_1'
    return stypy_return_type_388427

# Assigning a type to the variable '_fragment_2_1' (line 763)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 0), '_fragment_2_1', _fragment_2_1)

@norecursion
def _ell(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_ell'
    module_type_store = module_type_store.open_function_context('_ell', 812, 0, False)
    
    # Passed parameters checking function
    _ell.stypy_localization = localization
    _ell.stypy_type_of_self = None
    _ell.stypy_type_store = module_type_store
    _ell.stypy_function_name = '_ell'
    _ell.stypy_param_names_list = ['A', 'm']
    _ell.stypy_varargs_param_name = None
    _ell.stypy_kwargs_param_name = None
    _ell.stypy_call_defaults = defaults
    _ell.stypy_call_varargs = varargs
    _ell.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_ell', ['A', 'm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_ell', localization, ['A', 'm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_ell(...)' code ##################

    str_388428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, (-1)), 'str', '\n    A helper function for expm_2009.\n\n    Parameters\n    ----------\n    A : linear operator\n        A linear operator whose norm of power we care about.\n    m : int\n        The power of the linear operator\n\n    Returns\n    -------\n    value : int\n        A value related to a bound.\n\n    ')
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 829)
    # Processing the call arguments (line 829)
    # Getting the type of 'A' (line 829)
    A_388430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 829)
    shape_388431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 11), A_388430, 'shape')
    # Processing the call keyword arguments (line 829)
    kwargs_388432 = {}
    # Getting the type of 'len' (line 829)
    len_388429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 7), 'len', False)
    # Calling len(args, kwargs) (line 829)
    len_call_result_388433 = invoke(stypy.reporting.localization.Localization(__file__, 829, 7), len_388429, *[shape_388431], **kwargs_388432)
    
    int_388434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 23), 'int')
    # Applying the binary operator '!=' (line 829)
    result_ne_388435 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 7), '!=', len_call_result_388433, int_388434)
    
    
    
    # Obtaining the type of the subscript
    int_388436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 36), 'int')
    # Getting the type of 'A' (line 829)
    A_388437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 28), 'A')
    # Obtaining the member 'shape' of a type (line 829)
    shape_388438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 28), A_388437, 'shape')
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___388439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 28), shape_388438, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_388440 = invoke(stypy.reporting.localization.Localization(__file__, 829, 28), getitem___388439, int_388436)
    
    
    # Obtaining the type of the subscript
    int_388441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 50), 'int')
    # Getting the type of 'A' (line 829)
    A_388442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 42), 'A')
    # Obtaining the member 'shape' of a type (line 829)
    shape_388443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 42), A_388442, 'shape')
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___388444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 42), shape_388443, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_388445 = invoke(stypy.reporting.localization.Localization(__file__, 829, 42), getitem___388444, int_388441)
    
    # Applying the binary operator '!=' (line 829)
    result_ne_388446 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 28), '!=', subscript_call_result_388440, subscript_call_result_388445)
    
    # Applying the binary operator 'or' (line 829)
    result_or_keyword_388447 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 7), 'or', result_ne_388435, result_ne_388446)
    
    # Testing the type of an if condition (line 829)
    if_condition_388448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 829, 4), result_or_keyword_388447)
    # Assigning a type to the variable 'if_condition_388448' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 4), 'if_condition_388448', if_condition_388448)
    # SSA begins for if statement (line 829)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 830)
    # Processing the call arguments (line 830)
    str_388450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 25), 'str', 'expected A to be like a square matrix')
    # Processing the call keyword arguments (line 830)
    kwargs_388451 = {}
    # Getting the type of 'ValueError' (line 830)
    ValueError_388449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 830)
    ValueError_call_result_388452 = invoke(stypy.reporting.localization.Localization(__file__, 830, 14), ValueError_388449, *[str_388450], **kwargs_388451)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 830, 8), ValueError_call_result_388452, 'raise parameter', BaseException)
    # SSA join for if statement (line 829)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 832):
    
    # Assigning a BinOp to a Name (line 832):
    int_388453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 8), 'int')
    # Getting the type of 'm' (line 832)
    m_388454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 10), 'm')
    # Applying the binary operator '*' (line 832)
    result_mul_388455 = python_operator(stypy.reporting.localization.Localization(__file__, 832, 8), '*', int_388453, m_388454)
    
    int_388456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 14), 'int')
    # Applying the binary operator '+' (line 832)
    result_add_388457 = python_operator(stypy.reporting.localization.Localization(__file__, 832, 8), '+', result_mul_388455, int_388456)
    
    # Assigning a type to the variable 'p' (line 832)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 4), 'p', result_add_388457)
    
    # Assigning a Call to a Name (line 836):
    
    # Assigning a Call to a Name (line 836):
    
    # Call to comb(...): (line 836)
    # Processing the call arguments (line 836)
    int_388461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 37), 'int')
    # Getting the type of 'p' (line 836)
    p_388462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 39), 'p', False)
    # Applying the binary operator '*' (line 836)
    result_mul_388463 = python_operator(stypy.reporting.localization.Localization(__file__, 836, 37), '*', int_388461, p_388462)
    
    # Getting the type of 'p' (line 836)
    p_388464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 42), 'p', False)
    # Processing the call keyword arguments (line 836)
    # Getting the type of 'True' (line 836)
    True_388465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 51), 'True', False)
    keyword_388466 = True_388465
    kwargs_388467 = {'exact': keyword_388466}
    # Getting the type of 'scipy' (line 836)
    scipy_388458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 18), 'scipy', False)
    # Obtaining the member 'special' of a type (line 836)
    special_388459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 836, 18), scipy_388458, 'special')
    # Obtaining the member 'comb' of a type (line 836)
    comb_388460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 836, 18), special_388459, 'comb')
    # Calling comb(args, kwargs) (line 836)
    comb_call_result_388468 = invoke(stypy.reporting.localization.Localization(__file__, 836, 18), comb_388460, *[result_mul_388463, p_388464], **kwargs_388467)
    
    # Assigning a type to the variable 'choose_2p_p' (line 836)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 4), 'choose_2p_p', comb_call_result_388468)
    
    # Assigning a Call to a Name (line 837):
    
    # Assigning a Call to a Name (line 837):
    
    # Call to float(...): (line 837)
    # Processing the call arguments (line 837)
    # Getting the type of 'choose_2p_p' (line 837)
    choose_2p_p_388470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 24), 'choose_2p_p', False)
    
    # Call to factorial(...): (line 837)
    # Processing the call arguments (line 837)
    int_388473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 53), 'int')
    # Getting the type of 'p' (line 837)
    p_388474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 55), 'p', False)
    # Applying the binary operator '*' (line 837)
    result_mul_388475 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 53), '*', int_388473, p_388474)
    
    int_388476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 59), 'int')
    # Applying the binary operator '+' (line 837)
    result_add_388477 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 53), '+', result_mul_388475, int_388476)
    
    # Processing the call keyword arguments (line 837)
    kwargs_388478 = {}
    # Getting the type of 'math' (line 837)
    math_388471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 38), 'math', False)
    # Obtaining the member 'factorial' of a type (line 837)
    factorial_388472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 38), math_388471, 'factorial')
    # Calling factorial(args, kwargs) (line 837)
    factorial_call_result_388479 = invoke(stypy.reporting.localization.Localization(__file__, 837, 38), factorial_388472, *[result_add_388477], **kwargs_388478)
    
    # Applying the binary operator '*' (line 837)
    result_mul_388480 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 24), '*', choose_2p_p_388470, factorial_call_result_388479)
    
    # Processing the call keyword arguments (line 837)
    kwargs_388481 = {}
    # Getting the type of 'float' (line 837)
    float_388469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 18), 'float', False)
    # Calling float(args, kwargs) (line 837)
    float_call_result_388482 = invoke(stypy.reporting.localization.Localization(__file__, 837, 18), float_388469, *[result_mul_388480], **kwargs_388481)
    
    # Assigning a type to the variable 'abs_c_recip' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'abs_c_recip', float_call_result_388482)
    
    # Assigning a BinOp to a Name (line 841):
    
    # Assigning a BinOp to a Name (line 841):
    int_388483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 8), 'int')
    int_388484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 11), 'int')
    # Applying the binary operator '**' (line 841)
    result_pow_388485 = python_operator(stypy.reporting.localization.Localization(__file__, 841, 8), '**', int_388483, int_388484)
    
    # Assigning a type to the variable 'u' (line 841)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 4), 'u', result_pow_388485)
    
    # Assigning a Call to a Name (line 844):
    
    # Assigning a Call to a Name (line 844):
    
    # Call to _onenorm_matrix_power_nnm(...): (line 844)
    # Processing the call arguments (line 844)
    
    # Call to abs(...): (line 844)
    # Processing the call arguments (line 844)
    # Getting the type of 'A' (line 844)
    A_388488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 50), 'A', False)
    # Processing the call keyword arguments (line 844)
    kwargs_388489 = {}
    # Getting the type of 'abs' (line 844)
    abs_388487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 46), 'abs', False)
    # Calling abs(args, kwargs) (line 844)
    abs_call_result_388490 = invoke(stypy.reporting.localization.Localization(__file__, 844, 46), abs_388487, *[A_388488], **kwargs_388489)
    
    # Getting the type of 'p' (line 844)
    p_388491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 54), 'p', False)
    # Processing the call keyword arguments (line 844)
    kwargs_388492 = {}
    # Getting the type of '_onenorm_matrix_power_nnm' (line 844)
    _onenorm_matrix_power_nnm_388486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 20), '_onenorm_matrix_power_nnm', False)
    # Calling _onenorm_matrix_power_nnm(args, kwargs) (line 844)
    _onenorm_matrix_power_nnm_call_result_388493 = invoke(stypy.reporting.localization.Localization(__file__, 844, 20), _onenorm_matrix_power_nnm_388486, *[abs_call_result_388490, p_388491], **kwargs_388492)
    
    # Assigning a type to the variable 'A_abs_onenorm' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 4), 'A_abs_onenorm', _onenorm_matrix_power_nnm_call_result_388493)
    
    
    # Getting the type of 'A_abs_onenorm' (line 847)
    A_abs_onenorm_388494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 11), 'A_abs_onenorm')
    # Applying the 'not' unary operator (line 847)
    result_not__388495 = python_operator(stypy.reporting.localization.Localization(__file__, 847, 7), 'not', A_abs_onenorm_388494)
    
    # Testing the type of an if condition (line 847)
    if_condition_388496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 847, 4), result_not__388495)
    # Assigning a type to the variable 'if_condition_388496' (line 847)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 847, 4), 'if_condition_388496', if_condition_388496)
    # SSA begins for if statement (line 847)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_388497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 848)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'stypy_return_type', int_388497)
    # SSA join for if statement (line 847)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 850):
    
    # Assigning a BinOp to a Name (line 850):
    # Getting the type of 'A_abs_onenorm' (line 850)
    A_abs_onenorm_388498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 12), 'A_abs_onenorm')
    
    # Call to _onenorm(...): (line 850)
    # Processing the call arguments (line 850)
    # Getting the type of 'A' (line 850)
    A_388500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 38), 'A', False)
    # Processing the call keyword arguments (line 850)
    kwargs_388501 = {}
    # Getting the type of '_onenorm' (line 850)
    _onenorm_388499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 29), '_onenorm', False)
    # Calling _onenorm(args, kwargs) (line 850)
    _onenorm_call_result_388502 = invoke(stypy.reporting.localization.Localization(__file__, 850, 29), _onenorm_388499, *[A_388500], **kwargs_388501)
    
    # Getting the type of 'abs_c_recip' (line 850)
    abs_c_recip_388503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 43), 'abs_c_recip')
    # Applying the binary operator '*' (line 850)
    result_mul_388504 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 29), '*', _onenorm_call_result_388502, abs_c_recip_388503)
    
    # Applying the binary operator 'div' (line 850)
    result_div_388505 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 12), 'div', A_abs_onenorm_388498, result_mul_388504)
    
    # Assigning a type to the variable 'alpha' (line 850)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 4), 'alpha', result_div_388505)
    
    # Assigning a Call to a Name (line 851):
    
    # Assigning a Call to a Name (line 851):
    
    # Call to log2(...): (line 851)
    # Processing the call arguments (line 851)
    # Getting the type of 'alpha' (line 851)
    alpha_388508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 31), 'alpha', False)
    # Getting the type of 'u' (line 851)
    u_388509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 37), 'u', False)
    # Applying the binary operator 'div' (line 851)
    result_div_388510 = python_operator(stypy.reporting.localization.Localization(__file__, 851, 31), 'div', alpha_388508, u_388509)
    
    # Processing the call keyword arguments (line 851)
    kwargs_388511 = {}
    # Getting the type of 'np' (line 851)
    np_388506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 23), 'np', False)
    # Obtaining the member 'log2' of a type (line 851)
    log2_388507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 23), np_388506, 'log2')
    # Calling log2(args, kwargs) (line 851)
    log2_call_result_388512 = invoke(stypy.reporting.localization.Localization(__file__, 851, 23), log2_388507, *[result_div_388510], **kwargs_388511)
    
    # Assigning a type to the variable 'log2_alpha_div_u' (line 851)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 4), 'log2_alpha_div_u', log2_call_result_388512)
    
    # Assigning a Call to a Name (line 852):
    
    # Assigning a Call to a Name (line 852):
    
    # Call to int(...): (line 852)
    # Processing the call arguments (line 852)
    
    # Call to ceil(...): (line 852)
    # Processing the call arguments (line 852)
    # Getting the type of 'log2_alpha_div_u' (line 852)
    log2_alpha_div_u_388516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 24), 'log2_alpha_div_u', False)
    int_388517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 44), 'int')
    # Getting the type of 'm' (line 852)
    m_388518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 48), 'm', False)
    # Applying the binary operator '*' (line 852)
    result_mul_388519 = python_operator(stypy.reporting.localization.Localization(__file__, 852, 44), '*', int_388517, m_388518)
    
    # Applying the binary operator 'div' (line 852)
    result_div_388520 = python_operator(stypy.reporting.localization.Localization(__file__, 852, 24), 'div', log2_alpha_div_u_388516, result_mul_388519)
    
    # Processing the call keyword arguments (line 852)
    kwargs_388521 = {}
    # Getting the type of 'np' (line 852)
    np_388514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 16), 'np', False)
    # Obtaining the member 'ceil' of a type (line 852)
    ceil_388515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 852, 16), np_388514, 'ceil')
    # Calling ceil(args, kwargs) (line 852)
    ceil_call_result_388522 = invoke(stypy.reporting.localization.Localization(__file__, 852, 16), ceil_388515, *[result_div_388520], **kwargs_388521)
    
    # Processing the call keyword arguments (line 852)
    kwargs_388523 = {}
    # Getting the type of 'int' (line 852)
    int_388513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 12), 'int', False)
    # Calling int(args, kwargs) (line 852)
    int_call_result_388524 = invoke(stypy.reporting.localization.Localization(__file__, 852, 12), int_388513, *[ceil_call_result_388522], **kwargs_388523)
    
    # Assigning a type to the variable 'value' (line 852)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 4), 'value', int_call_result_388524)
    
    # Call to max(...): (line 853)
    # Processing the call arguments (line 853)
    # Getting the type of 'value' (line 853)
    value_388526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 15), 'value', False)
    int_388527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 22), 'int')
    # Processing the call keyword arguments (line 853)
    kwargs_388528 = {}
    # Getting the type of 'max' (line 853)
    max_388525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 11), 'max', False)
    # Calling max(args, kwargs) (line 853)
    max_call_result_388529 = invoke(stypy.reporting.localization.Localization(__file__, 853, 11), max_388525, *[value_388526, int_388527], **kwargs_388528)
    
    # Assigning a type to the variable 'stypy_return_type' (line 853)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 4), 'stypy_return_type', max_call_result_388529)
    
    # ################# End of '_ell(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_ell' in the type store
    # Getting the type of 'stypy_return_type' (line 812)
    stypy_return_type_388530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_388530)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_ell'
    return stypy_return_type_388530

# Assigning a type to the variable '_ell' (line 812)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 0), '_ell', _ell)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
