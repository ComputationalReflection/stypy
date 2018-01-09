
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import math
4: import numpy as np
5: from scipy._lib.six import xrange
6: from scipy._lib.six import string_types
7: 
8: 
9: __all__ = ['tri', 'tril', 'triu', 'toeplitz', 'circulant', 'hankel',
10:            'hadamard', 'leslie', 'kron', 'block_diag', 'companion',
11:            'helmert', 'hilbert', 'invhilbert', 'pascal', 'invpascal', 'dft']
12: 
13: 
14: #-----------------------------------------------------------------------------
15: # matrix construction functions
16: #-----------------------------------------------------------------------------
17: 
18: #
19: # *Note*: tri{,u,l} is implemented in numpy, but an important bug was fixed in
20: # 2.0.0.dev-1af2f3, the following tri{,u,l} definitions are here for backwards
21: # compatibility.
22: 
23: def tri(N, M=None, k=0, dtype=None):
24:     '''
25:     Construct (N, M) matrix filled with ones at and below the k-th diagonal.
26: 
27:     The matrix has A[i,j] == 1 for i <= j + k
28: 
29:     Parameters
30:     ----------
31:     N : int
32:         The size of the first dimension of the matrix.
33:     M : int or None, optional
34:         The size of the second dimension of the matrix. If `M` is None,
35:         `M = N` is assumed.
36:     k : int, optional
37:         Number of subdiagonal below which matrix is filled with ones.
38:         `k` = 0 is the main diagonal, `k` < 0 subdiagonal and `k` > 0
39:         superdiagonal.
40:     dtype : dtype, optional
41:         Data type of the matrix.
42: 
43:     Returns
44:     -------
45:     tri : (N, M) ndarray
46:         Tri matrix.
47: 
48:     Examples
49:     --------
50:     >>> from scipy.linalg import tri
51:     >>> tri(3, 5, 2, dtype=int)
52:     array([[1, 1, 1, 0, 0],
53:            [1, 1, 1, 1, 0],
54:            [1, 1, 1, 1, 1]])
55:     >>> tri(3, 5, -1, dtype=int)
56:     array([[0, 0, 0, 0, 0],
57:            [1, 0, 0, 0, 0],
58:            [1, 1, 0, 0, 0]])
59: 
60:     '''
61:     if M is None:
62:         M = N
63:     if isinstance(M, string_types):
64:         #pearu: any objections to remove this feature?
65:         #       As tri(N,'d') is equivalent to tri(N,dtype='d')
66:         dtype = M
67:         M = N
68:     m = np.greater_equal(np.subtract.outer(np.arange(N), np.arange(M)), -k)
69:     if dtype is None:
70:         return m
71:     else:
72:         return m.astype(dtype)
73: 
74: 
75: def tril(m, k=0):
76:     '''
77:     Make a copy of a matrix with elements above the k-th diagonal zeroed.
78: 
79:     Parameters
80:     ----------
81:     m : array_like
82:         Matrix whose elements to return
83:     k : int, optional
84:         Diagonal above which to zero elements.
85:         `k` == 0 is the main diagonal, `k` < 0 subdiagonal and
86:         `k` > 0 superdiagonal.
87: 
88:     Returns
89:     -------
90:     tril : ndarray
91:         Return is the same shape and type as `m`.
92: 
93:     Examples
94:     --------
95:     >>> from scipy.linalg import tril
96:     >>> tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
97:     array([[ 0,  0,  0],
98:            [ 4,  0,  0],
99:            [ 7,  8,  0],
100:            [10, 11, 12]])
101: 
102:     '''
103:     m = np.asarray(m)
104:     out = tri(m.shape[0], m.shape[1], k=k, dtype=m.dtype.char) * m
105:     return out
106: 
107: 
108: def triu(m, k=0):
109:     '''
110:     Make a copy of a matrix with elements below the k-th diagonal zeroed.
111: 
112:     Parameters
113:     ----------
114:     m : array_like
115:         Matrix whose elements to return
116:     k : int, optional
117:         Diagonal below which to zero elements.
118:         `k` == 0 is the main diagonal, `k` < 0 subdiagonal and
119:         `k` > 0 superdiagonal.
120: 
121:     Returns
122:     -------
123:     triu : ndarray
124:         Return matrix with zeroed elements below the k-th diagonal and has
125:         same shape and type as `m`.
126: 
127:     Examples
128:     --------
129:     >>> from scipy.linalg import triu
130:     >>> triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
131:     array([[ 1,  2,  3],
132:            [ 4,  5,  6],
133:            [ 0,  8,  9],
134:            [ 0,  0, 12]])
135: 
136:     '''
137:     m = np.asarray(m)
138:     out = (1 - tri(m.shape[0], m.shape[1], k - 1, m.dtype.char)) * m
139:     return out
140: 
141: 
142: def toeplitz(c, r=None):
143:     '''
144:     Construct a Toeplitz matrix.
145: 
146:     The Toeplitz matrix has constant diagonals, with c as its first column
147:     and r as its first row.  If r is not given, ``r == conjugate(c)`` is
148:     assumed.
149: 
150:     Parameters
151:     ----------
152:     c : array_like
153:         First column of the matrix.  Whatever the actual shape of `c`, it
154:         will be converted to a 1-D array.
155:     r : array_like, optional
156:         First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
157:         in this case, if c[0] is real, the result is a Hermitian matrix.
158:         r[0] is ignored; the first row of the returned matrix is
159:         ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
160:         converted to a 1-D array.
161: 
162:     Returns
163:     -------
164:     A : (len(c), len(r)) ndarray
165:         The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.
166: 
167:     See Also
168:     --------
169:     circulant : circulant matrix
170:     hankel : Hankel matrix
171:     solve_toeplitz : Solve a Toeplitz system.
172: 
173:     Notes
174:     -----
175:     The behavior when `c` or `r` is a scalar, or when `c` is complex and
176:     `r` is None, was changed in version 0.8.0.  The behavior in previous
177:     versions was undocumented and is no longer supported.
178: 
179:     Examples
180:     --------
181:     >>> from scipy.linalg import toeplitz
182:     >>> toeplitz([1,2,3], [1,4,5,6])
183:     array([[1, 4, 5, 6],
184:            [2, 1, 4, 5],
185:            [3, 2, 1, 4]])
186:     >>> toeplitz([1.0, 2+3j, 4-1j])
187:     array([[ 1.+0.j,  2.-3.j,  4.+1.j],
188:            [ 2.+3.j,  1.+0.j,  2.-3.j],
189:            [ 4.-1.j,  2.+3.j,  1.+0.j]])
190: 
191:     '''
192:     c = np.asarray(c).ravel()
193:     if r is None:
194:         r = c.conjugate()
195:     else:
196:         r = np.asarray(r).ravel()
197:     # Form a 1D array of values to be used in the matrix, containing a reversed
198:     # copy of r[1:], followed by c.
199:     vals = np.concatenate((r[-1:0:-1], c))
200:     a, b = np.ogrid[0:len(c), len(r) - 1:-1:-1]
201:     indx = a + b
202:     # `indx` is a 2D array of indices into the 1D array `vals`, arranged so
203:     # that `vals[indx]` is the Toeplitz matrix.
204:     return vals[indx]
205: 
206: 
207: def circulant(c):
208:     '''
209:     Construct a circulant matrix.
210: 
211:     Parameters
212:     ----------
213:     c : (N,) array_like
214:         1-D array, the first column of the matrix.
215: 
216:     Returns
217:     -------
218:     A : (N, N) ndarray
219:         A circulant matrix whose first column is `c`.
220: 
221:     See Also
222:     --------
223:     toeplitz : Toeplitz matrix
224:     hankel : Hankel matrix
225:     solve_circulant : Solve a circulant system.
226: 
227:     Notes
228:     -----
229:     .. versionadded:: 0.8.0
230: 
231:     Examples
232:     --------
233:     >>> from scipy.linalg import circulant
234:     >>> circulant([1, 2, 3])
235:     array([[1, 3, 2],
236:            [2, 1, 3],
237:            [3, 2, 1]])
238: 
239:     '''
240:     c = np.asarray(c).ravel()
241:     a, b = np.ogrid[0:len(c), 0:-len(c):-1]
242:     indx = a + b
243:     # `indx` is a 2D array of indices into `c`, arranged so that `c[indx]` is
244:     # the circulant matrix.
245:     return c[indx]
246: 
247: 
248: def hankel(c, r=None):
249:     '''
250:     Construct a Hankel matrix.
251: 
252:     The Hankel matrix has constant anti-diagonals, with `c` as its
253:     first column and `r` as its last row.  If `r` is not given, then
254:     `r = zeros_like(c)` is assumed.
255: 
256:     Parameters
257:     ----------
258:     c : array_like
259:         First column of the matrix.  Whatever the actual shape of `c`, it
260:         will be converted to a 1-D array.
261:     r : array_like, optional
262:         Last row of the matrix. If None, ``r = zeros_like(c)`` is assumed.
263:         r[0] is ignored; the last row of the returned matrix is
264:         ``[c[-1], r[1:]]``.  Whatever the actual shape of `r`, it will be
265:         converted to a 1-D array.
266: 
267:     Returns
268:     -------
269:     A : (len(c), len(r)) ndarray
270:         The Hankel matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.
271: 
272:     See Also
273:     --------
274:     toeplitz : Toeplitz matrix
275:     circulant : circulant matrix
276: 
277:     Examples
278:     --------
279:     >>> from scipy.linalg import hankel
280:     >>> hankel([1, 17, 99])
281:     array([[ 1, 17, 99],
282:            [17, 99,  0],
283:            [99,  0,  0]])
284:     >>> hankel([1,2,3,4], [4,7,7,8,9])
285:     array([[1, 2, 3, 4, 7],
286:            [2, 3, 4, 7, 7],
287:            [3, 4, 7, 7, 8],
288:            [4, 7, 7, 8, 9]])
289: 
290:     '''
291:     c = np.asarray(c).ravel()
292:     if r is None:
293:         r = np.zeros_like(c)
294:     else:
295:         r = np.asarray(r).ravel()
296:     # Form a 1D array of values to be used in the matrix, containing `c`
297:     # followed by r[1:].
298:     vals = np.concatenate((c, r[1:]))
299:     a, b = np.ogrid[0:len(c), 0:len(r)]
300:     indx = a + b
301:     # `indx` is a 2D array of indices into the 1D array `vals`, arranged so
302:     # that `vals[indx]` is the Hankel matrix.
303:     return vals[indx]
304: 
305: 
306: def hadamard(n, dtype=int):
307:     '''
308:     Construct a Hadamard matrix.
309: 
310:     Constructs an n-by-n Hadamard matrix, using Sylvester's
311:     construction.  `n` must be a power of 2.
312: 
313:     Parameters
314:     ----------
315:     n : int
316:         The order of the matrix.  `n` must be a power of 2.
317:     dtype : dtype, optional
318:         The data type of the array to be constructed.
319: 
320:     Returns
321:     -------
322:     H : (n, n) ndarray
323:         The Hadamard matrix.
324: 
325:     Notes
326:     -----
327:     .. versionadded:: 0.8.0
328: 
329:     Examples
330:     --------
331:     >>> from scipy.linalg import hadamard
332:     >>> hadamard(2, dtype=complex)
333:     array([[ 1.+0.j,  1.+0.j],
334:            [ 1.+0.j, -1.-0.j]])
335:     >>> hadamard(4)
336:     array([[ 1,  1,  1,  1],
337:            [ 1, -1,  1, -1],
338:            [ 1,  1, -1, -1],
339:            [ 1, -1, -1,  1]])
340: 
341:     '''
342: 
343:     # This function is a slightly modified version of the
344:     # function contributed by Ivo in ticket #675.
345: 
346:     if n < 1:
347:         lg2 = 0
348:     else:
349:         lg2 = int(math.log(n, 2))
350:     if 2 ** lg2 != n:
351:         raise ValueError("n must be an positive integer, and n must be "
352:                          "a power of 2")
353: 
354:     H = np.array([[1]], dtype=dtype)
355: 
356:     # Sylvester's construction
357:     for i in range(0, lg2):
358:         H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))
359: 
360:     return H
361: 
362: 
363: def leslie(f, s):
364:     '''
365:     Create a Leslie matrix.
366: 
367:     Given the length n array of fecundity coefficients `f` and the length
368:     n-1 array of survival coefficents `s`, return the associated Leslie matrix.
369: 
370:     Parameters
371:     ----------
372:     f : (N,) array_like
373:         The "fecundity" coefficients.
374:     s : (N-1,) array_like
375:         The "survival" coefficients, has to be 1-D.  The length of `s`
376:         must be one less than the length of `f`, and it must be at least 1.
377: 
378:     Returns
379:     -------
380:     L : (N, N) ndarray
381:         The array is zero except for the first row,
382:         which is `f`, and the first sub-diagonal, which is `s`.
383:         The data-type of the array will be the data-type of ``f[0]+s[0]``.
384: 
385:     Notes
386:     -----
387:     .. versionadded:: 0.8.0
388: 
389:     The Leslie matrix is used to model discrete-time, age-structured
390:     population growth [1]_ [2]_. In a population with `n` age classes, two sets
391:     of parameters define a Leslie matrix: the `n` "fecundity coefficients",
392:     which give the number of offspring per-capita produced by each age
393:     class, and the `n` - 1 "survival coefficients", which give the
394:     per-capita survival rate of each age class.
395: 
396:     References
397:     ----------
398:     .. [1] P. H. Leslie, On the use of matrices in certain population
399:            mathematics, Biometrika, Vol. 33, No. 3, 183--212 (Nov. 1945)
400:     .. [2] P. H. Leslie, Some further notes on the use of matrices in
401:            population mathematics, Biometrika, Vol. 35, No. 3/4, 213--245
402:            (Dec. 1948)
403: 
404:     Examples
405:     --------
406:     >>> from scipy.linalg import leslie
407:     >>> leslie([0.1, 2.0, 1.0, 0.1], [0.2, 0.8, 0.7])
408:     array([[ 0.1,  2. ,  1. ,  0.1],
409:            [ 0.2,  0. ,  0. ,  0. ],
410:            [ 0. ,  0.8,  0. ,  0. ],
411:            [ 0. ,  0. ,  0.7,  0. ]])
412: 
413:     '''
414:     f = np.atleast_1d(f)
415:     s = np.atleast_1d(s)
416:     if f.ndim != 1:
417:         raise ValueError("Incorrect shape for f.  f must be one-dimensional")
418:     if s.ndim != 1:
419:         raise ValueError("Incorrect shape for s.  s must be one-dimensional")
420:     if f.size != s.size + 1:
421:         raise ValueError("Incorrect lengths for f and s.  The length"
422:                          " of s must be one less than the length of f.")
423:     if s.size == 0:
424:         raise ValueError("The length of s must be at least 1.")
425: 
426:     tmp = f[0] + s[0]
427:     n = f.size
428:     a = np.zeros((n, n), dtype=tmp.dtype)
429:     a[0] = f
430:     a[list(range(1, n)), list(range(0, n - 1))] = s
431:     return a
432: 
433: 
434: def kron(a, b):
435:     '''
436:     Kronecker product.
437: 
438:     The result is the block matrix::
439: 
440:         a[0,0]*b    a[0,1]*b  ... a[0,-1]*b
441:         a[1,0]*b    a[1,1]*b  ... a[1,-1]*b
442:         ...
443:         a[-1,0]*b   a[-1,1]*b ... a[-1,-1]*b
444: 
445:     Parameters
446:     ----------
447:     a : (M, N) ndarray
448:         Input array
449:     b : (P, Q) ndarray
450:         Input array
451: 
452:     Returns
453:     -------
454:     A : (M*P, N*Q) ndarray
455:         Kronecker product of `a` and `b`.
456: 
457:     Examples
458:     --------
459:     >>> from numpy import array
460:     >>> from scipy.linalg import kron
461:     >>> kron(array([[1,2],[3,4]]), array([[1,1,1]]))
462:     array([[1, 1, 1, 2, 2, 2],
463:            [3, 3, 3, 4, 4, 4]])
464: 
465:     '''
466:     if not a.flags['CONTIGUOUS']:
467:         a = np.reshape(a, a.shape)
468:     if not b.flags['CONTIGUOUS']:
469:         b = np.reshape(b, b.shape)
470:     o = np.outer(a, b)
471:     o = o.reshape(a.shape + b.shape)
472:     return np.concatenate(np.concatenate(o, axis=1), axis=1)
473: 
474: 
475: def block_diag(*arrs):
476:     '''
477:     Create a block diagonal matrix from provided arrays.
478: 
479:     Given the inputs `A`, `B` and `C`, the output will have these
480:     arrays arranged on the diagonal::
481: 
482:         [[A, 0, 0],
483:          [0, B, 0],
484:          [0, 0, C]]
485: 
486:     Parameters
487:     ----------
488:     A, B, C, ... : array_like, up to 2-D
489:         Input arrays.  A 1-D array or array_like sequence of length `n` is
490:         treated as a 2-D array with shape ``(1,n)``.
491: 
492:     Returns
493:     -------
494:     D : ndarray
495:         Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
496:         same dtype as `A`.
497: 
498:     Notes
499:     -----
500:     If all the input arrays are square, the output is known as a
501:     block diagonal matrix.
502: 
503:     Empty sequences (i.e., array-likes of zero size) will not be ignored.
504:     Noteworthy, both [] and [[]] are treated as matrices with shape ``(1,0)``.
505: 
506:     Examples
507:     --------
508:     >>> from scipy.linalg import block_diag
509:     >>> A = [[1, 0],
510:     ...      [0, 1]]
511:     >>> B = [[3, 4, 5],
512:     ...      [6, 7, 8]]
513:     >>> C = [[7]]
514:     >>> P = np.zeros((2, 0), dtype='int32')
515:     >>> block_diag(A, B, C)
516:     array([[1, 0, 0, 0, 0, 0],
517:            [0, 1, 0, 0, 0, 0],
518:            [0, 0, 3, 4, 5, 0],
519:            [0, 0, 6, 7, 8, 0],
520:            [0, 0, 0, 0, 0, 7]])
521:     >>> block_diag(A, P, B, C)
522:     array([[1, 0, 0, 0, 0, 0],
523:            [0, 1, 0, 0, 0, 0],
524:            [0, 0, 0, 0, 0, 0],
525:            [0, 0, 0, 0, 0, 0],
526:            [0, 0, 3, 4, 5, 0],
527:            [0, 0, 6, 7, 8, 0],
528:            [0, 0, 0, 0, 0, 7]])
529:     >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
530:     array([[ 1.,  0.,  0.,  0.,  0.],
531:            [ 0.,  2.,  3.,  0.,  0.],
532:            [ 0.,  0.,  0.,  4.,  5.],
533:            [ 0.,  0.,  0.,  6.,  7.]])
534: 
535:     '''
536:     if arrs == ():
537:         arrs = ([],)
538:     arrs = [np.atleast_2d(a) for a in arrs]
539: 
540:     bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
541:     if bad_args:
542:         raise ValueError("arguments in the following positions have dimension "
543:                          "greater than 2: %s" % bad_args)
544: 
545:     shapes = np.array([a.shape for a in arrs])
546:     out_dtype = np.find_common_type([arr.dtype for arr in arrs], [])
547:     out = np.zeros(np.sum(shapes, axis=0), dtype=out_dtype)
548: 
549:     r, c = 0, 0
550:     for i, (rr, cc) in enumerate(shapes):
551:         out[r:r + rr, c:c + cc] = arrs[i]
552:         r += rr
553:         c += cc
554:     return out
555: 
556: 
557: def companion(a):
558:     '''
559:     Create a companion matrix.
560: 
561:     Create the companion matrix [1]_ associated with the polynomial whose
562:     coefficients are given in `a`.
563: 
564:     Parameters
565:     ----------
566:     a : (N,) array_like
567:         1-D array of polynomial coefficients.  The length of `a` must be
568:         at least two, and ``a[0]`` must not be zero.
569: 
570:     Returns
571:     -------
572:     c : (N-1, N-1) ndarray
573:         The first row of `c` is ``-a[1:]/a[0]``, and the first
574:         sub-diagonal is all ones.  The data-type of the array is the same
575:         as the data-type of ``1.0*a[0]``.
576: 
577:     Raises
578:     ------
579:     ValueError
580:         If any of the following are true: a) ``a.ndim != 1``;
581:         b) ``a.size < 2``; c) ``a[0] == 0``.
582: 
583:     Notes
584:     -----
585:     .. versionadded:: 0.8.0
586: 
587:     References
588:     ----------
589:     .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
590:         Cambridge University Press, 1999, pp. 146-7.
591: 
592:     Examples
593:     --------
594:     >>> from scipy.linalg import companion
595:     >>> companion([1, -10, 31, -30])
596:     array([[ 10., -31.,  30.],
597:            [  1.,   0.,   0.],
598:            [  0.,   1.,   0.]])
599: 
600:     '''
601:     a = np.atleast_1d(a)
602: 
603:     if a.ndim != 1:
604:         raise ValueError("Incorrect shape for `a`.  `a` must be "
605:                          "one-dimensional.")
606: 
607:     if a.size < 2:
608:         raise ValueError("The length of `a` must be at least 2.")
609: 
610:     if a[0] == 0:
611:         raise ValueError("The first coefficient in `a` must not be zero.")
612: 
613:     first_row = -a[1:] / (1.0 * a[0])
614:     n = a.size
615:     c = np.zeros((n - 1, n - 1), dtype=first_row.dtype)
616:     c[0] = first_row
617:     c[list(range(1, n - 1)), list(range(0, n - 2))] = 1
618:     return c
619: 
620: 
621: def helmert(n, full=False):
622:     '''
623:     Create a Helmert matrix of order `n`.
624: 
625:     This has applications in statistics, compositional or simplicial analysis,
626:     and in Aitchison geometry.
627: 
628:     Parameters
629:     ----------
630:     n : int
631:         The size of the array to create.
632:     full : bool, optional
633:         If True the (n, n) ndarray will be returned.
634:         Otherwise the submatrix that does not include the first
635:         row will be returned.
636:         Default: False.
637: 
638:     Returns
639:     -------
640:     M : ndarray
641:         The Helmert matrix.
642:         The shape is (n, n) or (n-1, n) depending on the `full` argument.
643: 
644:     Examples
645:     --------
646:     >>> from scipy.linalg import helmert
647:     >>> helmert(5, full=True)
648:     array([[ 0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ],
649:            [ 0.70710678, -0.70710678,  0.        ,  0.        ,  0.        ],
650:            [ 0.40824829,  0.40824829, -0.81649658,  0.        ,  0.        ],
651:            [ 0.28867513,  0.28867513,  0.28867513, -0.8660254 ,  0.        ],
652:            [ 0.2236068 ,  0.2236068 ,  0.2236068 ,  0.2236068 , -0.89442719]])
653: 
654:     '''
655:     H = np.tril(np.ones((n, n)), -1) - np.diag(np.arange(n))
656:     d = np.arange(n) * np.arange(1, n+1)
657:     H[0] = 1
658:     d[0] = n
659:     H_full = H / np.sqrt(d)[:, np.newaxis]
660:     if full:
661:         return H_full
662:     else:
663:         return H_full[1:]
664: 
665: 
666: def hilbert(n):
667:     '''
668:     Create a Hilbert matrix of order `n`.
669: 
670:     Returns the `n` by `n` array with entries `h[i,j] = 1 / (i + j + 1)`.
671: 
672:     Parameters
673:     ----------
674:     n : int
675:         The size of the array to create.
676: 
677:     Returns
678:     -------
679:     h : (n, n) ndarray
680:         The Hilbert matrix.
681: 
682:     See Also
683:     --------
684:     invhilbert : Compute the inverse of a Hilbert matrix.
685: 
686:     Notes
687:     -----
688:     .. versionadded:: 0.10.0
689: 
690:     Examples
691:     --------
692:     >>> from scipy.linalg import hilbert
693:     >>> hilbert(3)
694:     array([[ 1.        ,  0.5       ,  0.33333333],
695:            [ 0.5       ,  0.33333333,  0.25      ],
696:            [ 0.33333333,  0.25      ,  0.2       ]])
697: 
698:     '''
699:     values = 1.0 / (1.0 + np.arange(2 * n - 1))
700:     h = hankel(values[:n], r=values[n - 1:])
701:     return h
702: 
703: 
704: def invhilbert(n, exact=False):
705:     '''
706:     Compute the inverse of the Hilbert matrix of order `n`.
707: 
708:     The entries in the inverse of a Hilbert matrix are integers.  When `n`
709:     is greater than 14, some entries in the inverse exceed the upper limit
710:     of 64 bit integers.  The `exact` argument provides two options for
711:     dealing with these large integers.
712: 
713:     Parameters
714:     ----------
715:     n : int
716:         The order of the Hilbert matrix.
717:     exact : bool, optional
718:         If False, the data type of the array that is returned is np.float64,
719:         and the array is an approximation of the inverse.
720:         If True, the array is the exact integer inverse array.  To represent
721:         the exact inverse when n > 14, the returned array is an object array
722:         of long integers.  For n <= 14, the exact inverse is returned as an
723:         array with data type np.int64.
724: 
725:     Returns
726:     -------
727:     invh : (n, n) ndarray
728:         The data type of the array is np.float64 if `exact` is False.
729:         If `exact` is True, the data type is either np.int64 (for n <= 14)
730:         or object (for n > 14).  In the latter case, the objects in the
731:         array will be long integers.
732: 
733:     See Also
734:     --------
735:     hilbert : Create a Hilbert matrix.
736: 
737:     Notes
738:     -----
739:     .. versionadded:: 0.10.0
740: 
741:     Examples
742:     --------
743:     >>> from scipy.linalg import invhilbert
744:     >>> invhilbert(4)
745:     array([[   16.,  -120.,   240.,  -140.],
746:            [ -120.,  1200., -2700.,  1680.],
747:            [  240., -2700.,  6480., -4200.],
748:            [ -140.,  1680., -4200.,  2800.]])
749:     >>> invhilbert(4, exact=True)
750:     array([[   16,  -120,   240,  -140],
751:            [ -120,  1200, -2700,  1680],
752:            [  240, -2700,  6480, -4200],
753:            [ -140,  1680, -4200,  2800]], dtype=int64)
754:     >>> invhilbert(16)[7,7]
755:     4.2475099528537506e+19
756:     >>> invhilbert(16, exact=True)[7,7]
757:     42475099528537378560L
758: 
759:     '''
760:     from scipy.special import comb
761:     if exact:
762:         if n > 14:
763:             dtype = object
764:         else:
765:             dtype = np.int64
766:     else:
767:         dtype = np.float64
768:     invh = np.empty((n, n), dtype=dtype)
769:     for i in xrange(n):
770:         for j in xrange(0, i + 1):
771:             s = i + j
772:             invh[i, j] = ((-1) ** s * (s + 1) *
773:                           comb(n + i, n - j - 1, exact) *
774:                           comb(n + j, n - i - 1, exact) *
775:                           comb(s, i, exact) ** 2)
776:             if i != j:
777:                 invh[j, i] = invh[i, j]
778:     return invh
779: 
780: 
781: def pascal(n, kind='symmetric', exact=True):
782:     '''
783:     Returns the n x n Pascal matrix.
784: 
785:     The Pascal matrix is a matrix containing the binomial coefficients as
786:     its elements.
787: 
788:     Parameters
789:     ----------
790:     n : int
791:         The size of the matrix to create; that is, the result is an n x n
792:         matrix.
793:     kind : str, optional
794:         Must be one of 'symmetric', 'lower', or 'upper'.
795:         Default is 'symmetric'.
796:     exact : bool, optional
797:         If `exact` is True, the result is either an array of type
798:         numpy.uint64 (if n < 35) or an object array of Python long integers.
799:         If `exact` is False, the coefficients in the matrix are computed using
800:         `scipy.special.comb` with `exact=False`.  The result will be a floating
801:         point array, and the values in the array will not be the exact
802:         coefficients, but this version is much faster than `exact=True`.
803: 
804:     Returns
805:     -------
806:     p : (n, n) ndarray
807:         The Pascal matrix.
808: 
809:     See Also
810:     --------
811:     invpascal
812: 
813:     Notes
814:     -----
815:     See http://en.wikipedia.org/wiki/Pascal_matrix for more information
816:     about Pascal matrices.
817: 
818:     .. versionadded:: 0.11.0
819: 
820:     Examples
821:     --------
822:     >>> from scipy.linalg import pascal
823:     >>> pascal(4)
824:     array([[ 1,  1,  1,  1],
825:            [ 1,  2,  3,  4],
826:            [ 1,  3,  6, 10],
827:            [ 1,  4, 10, 20]], dtype=uint64)
828:     >>> pascal(4, kind='lower')
829:     array([[1, 0, 0, 0],
830:            [1, 1, 0, 0],
831:            [1, 2, 1, 0],
832:            [1, 3, 3, 1]], dtype=uint64)
833:     >>> pascal(50)[-1, -1]
834:     25477612258980856902730428600L
835:     >>> from scipy.special import comb
836:     >>> comb(98, 49, exact=True)
837:     25477612258980856902730428600L
838: 
839:     '''
840: 
841:     from scipy.special import comb
842:     if kind not in ['symmetric', 'lower', 'upper']:
843:         raise ValueError("kind must be 'symmetric', 'lower', or 'upper'")
844: 
845:     if exact:
846:         if n >= 35:
847:             L_n = np.empty((n, n), dtype=object)
848:             L_n.fill(0)
849:         else:
850:             L_n = np.zeros((n, n), dtype=np.uint64)
851:         for i in range(n):
852:             for j in range(i + 1):
853:                 L_n[i, j] = comb(i, j, exact=True)
854:     else:
855:         L_n = comb(*np.ogrid[:n, :n])
856: 
857:     if kind == 'lower':
858:         p = L_n
859:     elif kind == 'upper':
860:         p = L_n.T
861:     else:
862:         p = np.dot(L_n, L_n.T)
863: 
864:     return p
865: 
866: 
867: def invpascal(n, kind='symmetric', exact=True):
868:     '''
869:     Returns the inverse of the n x n Pascal matrix.
870: 
871:     The Pascal matrix is a matrix containing the binomial coefficients as
872:     its elements.
873: 
874:     Parameters
875:     ----------
876:     n : int
877:         The size of the matrix to create; that is, the result is an n x n
878:         matrix.
879:     kind : str, optional
880:         Must be one of 'symmetric', 'lower', or 'upper'.
881:         Default is 'symmetric'.
882:     exact : bool, optional
883:         If `exact` is True, the result is either an array of type
884:         `numpy.int64` (if `n` <= 35) or an object array of Python integers.
885:         If `exact` is False, the coefficients in the matrix are computed using
886:         `scipy.special.comb` with `exact=False`.  The result will be a floating
887:         point array, and for large `n`, the values in the array will not be the
888:         exact coefficients.
889: 
890:     Returns
891:     -------
892:     invp : (n, n) ndarray
893:         The inverse of the Pascal matrix.
894: 
895:     See Also
896:     --------
897:     pascal
898: 
899:     Notes
900:     -----
901: 
902:     .. versionadded:: 0.16.0
903: 
904:     References
905:     ----------
906:     .. [1] "Pascal matrix",  http://en.wikipedia.org/wiki/Pascal_matrix
907:     .. [2] Cohen, A. M., "The inverse of a Pascal matrix", Mathematical
908:            Gazette, 59(408), pp. 111-112, 1975.
909: 
910:     Examples
911:     --------
912:     >>> from scipy.linalg import invpascal, pascal
913:     >>> invp = invpascal(5)
914:     >>> invp
915:     array([[  5, -10,  10,  -5,   1],
916:            [-10,  30, -35,  19,  -4],
917:            [ 10, -35,  46, -27,   6],
918:            [ -5,  19, -27,  17,  -4],
919:            [  1,  -4,   6,  -4,   1]])
920: 
921:     >>> p = pascal(5)
922:     >>> p.dot(invp)
923:     array([[ 1.,  0.,  0.,  0.,  0.],
924:            [ 0.,  1.,  0.,  0.,  0.],
925:            [ 0.,  0.,  1.,  0.,  0.],
926:            [ 0.,  0.,  0.,  1.,  0.],
927:            [ 0.,  0.,  0.,  0.,  1.]])
928: 
929:     An example of the use of `kind` and `exact`:
930: 
931:     >>> invpascal(5, kind='lower', exact=False)
932:     array([[ 1., -0.,  0., -0.,  0.],
933:            [-1.,  1., -0.,  0., -0.],
934:            [ 1., -2.,  1., -0.,  0.],
935:            [-1.,  3., -3.,  1., -0.],
936:            [ 1., -4.,  6., -4.,  1.]])
937: 
938:     '''
939:     from scipy.special import comb
940: 
941:     if kind not in ['symmetric', 'lower', 'upper']:
942:         raise ValueError("'kind' must be 'symmetric', 'lower' or 'upper'.")
943: 
944:     if kind == 'symmetric':
945:         if exact:
946:             if n > 34:
947:                 dt = object
948:             else:
949:                 dt = np.int64
950:         else:
951:             dt = np.float64
952:         invp = np.empty((n, n), dtype=dt)
953:         for i in range(n):
954:             for j in range(0, i + 1):
955:                 v = 0
956:                 for k in range(n - i):
957:                     v += comb(i + k, k, exact=exact) * comb(i + k, i + k - j,
958:                                                             exact=exact)
959:                 invp[i, j] = (-1)**(i - j) * v
960:                 if i != j:
961:                     invp[j, i] = invp[i, j]
962:     else:
963:         # For the 'lower' and 'upper' cases, we computer the inverse by
964:         # changing the sign of every other diagonal of the pascal matrix.
965:         invp = pascal(n, kind=kind, exact=exact)
966:         if invp.dtype == np.uint64:
967:             # This cast from np.uint64 to int64 OK, because if `kind` is not
968:             # "symmetric", the values in invp are all much less than 2**63.
969:             invp = invp.view(np.int64)
970: 
971:         # The toeplitz matrix has alternating bands of 1 and -1.
972:         invp *= toeplitz((-1)**np.arange(n)).astype(invp.dtype)
973: 
974:     return invp
975: 
976: 
977: def dft(n, scale=None):
978:     '''
979:     Discrete Fourier transform matrix.
980: 
981:     Create the matrix that computes the discrete Fourier transform of a
982:     sequence [1]_.  The n-th primitive root of unity used to generate the
983:     matrix is exp(-2*pi*i/n), where i = sqrt(-1).
984: 
985:     Parameters
986:     ----------
987:     n : int
988:         Size the matrix to create.
989:     scale : str, optional
990:         Must be None, 'sqrtn', or 'n'.
991:         If `scale` is 'sqrtn', the matrix is divided by `sqrt(n)`.
992:         If `scale` is 'n', the matrix is divided by `n`.
993:         If `scale` is None (the default), the matrix is not normalized, and the
994:         return value is simply the Vandermonde matrix of the roots of unity.
995: 
996:     Returns
997:     -------
998:     m : (n, n) ndarray
999:         The DFT matrix.
1000: 
1001:     Notes
1002:     -----
1003:     When `scale` is None, multiplying a vector by the matrix returned by
1004:     `dft` is mathematically equivalent to (but much less efficient than)
1005:     the calculation performed by `scipy.fftpack.fft`.
1006: 
1007:     .. versionadded:: 0.14.0
1008: 
1009:     References
1010:     ----------
1011:     .. [1] "DFT matrix", http://en.wikipedia.org/wiki/DFT_matrix
1012: 
1013:     Examples
1014:     --------
1015:     >>> from scipy.linalg import dft
1016:     >>> np.set_printoptions(precision=5, suppress=True)
1017:     >>> x = np.array([1, 2, 3, 0, 3, 2, 1, 0])
1018:     >>> m = dft(8)
1019:     >>> m.dot(x)   # Compute the DFT of x
1020:     array([ 12.+0.j,  -2.-2.j,   0.-4.j,  -2.+2.j,   4.+0.j,  -2.-2.j,
1021:             -0.+4.j,  -2.+2.j])
1022: 
1023:     Verify that ``m.dot(x)`` is the same as ``fft(x)``.
1024: 
1025:     >>> from scipy.fftpack import fft
1026:     >>> fft(x)     # Same result as m.dot(x)
1027:     array([ 12.+0.j,  -2.-2.j,   0.-4.j,  -2.+2.j,   4.+0.j,  -2.-2.j,
1028:              0.+4.j,  -2.+2.j])
1029:     '''
1030:     if scale not in [None, 'sqrtn', 'n']:
1031:         raise ValueError("scale must be None, 'sqrtn', or 'n'; "
1032:                          "%r is not valid." % (scale,))
1033: 
1034:     omegas = np.exp(-2j * np.pi * np.arange(n) / n).reshape(-1, 1)
1035:     m = omegas ** np.arange(n)
1036:     if scale == 'sqrtn':
1037:         m /= math.sqrt(n)
1038:     elif scale == 'n':
1039:         m /= n
1040:     return m
1041: 

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

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_24038 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_24038) is not StypyTypeError):

    if (import_24038 != 'pyd_module'):
        __import__(import_24038)
        sys_modules_24039 = sys.modules[import_24038]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_24039.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_24038)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy._lib.six import xrange' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_24040 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib.six')

if (type(import_24040) is not StypyTypeError):

    if (import_24040 != 'pyd_module'):
        __import__(import_24040)
        sys_modules_24041 = sys.modules[import_24040]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib.six', sys_modules_24041.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_24041, sys_modules_24041.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib.six', import_24040)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy._lib.six import string_types' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_24042 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib.six')

if (type(import_24042) is not StypyTypeError):

    if (import_24042 != 'pyd_module'):
        __import__(import_24042)
        sys_modules_24043 = sys.modules[import_24042]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib.six', sys_modules_24043.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_24043, sys_modules_24043.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib.six', import_24042)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 9):

# Assigning a List to a Name (line 9):
__all__ = ['tri', 'tril', 'triu', 'toeplitz', 'circulant', 'hankel', 'hadamard', 'leslie', 'kron', 'block_diag', 'companion', 'helmert', 'hilbert', 'invhilbert', 'pascal', 'invpascal', 'dft']
module_type_store.set_exportable_members(['tri', 'tril', 'triu', 'toeplitz', 'circulant', 'hankel', 'hadamard', 'leslie', 'kron', 'block_diag', 'companion', 'helmert', 'hilbert', 'invhilbert', 'pascal', 'invpascal', 'dft'])

# Obtaining an instance of the builtin type 'list' (line 9)
list_24044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
str_24045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'tri')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24045)
# Adding element type (line 9)
str_24046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'str', 'tril')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24046)
# Adding element type (line 9)
str_24047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 26), 'str', 'triu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24047)
# Adding element type (line 9)
str_24048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 34), 'str', 'toeplitz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24048)
# Adding element type (line 9)
str_24049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 46), 'str', 'circulant')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24049)
# Adding element type (line 9)
str_24050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 59), 'str', 'hankel')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24050)
# Adding element type (line 9)
str_24051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'hadamard')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24051)
# Adding element type (line 9)
str_24052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 23), 'str', 'leslie')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24052)
# Adding element type (line 9)
str_24053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 33), 'str', 'kron')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24053)
# Adding element type (line 9)
str_24054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 41), 'str', 'block_diag')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24054)
# Adding element type (line 9)
str_24055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 55), 'str', 'companion')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24055)
# Adding element type (line 9)
str_24056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'str', 'helmert')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24056)
# Adding element type (line 9)
str_24057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'str', 'hilbert')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24057)
# Adding element type (line 9)
str_24058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 33), 'str', 'invhilbert')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24058)
# Adding element type (line 9)
str_24059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 47), 'str', 'pascal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24059)
# Adding element type (line 9)
str_24060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 57), 'str', 'invpascal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24060)
# Adding element type (line 9)
str_24061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 70), 'str', 'dft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_24044, str_24061)

# Assigning a type to the variable '__all__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__all__', list_24044)

@norecursion
def tri(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 23)
    None_24062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'None')
    int_24063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'int')
    # Getting the type of 'None' (line 23)
    None_24064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 30), 'None')
    defaults = [None_24062, int_24063, None_24064]
    # Create a new context for function 'tri'
    module_type_store = module_type_store.open_function_context('tri', 23, 0, False)
    
    # Passed parameters checking function
    tri.stypy_localization = localization
    tri.stypy_type_of_self = None
    tri.stypy_type_store = module_type_store
    tri.stypy_function_name = 'tri'
    tri.stypy_param_names_list = ['N', 'M', 'k', 'dtype']
    tri.stypy_varargs_param_name = None
    tri.stypy_kwargs_param_name = None
    tri.stypy_call_defaults = defaults
    tri.stypy_call_varargs = varargs
    tri.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tri', ['N', 'M', 'k', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tri', localization, ['N', 'M', 'k', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tri(...)' code ##################

    str_24065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, (-1)), 'str', '\n    Construct (N, M) matrix filled with ones at and below the k-th diagonal.\n\n    The matrix has A[i,j] == 1 for i <= j + k\n\n    Parameters\n    ----------\n    N : int\n        The size of the first dimension of the matrix.\n    M : int or None, optional\n        The size of the second dimension of the matrix. If `M` is None,\n        `M = N` is assumed.\n    k : int, optional\n        Number of subdiagonal below which matrix is filled with ones.\n        `k` = 0 is the main diagonal, `k` < 0 subdiagonal and `k` > 0\n        superdiagonal.\n    dtype : dtype, optional\n        Data type of the matrix.\n\n    Returns\n    -------\n    tri : (N, M) ndarray\n        Tri matrix.\n\n    Examples\n    --------\n    >>> from scipy.linalg import tri\n    >>> tri(3, 5, 2, dtype=int)\n    array([[1, 1, 1, 0, 0],\n           [1, 1, 1, 1, 0],\n           [1, 1, 1, 1, 1]])\n    >>> tri(3, 5, -1, dtype=int)\n    array([[0, 0, 0, 0, 0],\n           [1, 0, 0, 0, 0],\n           [1, 1, 0, 0, 0]])\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 61)
    # Getting the type of 'M' (line 61)
    M_24066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 7), 'M')
    # Getting the type of 'None' (line 61)
    None_24067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'None')
    
    (may_be_24068, more_types_in_union_24069) = may_be_none(M_24066, None_24067)

    if may_be_24068:

        if more_types_in_union_24069:
            # Runtime conditional SSA (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 62):
        
        # Assigning a Name to a Name (line 62):
        # Getting the type of 'N' (line 62)
        N_24070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'N')
        # Assigning a type to the variable 'M' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'M', N_24070)

        if more_types_in_union_24069:
            # SSA join for if statement (line 61)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to isinstance(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'M' (line 63)
    M_24072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'M', False)
    # Getting the type of 'string_types' (line 63)
    string_types_24073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 21), 'string_types', False)
    # Processing the call keyword arguments (line 63)
    kwargs_24074 = {}
    # Getting the type of 'isinstance' (line 63)
    isinstance_24071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 63)
    isinstance_call_result_24075 = invoke(stypy.reporting.localization.Localization(__file__, 63, 7), isinstance_24071, *[M_24072, string_types_24073], **kwargs_24074)
    
    # Testing the type of an if condition (line 63)
    if_condition_24076 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 4), isinstance_call_result_24075)
    # Assigning a type to the variable 'if_condition_24076' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'if_condition_24076', if_condition_24076)
    # SSA begins for if statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 66):
    
    # Assigning a Name to a Name (line 66):
    # Getting the type of 'M' (line 66)
    M_24077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'M')
    # Assigning a type to the variable 'dtype' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'dtype', M_24077)
    
    # Assigning a Name to a Name (line 67):
    
    # Assigning a Name to a Name (line 67):
    # Getting the type of 'N' (line 67)
    N_24078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'N')
    # Assigning a type to the variable 'M' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'M', N_24078)
    # SSA join for if statement (line 63)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 68):
    
    # Assigning a Call to a Name (line 68):
    
    # Call to greater_equal(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Call to outer(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Call to arange(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'N' (line 68)
    N_24086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 53), 'N', False)
    # Processing the call keyword arguments (line 68)
    kwargs_24087 = {}
    # Getting the type of 'np' (line 68)
    np_24084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 43), 'np', False)
    # Obtaining the member 'arange' of a type (line 68)
    arange_24085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 43), np_24084, 'arange')
    # Calling arange(args, kwargs) (line 68)
    arange_call_result_24088 = invoke(stypy.reporting.localization.Localization(__file__, 68, 43), arange_24085, *[N_24086], **kwargs_24087)
    
    
    # Call to arange(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'M' (line 68)
    M_24091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 67), 'M', False)
    # Processing the call keyword arguments (line 68)
    kwargs_24092 = {}
    # Getting the type of 'np' (line 68)
    np_24089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 57), 'np', False)
    # Obtaining the member 'arange' of a type (line 68)
    arange_24090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 57), np_24089, 'arange')
    # Calling arange(args, kwargs) (line 68)
    arange_call_result_24093 = invoke(stypy.reporting.localization.Localization(__file__, 68, 57), arange_24090, *[M_24091], **kwargs_24092)
    
    # Processing the call keyword arguments (line 68)
    kwargs_24094 = {}
    # Getting the type of 'np' (line 68)
    np_24081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'np', False)
    # Obtaining the member 'subtract' of a type (line 68)
    subtract_24082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 25), np_24081, 'subtract')
    # Obtaining the member 'outer' of a type (line 68)
    outer_24083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 25), subtract_24082, 'outer')
    # Calling outer(args, kwargs) (line 68)
    outer_call_result_24095 = invoke(stypy.reporting.localization.Localization(__file__, 68, 25), outer_24083, *[arange_call_result_24088, arange_call_result_24093], **kwargs_24094)
    
    
    # Getting the type of 'k' (line 68)
    k_24096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 73), 'k', False)
    # Applying the 'usub' unary operator (line 68)
    result___neg___24097 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 72), 'usub', k_24096)
    
    # Processing the call keyword arguments (line 68)
    kwargs_24098 = {}
    # Getting the type of 'np' (line 68)
    np_24079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'np', False)
    # Obtaining the member 'greater_equal' of a type (line 68)
    greater_equal_24080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), np_24079, 'greater_equal')
    # Calling greater_equal(args, kwargs) (line 68)
    greater_equal_call_result_24099 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), greater_equal_24080, *[outer_call_result_24095, result___neg___24097], **kwargs_24098)
    
    # Assigning a type to the variable 'm' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'm', greater_equal_call_result_24099)
    
    # Type idiom detected: calculating its left and rigth part (line 69)
    # Getting the type of 'dtype' (line 69)
    dtype_24100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 7), 'dtype')
    # Getting the type of 'None' (line 69)
    None_24101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'None')
    
    (may_be_24102, more_types_in_union_24103) = may_be_none(dtype_24100, None_24101)

    if may_be_24102:

        if more_types_in_union_24103:
            # Runtime conditional SSA (line 69)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'm' (line 70)
        m_24104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'm')
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type', m_24104)

        if more_types_in_union_24103:
            # Runtime conditional SSA for else branch (line 69)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_24102) or more_types_in_union_24103):
        
        # Call to astype(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'dtype' (line 72)
        dtype_24107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'dtype', False)
        # Processing the call keyword arguments (line 72)
        kwargs_24108 = {}
        # Getting the type of 'm' (line 72)
        m_24105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'm', False)
        # Obtaining the member 'astype' of a type (line 72)
        astype_24106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 15), m_24105, 'astype')
        # Calling astype(args, kwargs) (line 72)
        astype_call_result_24109 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), astype_24106, *[dtype_24107], **kwargs_24108)
        
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', astype_call_result_24109)

        if (may_be_24102 and more_types_in_union_24103):
            # SSA join for if statement (line 69)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'tri(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tri' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_24110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24110)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tri'
    return stypy_return_type_24110

# Assigning a type to the variable 'tri' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'tri', tri)

@norecursion
def tril(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_24111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 14), 'int')
    defaults = [int_24111]
    # Create a new context for function 'tril'
    module_type_store = module_type_store.open_function_context('tril', 75, 0, False)
    
    # Passed parameters checking function
    tril.stypy_localization = localization
    tril.stypy_type_of_self = None
    tril.stypy_type_store = module_type_store
    tril.stypy_function_name = 'tril'
    tril.stypy_param_names_list = ['m', 'k']
    tril.stypy_varargs_param_name = None
    tril.stypy_kwargs_param_name = None
    tril.stypy_call_defaults = defaults
    tril.stypy_call_varargs = varargs
    tril.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tril', ['m', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tril', localization, ['m', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tril(...)' code ##################

    str_24112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, (-1)), 'str', '\n    Make a copy of a matrix with elements above the k-th diagonal zeroed.\n\n    Parameters\n    ----------\n    m : array_like\n        Matrix whose elements to return\n    k : int, optional\n        Diagonal above which to zero elements.\n        `k` == 0 is the main diagonal, `k` < 0 subdiagonal and\n        `k` > 0 superdiagonal.\n\n    Returns\n    -------\n    tril : ndarray\n        Return is the same shape and type as `m`.\n\n    Examples\n    --------\n    >>> from scipy.linalg import tril\n    >>> tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)\n    array([[ 0,  0,  0],\n           [ 4,  0,  0],\n           [ 7,  8,  0],\n           [10, 11, 12]])\n\n    ')
    
    # Assigning a Call to a Name (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to asarray(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'm' (line 103)
    m_24115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'm', False)
    # Processing the call keyword arguments (line 103)
    kwargs_24116 = {}
    # Getting the type of 'np' (line 103)
    np_24113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 103)
    asarray_24114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), np_24113, 'asarray')
    # Calling asarray(args, kwargs) (line 103)
    asarray_call_result_24117 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), asarray_24114, *[m_24115], **kwargs_24116)
    
    # Assigning a type to the variable 'm' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'm', asarray_call_result_24117)
    
    # Assigning a BinOp to a Name (line 104):
    
    # Assigning a BinOp to a Name (line 104):
    
    # Call to tri(...): (line 104)
    # Processing the call arguments (line 104)
    
    # Obtaining the type of the subscript
    int_24119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 22), 'int')
    # Getting the type of 'm' (line 104)
    m_24120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 'm', False)
    # Obtaining the member 'shape' of a type (line 104)
    shape_24121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 14), m_24120, 'shape')
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___24122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 14), shape_24121, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_24123 = invoke(stypy.reporting.localization.Localization(__file__, 104, 14), getitem___24122, int_24119)
    
    
    # Obtaining the type of the subscript
    int_24124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 34), 'int')
    # Getting the type of 'm' (line 104)
    m_24125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 26), 'm', False)
    # Obtaining the member 'shape' of a type (line 104)
    shape_24126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 26), m_24125, 'shape')
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___24127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 26), shape_24126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_24128 = invoke(stypy.reporting.localization.Localization(__file__, 104, 26), getitem___24127, int_24124)
    
    # Processing the call keyword arguments (line 104)
    # Getting the type of 'k' (line 104)
    k_24129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 40), 'k', False)
    keyword_24130 = k_24129
    # Getting the type of 'm' (line 104)
    m_24131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 49), 'm', False)
    # Obtaining the member 'dtype' of a type (line 104)
    dtype_24132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 49), m_24131, 'dtype')
    # Obtaining the member 'char' of a type (line 104)
    char_24133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 49), dtype_24132, 'char')
    keyword_24134 = char_24133
    kwargs_24135 = {'dtype': keyword_24134, 'k': keyword_24130}
    # Getting the type of 'tri' (line 104)
    tri_24118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 10), 'tri', False)
    # Calling tri(args, kwargs) (line 104)
    tri_call_result_24136 = invoke(stypy.reporting.localization.Localization(__file__, 104, 10), tri_24118, *[subscript_call_result_24123, subscript_call_result_24128], **kwargs_24135)
    
    # Getting the type of 'm' (line 104)
    m_24137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 65), 'm')
    # Applying the binary operator '*' (line 104)
    result_mul_24138 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 10), '*', tri_call_result_24136, m_24137)
    
    # Assigning a type to the variable 'out' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'out', result_mul_24138)
    # Getting the type of 'out' (line 105)
    out_24139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', out_24139)
    
    # ################# End of 'tril(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tril' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_24140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24140)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tril'
    return stypy_return_type_24140

# Assigning a type to the variable 'tril' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'tril', tril)

@norecursion
def triu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_24141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 14), 'int')
    defaults = [int_24141]
    # Create a new context for function 'triu'
    module_type_store = module_type_store.open_function_context('triu', 108, 0, False)
    
    # Passed parameters checking function
    triu.stypy_localization = localization
    triu.stypy_type_of_self = None
    triu.stypy_type_store = module_type_store
    triu.stypy_function_name = 'triu'
    triu.stypy_param_names_list = ['m', 'k']
    triu.stypy_varargs_param_name = None
    triu.stypy_kwargs_param_name = None
    triu.stypy_call_defaults = defaults
    triu.stypy_call_varargs = varargs
    triu.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'triu', ['m', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'triu', localization, ['m', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'triu(...)' code ##################

    str_24142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, (-1)), 'str', '\n    Make a copy of a matrix with elements below the k-th diagonal zeroed.\n\n    Parameters\n    ----------\n    m : array_like\n        Matrix whose elements to return\n    k : int, optional\n        Diagonal below which to zero elements.\n        `k` == 0 is the main diagonal, `k` < 0 subdiagonal and\n        `k` > 0 superdiagonal.\n\n    Returns\n    -------\n    triu : ndarray\n        Return matrix with zeroed elements below the k-th diagonal and has\n        same shape and type as `m`.\n\n    Examples\n    --------\n    >>> from scipy.linalg import triu\n    >>> triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)\n    array([[ 1,  2,  3],\n           [ 4,  5,  6],\n           [ 0,  8,  9],\n           [ 0,  0, 12]])\n\n    ')
    
    # Assigning a Call to a Name (line 137):
    
    # Assigning a Call to a Name (line 137):
    
    # Call to asarray(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'm' (line 137)
    m_24145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 19), 'm', False)
    # Processing the call keyword arguments (line 137)
    kwargs_24146 = {}
    # Getting the type of 'np' (line 137)
    np_24143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 137)
    asarray_24144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), np_24143, 'asarray')
    # Calling asarray(args, kwargs) (line 137)
    asarray_call_result_24147 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), asarray_24144, *[m_24145], **kwargs_24146)
    
    # Assigning a type to the variable 'm' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'm', asarray_call_result_24147)
    
    # Assigning a BinOp to a Name (line 138):
    
    # Assigning a BinOp to a Name (line 138):
    int_24148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 11), 'int')
    
    # Call to tri(...): (line 138)
    # Processing the call arguments (line 138)
    
    # Obtaining the type of the subscript
    int_24150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 27), 'int')
    # Getting the type of 'm' (line 138)
    m_24151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'm', False)
    # Obtaining the member 'shape' of a type (line 138)
    shape_24152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 19), m_24151, 'shape')
    # Obtaining the member '__getitem__' of a type (line 138)
    getitem___24153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 19), shape_24152, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 138)
    subscript_call_result_24154 = invoke(stypy.reporting.localization.Localization(__file__, 138, 19), getitem___24153, int_24150)
    
    
    # Obtaining the type of the subscript
    int_24155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 39), 'int')
    # Getting the type of 'm' (line 138)
    m_24156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 31), 'm', False)
    # Obtaining the member 'shape' of a type (line 138)
    shape_24157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 31), m_24156, 'shape')
    # Obtaining the member '__getitem__' of a type (line 138)
    getitem___24158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 31), shape_24157, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 138)
    subscript_call_result_24159 = invoke(stypy.reporting.localization.Localization(__file__, 138, 31), getitem___24158, int_24155)
    
    # Getting the type of 'k' (line 138)
    k_24160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 43), 'k', False)
    int_24161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 47), 'int')
    # Applying the binary operator '-' (line 138)
    result_sub_24162 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 43), '-', k_24160, int_24161)
    
    # Getting the type of 'm' (line 138)
    m_24163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 50), 'm', False)
    # Obtaining the member 'dtype' of a type (line 138)
    dtype_24164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 50), m_24163, 'dtype')
    # Obtaining the member 'char' of a type (line 138)
    char_24165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 50), dtype_24164, 'char')
    # Processing the call keyword arguments (line 138)
    kwargs_24166 = {}
    # Getting the type of 'tri' (line 138)
    tri_24149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'tri', False)
    # Calling tri(args, kwargs) (line 138)
    tri_call_result_24167 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), tri_24149, *[subscript_call_result_24154, subscript_call_result_24159, result_sub_24162, char_24165], **kwargs_24166)
    
    # Applying the binary operator '-' (line 138)
    result_sub_24168 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 11), '-', int_24148, tri_call_result_24167)
    
    # Getting the type of 'm' (line 138)
    m_24169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 67), 'm')
    # Applying the binary operator '*' (line 138)
    result_mul_24170 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 10), '*', result_sub_24168, m_24169)
    
    # Assigning a type to the variable 'out' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'out', result_mul_24170)
    # Getting the type of 'out' (line 139)
    out_24171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type', out_24171)
    
    # ################# End of 'triu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'triu' in the type store
    # Getting the type of 'stypy_return_type' (line 108)
    stypy_return_type_24172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24172)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'triu'
    return stypy_return_type_24172

# Assigning a type to the variable 'triu' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'triu', triu)

@norecursion
def toeplitz(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 142)
    None_24173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'None')
    defaults = [None_24173]
    # Create a new context for function 'toeplitz'
    module_type_store = module_type_store.open_function_context('toeplitz', 142, 0, False)
    
    # Passed parameters checking function
    toeplitz.stypy_localization = localization
    toeplitz.stypy_type_of_self = None
    toeplitz.stypy_type_store = module_type_store
    toeplitz.stypy_function_name = 'toeplitz'
    toeplitz.stypy_param_names_list = ['c', 'r']
    toeplitz.stypy_varargs_param_name = None
    toeplitz.stypy_kwargs_param_name = None
    toeplitz.stypy_call_defaults = defaults
    toeplitz.stypy_call_varargs = varargs
    toeplitz.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'toeplitz', ['c', 'r'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'toeplitz', localization, ['c', 'r'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'toeplitz(...)' code ##################

    str_24174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, (-1)), 'str', '\n    Construct a Toeplitz matrix.\n\n    The Toeplitz matrix has constant diagonals, with c as its first column\n    and r as its first row.  If r is not given, ``r == conjugate(c)`` is\n    assumed.\n\n    Parameters\n    ----------\n    c : array_like\n        First column of the matrix.  Whatever the actual shape of `c`, it\n        will be converted to a 1-D array.\n    r : array_like, optional\n        First row of the matrix. If None, ``r = conjugate(c)`` is assumed;\n        in this case, if c[0] is real, the result is a Hermitian matrix.\n        r[0] is ignored; the first row of the returned matrix is\n        ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be\n        converted to a 1-D array.\n\n    Returns\n    -------\n    A : (len(c), len(r)) ndarray\n        The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.\n\n    See Also\n    --------\n    circulant : circulant matrix\n    hankel : Hankel matrix\n    solve_toeplitz : Solve a Toeplitz system.\n\n    Notes\n    -----\n    The behavior when `c` or `r` is a scalar, or when `c` is complex and\n    `r` is None, was changed in version 0.8.0.  The behavior in previous\n    versions was undocumented and is no longer supported.\n\n    Examples\n    --------\n    >>> from scipy.linalg import toeplitz\n    >>> toeplitz([1,2,3], [1,4,5,6])\n    array([[1, 4, 5, 6],\n           [2, 1, 4, 5],\n           [3, 2, 1, 4]])\n    >>> toeplitz([1.0, 2+3j, 4-1j])\n    array([[ 1.+0.j,  2.-3.j,  4.+1.j],\n           [ 2.+3.j,  1.+0.j,  2.-3.j],\n           [ 4.-1.j,  2.+3.j,  1.+0.j]])\n\n    ')
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to ravel(...): (line 192)
    # Processing the call keyword arguments (line 192)
    kwargs_24181 = {}
    
    # Call to asarray(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'c' (line 192)
    c_24177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 19), 'c', False)
    # Processing the call keyword arguments (line 192)
    kwargs_24178 = {}
    # Getting the type of 'np' (line 192)
    np_24175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 192)
    asarray_24176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), np_24175, 'asarray')
    # Calling asarray(args, kwargs) (line 192)
    asarray_call_result_24179 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), asarray_24176, *[c_24177], **kwargs_24178)
    
    # Obtaining the member 'ravel' of a type (line 192)
    ravel_24180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), asarray_call_result_24179, 'ravel')
    # Calling ravel(args, kwargs) (line 192)
    ravel_call_result_24182 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), ravel_24180, *[], **kwargs_24181)
    
    # Assigning a type to the variable 'c' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'c', ravel_call_result_24182)
    
    # Type idiom detected: calculating its left and rigth part (line 193)
    # Getting the type of 'r' (line 193)
    r_24183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 7), 'r')
    # Getting the type of 'None' (line 193)
    None_24184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'None')
    
    (may_be_24185, more_types_in_union_24186) = may_be_none(r_24183, None_24184)

    if may_be_24185:

        if more_types_in_union_24186:
            # Runtime conditional SSA (line 193)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 194):
        
        # Assigning a Call to a Name (line 194):
        
        # Call to conjugate(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_24189 = {}
        # Getting the type of 'c' (line 194)
        c_24187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'c', False)
        # Obtaining the member 'conjugate' of a type (line 194)
        conjugate_24188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), c_24187, 'conjugate')
        # Calling conjugate(args, kwargs) (line 194)
        conjugate_call_result_24190 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), conjugate_24188, *[], **kwargs_24189)
        
        # Assigning a type to the variable 'r' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'r', conjugate_call_result_24190)

        if more_types_in_union_24186:
            # Runtime conditional SSA for else branch (line 193)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_24185) or more_types_in_union_24186):
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to ravel(...): (line 196)
        # Processing the call keyword arguments (line 196)
        kwargs_24197 = {}
        
        # Call to asarray(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'r' (line 196)
        r_24193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 23), 'r', False)
        # Processing the call keyword arguments (line 196)
        kwargs_24194 = {}
        # Getting the type of 'np' (line 196)
        np_24191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 196)
        asarray_24192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), np_24191, 'asarray')
        # Calling asarray(args, kwargs) (line 196)
        asarray_call_result_24195 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), asarray_24192, *[r_24193], **kwargs_24194)
        
        # Obtaining the member 'ravel' of a type (line 196)
        ravel_24196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), asarray_call_result_24195, 'ravel')
        # Calling ravel(args, kwargs) (line 196)
        ravel_call_result_24198 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), ravel_24196, *[], **kwargs_24197)
        
        # Assigning a type to the variable 'r' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'r', ravel_call_result_24198)

        if (may_be_24185 and more_types_in_union_24186):
            # SSA join for if statement (line 193)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 199):
    
    # Assigning a Call to a Name (line 199):
    
    # Call to concatenate(...): (line 199)
    # Processing the call arguments (line 199)
    
    # Obtaining an instance of the builtin type 'tuple' (line 199)
    tuple_24201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 199)
    # Adding element type (line 199)
    
    # Obtaining the type of the subscript
    int_24202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 29), 'int')
    int_24203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 32), 'int')
    int_24204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 34), 'int')
    slice_24205 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 27), int_24202, int_24203, int_24204)
    # Getting the type of 'r' (line 199)
    r_24206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___24207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 27), r_24206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_24208 = invoke(stypy.reporting.localization.Localization(__file__, 199, 27), getitem___24207, slice_24205)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 27), tuple_24201, subscript_call_result_24208)
    # Adding element type (line 199)
    # Getting the type of 'c' (line 199)
    c_24209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 39), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 27), tuple_24201, c_24209)
    
    # Processing the call keyword arguments (line 199)
    kwargs_24210 = {}
    # Getting the type of 'np' (line 199)
    np_24199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 199)
    concatenate_24200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 11), np_24199, 'concatenate')
    # Calling concatenate(args, kwargs) (line 199)
    concatenate_call_result_24211 = invoke(stypy.reporting.localization.Localization(__file__, 199, 11), concatenate_24200, *[tuple_24201], **kwargs_24210)
    
    # Assigning a type to the variable 'vals' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'vals', concatenate_call_result_24211)
    
    # Assigning a Subscript to a Tuple (line 200):
    
    # Assigning a Subscript to a Name (line 200):
    
    # Obtaining the type of the subscript
    int_24212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 4), 'int')
    
    # Obtaining the type of the subscript
    int_24213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 20), 'int')
    
    # Call to len(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'c' (line 200)
    c_24215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'c', False)
    # Processing the call keyword arguments (line 200)
    kwargs_24216 = {}
    # Getting the type of 'len' (line 200)
    len_24214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'len', False)
    # Calling len(args, kwargs) (line 200)
    len_call_result_24217 = invoke(stypy.reporting.localization.Localization(__file__, 200, 22), len_24214, *[c_24215], **kwargs_24216)
    
    slice_24218 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 11), int_24213, len_call_result_24217, None)
    
    # Call to len(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'r' (line 200)
    r_24220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 34), 'r', False)
    # Processing the call keyword arguments (line 200)
    kwargs_24221 = {}
    # Getting the type of 'len' (line 200)
    len_24219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 30), 'len', False)
    # Calling len(args, kwargs) (line 200)
    len_call_result_24222 = invoke(stypy.reporting.localization.Localization(__file__, 200, 30), len_24219, *[r_24220], **kwargs_24221)
    
    int_24223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 39), 'int')
    # Applying the binary operator '-' (line 200)
    result_sub_24224 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 30), '-', len_call_result_24222, int_24223)
    
    int_24225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 41), 'int')
    int_24226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 44), 'int')
    slice_24227 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 11), result_sub_24224, int_24225, int_24226)
    # Getting the type of 'np' (line 200)
    np_24228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'np')
    # Obtaining the member 'ogrid' of a type (line 200)
    ogrid_24229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 11), np_24228, 'ogrid')
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___24230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 11), ogrid_24229, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_24231 = invoke(stypy.reporting.localization.Localization(__file__, 200, 11), getitem___24230, (slice_24218, slice_24227))
    
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___24232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 4), subscript_call_result_24231, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_24233 = invoke(stypy.reporting.localization.Localization(__file__, 200, 4), getitem___24232, int_24212)
    
    # Assigning a type to the variable 'tuple_var_assignment_24030' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'tuple_var_assignment_24030', subscript_call_result_24233)
    
    # Assigning a Subscript to a Name (line 200):
    
    # Obtaining the type of the subscript
    int_24234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 4), 'int')
    
    # Obtaining the type of the subscript
    int_24235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 20), 'int')
    
    # Call to len(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'c' (line 200)
    c_24237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'c', False)
    # Processing the call keyword arguments (line 200)
    kwargs_24238 = {}
    # Getting the type of 'len' (line 200)
    len_24236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'len', False)
    # Calling len(args, kwargs) (line 200)
    len_call_result_24239 = invoke(stypy.reporting.localization.Localization(__file__, 200, 22), len_24236, *[c_24237], **kwargs_24238)
    
    slice_24240 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 11), int_24235, len_call_result_24239, None)
    
    # Call to len(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'r' (line 200)
    r_24242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 34), 'r', False)
    # Processing the call keyword arguments (line 200)
    kwargs_24243 = {}
    # Getting the type of 'len' (line 200)
    len_24241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 30), 'len', False)
    # Calling len(args, kwargs) (line 200)
    len_call_result_24244 = invoke(stypy.reporting.localization.Localization(__file__, 200, 30), len_24241, *[r_24242], **kwargs_24243)
    
    int_24245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 39), 'int')
    # Applying the binary operator '-' (line 200)
    result_sub_24246 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 30), '-', len_call_result_24244, int_24245)
    
    int_24247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 41), 'int')
    int_24248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 44), 'int')
    slice_24249 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 11), result_sub_24246, int_24247, int_24248)
    # Getting the type of 'np' (line 200)
    np_24250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'np')
    # Obtaining the member 'ogrid' of a type (line 200)
    ogrid_24251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 11), np_24250, 'ogrid')
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___24252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 11), ogrid_24251, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_24253 = invoke(stypy.reporting.localization.Localization(__file__, 200, 11), getitem___24252, (slice_24240, slice_24249))
    
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___24254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 4), subscript_call_result_24253, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_24255 = invoke(stypy.reporting.localization.Localization(__file__, 200, 4), getitem___24254, int_24234)
    
    # Assigning a type to the variable 'tuple_var_assignment_24031' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'tuple_var_assignment_24031', subscript_call_result_24255)
    
    # Assigning a Name to a Name (line 200):
    # Getting the type of 'tuple_var_assignment_24030' (line 200)
    tuple_var_assignment_24030_24256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'tuple_var_assignment_24030')
    # Assigning a type to the variable 'a' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'a', tuple_var_assignment_24030_24256)
    
    # Assigning a Name to a Name (line 200):
    # Getting the type of 'tuple_var_assignment_24031' (line 200)
    tuple_var_assignment_24031_24257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'tuple_var_assignment_24031')
    # Assigning a type to the variable 'b' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 7), 'b', tuple_var_assignment_24031_24257)
    
    # Assigning a BinOp to a Name (line 201):
    
    # Assigning a BinOp to a Name (line 201):
    # Getting the type of 'a' (line 201)
    a_24258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'a')
    # Getting the type of 'b' (line 201)
    b_24259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'b')
    # Applying the binary operator '+' (line 201)
    result_add_24260 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 11), '+', a_24258, b_24259)
    
    # Assigning a type to the variable 'indx' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'indx', result_add_24260)
    
    # Obtaining the type of the subscript
    # Getting the type of 'indx' (line 204)
    indx_24261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'indx')
    # Getting the type of 'vals' (line 204)
    vals_24262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'vals')
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___24263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 11), vals_24262, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_24264 = invoke(stypy.reporting.localization.Localization(__file__, 204, 11), getitem___24263, indx_24261)
    
    # Assigning a type to the variable 'stypy_return_type' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type', subscript_call_result_24264)
    
    # ################# End of 'toeplitz(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'toeplitz' in the type store
    # Getting the type of 'stypy_return_type' (line 142)
    stypy_return_type_24265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24265)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'toeplitz'
    return stypy_return_type_24265

# Assigning a type to the variable 'toeplitz' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'toeplitz', toeplitz)

@norecursion
def circulant(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'circulant'
    module_type_store = module_type_store.open_function_context('circulant', 207, 0, False)
    
    # Passed parameters checking function
    circulant.stypy_localization = localization
    circulant.stypy_type_of_self = None
    circulant.stypy_type_store = module_type_store
    circulant.stypy_function_name = 'circulant'
    circulant.stypy_param_names_list = ['c']
    circulant.stypy_varargs_param_name = None
    circulant.stypy_kwargs_param_name = None
    circulant.stypy_call_defaults = defaults
    circulant.stypy_call_varargs = varargs
    circulant.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'circulant', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'circulant', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'circulant(...)' code ##################

    str_24266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, (-1)), 'str', '\n    Construct a circulant matrix.\n\n    Parameters\n    ----------\n    c : (N,) array_like\n        1-D array, the first column of the matrix.\n\n    Returns\n    -------\n    A : (N, N) ndarray\n        A circulant matrix whose first column is `c`.\n\n    See Also\n    --------\n    toeplitz : Toeplitz matrix\n    hankel : Hankel matrix\n    solve_circulant : Solve a circulant system.\n\n    Notes\n    -----\n    .. versionadded:: 0.8.0\n\n    Examples\n    --------\n    >>> from scipy.linalg import circulant\n    >>> circulant([1, 2, 3])\n    array([[1, 3, 2],\n           [2, 1, 3],\n           [3, 2, 1]])\n\n    ')
    
    # Assigning a Call to a Name (line 240):
    
    # Assigning a Call to a Name (line 240):
    
    # Call to ravel(...): (line 240)
    # Processing the call keyword arguments (line 240)
    kwargs_24273 = {}
    
    # Call to asarray(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'c' (line 240)
    c_24269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), 'c', False)
    # Processing the call keyword arguments (line 240)
    kwargs_24270 = {}
    # Getting the type of 'np' (line 240)
    np_24267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 240)
    asarray_24268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), np_24267, 'asarray')
    # Calling asarray(args, kwargs) (line 240)
    asarray_call_result_24271 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), asarray_24268, *[c_24269], **kwargs_24270)
    
    # Obtaining the member 'ravel' of a type (line 240)
    ravel_24272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), asarray_call_result_24271, 'ravel')
    # Calling ravel(args, kwargs) (line 240)
    ravel_call_result_24274 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), ravel_24272, *[], **kwargs_24273)
    
    # Assigning a type to the variable 'c' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'c', ravel_call_result_24274)
    
    # Assigning a Subscript to a Tuple (line 241):
    
    # Assigning a Subscript to a Name (line 241):
    
    # Obtaining the type of the subscript
    int_24275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 4), 'int')
    
    # Obtaining the type of the subscript
    int_24276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 20), 'int')
    
    # Call to len(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'c' (line 241)
    c_24278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 26), 'c', False)
    # Processing the call keyword arguments (line 241)
    kwargs_24279 = {}
    # Getting the type of 'len' (line 241)
    len_24277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 22), 'len', False)
    # Calling len(args, kwargs) (line 241)
    len_call_result_24280 = invoke(stypy.reporting.localization.Localization(__file__, 241, 22), len_24277, *[c_24278], **kwargs_24279)
    
    slice_24281 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 241, 11), int_24276, len_call_result_24280, None)
    int_24282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 30), 'int')
    
    
    # Call to len(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'c' (line 241)
    c_24284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 37), 'c', False)
    # Processing the call keyword arguments (line 241)
    kwargs_24285 = {}
    # Getting the type of 'len' (line 241)
    len_24283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 33), 'len', False)
    # Calling len(args, kwargs) (line 241)
    len_call_result_24286 = invoke(stypy.reporting.localization.Localization(__file__, 241, 33), len_24283, *[c_24284], **kwargs_24285)
    
    # Applying the 'usub' unary operator (line 241)
    result___neg___24287 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 32), 'usub', len_call_result_24286)
    
    int_24288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 40), 'int')
    slice_24289 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 241, 11), int_24282, result___neg___24287, int_24288)
    # Getting the type of 'np' (line 241)
    np_24290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 'np')
    # Obtaining the member 'ogrid' of a type (line 241)
    ogrid_24291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 11), np_24290, 'ogrid')
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___24292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 11), ogrid_24291, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_24293 = invoke(stypy.reporting.localization.Localization(__file__, 241, 11), getitem___24292, (slice_24281, slice_24289))
    
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___24294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 4), subscript_call_result_24293, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_24295 = invoke(stypy.reporting.localization.Localization(__file__, 241, 4), getitem___24294, int_24275)
    
    # Assigning a type to the variable 'tuple_var_assignment_24032' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_24032', subscript_call_result_24295)
    
    # Assigning a Subscript to a Name (line 241):
    
    # Obtaining the type of the subscript
    int_24296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 4), 'int')
    
    # Obtaining the type of the subscript
    int_24297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 20), 'int')
    
    # Call to len(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'c' (line 241)
    c_24299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 26), 'c', False)
    # Processing the call keyword arguments (line 241)
    kwargs_24300 = {}
    # Getting the type of 'len' (line 241)
    len_24298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 22), 'len', False)
    # Calling len(args, kwargs) (line 241)
    len_call_result_24301 = invoke(stypy.reporting.localization.Localization(__file__, 241, 22), len_24298, *[c_24299], **kwargs_24300)
    
    slice_24302 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 241, 11), int_24297, len_call_result_24301, None)
    int_24303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 30), 'int')
    
    
    # Call to len(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'c' (line 241)
    c_24305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 37), 'c', False)
    # Processing the call keyword arguments (line 241)
    kwargs_24306 = {}
    # Getting the type of 'len' (line 241)
    len_24304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 33), 'len', False)
    # Calling len(args, kwargs) (line 241)
    len_call_result_24307 = invoke(stypy.reporting.localization.Localization(__file__, 241, 33), len_24304, *[c_24305], **kwargs_24306)
    
    # Applying the 'usub' unary operator (line 241)
    result___neg___24308 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 32), 'usub', len_call_result_24307)
    
    int_24309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 40), 'int')
    slice_24310 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 241, 11), int_24303, result___neg___24308, int_24309)
    # Getting the type of 'np' (line 241)
    np_24311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 'np')
    # Obtaining the member 'ogrid' of a type (line 241)
    ogrid_24312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 11), np_24311, 'ogrid')
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___24313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 11), ogrid_24312, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_24314 = invoke(stypy.reporting.localization.Localization(__file__, 241, 11), getitem___24313, (slice_24302, slice_24310))
    
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___24315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 4), subscript_call_result_24314, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_24316 = invoke(stypy.reporting.localization.Localization(__file__, 241, 4), getitem___24315, int_24296)
    
    # Assigning a type to the variable 'tuple_var_assignment_24033' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_24033', subscript_call_result_24316)
    
    # Assigning a Name to a Name (line 241):
    # Getting the type of 'tuple_var_assignment_24032' (line 241)
    tuple_var_assignment_24032_24317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_24032')
    # Assigning a type to the variable 'a' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'a', tuple_var_assignment_24032_24317)
    
    # Assigning a Name to a Name (line 241):
    # Getting the type of 'tuple_var_assignment_24033' (line 241)
    tuple_var_assignment_24033_24318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'tuple_var_assignment_24033')
    # Assigning a type to the variable 'b' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 7), 'b', tuple_var_assignment_24033_24318)
    
    # Assigning a BinOp to a Name (line 242):
    
    # Assigning a BinOp to a Name (line 242):
    # Getting the type of 'a' (line 242)
    a_24319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'a')
    # Getting the type of 'b' (line 242)
    b_24320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'b')
    # Applying the binary operator '+' (line 242)
    result_add_24321 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 11), '+', a_24319, b_24320)
    
    # Assigning a type to the variable 'indx' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'indx', result_add_24321)
    
    # Obtaining the type of the subscript
    # Getting the type of 'indx' (line 245)
    indx_24322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 13), 'indx')
    # Getting the type of 'c' (line 245)
    c_24323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 11), 'c')
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___24324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 11), c_24323, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 245)
    subscript_call_result_24325 = invoke(stypy.reporting.localization.Localization(__file__, 245, 11), getitem___24324, indx_24322)
    
    # Assigning a type to the variable 'stypy_return_type' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'stypy_return_type', subscript_call_result_24325)
    
    # ################# End of 'circulant(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'circulant' in the type store
    # Getting the type of 'stypy_return_type' (line 207)
    stypy_return_type_24326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24326)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'circulant'
    return stypy_return_type_24326

# Assigning a type to the variable 'circulant' (line 207)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 0), 'circulant', circulant)

@norecursion
def hankel(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 248)
    None_24327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'None')
    defaults = [None_24327]
    # Create a new context for function 'hankel'
    module_type_store = module_type_store.open_function_context('hankel', 248, 0, False)
    
    # Passed parameters checking function
    hankel.stypy_localization = localization
    hankel.stypy_type_of_self = None
    hankel.stypy_type_store = module_type_store
    hankel.stypy_function_name = 'hankel'
    hankel.stypy_param_names_list = ['c', 'r']
    hankel.stypy_varargs_param_name = None
    hankel.stypy_kwargs_param_name = None
    hankel.stypy_call_defaults = defaults
    hankel.stypy_call_varargs = varargs
    hankel.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hankel', ['c', 'r'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hankel', localization, ['c', 'r'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hankel(...)' code ##################

    str_24328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, (-1)), 'str', '\n    Construct a Hankel matrix.\n\n    The Hankel matrix has constant anti-diagonals, with `c` as its\n    first column and `r` as its last row.  If `r` is not given, then\n    `r = zeros_like(c)` is assumed.\n\n    Parameters\n    ----------\n    c : array_like\n        First column of the matrix.  Whatever the actual shape of `c`, it\n        will be converted to a 1-D array.\n    r : array_like, optional\n        Last row of the matrix. If None, ``r = zeros_like(c)`` is assumed.\n        r[0] is ignored; the last row of the returned matrix is\n        ``[c[-1], r[1:]]``.  Whatever the actual shape of `r`, it will be\n        converted to a 1-D array.\n\n    Returns\n    -------\n    A : (len(c), len(r)) ndarray\n        The Hankel matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.\n\n    See Also\n    --------\n    toeplitz : Toeplitz matrix\n    circulant : circulant matrix\n\n    Examples\n    --------\n    >>> from scipy.linalg import hankel\n    >>> hankel([1, 17, 99])\n    array([[ 1, 17, 99],\n           [17, 99,  0],\n           [99,  0,  0]])\n    >>> hankel([1,2,3,4], [4,7,7,8,9])\n    array([[1, 2, 3, 4, 7],\n           [2, 3, 4, 7, 7],\n           [3, 4, 7, 7, 8],\n           [4, 7, 7, 8, 9]])\n\n    ')
    
    # Assigning a Call to a Name (line 291):
    
    # Assigning a Call to a Name (line 291):
    
    # Call to ravel(...): (line 291)
    # Processing the call keyword arguments (line 291)
    kwargs_24335 = {}
    
    # Call to asarray(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'c' (line 291)
    c_24331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 19), 'c', False)
    # Processing the call keyword arguments (line 291)
    kwargs_24332 = {}
    # Getting the type of 'np' (line 291)
    np_24329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 291)
    asarray_24330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), np_24329, 'asarray')
    # Calling asarray(args, kwargs) (line 291)
    asarray_call_result_24333 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), asarray_24330, *[c_24331], **kwargs_24332)
    
    # Obtaining the member 'ravel' of a type (line 291)
    ravel_24334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), asarray_call_result_24333, 'ravel')
    # Calling ravel(args, kwargs) (line 291)
    ravel_call_result_24336 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), ravel_24334, *[], **kwargs_24335)
    
    # Assigning a type to the variable 'c' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'c', ravel_call_result_24336)
    
    # Type idiom detected: calculating its left and rigth part (line 292)
    # Getting the type of 'r' (line 292)
    r_24337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 7), 'r')
    # Getting the type of 'None' (line 292)
    None_24338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'None')
    
    (may_be_24339, more_types_in_union_24340) = may_be_none(r_24337, None_24338)

    if may_be_24339:

        if more_types_in_union_24340:
            # Runtime conditional SSA (line 292)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to zeros_like(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'c' (line 293)
        c_24343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'c', False)
        # Processing the call keyword arguments (line 293)
        kwargs_24344 = {}
        # Getting the type of 'np' (line 293)
        np_24341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'np', False)
        # Obtaining the member 'zeros_like' of a type (line 293)
        zeros_like_24342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 12), np_24341, 'zeros_like')
        # Calling zeros_like(args, kwargs) (line 293)
        zeros_like_call_result_24345 = invoke(stypy.reporting.localization.Localization(__file__, 293, 12), zeros_like_24342, *[c_24343], **kwargs_24344)
        
        # Assigning a type to the variable 'r' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'r', zeros_like_call_result_24345)

        if more_types_in_union_24340:
            # Runtime conditional SSA for else branch (line 292)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_24339) or more_types_in_union_24340):
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to ravel(...): (line 295)
        # Processing the call keyword arguments (line 295)
        kwargs_24352 = {}
        
        # Call to asarray(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'r' (line 295)
        r_24348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 23), 'r', False)
        # Processing the call keyword arguments (line 295)
        kwargs_24349 = {}
        # Getting the type of 'np' (line 295)
        np_24346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 295)
        asarray_24347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 12), np_24346, 'asarray')
        # Calling asarray(args, kwargs) (line 295)
        asarray_call_result_24350 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), asarray_24347, *[r_24348], **kwargs_24349)
        
        # Obtaining the member 'ravel' of a type (line 295)
        ravel_24351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 12), asarray_call_result_24350, 'ravel')
        # Calling ravel(args, kwargs) (line 295)
        ravel_call_result_24353 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), ravel_24351, *[], **kwargs_24352)
        
        # Assigning a type to the variable 'r' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'r', ravel_call_result_24353)

        if (may_be_24339 and more_types_in_union_24340):
            # SSA join for if statement (line 292)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 298):
    
    # Assigning a Call to a Name (line 298):
    
    # Call to concatenate(...): (line 298)
    # Processing the call arguments (line 298)
    
    # Obtaining an instance of the builtin type 'tuple' (line 298)
    tuple_24356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 298)
    # Adding element type (line 298)
    # Getting the type of 'c' (line 298)
    c_24357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 27), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 27), tuple_24356, c_24357)
    # Adding element type (line 298)
    
    # Obtaining the type of the subscript
    int_24358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 32), 'int')
    slice_24359 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 298, 30), int_24358, None, None)
    # Getting the type of 'r' (line 298)
    r_24360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 30), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___24361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 30), r_24360, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_24362 = invoke(stypy.reporting.localization.Localization(__file__, 298, 30), getitem___24361, slice_24359)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 27), tuple_24356, subscript_call_result_24362)
    
    # Processing the call keyword arguments (line 298)
    kwargs_24363 = {}
    # Getting the type of 'np' (line 298)
    np_24354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 298)
    concatenate_24355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 11), np_24354, 'concatenate')
    # Calling concatenate(args, kwargs) (line 298)
    concatenate_call_result_24364 = invoke(stypy.reporting.localization.Localization(__file__, 298, 11), concatenate_24355, *[tuple_24356], **kwargs_24363)
    
    # Assigning a type to the variable 'vals' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'vals', concatenate_call_result_24364)
    
    # Assigning a Subscript to a Tuple (line 299):
    
    # Assigning a Subscript to a Name (line 299):
    
    # Obtaining the type of the subscript
    int_24365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 4), 'int')
    
    # Obtaining the type of the subscript
    int_24366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 20), 'int')
    
    # Call to len(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'c' (line 299)
    c_24368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 26), 'c', False)
    # Processing the call keyword arguments (line 299)
    kwargs_24369 = {}
    # Getting the type of 'len' (line 299)
    len_24367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 22), 'len', False)
    # Calling len(args, kwargs) (line 299)
    len_call_result_24370 = invoke(stypy.reporting.localization.Localization(__file__, 299, 22), len_24367, *[c_24368], **kwargs_24369)
    
    slice_24371 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 299, 11), int_24366, len_call_result_24370, None)
    int_24372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 30), 'int')
    
    # Call to len(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'r' (line 299)
    r_24374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 36), 'r', False)
    # Processing the call keyword arguments (line 299)
    kwargs_24375 = {}
    # Getting the type of 'len' (line 299)
    len_24373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 32), 'len', False)
    # Calling len(args, kwargs) (line 299)
    len_call_result_24376 = invoke(stypy.reporting.localization.Localization(__file__, 299, 32), len_24373, *[r_24374], **kwargs_24375)
    
    slice_24377 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 299, 11), int_24372, len_call_result_24376, None)
    # Getting the type of 'np' (line 299)
    np_24378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 11), 'np')
    # Obtaining the member 'ogrid' of a type (line 299)
    ogrid_24379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 11), np_24378, 'ogrid')
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___24380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 11), ogrid_24379, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_24381 = invoke(stypy.reporting.localization.Localization(__file__, 299, 11), getitem___24380, (slice_24371, slice_24377))
    
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___24382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 4), subscript_call_result_24381, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_24383 = invoke(stypy.reporting.localization.Localization(__file__, 299, 4), getitem___24382, int_24365)
    
    # Assigning a type to the variable 'tuple_var_assignment_24034' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'tuple_var_assignment_24034', subscript_call_result_24383)
    
    # Assigning a Subscript to a Name (line 299):
    
    # Obtaining the type of the subscript
    int_24384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 4), 'int')
    
    # Obtaining the type of the subscript
    int_24385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 20), 'int')
    
    # Call to len(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'c' (line 299)
    c_24387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 26), 'c', False)
    # Processing the call keyword arguments (line 299)
    kwargs_24388 = {}
    # Getting the type of 'len' (line 299)
    len_24386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 22), 'len', False)
    # Calling len(args, kwargs) (line 299)
    len_call_result_24389 = invoke(stypy.reporting.localization.Localization(__file__, 299, 22), len_24386, *[c_24387], **kwargs_24388)
    
    slice_24390 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 299, 11), int_24385, len_call_result_24389, None)
    int_24391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 30), 'int')
    
    # Call to len(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'r' (line 299)
    r_24393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 36), 'r', False)
    # Processing the call keyword arguments (line 299)
    kwargs_24394 = {}
    # Getting the type of 'len' (line 299)
    len_24392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 32), 'len', False)
    # Calling len(args, kwargs) (line 299)
    len_call_result_24395 = invoke(stypy.reporting.localization.Localization(__file__, 299, 32), len_24392, *[r_24393], **kwargs_24394)
    
    slice_24396 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 299, 11), int_24391, len_call_result_24395, None)
    # Getting the type of 'np' (line 299)
    np_24397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 11), 'np')
    # Obtaining the member 'ogrid' of a type (line 299)
    ogrid_24398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 11), np_24397, 'ogrid')
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___24399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 11), ogrid_24398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_24400 = invoke(stypy.reporting.localization.Localization(__file__, 299, 11), getitem___24399, (slice_24390, slice_24396))
    
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___24401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 4), subscript_call_result_24400, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_24402 = invoke(stypy.reporting.localization.Localization(__file__, 299, 4), getitem___24401, int_24384)
    
    # Assigning a type to the variable 'tuple_var_assignment_24035' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'tuple_var_assignment_24035', subscript_call_result_24402)
    
    # Assigning a Name to a Name (line 299):
    # Getting the type of 'tuple_var_assignment_24034' (line 299)
    tuple_var_assignment_24034_24403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'tuple_var_assignment_24034')
    # Assigning a type to the variable 'a' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'a', tuple_var_assignment_24034_24403)
    
    # Assigning a Name to a Name (line 299):
    # Getting the type of 'tuple_var_assignment_24035' (line 299)
    tuple_var_assignment_24035_24404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'tuple_var_assignment_24035')
    # Assigning a type to the variable 'b' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 7), 'b', tuple_var_assignment_24035_24404)
    
    # Assigning a BinOp to a Name (line 300):
    
    # Assigning a BinOp to a Name (line 300):
    # Getting the type of 'a' (line 300)
    a_24405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 11), 'a')
    # Getting the type of 'b' (line 300)
    b_24406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'b')
    # Applying the binary operator '+' (line 300)
    result_add_24407 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 11), '+', a_24405, b_24406)
    
    # Assigning a type to the variable 'indx' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'indx', result_add_24407)
    
    # Obtaining the type of the subscript
    # Getting the type of 'indx' (line 303)
    indx_24408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'indx')
    # Getting the type of 'vals' (line 303)
    vals_24409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 11), 'vals')
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___24410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 11), vals_24409, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_24411 = invoke(stypy.reporting.localization.Localization(__file__, 303, 11), getitem___24410, indx_24408)
    
    # Assigning a type to the variable 'stypy_return_type' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type', subscript_call_result_24411)
    
    # ################# End of 'hankel(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hankel' in the type store
    # Getting the type of 'stypy_return_type' (line 248)
    stypy_return_type_24412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24412)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hankel'
    return stypy_return_type_24412

# Assigning a type to the variable 'hankel' (line 248)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), 'hankel', hankel)

@norecursion
def hadamard(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'int' (line 306)
    int_24413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 22), 'int')
    defaults = [int_24413]
    # Create a new context for function 'hadamard'
    module_type_store = module_type_store.open_function_context('hadamard', 306, 0, False)
    
    # Passed parameters checking function
    hadamard.stypy_localization = localization
    hadamard.stypy_type_of_self = None
    hadamard.stypy_type_store = module_type_store
    hadamard.stypy_function_name = 'hadamard'
    hadamard.stypy_param_names_list = ['n', 'dtype']
    hadamard.stypy_varargs_param_name = None
    hadamard.stypy_kwargs_param_name = None
    hadamard.stypy_call_defaults = defaults
    hadamard.stypy_call_varargs = varargs
    hadamard.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hadamard', ['n', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hadamard', localization, ['n', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hadamard(...)' code ##################

    str_24414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, (-1)), 'str', "\n    Construct a Hadamard matrix.\n\n    Constructs an n-by-n Hadamard matrix, using Sylvester's\n    construction.  `n` must be a power of 2.\n\n    Parameters\n    ----------\n    n : int\n        The order of the matrix.  `n` must be a power of 2.\n    dtype : dtype, optional\n        The data type of the array to be constructed.\n\n    Returns\n    -------\n    H : (n, n) ndarray\n        The Hadamard matrix.\n\n    Notes\n    -----\n    .. versionadded:: 0.8.0\n\n    Examples\n    --------\n    >>> from scipy.linalg import hadamard\n    >>> hadamard(2, dtype=complex)\n    array([[ 1.+0.j,  1.+0.j],\n           [ 1.+0.j, -1.-0.j]])\n    >>> hadamard(4)\n    array([[ 1,  1,  1,  1],\n           [ 1, -1,  1, -1],\n           [ 1,  1, -1, -1],\n           [ 1, -1, -1,  1]])\n\n    ")
    
    
    # Getting the type of 'n' (line 346)
    n_24415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 7), 'n')
    int_24416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 11), 'int')
    # Applying the binary operator '<' (line 346)
    result_lt_24417 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 7), '<', n_24415, int_24416)
    
    # Testing the type of an if condition (line 346)
    if_condition_24418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 346, 4), result_lt_24417)
    # Assigning a type to the variable 'if_condition_24418' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'if_condition_24418', if_condition_24418)
    # SSA begins for if statement (line 346)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 347):
    
    # Assigning a Num to a Name (line 347):
    int_24419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 14), 'int')
    # Assigning a type to the variable 'lg2' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'lg2', int_24419)
    # SSA branch for the else part of an if statement (line 346)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 349):
    
    # Assigning a Call to a Name (line 349):
    
    # Call to int(...): (line 349)
    # Processing the call arguments (line 349)
    
    # Call to log(...): (line 349)
    # Processing the call arguments (line 349)
    # Getting the type of 'n' (line 349)
    n_24423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 27), 'n', False)
    int_24424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 30), 'int')
    # Processing the call keyword arguments (line 349)
    kwargs_24425 = {}
    # Getting the type of 'math' (line 349)
    math_24421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 18), 'math', False)
    # Obtaining the member 'log' of a type (line 349)
    log_24422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 18), math_24421, 'log')
    # Calling log(args, kwargs) (line 349)
    log_call_result_24426 = invoke(stypy.reporting.localization.Localization(__file__, 349, 18), log_24422, *[n_24423, int_24424], **kwargs_24425)
    
    # Processing the call keyword arguments (line 349)
    kwargs_24427 = {}
    # Getting the type of 'int' (line 349)
    int_24420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 14), 'int', False)
    # Calling int(args, kwargs) (line 349)
    int_call_result_24428 = invoke(stypy.reporting.localization.Localization(__file__, 349, 14), int_24420, *[log_call_result_24426], **kwargs_24427)
    
    # Assigning a type to the variable 'lg2' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'lg2', int_call_result_24428)
    # SSA join for if statement (line 346)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    int_24429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 7), 'int')
    # Getting the type of 'lg2' (line 350)
    lg2_24430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'lg2')
    # Applying the binary operator '**' (line 350)
    result_pow_24431 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 7), '**', int_24429, lg2_24430)
    
    # Getting the type of 'n' (line 350)
    n_24432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 19), 'n')
    # Applying the binary operator '!=' (line 350)
    result_ne_24433 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 7), '!=', result_pow_24431, n_24432)
    
    # Testing the type of an if condition (line 350)
    if_condition_24434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 4), result_ne_24433)
    # Assigning a type to the variable 'if_condition_24434' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'if_condition_24434', if_condition_24434)
    # SSA begins for if statement (line 350)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 351)
    # Processing the call arguments (line 351)
    str_24436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 25), 'str', 'n must be an positive integer, and n must be a power of 2')
    # Processing the call keyword arguments (line 351)
    kwargs_24437 = {}
    # Getting the type of 'ValueError' (line 351)
    ValueError_24435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 351)
    ValueError_call_result_24438 = invoke(stypy.reporting.localization.Localization(__file__, 351, 14), ValueError_24435, *[str_24436], **kwargs_24437)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 351, 8), ValueError_call_result_24438, 'raise parameter', BaseException)
    # SSA join for if statement (line 350)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 354):
    
    # Assigning a Call to a Name (line 354):
    
    # Call to array(...): (line 354)
    # Processing the call arguments (line 354)
    
    # Obtaining an instance of the builtin type 'list' (line 354)
    list_24441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 354)
    # Adding element type (line 354)
    
    # Obtaining an instance of the builtin type 'list' (line 354)
    list_24442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 354)
    # Adding element type (line 354)
    int_24443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 18), list_24442, int_24443)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 17), list_24441, list_24442)
    
    # Processing the call keyword arguments (line 354)
    # Getting the type of 'dtype' (line 354)
    dtype_24444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 30), 'dtype', False)
    keyword_24445 = dtype_24444
    kwargs_24446 = {'dtype': keyword_24445}
    # Getting the type of 'np' (line 354)
    np_24439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 354)
    array_24440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), np_24439, 'array')
    # Calling array(args, kwargs) (line 354)
    array_call_result_24447 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), array_24440, *[list_24441], **kwargs_24446)
    
    # Assigning a type to the variable 'H' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'H', array_call_result_24447)
    
    
    # Call to range(...): (line 357)
    # Processing the call arguments (line 357)
    int_24449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 19), 'int')
    # Getting the type of 'lg2' (line 357)
    lg2_24450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 22), 'lg2', False)
    # Processing the call keyword arguments (line 357)
    kwargs_24451 = {}
    # Getting the type of 'range' (line 357)
    range_24448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'range', False)
    # Calling range(args, kwargs) (line 357)
    range_call_result_24452 = invoke(stypy.reporting.localization.Localization(__file__, 357, 13), range_24448, *[int_24449, lg2_24450], **kwargs_24451)
    
    # Testing the type of a for loop iterable (line 357)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 357, 4), range_call_result_24452)
    # Getting the type of the for loop variable (line 357)
    for_loop_var_24453 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 357, 4), range_call_result_24452)
    # Assigning a type to the variable 'i' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'i', for_loop_var_24453)
    # SSA begins for a for statement (line 357)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 358):
    
    # Assigning a Call to a Name (line 358):
    
    # Call to vstack(...): (line 358)
    # Processing the call arguments (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 358)
    tuple_24456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 358)
    # Adding element type (line 358)
    
    # Call to hstack(...): (line 358)
    # Processing the call arguments (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 358)
    tuple_24459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 358)
    # Adding element type (line 358)
    # Getting the type of 'H' (line 358)
    H_24460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 34), 'H', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 34), tuple_24459, H_24460)
    # Adding element type (line 358)
    # Getting the type of 'H' (line 358)
    H_24461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 37), 'H', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 34), tuple_24459, H_24461)
    
    # Processing the call keyword arguments (line 358)
    kwargs_24462 = {}
    # Getting the type of 'np' (line 358)
    np_24457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 23), 'np', False)
    # Obtaining the member 'hstack' of a type (line 358)
    hstack_24458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 23), np_24457, 'hstack')
    # Calling hstack(args, kwargs) (line 358)
    hstack_call_result_24463 = invoke(stypy.reporting.localization.Localization(__file__, 358, 23), hstack_24458, *[tuple_24459], **kwargs_24462)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 23), tuple_24456, hstack_call_result_24463)
    # Adding element type (line 358)
    
    # Call to hstack(...): (line 358)
    # Processing the call arguments (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 358)
    tuple_24466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 358)
    # Adding element type (line 358)
    # Getting the type of 'H' (line 358)
    H_24467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 53), 'H', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 53), tuple_24466, H_24467)
    # Adding element type (line 358)
    
    # Getting the type of 'H' (line 358)
    H_24468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 57), 'H', False)
    # Applying the 'usub' unary operator (line 358)
    result___neg___24469 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 56), 'usub', H_24468)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 53), tuple_24466, result___neg___24469)
    
    # Processing the call keyword arguments (line 358)
    kwargs_24470 = {}
    # Getting the type of 'np' (line 358)
    np_24464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 42), 'np', False)
    # Obtaining the member 'hstack' of a type (line 358)
    hstack_24465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 42), np_24464, 'hstack')
    # Calling hstack(args, kwargs) (line 358)
    hstack_call_result_24471 = invoke(stypy.reporting.localization.Localization(__file__, 358, 42), hstack_24465, *[tuple_24466], **kwargs_24470)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 23), tuple_24456, hstack_call_result_24471)
    
    # Processing the call keyword arguments (line 358)
    kwargs_24472 = {}
    # Getting the type of 'np' (line 358)
    np_24454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'np', False)
    # Obtaining the member 'vstack' of a type (line 358)
    vstack_24455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), np_24454, 'vstack')
    # Calling vstack(args, kwargs) (line 358)
    vstack_call_result_24473 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), vstack_24455, *[tuple_24456], **kwargs_24472)
    
    # Assigning a type to the variable 'H' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'H', vstack_call_result_24473)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'H' (line 360)
    H_24474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 11), 'H')
    # Assigning a type to the variable 'stypy_return_type' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'stypy_return_type', H_24474)
    
    # ################# End of 'hadamard(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hadamard' in the type store
    # Getting the type of 'stypy_return_type' (line 306)
    stypy_return_type_24475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24475)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hadamard'
    return stypy_return_type_24475

# Assigning a type to the variable 'hadamard' (line 306)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 0), 'hadamard', hadamard)

@norecursion
def leslie(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'leslie'
    module_type_store = module_type_store.open_function_context('leslie', 363, 0, False)
    
    # Passed parameters checking function
    leslie.stypy_localization = localization
    leslie.stypy_type_of_self = None
    leslie.stypy_type_store = module_type_store
    leslie.stypy_function_name = 'leslie'
    leslie.stypy_param_names_list = ['f', 's']
    leslie.stypy_varargs_param_name = None
    leslie.stypy_kwargs_param_name = None
    leslie.stypy_call_defaults = defaults
    leslie.stypy_call_varargs = varargs
    leslie.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'leslie', ['f', 's'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'leslie', localization, ['f', 's'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'leslie(...)' code ##################

    str_24476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, (-1)), 'str', '\n    Create a Leslie matrix.\n\n    Given the length n array of fecundity coefficients `f` and the length\n    n-1 array of survival coefficents `s`, return the associated Leslie matrix.\n\n    Parameters\n    ----------\n    f : (N,) array_like\n        The "fecundity" coefficients.\n    s : (N-1,) array_like\n        The "survival" coefficients, has to be 1-D.  The length of `s`\n        must be one less than the length of `f`, and it must be at least 1.\n\n    Returns\n    -------\n    L : (N, N) ndarray\n        The array is zero except for the first row,\n        which is `f`, and the first sub-diagonal, which is `s`.\n        The data-type of the array will be the data-type of ``f[0]+s[0]``.\n\n    Notes\n    -----\n    .. versionadded:: 0.8.0\n\n    The Leslie matrix is used to model discrete-time, age-structured\n    population growth [1]_ [2]_. In a population with `n` age classes, two sets\n    of parameters define a Leslie matrix: the `n` "fecundity coefficients",\n    which give the number of offspring per-capita produced by each age\n    class, and the `n` - 1 "survival coefficients", which give the\n    per-capita survival rate of each age class.\n\n    References\n    ----------\n    .. [1] P. H. Leslie, On the use of matrices in certain population\n           mathematics, Biometrika, Vol. 33, No. 3, 183--212 (Nov. 1945)\n    .. [2] P. H. Leslie, Some further notes on the use of matrices in\n           population mathematics, Biometrika, Vol. 35, No. 3/4, 213--245\n           (Dec. 1948)\n\n    Examples\n    --------\n    >>> from scipy.linalg import leslie\n    >>> leslie([0.1, 2.0, 1.0, 0.1], [0.2, 0.8, 0.7])\n    array([[ 0.1,  2. ,  1. ,  0.1],\n           [ 0.2,  0. ,  0. ,  0. ],\n           [ 0. ,  0.8,  0. ,  0. ],\n           [ 0. ,  0. ,  0.7,  0. ]])\n\n    ')
    
    # Assigning a Call to a Name (line 414):
    
    # Assigning a Call to a Name (line 414):
    
    # Call to atleast_1d(...): (line 414)
    # Processing the call arguments (line 414)
    # Getting the type of 'f' (line 414)
    f_24479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 22), 'f', False)
    # Processing the call keyword arguments (line 414)
    kwargs_24480 = {}
    # Getting the type of 'np' (line 414)
    np_24477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 414)
    atleast_1d_24478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 8), np_24477, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 414)
    atleast_1d_call_result_24481 = invoke(stypy.reporting.localization.Localization(__file__, 414, 8), atleast_1d_24478, *[f_24479], **kwargs_24480)
    
    # Assigning a type to the variable 'f' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'f', atleast_1d_call_result_24481)
    
    # Assigning a Call to a Name (line 415):
    
    # Assigning a Call to a Name (line 415):
    
    # Call to atleast_1d(...): (line 415)
    # Processing the call arguments (line 415)
    # Getting the type of 's' (line 415)
    s_24484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 22), 's', False)
    # Processing the call keyword arguments (line 415)
    kwargs_24485 = {}
    # Getting the type of 'np' (line 415)
    np_24482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 415)
    atleast_1d_24483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 8), np_24482, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 415)
    atleast_1d_call_result_24486 = invoke(stypy.reporting.localization.Localization(__file__, 415, 8), atleast_1d_24483, *[s_24484], **kwargs_24485)
    
    # Assigning a type to the variable 's' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 's', atleast_1d_call_result_24486)
    
    
    # Getting the type of 'f' (line 416)
    f_24487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 7), 'f')
    # Obtaining the member 'ndim' of a type (line 416)
    ndim_24488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 7), f_24487, 'ndim')
    int_24489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 17), 'int')
    # Applying the binary operator '!=' (line 416)
    result_ne_24490 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 7), '!=', ndim_24488, int_24489)
    
    # Testing the type of an if condition (line 416)
    if_condition_24491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 416, 4), result_ne_24490)
    # Assigning a type to the variable 'if_condition_24491' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'if_condition_24491', if_condition_24491)
    # SSA begins for if statement (line 416)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 417)
    # Processing the call arguments (line 417)
    str_24493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 25), 'str', 'Incorrect shape for f.  f must be one-dimensional')
    # Processing the call keyword arguments (line 417)
    kwargs_24494 = {}
    # Getting the type of 'ValueError' (line 417)
    ValueError_24492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 417)
    ValueError_call_result_24495 = invoke(stypy.reporting.localization.Localization(__file__, 417, 14), ValueError_24492, *[str_24493], **kwargs_24494)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 417, 8), ValueError_call_result_24495, 'raise parameter', BaseException)
    # SSA join for if statement (line 416)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 's' (line 418)
    s_24496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 7), 's')
    # Obtaining the member 'ndim' of a type (line 418)
    ndim_24497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 7), s_24496, 'ndim')
    int_24498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 17), 'int')
    # Applying the binary operator '!=' (line 418)
    result_ne_24499 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 7), '!=', ndim_24497, int_24498)
    
    # Testing the type of an if condition (line 418)
    if_condition_24500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 418, 4), result_ne_24499)
    # Assigning a type to the variable 'if_condition_24500' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'if_condition_24500', if_condition_24500)
    # SSA begins for if statement (line 418)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 419)
    # Processing the call arguments (line 419)
    str_24502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 25), 'str', 'Incorrect shape for s.  s must be one-dimensional')
    # Processing the call keyword arguments (line 419)
    kwargs_24503 = {}
    # Getting the type of 'ValueError' (line 419)
    ValueError_24501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 419)
    ValueError_call_result_24504 = invoke(stypy.reporting.localization.Localization(__file__, 419, 14), ValueError_24501, *[str_24502], **kwargs_24503)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 419, 8), ValueError_call_result_24504, 'raise parameter', BaseException)
    # SSA join for if statement (line 418)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'f' (line 420)
    f_24505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 7), 'f')
    # Obtaining the member 'size' of a type (line 420)
    size_24506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 7), f_24505, 'size')
    # Getting the type of 's' (line 420)
    s_24507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 17), 's')
    # Obtaining the member 'size' of a type (line 420)
    size_24508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 17), s_24507, 'size')
    int_24509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 26), 'int')
    # Applying the binary operator '+' (line 420)
    result_add_24510 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 17), '+', size_24508, int_24509)
    
    # Applying the binary operator '!=' (line 420)
    result_ne_24511 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 7), '!=', size_24506, result_add_24510)
    
    # Testing the type of an if condition (line 420)
    if_condition_24512 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 4), result_ne_24511)
    # Assigning a type to the variable 'if_condition_24512' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'if_condition_24512', if_condition_24512)
    # SSA begins for if statement (line 420)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 421)
    # Processing the call arguments (line 421)
    str_24514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 25), 'str', 'Incorrect lengths for f and s.  The length of s must be one less than the length of f.')
    # Processing the call keyword arguments (line 421)
    kwargs_24515 = {}
    # Getting the type of 'ValueError' (line 421)
    ValueError_24513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 421)
    ValueError_call_result_24516 = invoke(stypy.reporting.localization.Localization(__file__, 421, 14), ValueError_24513, *[str_24514], **kwargs_24515)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 421, 8), ValueError_call_result_24516, 'raise parameter', BaseException)
    # SSA join for if statement (line 420)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 's' (line 423)
    s_24517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 7), 's')
    # Obtaining the member 'size' of a type (line 423)
    size_24518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 7), s_24517, 'size')
    int_24519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 17), 'int')
    # Applying the binary operator '==' (line 423)
    result_eq_24520 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 7), '==', size_24518, int_24519)
    
    # Testing the type of an if condition (line 423)
    if_condition_24521 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 4), result_eq_24520)
    # Assigning a type to the variable 'if_condition_24521' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'if_condition_24521', if_condition_24521)
    # SSA begins for if statement (line 423)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 424)
    # Processing the call arguments (line 424)
    str_24523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 25), 'str', 'The length of s must be at least 1.')
    # Processing the call keyword arguments (line 424)
    kwargs_24524 = {}
    # Getting the type of 'ValueError' (line 424)
    ValueError_24522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 424)
    ValueError_call_result_24525 = invoke(stypy.reporting.localization.Localization(__file__, 424, 14), ValueError_24522, *[str_24523], **kwargs_24524)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 424, 8), ValueError_call_result_24525, 'raise parameter', BaseException)
    # SSA join for if statement (line 423)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 426):
    
    # Assigning a BinOp to a Name (line 426):
    
    # Obtaining the type of the subscript
    int_24526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 12), 'int')
    # Getting the type of 'f' (line 426)
    f_24527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 10), 'f')
    # Obtaining the member '__getitem__' of a type (line 426)
    getitem___24528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 10), f_24527, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 426)
    subscript_call_result_24529 = invoke(stypy.reporting.localization.Localization(__file__, 426, 10), getitem___24528, int_24526)
    
    
    # Obtaining the type of the subscript
    int_24530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 19), 'int')
    # Getting the type of 's' (line 426)
    s_24531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 17), 's')
    # Obtaining the member '__getitem__' of a type (line 426)
    getitem___24532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 17), s_24531, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 426)
    subscript_call_result_24533 = invoke(stypy.reporting.localization.Localization(__file__, 426, 17), getitem___24532, int_24530)
    
    # Applying the binary operator '+' (line 426)
    result_add_24534 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 10), '+', subscript_call_result_24529, subscript_call_result_24533)
    
    # Assigning a type to the variable 'tmp' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'tmp', result_add_24534)
    
    # Assigning a Attribute to a Name (line 427):
    
    # Assigning a Attribute to a Name (line 427):
    # Getting the type of 'f' (line 427)
    f_24535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'f')
    # Obtaining the member 'size' of a type (line 427)
    size_24536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), f_24535, 'size')
    # Assigning a type to the variable 'n' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'n', size_24536)
    
    # Assigning a Call to a Name (line 428):
    
    # Assigning a Call to a Name (line 428):
    
    # Call to zeros(...): (line 428)
    # Processing the call arguments (line 428)
    
    # Obtaining an instance of the builtin type 'tuple' (line 428)
    tuple_24539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 428)
    # Adding element type (line 428)
    # Getting the type of 'n' (line 428)
    n_24540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 18), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 18), tuple_24539, n_24540)
    # Adding element type (line 428)
    # Getting the type of 'n' (line 428)
    n_24541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 21), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 18), tuple_24539, n_24541)
    
    # Processing the call keyword arguments (line 428)
    # Getting the type of 'tmp' (line 428)
    tmp_24542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 31), 'tmp', False)
    # Obtaining the member 'dtype' of a type (line 428)
    dtype_24543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 31), tmp_24542, 'dtype')
    keyword_24544 = dtype_24543
    kwargs_24545 = {'dtype': keyword_24544}
    # Getting the type of 'np' (line 428)
    np_24537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 428)
    zeros_24538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 8), np_24537, 'zeros')
    # Calling zeros(args, kwargs) (line 428)
    zeros_call_result_24546 = invoke(stypy.reporting.localization.Localization(__file__, 428, 8), zeros_24538, *[tuple_24539], **kwargs_24545)
    
    # Assigning a type to the variable 'a' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'a', zeros_call_result_24546)
    
    # Assigning a Name to a Subscript (line 429):
    
    # Assigning a Name to a Subscript (line 429):
    # Getting the type of 'f' (line 429)
    f_24547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 11), 'f')
    # Getting the type of 'a' (line 429)
    a_24548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'a')
    int_24549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 6), 'int')
    # Storing an element on a container (line 429)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 4), a_24548, (int_24549, f_24547))
    
    # Assigning a Name to a Subscript (line 430):
    
    # Assigning a Name to a Subscript (line 430):
    # Getting the type of 's' (line 430)
    s_24550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 50), 's')
    # Getting the type of 'a' (line 430)
    a_24551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'a')
    
    # Obtaining an instance of the builtin type 'tuple' (line 430)
    tuple_24552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 6), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 430)
    # Adding element type (line 430)
    
    # Call to list(...): (line 430)
    # Processing the call arguments (line 430)
    
    # Call to range(...): (line 430)
    # Processing the call arguments (line 430)
    int_24555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 17), 'int')
    # Getting the type of 'n' (line 430)
    n_24556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 20), 'n', False)
    # Processing the call keyword arguments (line 430)
    kwargs_24557 = {}
    # Getting the type of 'range' (line 430)
    range_24554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 11), 'range', False)
    # Calling range(args, kwargs) (line 430)
    range_call_result_24558 = invoke(stypy.reporting.localization.Localization(__file__, 430, 11), range_24554, *[int_24555, n_24556], **kwargs_24557)
    
    # Processing the call keyword arguments (line 430)
    kwargs_24559 = {}
    # Getting the type of 'list' (line 430)
    list_24553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 6), 'list', False)
    # Calling list(args, kwargs) (line 430)
    list_call_result_24560 = invoke(stypy.reporting.localization.Localization(__file__, 430, 6), list_24553, *[range_call_result_24558], **kwargs_24559)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 6), tuple_24552, list_call_result_24560)
    # Adding element type (line 430)
    
    # Call to list(...): (line 430)
    # Processing the call arguments (line 430)
    
    # Call to range(...): (line 430)
    # Processing the call arguments (line 430)
    int_24563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 36), 'int')
    # Getting the type of 'n' (line 430)
    n_24564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 39), 'n', False)
    int_24565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 43), 'int')
    # Applying the binary operator '-' (line 430)
    result_sub_24566 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 39), '-', n_24564, int_24565)
    
    # Processing the call keyword arguments (line 430)
    kwargs_24567 = {}
    # Getting the type of 'range' (line 430)
    range_24562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 30), 'range', False)
    # Calling range(args, kwargs) (line 430)
    range_call_result_24568 = invoke(stypy.reporting.localization.Localization(__file__, 430, 30), range_24562, *[int_24563, result_sub_24566], **kwargs_24567)
    
    # Processing the call keyword arguments (line 430)
    kwargs_24569 = {}
    # Getting the type of 'list' (line 430)
    list_24561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 25), 'list', False)
    # Calling list(args, kwargs) (line 430)
    list_call_result_24570 = invoke(stypy.reporting.localization.Localization(__file__, 430, 25), list_24561, *[range_call_result_24568], **kwargs_24569)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 6), tuple_24552, list_call_result_24570)
    
    # Storing an element on a container (line 430)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 4), a_24551, (tuple_24552, s_24550))
    # Getting the type of 'a' (line 431)
    a_24571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'stypy_return_type', a_24571)
    
    # ################# End of 'leslie(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'leslie' in the type store
    # Getting the type of 'stypy_return_type' (line 363)
    stypy_return_type_24572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24572)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'leslie'
    return stypy_return_type_24572

# Assigning a type to the variable 'leslie' (line 363)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 0), 'leslie', leslie)

@norecursion
def kron(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'kron'
    module_type_store = module_type_store.open_function_context('kron', 434, 0, False)
    
    # Passed parameters checking function
    kron.stypy_localization = localization
    kron.stypy_type_of_self = None
    kron.stypy_type_store = module_type_store
    kron.stypy_function_name = 'kron'
    kron.stypy_param_names_list = ['a', 'b']
    kron.stypy_varargs_param_name = None
    kron.stypy_kwargs_param_name = None
    kron.stypy_call_defaults = defaults
    kron.stypy_call_varargs = varargs
    kron.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'kron', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'kron', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'kron(...)' code ##################

    str_24573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, (-1)), 'str', '\n    Kronecker product.\n\n    The result is the block matrix::\n\n        a[0,0]*b    a[0,1]*b  ... a[0,-1]*b\n        a[1,0]*b    a[1,1]*b  ... a[1,-1]*b\n        ...\n        a[-1,0]*b   a[-1,1]*b ... a[-1,-1]*b\n\n    Parameters\n    ----------\n    a : (M, N) ndarray\n        Input array\n    b : (P, Q) ndarray\n        Input array\n\n    Returns\n    -------\n    A : (M*P, N*Q) ndarray\n        Kronecker product of `a` and `b`.\n\n    Examples\n    --------\n    >>> from numpy import array\n    >>> from scipy.linalg import kron\n    >>> kron(array([[1,2],[3,4]]), array([[1,1,1]]))\n    array([[1, 1, 1, 2, 2, 2],\n           [3, 3, 3, 4, 4, 4]])\n\n    ')
    
    
    
    # Obtaining the type of the subscript
    str_24574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 19), 'str', 'CONTIGUOUS')
    # Getting the type of 'a' (line 466)
    a_24575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 11), 'a')
    # Obtaining the member 'flags' of a type (line 466)
    flags_24576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 11), a_24575, 'flags')
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___24577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 11), flags_24576, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_24578 = invoke(stypy.reporting.localization.Localization(__file__, 466, 11), getitem___24577, str_24574)
    
    # Applying the 'not' unary operator (line 466)
    result_not__24579 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 7), 'not', subscript_call_result_24578)
    
    # Testing the type of an if condition (line 466)
    if_condition_24580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 466, 4), result_not__24579)
    # Assigning a type to the variable 'if_condition_24580' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'if_condition_24580', if_condition_24580)
    # SSA begins for if statement (line 466)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 467):
    
    # Assigning a Call to a Name (line 467):
    
    # Call to reshape(...): (line 467)
    # Processing the call arguments (line 467)
    # Getting the type of 'a' (line 467)
    a_24583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 23), 'a', False)
    # Getting the type of 'a' (line 467)
    a_24584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 26), 'a', False)
    # Obtaining the member 'shape' of a type (line 467)
    shape_24585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 26), a_24584, 'shape')
    # Processing the call keyword arguments (line 467)
    kwargs_24586 = {}
    # Getting the type of 'np' (line 467)
    np_24581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'np', False)
    # Obtaining the member 'reshape' of a type (line 467)
    reshape_24582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 12), np_24581, 'reshape')
    # Calling reshape(args, kwargs) (line 467)
    reshape_call_result_24587 = invoke(stypy.reporting.localization.Localization(__file__, 467, 12), reshape_24582, *[a_24583, shape_24585], **kwargs_24586)
    
    # Assigning a type to the variable 'a' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'a', reshape_call_result_24587)
    # SSA join for if statement (line 466)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    str_24588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 19), 'str', 'CONTIGUOUS')
    # Getting the type of 'b' (line 468)
    b_24589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 11), 'b')
    # Obtaining the member 'flags' of a type (line 468)
    flags_24590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 11), b_24589, 'flags')
    # Obtaining the member '__getitem__' of a type (line 468)
    getitem___24591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 11), flags_24590, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 468)
    subscript_call_result_24592 = invoke(stypy.reporting.localization.Localization(__file__, 468, 11), getitem___24591, str_24588)
    
    # Applying the 'not' unary operator (line 468)
    result_not__24593 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 7), 'not', subscript_call_result_24592)
    
    # Testing the type of an if condition (line 468)
    if_condition_24594 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 4), result_not__24593)
    # Assigning a type to the variable 'if_condition_24594' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'if_condition_24594', if_condition_24594)
    # SSA begins for if statement (line 468)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 469):
    
    # Assigning a Call to a Name (line 469):
    
    # Call to reshape(...): (line 469)
    # Processing the call arguments (line 469)
    # Getting the type of 'b' (line 469)
    b_24597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 23), 'b', False)
    # Getting the type of 'b' (line 469)
    b_24598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 26), 'b', False)
    # Obtaining the member 'shape' of a type (line 469)
    shape_24599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 26), b_24598, 'shape')
    # Processing the call keyword arguments (line 469)
    kwargs_24600 = {}
    # Getting the type of 'np' (line 469)
    np_24595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'np', False)
    # Obtaining the member 'reshape' of a type (line 469)
    reshape_24596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 12), np_24595, 'reshape')
    # Calling reshape(args, kwargs) (line 469)
    reshape_call_result_24601 = invoke(stypy.reporting.localization.Localization(__file__, 469, 12), reshape_24596, *[b_24597, shape_24599], **kwargs_24600)
    
    # Assigning a type to the variable 'b' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'b', reshape_call_result_24601)
    # SSA join for if statement (line 468)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 470):
    
    # Assigning a Call to a Name (line 470):
    
    # Call to outer(...): (line 470)
    # Processing the call arguments (line 470)
    # Getting the type of 'a' (line 470)
    a_24604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 17), 'a', False)
    # Getting the type of 'b' (line 470)
    b_24605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 20), 'b', False)
    # Processing the call keyword arguments (line 470)
    kwargs_24606 = {}
    # Getting the type of 'np' (line 470)
    np_24602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'np', False)
    # Obtaining the member 'outer' of a type (line 470)
    outer_24603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 8), np_24602, 'outer')
    # Calling outer(args, kwargs) (line 470)
    outer_call_result_24607 = invoke(stypy.reporting.localization.Localization(__file__, 470, 8), outer_24603, *[a_24604, b_24605], **kwargs_24606)
    
    # Assigning a type to the variable 'o' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'o', outer_call_result_24607)
    
    # Assigning a Call to a Name (line 471):
    
    # Assigning a Call to a Name (line 471):
    
    # Call to reshape(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'a' (line 471)
    a_24610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 18), 'a', False)
    # Obtaining the member 'shape' of a type (line 471)
    shape_24611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 18), a_24610, 'shape')
    # Getting the type of 'b' (line 471)
    b_24612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 28), 'b', False)
    # Obtaining the member 'shape' of a type (line 471)
    shape_24613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 28), b_24612, 'shape')
    # Applying the binary operator '+' (line 471)
    result_add_24614 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 18), '+', shape_24611, shape_24613)
    
    # Processing the call keyword arguments (line 471)
    kwargs_24615 = {}
    # Getting the type of 'o' (line 471)
    o_24608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'o', False)
    # Obtaining the member 'reshape' of a type (line 471)
    reshape_24609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 8), o_24608, 'reshape')
    # Calling reshape(args, kwargs) (line 471)
    reshape_call_result_24616 = invoke(stypy.reporting.localization.Localization(__file__, 471, 8), reshape_24609, *[result_add_24614], **kwargs_24615)
    
    # Assigning a type to the variable 'o' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'o', reshape_call_result_24616)
    
    # Call to concatenate(...): (line 472)
    # Processing the call arguments (line 472)
    
    # Call to concatenate(...): (line 472)
    # Processing the call arguments (line 472)
    # Getting the type of 'o' (line 472)
    o_24621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 41), 'o', False)
    # Processing the call keyword arguments (line 472)
    int_24622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 49), 'int')
    keyword_24623 = int_24622
    kwargs_24624 = {'axis': keyword_24623}
    # Getting the type of 'np' (line 472)
    np_24619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 26), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 472)
    concatenate_24620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 26), np_24619, 'concatenate')
    # Calling concatenate(args, kwargs) (line 472)
    concatenate_call_result_24625 = invoke(stypy.reporting.localization.Localization(__file__, 472, 26), concatenate_24620, *[o_24621], **kwargs_24624)
    
    # Processing the call keyword arguments (line 472)
    int_24626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 58), 'int')
    keyword_24627 = int_24626
    kwargs_24628 = {'axis': keyword_24627}
    # Getting the type of 'np' (line 472)
    np_24617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 472)
    concatenate_24618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 11), np_24617, 'concatenate')
    # Calling concatenate(args, kwargs) (line 472)
    concatenate_call_result_24629 = invoke(stypy.reporting.localization.Localization(__file__, 472, 11), concatenate_24618, *[concatenate_call_result_24625], **kwargs_24628)
    
    # Assigning a type to the variable 'stypy_return_type' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'stypy_return_type', concatenate_call_result_24629)
    
    # ################# End of 'kron(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'kron' in the type store
    # Getting the type of 'stypy_return_type' (line 434)
    stypy_return_type_24630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24630)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'kron'
    return stypy_return_type_24630

# Assigning a type to the variable 'kron' (line 434)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 0), 'kron', kron)

@norecursion
def block_diag(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'block_diag'
    module_type_store = module_type_store.open_function_context('block_diag', 475, 0, False)
    
    # Passed parameters checking function
    block_diag.stypy_localization = localization
    block_diag.stypy_type_of_self = None
    block_diag.stypy_type_store = module_type_store
    block_diag.stypy_function_name = 'block_diag'
    block_diag.stypy_param_names_list = []
    block_diag.stypy_varargs_param_name = 'arrs'
    block_diag.stypy_kwargs_param_name = None
    block_diag.stypy_call_defaults = defaults
    block_diag.stypy_call_varargs = varargs
    block_diag.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'block_diag', [], 'arrs', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'block_diag', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'block_diag(...)' code ##################

    str_24631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, (-1)), 'str', "\n    Create a block diagonal matrix from provided arrays.\n\n    Given the inputs `A`, `B` and `C`, the output will have these\n    arrays arranged on the diagonal::\n\n        [[A, 0, 0],\n         [0, B, 0],\n         [0, 0, C]]\n\n    Parameters\n    ----------\n    A, B, C, ... : array_like, up to 2-D\n        Input arrays.  A 1-D array or array_like sequence of length `n` is\n        treated as a 2-D array with shape ``(1,n)``.\n\n    Returns\n    -------\n    D : ndarray\n        Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the\n        same dtype as `A`.\n\n    Notes\n    -----\n    If all the input arrays are square, the output is known as a\n    block diagonal matrix.\n\n    Empty sequences (i.e., array-likes of zero size) will not be ignored.\n    Noteworthy, both [] and [[]] are treated as matrices with shape ``(1,0)``.\n\n    Examples\n    --------\n    >>> from scipy.linalg import block_diag\n    >>> A = [[1, 0],\n    ...      [0, 1]]\n    >>> B = [[3, 4, 5],\n    ...      [6, 7, 8]]\n    >>> C = [[7]]\n    >>> P = np.zeros((2, 0), dtype='int32')\n    >>> block_diag(A, B, C)\n    array([[1, 0, 0, 0, 0, 0],\n           [0, 1, 0, 0, 0, 0],\n           [0, 0, 3, 4, 5, 0],\n           [0, 0, 6, 7, 8, 0],\n           [0, 0, 0, 0, 0, 7]])\n    >>> block_diag(A, P, B, C)\n    array([[1, 0, 0, 0, 0, 0],\n           [0, 1, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0],\n           [0, 0, 3, 4, 5, 0],\n           [0, 0, 6, 7, 8, 0],\n           [0, 0, 0, 0, 0, 7]])\n    >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])\n    array([[ 1.,  0.,  0.,  0.,  0.],\n           [ 0.,  2.,  3.,  0.,  0.],\n           [ 0.,  0.,  0.,  4.,  5.],\n           [ 0.,  0.,  0.,  6.,  7.]])\n\n    ")
    
    
    # Getting the type of 'arrs' (line 536)
    arrs_24632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 7), 'arrs')
    
    # Obtaining an instance of the builtin type 'tuple' (line 536)
    tuple_24633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 536)
    
    # Applying the binary operator '==' (line 536)
    result_eq_24634 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 7), '==', arrs_24632, tuple_24633)
    
    # Testing the type of an if condition (line 536)
    if_condition_24635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 4), result_eq_24634)
    # Assigning a type to the variable 'if_condition_24635' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'if_condition_24635', if_condition_24635)
    # SSA begins for if statement (line 536)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 537):
    
    # Assigning a Tuple to a Name (line 537):
    
    # Obtaining an instance of the builtin type 'tuple' (line 537)
    tuple_24636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 537)
    # Adding element type (line 537)
    
    # Obtaining an instance of the builtin type 'list' (line 537)
    list_24637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 537)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 16), tuple_24636, list_24637)
    
    # Assigning a type to the variable 'arrs' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'arrs', tuple_24636)
    # SSA join for if statement (line 536)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 538):
    
    # Assigning a ListComp to a Name (line 538):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrs' (line 538)
    arrs_24643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 38), 'arrs')
    comprehension_24644 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 12), arrs_24643)
    # Assigning a type to the variable 'a' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'a', comprehension_24644)
    
    # Call to atleast_2d(...): (line 538)
    # Processing the call arguments (line 538)
    # Getting the type of 'a' (line 538)
    a_24640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 26), 'a', False)
    # Processing the call keyword arguments (line 538)
    kwargs_24641 = {}
    # Getting the type of 'np' (line 538)
    np_24638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 538)
    atleast_2d_24639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 12), np_24638, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 538)
    atleast_2d_call_result_24642 = invoke(stypy.reporting.localization.Localization(__file__, 538, 12), atleast_2d_24639, *[a_24640], **kwargs_24641)
    
    list_24645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 12), list_24645, atleast_2d_call_result_24642)
    # Assigning a type to the variable 'arrs' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'arrs', list_24645)
    
    # Assigning a ListComp to a Name (line 540):
    
    # Assigning a ListComp to a Name (line 540):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 540)
    # Processing the call arguments (line 540)
    
    # Call to len(...): (line 540)
    # Processing the call arguments (line 540)
    # Getting the type of 'arrs' (line 540)
    arrs_24656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 37), 'arrs', False)
    # Processing the call keyword arguments (line 540)
    kwargs_24657 = {}
    # Getting the type of 'len' (line 540)
    len_24655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 33), 'len', False)
    # Calling len(args, kwargs) (line 540)
    len_call_result_24658 = invoke(stypy.reporting.localization.Localization(__file__, 540, 33), len_24655, *[arrs_24656], **kwargs_24657)
    
    # Processing the call keyword arguments (line 540)
    kwargs_24659 = {}
    # Getting the type of 'range' (line 540)
    range_24654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 27), 'range', False)
    # Calling range(args, kwargs) (line 540)
    range_call_result_24660 = invoke(stypy.reporting.localization.Localization(__file__, 540, 27), range_24654, *[len_call_result_24658], **kwargs_24659)
    
    comprehension_24661 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 16), range_call_result_24660)
    # Assigning a type to the variable 'k' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 16), 'k', comprehension_24661)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 540)
    k_24647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 52), 'k')
    # Getting the type of 'arrs' (line 540)
    arrs_24648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 47), 'arrs')
    # Obtaining the member '__getitem__' of a type (line 540)
    getitem___24649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 47), arrs_24648, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 540)
    subscript_call_result_24650 = invoke(stypy.reporting.localization.Localization(__file__, 540, 47), getitem___24649, k_24647)
    
    # Obtaining the member 'ndim' of a type (line 540)
    ndim_24651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 47), subscript_call_result_24650, 'ndim')
    int_24652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 62), 'int')
    # Applying the binary operator '>' (line 540)
    result_gt_24653 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 47), '>', ndim_24651, int_24652)
    
    # Getting the type of 'k' (line 540)
    k_24646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 16), 'k')
    list_24662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 16), list_24662, k_24646)
    # Assigning a type to the variable 'bad_args' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'bad_args', list_24662)
    
    # Getting the type of 'bad_args' (line 541)
    bad_args_24663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 7), 'bad_args')
    # Testing the type of an if condition (line 541)
    if_condition_24664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 541, 4), bad_args_24663)
    # Assigning a type to the variable 'if_condition_24664' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'if_condition_24664', if_condition_24664)
    # SSA begins for if statement (line 541)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 542)
    # Processing the call arguments (line 542)
    str_24666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 25), 'str', 'arguments in the following positions have dimension greater than 2: %s')
    # Getting the type of 'bad_args' (line 543)
    bad_args_24667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 48), 'bad_args', False)
    # Applying the binary operator '%' (line 542)
    result_mod_24668 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 25), '%', str_24666, bad_args_24667)
    
    # Processing the call keyword arguments (line 542)
    kwargs_24669 = {}
    # Getting the type of 'ValueError' (line 542)
    ValueError_24665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 542)
    ValueError_call_result_24670 = invoke(stypy.reporting.localization.Localization(__file__, 542, 14), ValueError_24665, *[result_mod_24668], **kwargs_24669)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 542, 8), ValueError_call_result_24670, 'raise parameter', BaseException)
    # SSA join for if statement (line 541)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 545):
    
    # Assigning a Call to a Name (line 545):
    
    # Call to array(...): (line 545)
    # Processing the call arguments (line 545)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrs' (line 545)
    arrs_24675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 40), 'arrs', False)
    comprehension_24676 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 23), arrs_24675)
    # Assigning a type to the variable 'a' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 23), 'a', comprehension_24676)
    # Getting the type of 'a' (line 545)
    a_24673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 23), 'a', False)
    # Obtaining the member 'shape' of a type (line 545)
    shape_24674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 23), a_24673, 'shape')
    list_24677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 23), list_24677, shape_24674)
    # Processing the call keyword arguments (line 545)
    kwargs_24678 = {}
    # Getting the type of 'np' (line 545)
    np_24671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 545)
    array_24672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 13), np_24671, 'array')
    # Calling array(args, kwargs) (line 545)
    array_call_result_24679 = invoke(stypy.reporting.localization.Localization(__file__, 545, 13), array_24672, *[list_24677], **kwargs_24678)
    
    # Assigning a type to the variable 'shapes' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'shapes', array_call_result_24679)
    
    # Assigning a Call to a Name (line 546):
    
    # Assigning a Call to a Name (line 546):
    
    # Call to find_common_type(...): (line 546)
    # Processing the call arguments (line 546)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrs' (line 546)
    arrs_24684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 58), 'arrs', False)
    comprehension_24685 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 37), arrs_24684)
    # Assigning a type to the variable 'arr' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 37), 'arr', comprehension_24685)
    # Getting the type of 'arr' (line 546)
    arr_24682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 37), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 546)
    dtype_24683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 37), arr_24682, 'dtype')
    list_24686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 37), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 37), list_24686, dtype_24683)
    
    # Obtaining an instance of the builtin type 'list' (line 546)
    list_24687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 65), 'list')
    # Adding type elements to the builtin type 'list' instance (line 546)
    
    # Processing the call keyword arguments (line 546)
    kwargs_24688 = {}
    # Getting the type of 'np' (line 546)
    np_24680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'np', False)
    # Obtaining the member 'find_common_type' of a type (line 546)
    find_common_type_24681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 16), np_24680, 'find_common_type')
    # Calling find_common_type(args, kwargs) (line 546)
    find_common_type_call_result_24689 = invoke(stypy.reporting.localization.Localization(__file__, 546, 16), find_common_type_24681, *[list_24686, list_24687], **kwargs_24688)
    
    # Assigning a type to the variable 'out_dtype' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'out_dtype', find_common_type_call_result_24689)
    
    # Assigning a Call to a Name (line 547):
    
    # Assigning a Call to a Name (line 547):
    
    # Call to zeros(...): (line 547)
    # Processing the call arguments (line 547)
    
    # Call to sum(...): (line 547)
    # Processing the call arguments (line 547)
    # Getting the type of 'shapes' (line 547)
    shapes_24694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 26), 'shapes', False)
    # Processing the call keyword arguments (line 547)
    int_24695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 39), 'int')
    keyword_24696 = int_24695
    kwargs_24697 = {'axis': keyword_24696}
    # Getting the type of 'np' (line 547)
    np_24692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 19), 'np', False)
    # Obtaining the member 'sum' of a type (line 547)
    sum_24693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 19), np_24692, 'sum')
    # Calling sum(args, kwargs) (line 547)
    sum_call_result_24698 = invoke(stypy.reporting.localization.Localization(__file__, 547, 19), sum_24693, *[shapes_24694], **kwargs_24697)
    
    # Processing the call keyword arguments (line 547)
    # Getting the type of 'out_dtype' (line 547)
    out_dtype_24699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 49), 'out_dtype', False)
    keyword_24700 = out_dtype_24699
    kwargs_24701 = {'dtype': keyword_24700}
    # Getting the type of 'np' (line 547)
    np_24690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 547)
    zeros_24691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 10), np_24690, 'zeros')
    # Calling zeros(args, kwargs) (line 547)
    zeros_call_result_24702 = invoke(stypy.reporting.localization.Localization(__file__, 547, 10), zeros_24691, *[sum_call_result_24698], **kwargs_24701)
    
    # Assigning a type to the variable 'out' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'out', zeros_call_result_24702)
    
    # Assigning a Tuple to a Tuple (line 549):
    
    # Assigning a Num to a Name (line 549):
    int_24703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_24036' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_assignment_24036', int_24703)
    
    # Assigning a Num to a Name (line 549):
    int_24704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 14), 'int')
    # Assigning a type to the variable 'tuple_assignment_24037' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_assignment_24037', int_24704)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'tuple_assignment_24036' (line 549)
    tuple_assignment_24036_24705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_assignment_24036')
    # Assigning a type to the variable 'r' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'r', tuple_assignment_24036_24705)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'tuple_assignment_24037' (line 549)
    tuple_assignment_24037_24706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_assignment_24037')
    # Assigning a type to the variable 'c' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 7), 'c', tuple_assignment_24037_24706)
    
    
    # Call to enumerate(...): (line 550)
    # Processing the call arguments (line 550)
    # Getting the type of 'shapes' (line 550)
    shapes_24708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 33), 'shapes', False)
    # Processing the call keyword arguments (line 550)
    kwargs_24709 = {}
    # Getting the type of 'enumerate' (line 550)
    enumerate_24707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 23), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 550)
    enumerate_call_result_24710 = invoke(stypy.reporting.localization.Localization(__file__, 550, 23), enumerate_24707, *[shapes_24708], **kwargs_24709)
    
    # Testing the type of a for loop iterable (line 550)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 550, 4), enumerate_call_result_24710)
    # Getting the type of the for loop variable (line 550)
    for_loop_var_24711 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 550, 4), enumerate_call_result_24710)
    # Assigning a type to the variable 'i' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 4), for_loop_var_24711))
    # Assigning a type to the variable 'rr' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'rr', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 4), for_loop_var_24711))
    # Assigning a type to the variable 'cc' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'cc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 4), for_loop_var_24711))
    # SSA begins for a for statement (line 550)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Subscript (line 551):
    
    # Assigning a Subscript to a Subscript (line 551):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 551)
    i_24712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 39), 'i')
    # Getting the type of 'arrs' (line 551)
    arrs_24713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 34), 'arrs')
    # Obtaining the member '__getitem__' of a type (line 551)
    getitem___24714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 34), arrs_24713, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 551)
    subscript_call_result_24715 = invoke(stypy.reporting.localization.Localization(__file__, 551, 34), getitem___24714, i_24712)
    
    # Getting the type of 'out' (line 551)
    out_24716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'out')
    # Getting the type of 'r' (line 551)
    r_24717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'r')
    # Getting the type of 'r' (line 551)
    r_24718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 14), 'r')
    # Getting the type of 'rr' (line 551)
    rr_24719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 18), 'rr')
    # Applying the binary operator '+' (line 551)
    result_add_24720 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 14), '+', r_24718, rr_24719)
    
    slice_24721 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 551, 8), r_24717, result_add_24720, None)
    # Getting the type of 'c' (line 551)
    c_24722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 22), 'c')
    # Getting the type of 'c' (line 551)
    c_24723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 24), 'c')
    # Getting the type of 'cc' (line 551)
    cc_24724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 28), 'cc')
    # Applying the binary operator '+' (line 551)
    result_add_24725 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 24), '+', c_24723, cc_24724)
    
    slice_24726 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 551, 8), c_24722, result_add_24725, None)
    # Storing an element on a container (line 551)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 8), out_24716, ((slice_24721, slice_24726), subscript_call_result_24715))
    
    # Getting the type of 'r' (line 552)
    r_24727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'r')
    # Getting the type of 'rr' (line 552)
    rr_24728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 13), 'rr')
    # Applying the binary operator '+=' (line 552)
    result_iadd_24729 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 8), '+=', r_24727, rr_24728)
    # Assigning a type to the variable 'r' (line 552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'r', result_iadd_24729)
    
    
    # Getting the type of 'c' (line 553)
    c_24730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'c')
    # Getting the type of 'cc' (line 553)
    cc_24731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 13), 'cc')
    # Applying the binary operator '+=' (line 553)
    result_iadd_24732 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 8), '+=', c_24730, cc_24731)
    # Assigning a type to the variable 'c' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'c', result_iadd_24732)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'out' (line 554)
    out_24733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'stypy_return_type', out_24733)
    
    # ################# End of 'block_diag(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'block_diag' in the type store
    # Getting the type of 'stypy_return_type' (line 475)
    stypy_return_type_24734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24734)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'block_diag'
    return stypy_return_type_24734

# Assigning a type to the variable 'block_diag' (line 475)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 0), 'block_diag', block_diag)

@norecursion
def companion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'companion'
    module_type_store = module_type_store.open_function_context('companion', 557, 0, False)
    
    # Passed parameters checking function
    companion.stypy_localization = localization
    companion.stypy_type_of_self = None
    companion.stypy_type_store = module_type_store
    companion.stypy_function_name = 'companion'
    companion.stypy_param_names_list = ['a']
    companion.stypy_varargs_param_name = None
    companion.stypy_kwargs_param_name = None
    companion.stypy_call_defaults = defaults
    companion.stypy_call_varargs = varargs
    companion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'companion', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'companion', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'companion(...)' code ##################

    str_24735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, (-1)), 'str', '\n    Create a companion matrix.\n\n    Create the companion matrix [1]_ associated with the polynomial whose\n    coefficients are given in `a`.\n\n    Parameters\n    ----------\n    a : (N,) array_like\n        1-D array of polynomial coefficients.  The length of `a` must be\n        at least two, and ``a[0]`` must not be zero.\n\n    Returns\n    -------\n    c : (N-1, N-1) ndarray\n        The first row of `c` is ``-a[1:]/a[0]``, and the first\n        sub-diagonal is all ones.  The data-type of the array is the same\n        as the data-type of ``1.0*a[0]``.\n\n    Raises\n    ------\n    ValueError\n        If any of the following are true: a) ``a.ndim != 1``;\n        b) ``a.size < 2``; c) ``a[0] == 0``.\n\n    Notes\n    -----\n    .. versionadded:: 0.8.0\n\n    References\n    ----------\n    .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:\n        Cambridge University Press, 1999, pp. 146-7.\n\n    Examples\n    --------\n    >>> from scipy.linalg import companion\n    >>> companion([1, -10, 31, -30])\n    array([[ 10., -31.,  30.],\n           [  1.,   0.,   0.],\n           [  0.,   1.,   0.]])\n\n    ')
    
    # Assigning a Call to a Name (line 601):
    
    # Assigning a Call to a Name (line 601):
    
    # Call to atleast_1d(...): (line 601)
    # Processing the call arguments (line 601)
    # Getting the type of 'a' (line 601)
    a_24738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 22), 'a', False)
    # Processing the call keyword arguments (line 601)
    kwargs_24739 = {}
    # Getting the type of 'np' (line 601)
    np_24736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 601)
    atleast_1d_24737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 8), np_24736, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 601)
    atleast_1d_call_result_24740 = invoke(stypy.reporting.localization.Localization(__file__, 601, 8), atleast_1d_24737, *[a_24738], **kwargs_24739)
    
    # Assigning a type to the variable 'a' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'a', atleast_1d_call_result_24740)
    
    
    # Getting the type of 'a' (line 603)
    a_24741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 7), 'a')
    # Obtaining the member 'ndim' of a type (line 603)
    ndim_24742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 7), a_24741, 'ndim')
    int_24743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 17), 'int')
    # Applying the binary operator '!=' (line 603)
    result_ne_24744 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 7), '!=', ndim_24742, int_24743)
    
    # Testing the type of an if condition (line 603)
    if_condition_24745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 603, 4), result_ne_24744)
    # Assigning a type to the variable 'if_condition_24745' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'if_condition_24745', if_condition_24745)
    # SSA begins for if statement (line 603)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 604)
    # Processing the call arguments (line 604)
    str_24747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 25), 'str', 'Incorrect shape for `a`.  `a` must be one-dimensional.')
    # Processing the call keyword arguments (line 604)
    kwargs_24748 = {}
    # Getting the type of 'ValueError' (line 604)
    ValueError_24746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 604)
    ValueError_call_result_24749 = invoke(stypy.reporting.localization.Localization(__file__, 604, 14), ValueError_24746, *[str_24747], **kwargs_24748)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 604, 8), ValueError_call_result_24749, 'raise parameter', BaseException)
    # SSA join for if statement (line 603)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a' (line 607)
    a_24750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 7), 'a')
    # Obtaining the member 'size' of a type (line 607)
    size_24751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 7), a_24750, 'size')
    int_24752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 16), 'int')
    # Applying the binary operator '<' (line 607)
    result_lt_24753 = python_operator(stypy.reporting.localization.Localization(__file__, 607, 7), '<', size_24751, int_24752)
    
    # Testing the type of an if condition (line 607)
    if_condition_24754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 607, 4), result_lt_24753)
    # Assigning a type to the variable 'if_condition_24754' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'if_condition_24754', if_condition_24754)
    # SSA begins for if statement (line 607)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 608)
    # Processing the call arguments (line 608)
    str_24756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 25), 'str', 'The length of `a` must be at least 2.')
    # Processing the call keyword arguments (line 608)
    kwargs_24757 = {}
    # Getting the type of 'ValueError' (line 608)
    ValueError_24755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 608)
    ValueError_call_result_24758 = invoke(stypy.reporting.localization.Localization(__file__, 608, 14), ValueError_24755, *[str_24756], **kwargs_24757)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 608, 8), ValueError_call_result_24758, 'raise parameter', BaseException)
    # SSA join for if statement (line 607)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_24759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 9), 'int')
    # Getting the type of 'a' (line 610)
    a_24760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 7), 'a')
    # Obtaining the member '__getitem__' of a type (line 610)
    getitem___24761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 7), a_24760, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 610)
    subscript_call_result_24762 = invoke(stypy.reporting.localization.Localization(__file__, 610, 7), getitem___24761, int_24759)
    
    int_24763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 15), 'int')
    # Applying the binary operator '==' (line 610)
    result_eq_24764 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 7), '==', subscript_call_result_24762, int_24763)
    
    # Testing the type of an if condition (line 610)
    if_condition_24765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 610, 4), result_eq_24764)
    # Assigning a type to the variable 'if_condition_24765' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'if_condition_24765', if_condition_24765)
    # SSA begins for if statement (line 610)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 611)
    # Processing the call arguments (line 611)
    str_24767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 25), 'str', 'The first coefficient in `a` must not be zero.')
    # Processing the call keyword arguments (line 611)
    kwargs_24768 = {}
    # Getting the type of 'ValueError' (line 611)
    ValueError_24766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 611)
    ValueError_call_result_24769 = invoke(stypy.reporting.localization.Localization(__file__, 611, 14), ValueError_24766, *[str_24767], **kwargs_24768)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 611, 8), ValueError_call_result_24769, 'raise parameter', BaseException)
    # SSA join for if statement (line 610)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 613):
    
    # Assigning a BinOp to a Name (line 613):
    
    
    # Obtaining the type of the subscript
    int_24770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 19), 'int')
    slice_24771 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 613, 17), int_24770, None, None)
    # Getting the type of 'a' (line 613)
    a_24772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 17), 'a')
    # Obtaining the member '__getitem__' of a type (line 613)
    getitem___24773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 17), a_24772, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 613)
    subscript_call_result_24774 = invoke(stypy.reporting.localization.Localization(__file__, 613, 17), getitem___24773, slice_24771)
    
    # Applying the 'usub' unary operator (line 613)
    result___neg___24775 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 16), 'usub', subscript_call_result_24774)
    
    float_24776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 26), 'float')
    
    # Obtaining the type of the subscript
    int_24777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 34), 'int')
    # Getting the type of 'a' (line 613)
    a_24778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 32), 'a')
    # Obtaining the member '__getitem__' of a type (line 613)
    getitem___24779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 32), a_24778, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 613)
    subscript_call_result_24780 = invoke(stypy.reporting.localization.Localization(__file__, 613, 32), getitem___24779, int_24777)
    
    # Applying the binary operator '*' (line 613)
    result_mul_24781 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 26), '*', float_24776, subscript_call_result_24780)
    
    # Applying the binary operator 'div' (line 613)
    result_div_24782 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 16), 'div', result___neg___24775, result_mul_24781)
    
    # Assigning a type to the variable 'first_row' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'first_row', result_div_24782)
    
    # Assigning a Attribute to a Name (line 614):
    
    # Assigning a Attribute to a Name (line 614):
    # Getting the type of 'a' (line 614)
    a_24783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'a')
    # Obtaining the member 'size' of a type (line 614)
    size_24784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 8), a_24783, 'size')
    # Assigning a type to the variable 'n' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'n', size_24784)
    
    # Assigning a Call to a Name (line 615):
    
    # Assigning a Call to a Name (line 615):
    
    # Call to zeros(...): (line 615)
    # Processing the call arguments (line 615)
    
    # Obtaining an instance of the builtin type 'tuple' (line 615)
    tuple_24787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 615)
    # Adding element type (line 615)
    # Getting the type of 'n' (line 615)
    n_24788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 18), 'n', False)
    int_24789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 22), 'int')
    # Applying the binary operator '-' (line 615)
    result_sub_24790 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 18), '-', n_24788, int_24789)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 18), tuple_24787, result_sub_24790)
    # Adding element type (line 615)
    # Getting the type of 'n' (line 615)
    n_24791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 25), 'n', False)
    int_24792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 29), 'int')
    # Applying the binary operator '-' (line 615)
    result_sub_24793 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 25), '-', n_24791, int_24792)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 18), tuple_24787, result_sub_24793)
    
    # Processing the call keyword arguments (line 615)
    # Getting the type of 'first_row' (line 615)
    first_row_24794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 39), 'first_row', False)
    # Obtaining the member 'dtype' of a type (line 615)
    dtype_24795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 39), first_row_24794, 'dtype')
    keyword_24796 = dtype_24795
    kwargs_24797 = {'dtype': keyword_24796}
    # Getting the type of 'np' (line 615)
    np_24785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 615)
    zeros_24786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 8), np_24785, 'zeros')
    # Calling zeros(args, kwargs) (line 615)
    zeros_call_result_24798 = invoke(stypy.reporting.localization.Localization(__file__, 615, 8), zeros_24786, *[tuple_24787], **kwargs_24797)
    
    # Assigning a type to the variable 'c' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'c', zeros_call_result_24798)
    
    # Assigning a Name to a Subscript (line 616):
    
    # Assigning a Name to a Subscript (line 616):
    # Getting the type of 'first_row' (line 616)
    first_row_24799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 11), 'first_row')
    # Getting the type of 'c' (line 616)
    c_24800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 4), 'c')
    int_24801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 6), 'int')
    # Storing an element on a container (line 616)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 4), c_24800, (int_24801, first_row_24799))
    
    # Assigning a Num to a Subscript (line 617):
    
    # Assigning a Num to a Subscript (line 617):
    int_24802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 54), 'int')
    # Getting the type of 'c' (line 617)
    c_24803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 4), 'c')
    
    # Obtaining an instance of the builtin type 'tuple' (line 617)
    tuple_24804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 6), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 617)
    # Adding element type (line 617)
    
    # Call to list(...): (line 617)
    # Processing the call arguments (line 617)
    
    # Call to range(...): (line 617)
    # Processing the call arguments (line 617)
    int_24807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 17), 'int')
    # Getting the type of 'n' (line 617)
    n_24808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 20), 'n', False)
    int_24809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 24), 'int')
    # Applying the binary operator '-' (line 617)
    result_sub_24810 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 20), '-', n_24808, int_24809)
    
    # Processing the call keyword arguments (line 617)
    kwargs_24811 = {}
    # Getting the type of 'range' (line 617)
    range_24806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 11), 'range', False)
    # Calling range(args, kwargs) (line 617)
    range_call_result_24812 = invoke(stypy.reporting.localization.Localization(__file__, 617, 11), range_24806, *[int_24807, result_sub_24810], **kwargs_24811)
    
    # Processing the call keyword arguments (line 617)
    kwargs_24813 = {}
    # Getting the type of 'list' (line 617)
    list_24805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 6), 'list', False)
    # Calling list(args, kwargs) (line 617)
    list_call_result_24814 = invoke(stypy.reporting.localization.Localization(__file__, 617, 6), list_24805, *[range_call_result_24812], **kwargs_24813)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 6), tuple_24804, list_call_result_24814)
    # Adding element type (line 617)
    
    # Call to list(...): (line 617)
    # Processing the call arguments (line 617)
    
    # Call to range(...): (line 617)
    # Processing the call arguments (line 617)
    int_24817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 40), 'int')
    # Getting the type of 'n' (line 617)
    n_24818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 43), 'n', False)
    int_24819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 47), 'int')
    # Applying the binary operator '-' (line 617)
    result_sub_24820 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 43), '-', n_24818, int_24819)
    
    # Processing the call keyword arguments (line 617)
    kwargs_24821 = {}
    # Getting the type of 'range' (line 617)
    range_24816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 34), 'range', False)
    # Calling range(args, kwargs) (line 617)
    range_call_result_24822 = invoke(stypy.reporting.localization.Localization(__file__, 617, 34), range_24816, *[int_24817, result_sub_24820], **kwargs_24821)
    
    # Processing the call keyword arguments (line 617)
    kwargs_24823 = {}
    # Getting the type of 'list' (line 617)
    list_24815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 29), 'list', False)
    # Calling list(args, kwargs) (line 617)
    list_call_result_24824 = invoke(stypy.reporting.localization.Localization(__file__, 617, 29), list_24815, *[range_call_result_24822], **kwargs_24823)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 6), tuple_24804, list_call_result_24824)
    
    # Storing an element on a container (line 617)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 4), c_24803, (tuple_24804, int_24802))
    # Getting the type of 'c' (line 618)
    c_24825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'stypy_return_type', c_24825)
    
    # ################# End of 'companion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'companion' in the type store
    # Getting the type of 'stypy_return_type' (line 557)
    stypy_return_type_24826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24826)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'companion'
    return stypy_return_type_24826

# Assigning a type to the variable 'companion' (line 557)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 0), 'companion', companion)

@norecursion
def helmert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 621)
    False_24827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 20), 'False')
    defaults = [False_24827]
    # Create a new context for function 'helmert'
    module_type_store = module_type_store.open_function_context('helmert', 621, 0, False)
    
    # Passed parameters checking function
    helmert.stypy_localization = localization
    helmert.stypy_type_of_self = None
    helmert.stypy_type_store = module_type_store
    helmert.stypy_function_name = 'helmert'
    helmert.stypy_param_names_list = ['n', 'full']
    helmert.stypy_varargs_param_name = None
    helmert.stypy_kwargs_param_name = None
    helmert.stypy_call_defaults = defaults
    helmert.stypy_call_varargs = varargs
    helmert.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'helmert', ['n', 'full'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'helmert', localization, ['n', 'full'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'helmert(...)' code ##################

    str_24828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, (-1)), 'str', '\n    Create a Helmert matrix of order `n`.\n\n    This has applications in statistics, compositional or simplicial analysis,\n    and in Aitchison geometry.\n\n    Parameters\n    ----------\n    n : int\n        The size of the array to create.\n    full : bool, optional\n        If True the (n, n) ndarray will be returned.\n        Otherwise the submatrix that does not include the first\n        row will be returned.\n        Default: False.\n\n    Returns\n    -------\n    M : ndarray\n        The Helmert matrix.\n        The shape is (n, n) or (n-1, n) depending on the `full` argument.\n\n    Examples\n    --------\n    >>> from scipy.linalg import helmert\n    >>> helmert(5, full=True)\n    array([[ 0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ],\n           [ 0.70710678, -0.70710678,  0.        ,  0.        ,  0.        ],\n           [ 0.40824829,  0.40824829, -0.81649658,  0.        ,  0.        ],\n           [ 0.28867513,  0.28867513,  0.28867513, -0.8660254 ,  0.        ],\n           [ 0.2236068 ,  0.2236068 ,  0.2236068 ,  0.2236068 , -0.89442719]])\n\n    ')
    
    # Assigning a BinOp to a Name (line 655):
    
    # Assigning a BinOp to a Name (line 655):
    
    # Call to tril(...): (line 655)
    # Processing the call arguments (line 655)
    
    # Call to ones(...): (line 655)
    # Processing the call arguments (line 655)
    
    # Obtaining an instance of the builtin type 'tuple' (line 655)
    tuple_24833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 655)
    # Adding element type (line 655)
    # Getting the type of 'n' (line 655)
    n_24834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 25), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 655, 25), tuple_24833, n_24834)
    # Adding element type (line 655)
    # Getting the type of 'n' (line 655)
    n_24835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 655, 25), tuple_24833, n_24835)
    
    # Processing the call keyword arguments (line 655)
    kwargs_24836 = {}
    # Getting the type of 'np' (line 655)
    np_24831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 16), 'np', False)
    # Obtaining the member 'ones' of a type (line 655)
    ones_24832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 16), np_24831, 'ones')
    # Calling ones(args, kwargs) (line 655)
    ones_call_result_24837 = invoke(stypy.reporting.localization.Localization(__file__, 655, 16), ones_24832, *[tuple_24833], **kwargs_24836)
    
    int_24838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 33), 'int')
    # Processing the call keyword arguments (line 655)
    kwargs_24839 = {}
    # Getting the type of 'np' (line 655)
    np_24829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'np', False)
    # Obtaining the member 'tril' of a type (line 655)
    tril_24830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 8), np_24829, 'tril')
    # Calling tril(args, kwargs) (line 655)
    tril_call_result_24840 = invoke(stypy.reporting.localization.Localization(__file__, 655, 8), tril_24830, *[ones_call_result_24837, int_24838], **kwargs_24839)
    
    
    # Call to diag(...): (line 655)
    # Processing the call arguments (line 655)
    
    # Call to arange(...): (line 655)
    # Processing the call arguments (line 655)
    # Getting the type of 'n' (line 655)
    n_24845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 57), 'n', False)
    # Processing the call keyword arguments (line 655)
    kwargs_24846 = {}
    # Getting the type of 'np' (line 655)
    np_24843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 47), 'np', False)
    # Obtaining the member 'arange' of a type (line 655)
    arange_24844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 47), np_24843, 'arange')
    # Calling arange(args, kwargs) (line 655)
    arange_call_result_24847 = invoke(stypy.reporting.localization.Localization(__file__, 655, 47), arange_24844, *[n_24845], **kwargs_24846)
    
    # Processing the call keyword arguments (line 655)
    kwargs_24848 = {}
    # Getting the type of 'np' (line 655)
    np_24841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 39), 'np', False)
    # Obtaining the member 'diag' of a type (line 655)
    diag_24842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 39), np_24841, 'diag')
    # Calling diag(args, kwargs) (line 655)
    diag_call_result_24849 = invoke(stypy.reporting.localization.Localization(__file__, 655, 39), diag_24842, *[arange_call_result_24847], **kwargs_24848)
    
    # Applying the binary operator '-' (line 655)
    result_sub_24850 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 8), '-', tril_call_result_24840, diag_call_result_24849)
    
    # Assigning a type to the variable 'H' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'H', result_sub_24850)
    
    # Assigning a BinOp to a Name (line 656):
    
    # Assigning a BinOp to a Name (line 656):
    
    # Call to arange(...): (line 656)
    # Processing the call arguments (line 656)
    # Getting the type of 'n' (line 656)
    n_24853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 18), 'n', False)
    # Processing the call keyword arguments (line 656)
    kwargs_24854 = {}
    # Getting the type of 'np' (line 656)
    np_24851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 656)
    arange_24852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 8), np_24851, 'arange')
    # Calling arange(args, kwargs) (line 656)
    arange_call_result_24855 = invoke(stypy.reporting.localization.Localization(__file__, 656, 8), arange_24852, *[n_24853], **kwargs_24854)
    
    
    # Call to arange(...): (line 656)
    # Processing the call arguments (line 656)
    int_24858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 33), 'int')
    # Getting the type of 'n' (line 656)
    n_24859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 36), 'n', False)
    int_24860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 38), 'int')
    # Applying the binary operator '+' (line 656)
    result_add_24861 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 36), '+', n_24859, int_24860)
    
    # Processing the call keyword arguments (line 656)
    kwargs_24862 = {}
    # Getting the type of 'np' (line 656)
    np_24856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 23), 'np', False)
    # Obtaining the member 'arange' of a type (line 656)
    arange_24857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 23), np_24856, 'arange')
    # Calling arange(args, kwargs) (line 656)
    arange_call_result_24863 = invoke(stypy.reporting.localization.Localization(__file__, 656, 23), arange_24857, *[int_24858, result_add_24861], **kwargs_24862)
    
    # Applying the binary operator '*' (line 656)
    result_mul_24864 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 8), '*', arange_call_result_24855, arange_call_result_24863)
    
    # Assigning a type to the variable 'd' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'd', result_mul_24864)
    
    # Assigning a Num to a Subscript (line 657):
    
    # Assigning a Num to a Subscript (line 657):
    int_24865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 11), 'int')
    # Getting the type of 'H' (line 657)
    H_24866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'H')
    int_24867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 6), 'int')
    # Storing an element on a container (line 657)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 657, 4), H_24866, (int_24867, int_24865))
    
    # Assigning a Name to a Subscript (line 658):
    
    # Assigning a Name to a Subscript (line 658):
    # Getting the type of 'n' (line 658)
    n_24868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 11), 'n')
    # Getting the type of 'd' (line 658)
    d_24869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'd')
    int_24870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 6), 'int')
    # Storing an element on a container (line 658)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 4), d_24869, (int_24870, n_24868))
    
    # Assigning a BinOp to a Name (line 659):
    
    # Assigning a BinOp to a Name (line 659):
    # Getting the type of 'H' (line 659)
    H_24871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 13), 'H')
    
    # Obtaining the type of the subscript
    slice_24872 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 659, 17), None, None, None)
    # Getting the type of 'np' (line 659)
    np_24873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 31), 'np')
    # Obtaining the member 'newaxis' of a type (line 659)
    newaxis_24874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 31), np_24873, 'newaxis')
    
    # Call to sqrt(...): (line 659)
    # Processing the call arguments (line 659)
    # Getting the type of 'd' (line 659)
    d_24877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 25), 'd', False)
    # Processing the call keyword arguments (line 659)
    kwargs_24878 = {}
    # Getting the type of 'np' (line 659)
    np_24875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 17), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 659)
    sqrt_24876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 17), np_24875, 'sqrt')
    # Calling sqrt(args, kwargs) (line 659)
    sqrt_call_result_24879 = invoke(stypy.reporting.localization.Localization(__file__, 659, 17), sqrt_24876, *[d_24877], **kwargs_24878)
    
    # Obtaining the member '__getitem__' of a type (line 659)
    getitem___24880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 17), sqrt_call_result_24879, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 659)
    subscript_call_result_24881 = invoke(stypy.reporting.localization.Localization(__file__, 659, 17), getitem___24880, (slice_24872, newaxis_24874))
    
    # Applying the binary operator 'div' (line 659)
    result_div_24882 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 13), 'div', H_24871, subscript_call_result_24881)
    
    # Assigning a type to the variable 'H_full' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'H_full', result_div_24882)
    
    # Getting the type of 'full' (line 660)
    full_24883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 7), 'full')
    # Testing the type of an if condition (line 660)
    if_condition_24884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 660, 4), full_24883)
    # Assigning a type to the variable 'if_condition_24884' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'if_condition_24884', if_condition_24884)
    # SSA begins for if statement (line 660)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'H_full' (line 661)
    H_full_24885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 15), 'H_full')
    # Assigning a type to the variable 'stypy_return_type' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 8), 'stypy_return_type', H_full_24885)
    # SSA branch for the else part of an if statement (line 660)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining the type of the subscript
    int_24886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 22), 'int')
    slice_24887 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 663, 15), int_24886, None, None)
    # Getting the type of 'H_full' (line 663)
    H_full_24888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 15), 'H_full')
    # Obtaining the member '__getitem__' of a type (line 663)
    getitem___24889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 15), H_full_24888, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 663)
    subscript_call_result_24890 = invoke(stypy.reporting.localization.Localization(__file__, 663, 15), getitem___24889, slice_24887)
    
    # Assigning a type to the variable 'stypy_return_type' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'stypy_return_type', subscript_call_result_24890)
    # SSA join for if statement (line 660)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'helmert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'helmert' in the type store
    # Getting the type of 'stypy_return_type' (line 621)
    stypy_return_type_24891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24891)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'helmert'
    return stypy_return_type_24891

# Assigning a type to the variable 'helmert' (line 621)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 0), 'helmert', helmert)

@norecursion
def hilbert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hilbert'
    module_type_store = module_type_store.open_function_context('hilbert', 666, 0, False)
    
    # Passed parameters checking function
    hilbert.stypy_localization = localization
    hilbert.stypy_type_of_self = None
    hilbert.stypy_type_store = module_type_store
    hilbert.stypy_function_name = 'hilbert'
    hilbert.stypy_param_names_list = ['n']
    hilbert.stypy_varargs_param_name = None
    hilbert.stypy_kwargs_param_name = None
    hilbert.stypy_call_defaults = defaults
    hilbert.stypy_call_varargs = varargs
    hilbert.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hilbert', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hilbert', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hilbert(...)' code ##################

    str_24892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, (-1)), 'str', '\n    Create a Hilbert matrix of order `n`.\n\n    Returns the `n` by `n` array with entries `h[i,j] = 1 / (i + j + 1)`.\n\n    Parameters\n    ----------\n    n : int\n        The size of the array to create.\n\n    Returns\n    -------\n    h : (n, n) ndarray\n        The Hilbert matrix.\n\n    See Also\n    --------\n    invhilbert : Compute the inverse of a Hilbert matrix.\n\n    Notes\n    -----\n    .. versionadded:: 0.10.0\n\n    Examples\n    --------\n    >>> from scipy.linalg import hilbert\n    >>> hilbert(3)\n    array([[ 1.        ,  0.5       ,  0.33333333],\n           [ 0.5       ,  0.33333333,  0.25      ],\n           [ 0.33333333,  0.25      ,  0.2       ]])\n\n    ')
    
    # Assigning a BinOp to a Name (line 699):
    
    # Assigning a BinOp to a Name (line 699):
    float_24893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 13), 'float')
    float_24894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 20), 'float')
    
    # Call to arange(...): (line 699)
    # Processing the call arguments (line 699)
    int_24897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 36), 'int')
    # Getting the type of 'n' (line 699)
    n_24898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 40), 'n', False)
    # Applying the binary operator '*' (line 699)
    result_mul_24899 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 36), '*', int_24897, n_24898)
    
    int_24900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 44), 'int')
    # Applying the binary operator '-' (line 699)
    result_sub_24901 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 36), '-', result_mul_24899, int_24900)
    
    # Processing the call keyword arguments (line 699)
    kwargs_24902 = {}
    # Getting the type of 'np' (line 699)
    np_24895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 26), 'np', False)
    # Obtaining the member 'arange' of a type (line 699)
    arange_24896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 26), np_24895, 'arange')
    # Calling arange(args, kwargs) (line 699)
    arange_call_result_24903 = invoke(stypy.reporting.localization.Localization(__file__, 699, 26), arange_24896, *[result_sub_24901], **kwargs_24902)
    
    # Applying the binary operator '+' (line 699)
    result_add_24904 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 20), '+', float_24894, arange_call_result_24903)
    
    # Applying the binary operator 'div' (line 699)
    result_div_24905 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 13), 'div', float_24893, result_add_24904)
    
    # Assigning a type to the variable 'values' (line 699)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'values', result_div_24905)
    
    # Assigning a Call to a Name (line 700):
    
    # Assigning a Call to a Name (line 700):
    
    # Call to hankel(...): (line 700)
    # Processing the call arguments (line 700)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 700)
    n_24907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 23), 'n', False)
    slice_24908 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 700, 15), None, n_24907, None)
    # Getting the type of 'values' (line 700)
    values_24909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 15), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 700)
    getitem___24910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 15), values_24909, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 700)
    subscript_call_result_24911 = invoke(stypy.reporting.localization.Localization(__file__, 700, 15), getitem___24910, slice_24908)
    
    # Processing the call keyword arguments (line 700)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 700)
    n_24912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 36), 'n', False)
    int_24913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 40), 'int')
    # Applying the binary operator '-' (line 700)
    result_sub_24914 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 36), '-', n_24912, int_24913)
    
    slice_24915 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 700, 29), result_sub_24914, None, None)
    # Getting the type of 'values' (line 700)
    values_24916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 29), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 700)
    getitem___24917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 29), values_24916, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 700)
    subscript_call_result_24918 = invoke(stypy.reporting.localization.Localization(__file__, 700, 29), getitem___24917, slice_24915)
    
    keyword_24919 = subscript_call_result_24918
    kwargs_24920 = {'r': keyword_24919}
    # Getting the type of 'hankel' (line 700)
    hankel_24906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 8), 'hankel', False)
    # Calling hankel(args, kwargs) (line 700)
    hankel_call_result_24921 = invoke(stypy.reporting.localization.Localization(__file__, 700, 8), hankel_24906, *[subscript_call_result_24911], **kwargs_24920)
    
    # Assigning a type to the variable 'h' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'h', hankel_call_result_24921)
    # Getting the type of 'h' (line 701)
    h_24922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 11), 'h')
    # Assigning a type to the variable 'stypy_return_type' (line 701)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'stypy_return_type', h_24922)
    
    # ################# End of 'hilbert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hilbert' in the type store
    # Getting the type of 'stypy_return_type' (line 666)
    stypy_return_type_24923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24923)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hilbert'
    return stypy_return_type_24923

# Assigning a type to the variable 'hilbert' (line 666)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 0), 'hilbert', hilbert)

@norecursion
def invhilbert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 704)
    False_24924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 24), 'False')
    defaults = [False_24924]
    # Create a new context for function 'invhilbert'
    module_type_store = module_type_store.open_function_context('invhilbert', 704, 0, False)
    
    # Passed parameters checking function
    invhilbert.stypy_localization = localization
    invhilbert.stypy_type_of_self = None
    invhilbert.stypy_type_store = module_type_store
    invhilbert.stypy_function_name = 'invhilbert'
    invhilbert.stypy_param_names_list = ['n', 'exact']
    invhilbert.stypy_varargs_param_name = None
    invhilbert.stypy_kwargs_param_name = None
    invhilbert.stypy_call_defaults = defaults
    invhilbert.stypy_call_varargs = varargs
    invhilbert.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'invhilbert', ['n', 'exact'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'invhilbert', localization, ['n', 'exact'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'invhilbert(...)' code ##################

    str_24925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, (-1)), 'str', '\n    Compute the inverse of the Hilbert matrix of order `n`.\n\n    The entries in the inverse of a Hilbert matrix are integers.  When `n`\n    is greater than 14, some entries in the inverse exceed the upper limit\n    of 64 bit integers.  The `exact` argument provides two options for\n    dealing with these large integers.\n\n    Parameters\n    ----------\n    n : int\n        The order of the Hilbert matrix.\n    exact : bool, optional\n        If False, the data type of the array that is returned is np.float64,\n        and the array is an approximation of the inverse.\n        If True, the array is the exact integer inverse array.  To represent\n        the exact inverse when n > 14, the returned array is an object array\n        of long integers.  For n <= 14, the exact inverse is returned as an\n        array with data type np.int64.\n\n    Returns\n    -------\n    invh : (n, n) ndarray\n        The data type of the array is np.float64 if `exact` is False.\n        If `exact` is True, the data type is either np.int64 (for n <= 14)\n        or object (for n > 14).  In the latter case, the objects in the\n        array will be long integers.\n\n    See Also\n    --------\n    hilbert : Create a Hilbert matrix.\n\n    Notes\n    -----\n    .. versionadded:: 0.10.0\n\n    Examples\n    --------\n    >>> from scipy.linalg import invhilbert\n    >>> invhilbert(4)\n    array([[   16.,  -120.,   240.,  -140.],\n           [ -120.,  1200., -2700.,  1680.],\n           [  240., -2700.,  6480., -4200.],\n           [ -140.,  1680., -4200.,  2800.]])\n    >>> invhilbert(4, exact=True)\n    array([[   16,  -120,   240,  -140],\n           [ -120,  1200, -2700,  1680],\n           [  240, -2700,  6480, -4200],\n           [ -140,  1680, -4200,  2800]], dtype=int64)\n    >>> invhilbert(16)[7,7]\n    4.2475099528537506e+19\n    >>> invhilbert(16, exact=True)[7,7]\n    42475099528537378560L\n\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 760, 4))
    
    # 'from scipy.special import comb' statement (line 760)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_24926 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 760, 4), 'scipy.special')

    if (type(import_24926) is not StypyTypeError):

        if (import_24926 != 'pyd_module'):
            __import__(import_24926)
            sys_modules_24927 = sys.modules[import_24926]
            import_from_module(stypy.reporting.localization.Localization(__file__, 760, 4), 'scipy.special', sys_modules_24927.module_type_store, module_type_store, ['comb'])
            nest_module(stypy.reporting.localization.Localization(__file__, 760, 4), __file__, sys_modules_24927, sys_modules_24927.module_type_store, module_type_store)
        else:
            from scipy.special import comb

            import_from_module(stypy.reporting.localization.Localization(__file__, 760, 4), 'scipy.special', None, module_type_store, ['comb'], [comb])

    else:
        # Assigning a type to the variable 'scipy.special' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 4), 'scipy.special', import_24926)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    # Getting the type of 'exact' (line 761)
    exact_24928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 7), 'exact')
    # Testing the type of an if condition (line 761)
    if_condition_24929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 761, 4), exact_24928)
    # Assigning a type to the variable 'if_condition_24929' (line 761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 4), 'if_condition_24929', if_condition_24929)
    # SSA begins for if statement (line 761)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'n' (line 762)
    n_24930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 11), 'n')
    int_24931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 15), 'int')
    # Applying the binary operator '>' (line 762)
    result_gt_24932 = python_operator(stypy.reporting.localization.Localization(__file__, 762, 11), '>', n_24930, int_24931)
    
    # Testing the type of an if condition (line 762)
    if_condition_24933 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 762, 8), result_gt_24932)
    # Assigning a type to the variable 'if_condition_24933' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'if_condition_24933', if_condition_24933)
    # SSA begins for if statement (line 762)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 763):
    
    # Assigning a Name to a Name (line 763):
    # Getting the type of 'object' (line 763)
    object_24934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 20), 'object')
    # Assigning a type to the variable 'dtype' (line 763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'dtype', object_24934)
    # SSA branch for the else part of an if statement (line 762)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 765):
    
    # Assigning a Attribute to a Name (line 765):
    # Getting the type of 'np' (line 765)
    np_24935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 20), 'np')
    # Obtaining the member 'int64' of a type (line 765)
    int64_24936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 20), np_24935, 'int64')
    # Assigning a type to the variable 'dtype' (line 765)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 12), 'dtype', int64_24936)
    # SSA join for if statement (line 762)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 761)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 767):
    
    # Assigning a Attribute to a Name (line 767):
    # Getting the type of 'np' (line 767)
    np_24937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 16), 'np')
    # Obtaining the member 'float64' of a type (line 767)
    float64_24938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 16), np_24937, 'float64')
    # Assigning a type to the variable 'dtype' (line 767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 8), 'dtype', float64_24938)
    # SSA join for if statement (line 761)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 768):
    
    # Assigning a Call to a Name (line 768):
    
    # Call to empty(...): (line 768)
    # Processing the call arguments (line 768)
    
    # Obtaining an instance of the builtin type 'tuple' (line 768)
    tuple_24941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 768)
    # Adding element type (line 768)
    # Getting the type of 'n' (line 768)
    n_24942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 21), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 768, 21), tuple_24941, n_24942)
    # Adding element type (line 768)
    # Getting the type of 'n' (line 768)
    n_24943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 768, 21), tuple_24941, n_24943)
    
    # Processing the call keyword arguments (line 768)
    # Getting the type of 'dtype' (line 768)
    dtype_24944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 34), 'dtype', False)
    keyword_24945 = dtype_24944
    kwargs_24946 = {'dtype': keyword_24945}
    # Getting the type of 'np' (line 768)
    np_24939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 11), 'np', False)
    # Obtaining the member 'empty' of a type (line 768)
    empty_24940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 11), np_24939, 'empty')
    # Calling empty(args, kwargs) (line 768)
    empty_call_result_24947 = invoke(stypy.reporting.localization.Localization(__file__, 768, 11), empty_24940, *[tuple_24941], **kwargs_24946)
    
    # Assigning a type to the variable 'invh' (line 768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 4), 'invh', empty_call_result_24947)
    
    
    # Call to xrange(...): (line 769)
    # Processing the call arguments (line 769)
    # Getting the type of 'n' (line 769)
    n_24949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 20), 'n', False)
    # Processing the call keyword arguments (line 769)
    kwargs_24950 = {}
    # Getting the type of 'xrange' (line 769)
    xrange_24948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 769)
    xrange_call_result_24951 = invoke(stypy.reporting.localization.Localization(__file__, 769, 13), xrange_24948, *[n_24949], **kwargs_24950)
    
    # Testing the type of a for loop iterable (line 769)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 769, 4), xrange_call_result_24951)
    # Getting the type of the for loop variable (line 769)
    for_loop_var_24952 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 769, 4), xrange_call_result_24951)
    # Assigning a type to the variable 'i' (line 769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'i', for_loop_var_24952)
    # SSA begins for a for statement (line 769)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to xrange(...): (line 770)
    # Processing the call arguments (line 770)
    int_24954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 24), 'int')
    # Getting the type of 'i' (line 770)
    i_24955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 27), 'i', False)
    int_24956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 31), 'int')
    # Applying the binary operator '+' (line 770)
    result_add_24957 = python_operator(stypy.reporting.localization.Localization(__file__, 770, 27), '+', i_24955, int_24956)
    
    # Processing the call keyword arguments (line 770)
    kwargs_24958 = {}
    # Getting the type of 'xrange' (line 770)
    xrange_24953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 17), 'xrange', False)
    # Calling xrange(args, kwargs) (line 770)
    xrange_call_result_24959 = invoke(stypy.reporting.localization.Localization(__file__, 770, 17), xrange_24953, *[int_24954, result_add_24957], **kwargs_24958)
    
    # Testing the type of a for loop iterable (line 770)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 770, 8), xrange_call_result_24959)
    # Getting the type of the for loop variable (line 770)
    for_loop_var_24960 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 770, 8), xrange_call_result_24959)
    # Assigning a type to the variable 'j' (line 770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 8), 'j', for_loop_var_24960)
    # SSA begins for a for statement (line 770)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 771):
    
    # Assigning a BinOp to a Name (line 771):
    # Getting the type of 'i' (line 771)
    i_24961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 16), 'i')
    # Getting the type of 'j' (line 771)
    j_24962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 20), 'j')
    # Applying the binary operator '+' (line 771)
    result_add_24963 = python_operator(stypy.reporting.localization.Localization(__file__, 771, 16), '+', i_24961, j_24962)
    
    # Assigning a type to the variable 's' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 12), 's', result_add_24963)
    
    # Assigning a BinOp to a Subscript (line 772):
    
    # Assigning a BinOp to a Subscript (line 772):
    int_24964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 27), 'int')
    # Getting the type of 's' (line 772)
    s_24965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 34), 's')
    # Applying the binary operator '**' (line 772)
    result_pow_24966 = python_operator(stypy.reporting.localization.Localization(__file__, 772, 26), '**', int_24964, s_24965)
    
    # Getting the type of 's' (line 772)
    s_24967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 39), 's')
    int_24968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 43), 'int')
    # Applying the binary operator '+' (line 772)
    result_add_24969 = python_operator(stypy.reporting.localization.Localization(__file__, 772, 39), '+', s_24967, int_24968)
    
    # Applying the binary operator '*' (line 772)
    result_mul_24970 = python_operator(stypy.reporting.localization.Localization(__file__, 772, 26), '*', result_pow_24966, result_add_24969)
    
    
    # Call to comb(...): (line 773)
    # Processing the call arguments (line 773)
    # Getting the type of 'n' (line 773)
    n_24972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 31), 'n', False)
    # Getting the type of 'i' (line 773)
    i_24973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 35), 'i', False)
    # Applying the binary operator '+' (line 773)
    result_add_24974 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 31), '+', n_24972, i_24973)
    
    # Getting the type of 'n' (line 773)
    n_24975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 38), 'n', False)
    # Getting the type of 'j' (line 773)
    j_24976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 42), 'j', False)
    # Applying the binary operator '-' (line 773)
    result_sub_24977 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 38), '-', n_24975, j_24976)
    
    int_24978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 46), 'int')
    # Applying the binary operator '-' (line 773)
    result_sub_24979 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 44), '-', result_sub_24977, int_24978)
    
    # Getting the type of 'exact' (line 773)
    exact_24980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 49), 'exact', False)
    # Processing the call keyword arguments (line 773)
    kwargs_24981 = {}
    # Getting the type of 'comb' (line 773)
    comb_24971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 26), 'comb', False)
    # Calling comb(args, kwargs) (line 773)
    comb_call_result_24982 = invoke(stypy.reporting.localization.Localization(__file__, 773, 26), comb_24971, *[result_add_24974, result_sub_24979, exact_24980], **kwargs_24981)
    
    # Applying the binary operator '*' (line 772)
    result_mul_24983 = python_operator(stypy.reporting.localization.Localization(__file__, 772, 46), '*', result_mul_24970, comb_call_result_24982)
    
    
    # Call to comb(...): (line 774)
    # Processing the call arguments (line 774)
    # Getting the type of 'n' (line 774)
    n_24985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 31), 'n', False)
    # Getting the type of 'j' (line 774)
    j_24986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 35), 'j', False)
    # Applying the binary operator '+' (line 774)
    result_add_24987 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 31), '+', n_24985, j_24986)
    
    # Getting the type of 'n' (line 774)
    n_24988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 38), 'n', False)
    # Getting the type of 'i' (line 774)
    i_24989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 42), 'i', False)
    # Applying the binary operator '-' (line 774)
    result_sub_24990 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 38), '-', n_24988, i_24989)
    
    int_24991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 46), 'int')
    # Applying the binary operator '-' (line 774)
    result_sub_24992 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 44), '-', result_sub_24990, int_24991)
    
    # Getting the type of 'exact' (line 774)
    exact_24993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 49), 'exact', False)
    # Processing the call keyword arguments (line 774)
    kwargs_24994 = {}
    # Getting the type of 'comb' (line 774)
    comb_24984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 26), 'comb', False)
    # Calling comb(args, kwargs) (line 774)
    comb_call_result_24995 = invoke(stypy.reporting.localization.Localization(__file__, 774, 26), comb_24984, *[result_add_24987, result_sub_24992, exact_24993], **kwargs_24994)
    
    # Applying the binary operator '*' (line 773)
    result_mul_24996 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 56), '*', result_mul_24983, comb_call_result_24995)
    
    
    # Call to comb(...): (line 775)
    # Processing the call arguments (line 775)
    # Getting the type of 's' (line 775)
    s_24998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 31), 's', False)
    # Getting the type of 'i' (line 775)
    i_24999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 34), 'i', False)
    # Getting the type of 'exact' (line 775)
    exact_25000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 37), 'exact', False)
    # Processing the call keyword arguments (line 775)
    kwargs_25001 = {}
    # Getting the type of 'comb' (line 775)
    comb_24997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 26), 'comb', False)
    # Calling comb(args, kwargs) (line 775)
    comb_call_result_25002 = invoke(stypy.reporting.localization.Localization(__file__, 775, 26), comb_24997, *[s_24998, i_24999, exact_25000], **kwargs_25001)
    
    int_25003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 47), 'int')
    # Applying the binary operator '**' (line 775)
    result_pow_25004 = python_operator(stypy.reporting.localization.Localization(__file__, 775, 26), '**', comb_call_result_25002, int_25003)
    
    # Applying the binary operator '*' (line 774)
    result_mul_25005 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 56), '*', result_mul_24996, result_pow_25004)
    
    # Getting the type of 'invh' (line 772)
    invh_25006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 12), 'invh')
    
    # Obtaining an instance of the builtin type 'tuple' (line 772)
    tuple_25007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 772)
    # Adding element type (line 772)
    # Getting the type of 'i' (line 772)
    i_25008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 17), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 772, 17), tuple_25007, i_25008)
    # Adding element type (line 772)
    # Getting the type of 'j' (line 772)
    j_25009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 20), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 772, 17), tuple_25007, j_25009)
    
    # Storing an element on a container (line 772)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 772, 12), invh_25006, (tuple_25007, result_mul_25005))
    
    
    # Getting the type of 'i' (line 776)
    i_25010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 15), 'i')
    # Getting the type of 'j' (line 776)
    j_25011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 20), 'j')
    # Applying the binary operator '!=' (line 776)
    result_ne_25012 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 15), '!=', i_25010, j_25011)
    
    # Testing the type of an if condition (line 776)
    if_condition_25013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 776, 12), result_ne_25012)
    # Assigning a type to the variable 'if_condition_25013' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 12), 'if_condition_25013', if_condition_25013)
    # SSA begins for if statement (line 776)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 777):
    
    # Assigning a Subscript to a Subscript (line 777):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 777)
    tuple_25014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 777)
    # Adding element type (line 777)
    # Getting the type of 'i' (line 777)
    i_25015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 34), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 34), tuple_25014, i_25015)
    # Adding element type (line 777)
    # Getting the type of 'j' (line 777)
    j_25016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 37), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 34), tuple_25014, j_25016)
    
    # Getting the type of 'invh' (line 777)
    invh_25017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 29), 'invh')
    # Obtaining the member '__getitem__' of a type (line 777)
    getitem___25018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 29), invh_25017, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 777)
    subscript_call_result_25019 = invoke(stypy.reporting.localization.Localization(__file__, 777, 29), getitem___25018, tuple_25014)
    
    # Getting the type of 'invh' (line 777)
    invh_25020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 16), 'invh')
    
    # Obtaining an instance of the builtin type 'tuple' (line 777)
    tuple_25021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 777)
    # Adding element type (line 777)
    # Getting the type of 'j' (line 777)
    j_25022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 21), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 21), tuple_25021, j_25022)
    # Adding element type (line 777)
    # Getting the type of 'i' (line 777)
    i_25023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 24), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 21), tuple_25021, i_25023)
    
    # Storing an element on a container (line 777)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 16), invh_25020, (tuple_25021, subscript_call_result_25019))
    # SSA join for if statement (line 776)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'invh' (line 778)
    invh_25024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 11), 'invh')
    # Assigning a type to the variable 'stypy_return_type' (line 778)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 4), 'stypy_return_type', invh_25024)
    
    # ################# End of 'invhilbert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'invhilbert' in the type store
    # Getting the type of 'stypy_return_type' (line 704)
    stypy_return_type_25025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25025)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'invhilbert'
    return stypy_return_type_25025

# Assigning a type to the variable 'invhilbert' (line 704)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 0), 'invhilbert', invhilbert)

@norecursion
def pascal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_25026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 19), 'str', 'symmetric')
    # Getting the type of 'True' (line 781)
    True_25027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 38), 'True')
    defaults = [str_25026, True_25027]
    # Create a new context for function 'pascal'
    module_type_store = module_type_store.open_function_context('pascal', 781, 0, False)
    
    # Passed parameters checking function
    pascal.stypy_localization = localization
    pascal.stypy_type_of_self = None
    pascal.stypy_type_store = module_type_store
    pascal.stypy_function_name = 'pascal'
    pascal.stypy_param_names_list = ['n', 'kind', 'exact']
    pascal.stypy_varargs_param_name = None
    pascal.stypy_kwargs_param_name = None
    pascal.stypy_call_defaults = defaults
    pascal.stypy_call_varargs = varargs
    pascal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pascal', ['n', 'kind', 'exact'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pascal', localization, ['n', 'kind', 'exact'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pascal(...)' code ##################

    str_25028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, (-1)), 'str', "\n    Returns the n x n Pascal matrix.\n\n    The Pascal matrix is a matrix containing the binomial coefficients as\n    its elements.\n\n    Parameters\n    ----------\n    n : int\n        The size of the matrix to create; that is, the result is an n x n\n        matrix.\n    kind : str, optional\n        Must be one of 'symmetric', 'lower', or 'upper'.\n        Default is 'symmetric'.\n    exact : bool, optional\n        If `exact` is True, the result is either an array of type\n        numpy.uint64 (if n < 35) or an object array of Python long integers.\n        If `exact` is False, the coefficients in the matrix are computed using\n        `scipy.special.comb` with `exact=False`.  The result will be a floating\n        point array, and the values in the array will not be the exact\n        coefficients, but this version is much faster than `exact=True`.\n\n    Returns\n    -------\n    p : (n, n) ndarray\n        The Pascal matrix.\n\n    See Also\n    --------\n    invpascal\n\n    Notes\n    -----\n    See http://en.wikipedia.org/wiki/Pascal_matrix for more information\n    about Pascal matrices.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    >>> from scipy.linalg import pascal\n    >>> pascal(4)\n    array([[ 1,  1,  1,  1],\n           [ 1,  2,  3,  4],\n           [ 1,  3,  6, 10],\n           [ 1,  4, 10, 20]], dtype=uint64)\n    >>> pascal(4, kind='lower')\n    array([[1, 0, 0, 0],\n           [1, 1, 0, 0],\n           [1, 2, 1, 0],\n           [1, 3, 3, 1]], dtype=uint64)\n    >>> pascal(50)[-1, -1]\n    25477612258980856902730428600L\n    >>> from scipy.special import comb\n    >>> comb(98, 49, exact=True)\n    25477612258980856902730428600L\n\n    ")
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 841, 4))
    
    # 'from scipy.special import comb' statement (line 841)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_25029 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 841, 4), 'scipy.special')

    if (type(import_25029) is not StypyTypeError):

        if (import_25029 != 'pyd_module'):
            __import__(import_25029)
            sys_modules_25030 = sys.modules[import_25029]
            import_from_module(stypy.reporting.localization.Localization(__file__, 841, 4), 'scipy.special', sys_modules_25030.module_type_store, module_type_store, ['comb'])
            nest_module(stypy.reporting.localization.Localization(__file__, 841, 4), __file__, sys_modules_25030, sys_modules_25030.module_type_store, module_type_store)
        else:
            from scipy.special import comb

            import_from_module(stypy.reporting.localization.Localization(__file__, 841, 4), 'scipy.special', None, module_type_store, ['comb'], [comb])

    else:
        # Assigning a type to the variable 'scipy.special' (line 841)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 4), 'scipy.special', import_25029)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    
    # Getting the type of 'kind' (line 842)
    kind_25031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 7), 'kind')
    
    # Obtaining an instance of the builtin type 'list' (line 842)
    list_25032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 842)
    # Adding element type (line 842)
    str_25033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 20), 'str', 'symmetric')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 842, 19), list_25032, str_25033)
    # Adding element type (line 842)
    str_25034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 33), 'str', 'lower')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 842, 19), list_25032, str_25034)
    # Adding element type (line 842)
    str_25035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 42), 'str', 'upper')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 842, 19), list_25032, str_25035)
    
    # Applying the binary operator 'notin' (line 842)
    result_contains_25036 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 7), 'notin', kind_25031, list_25032)
    
    # Testing the type of an if condition (line 842)
    if_condition_25037 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 842, 4), result_contains_25036)
    # Assigning a type to the variable 'if_condition_25037' (line 842)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 4), 'if_condition_25037', if_condition_25037)
    # SSA begins for if statement (line 842)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 843)
    # Processing the call arguments (line 843)
    str_25039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 25), 'str', "kind must be 'symmetric', 'lower', or 'upper'")
    # Processing the call keyword arguments (line 843)
    kwargs_25040 = {}
    # Getting the type of 'ValueError' (line 843)
    ValueError_25038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 843)
    ValueError_call_result_25041 = invoke(stypy.reporting.localization.Localization(__file__, 843, 14), ValueError_25038, *[str_25039], **kwargs_25040)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 843, 8), ValueError_call_result_25041, 'raise parameter', BaseException)
    # SSA join for if statement (line 842)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'exact' (line 845)
    exact_25042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 7), 'exact')
    # Testing the type of an if condition (line 845)
    if_condition_25043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 845, 4), exact_25042)
    # Assigning a type to the variable 'if_condition_25043' (line 845)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 4), 'if_condition_25043', if_condition_25043)
    # SSA begins for if statement (line 845)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'n' (line 846)
    n_25044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 11), 'n')
    int_25045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 16), 'int')
    # Applying the binary operator '>=' (line 846)
    result_ge_25046 = python_operator(stypy.reporting.localization.Localization(__file__, 846, 11), '>=', n_25044, int_25045)
    
    # Testing the type of an if condition (line 846)
    if_condition_25047 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 846, 8), result_ge_25046)
    # Assigning a type to the variable 'if_condition_25047' (line 846)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'if_condition_25047', if_condition_25047)
    # SSA begins for if statement (line 846)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 847):
    
    # Assigning a Call to a Name (line 847):
    
    # Call to empty(...): (line 847)
    # Processing the call arguments (line 847)
    
    # Obtaining an instance of the builtin type 'tuple' (line 847)
    tuple_25050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 847)
    # Adding element type (line 847)
    # Getting the type of 'n' (line 847)
    n_25051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 847, 28), tuple_25050, n_25051)
    # Adding element type (line 847)
    # Getting the type of 'n' (line 847)
    n_25052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 31), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 847, 28), tuple_25050, n_25052)
    
    # Processing the call keyword arguments (line 847)
    # Getting the type of 'object' (line 847)
    object_25053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 41), 'object', False)
    keyword_25054 = object_25053
    kwargs_25055 = {'dtype': keyword_25054}
    # Getting the type of 'np' (line 847)
    np_25048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 18), 'np', False)
    # Obtaining the member 'empty' of a type (line 847)
    empty_25049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 18), np_25048, 'empty')
    # Calling empty(args, kwargs) (line 847)
    empty_call_result_25056 = invoke(stypy.reporting.localization.Localization(__file__, 847, 18), empty_25049, *[tuple_25050], **kwargs_25055)
    
    # Assigning a type to the variable 'L_n' (line 847)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 847, 12), 'L_n', empty_call_result_25056)
    
    # Call to fill(...): (line 848)
    # Processing the call arguments (line 848)
    int_25059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 21), 'int')
    # Processing the call keyword arguments (line 848)
    kwargs_25060 = {}
    # Getting the type of 'L_n' (line 848)
    L_n_25057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 12), 'L_n', False)
    # Obtaining the member 'fill' of a type (line 848)
    fill_25058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 12), L_n_25057, 'fill')
    # Calling fill(args, kwargs) (line 848)
    fill_call_result_25061 = invoke(stypy.reporting.localization.Localization(__file__, 848, 12), fill_25058, *[int_25059], **kwargs_25060)
    
    # SSA branch for the else part of an if statement (line 846)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 850):
    
    # Assigning a Call to a Name (line 850):
    
    # Call to zeros(...): (line 850)
    # Processing the call arguments (line 850)
    
    # Obtaining an instance of the builtin type 'tuple' (line 850)
    tuple_25064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 850)
    # Adding element type (line 850)
    # Getting the type of 'n' (line 850)
    n_25065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 850, 28), tuple_25064, n_25065)
    # Adding element type (line 850)
    # Getting the type of 'n' (line 850)
    n_25066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 31), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 850, 28), tuple_25064, n_25066)
    
    # Processing the call keyword arguments (line 850)
    # Getting the type of 'np' (line 850)
    np_25067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 41), 'np', False)
    # Obtaining the member 'uint64' of a type (line 850)
    uint64_25068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 41), np_25067, 'uint64')
    keyword_25069 = uint64_25068
    kwargs_25070 = {'dtype': keyword_25069}
    # Getting the type of 'np' (line 850)
    np_25062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 18), 'np', False)
    # Obtaining the member 'zeros' of a type (line 850)
    zeros_25063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 18), np_25062, 'zeros')
    # Calling zeros(args, kwargs) (line 850)
    zeros_call_result_25071 = invoke(stypy.reporting.localization.Localization(__file__, 850, 18), zeros_25063, *[tuple_25064], **kwargs_25070)
    
    # Assigning a type to the variable 'L_n' (line 850)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 12), 'L_n', zeros_call_result_25071)
    # SSA join for if statement (line 846)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 851)
    # Processing the call arguments (line 851)
    # Getting the type of 'n' (line 851)
    n_25073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 23), 'n', False)
    # Processing the call keyword arguments (line 851)
    kwargs_25074 = {}
    # Getting the type of 'range' (line 851)
    range_25072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 17), 'range', False)
    # Calling range(args, kwargs) (line 851)
    range_call_result_25075 = invoke(stypy.reporting.localization.Localization(__file__, 851, 17), range_25072, *[n_25073], **kwargs_25074)
    
    # Testing the type of a for loop iterable (line 851)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 851, 8), range_call_result_25075)
    # Getting the type of the for loop variable (line 851)
    for_loop_var_25076 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 851, 8), range_call_result_25075)
    # Assigning a type to the variable 'i' (line 851)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'i', for_loop_var_25076)
    # SSA begins for a for statement (line 851)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 852)
    # Processing the call arguments (line 852)
    # Getting the type of 'i' (line 852)
    i_25078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 27), 'i', False)
    int_25079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 31), 'int')
    # Applying the binary operator '+' (line 852)
    result_add_25080 = python_operator(stypy.reporting.localization.Localization(__file__, 852, 27), '+', i_25078, int_25079)
    
    # Processing the call keyword arguments (line 852)
    kwargs_25081 = {}
    # Getting the type of 'range' (line 852)
    range_25077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 21), 'range', False)
    # Calling range(args, kwargs) (line 852)
    range_call_result_25082 = invoke(stypy.reporting.localization.Localization(__file__, 852, 21), range_25077, *[result_add_25080], **kwargs_25081)
    
    # Testing the type of a for loop iterable (line 852)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 852, 12), range_call_result_25082)
    # Getting the type of the for loop variable (line 852)
    for_loop_var_25083 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 852, 12), range_call_result_25082)
    # Assigning a type to the variable 'j' (line 852)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 12), 'j', for_loop_var_25083)
    # SSA begins for a for statement (line 852)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 853):
    
    # Assigning a Call to a Subscript (line 853):
    
    # Call to comb(...): (line 853)
    # Processing the call arguments (line 853)
    # Getting the type of 'i' (line 853)
    i_25085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 33), 'i', False)
    # Getting the type of 'j' (line 853)
    j_25086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 36), 'j', False)
    # Processing the call keyword arguments (line 853)
    # Getting the type of 'True' (line 853)
    True_25087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 45), 'True', False)
    keyword_25088 = True_25087
    kwargs_25089 = {'exact': keyword_25088}
    # Getting the type of 'comb' (line 853)
    comb_25084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 28), 'comb', False)
    # Calling comb(args, kwargs) (line 853)
    comb_call_result_25090 = invoke(stypy.reporting.localization.Localization(__file__, 853, 28), comb_25084, *[i_25085, j_25086], **kwargs_25089)
    
    # Getting the type of 'L_n' (line 853)
    L_n_25091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 16), 'L_n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 853)
    tuple_25092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 853)
    # Adding element type (line 853)
    # Getting the type of 'i' (line 853)
    i_25093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 20), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 853, 20), tuple_25092, i_25093)
    # Adding element type (line 853)
    # Getting the type of 'j' (line 853)
    j_25094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 23), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 853, 20), tuple_25092, j_25094)
    
    # Storing an element on a container (line 853)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 853, 16), L_n_25091, (tuple_25092, comb_call_result_25090))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 845)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 855):
    
    # Assigning a Call to a Name (line 855):
    
    # Call to comb(...): (line 855)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 855)
    n_25096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 30), 'n', False)
    slice_25097 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 855, 20), None, n_25096, None)
    # Getting the type of 'n' (line 855)
    n_25098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 34), 'n', False)
    slice_25099 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 855, 20), None, n_25098, None)
    # Getting the type of 'np' (line 855)
    np_25100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 20), 'np', False)
    # Obtaining the member 'ogrid' of a type (line 855)
    ogrid_25101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 20), np_25100, 'ogrid')
    # Obtaining the member '__getitem__' of a type (line 855)
    getitem___25102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 20), ogrid_25101, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 855)
    subscript_call_result_25103 = invoke(stypy.reporting.localization.Localization(__file__, 855, 20), getitem___25102, (slice_25097, slice_25099))
    
    # Processing the call keyword arguments (line 855)
    kwargs_25104 = {}
    # Getting the type of 'comb' (line 855)
    comb_25095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 14), 'comb', False)
    # Calling comb(args, kwargs) (line 855)
    comb_call_result_25105 = invoke(stypy.reporting.localization.Localization(__file__, 855, 14), comb_25095, *[subscript_call_result_25103], **kwargs_25104)
    
    # Assigning a type to the variable 'L_n' (line 855)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 8), 'L_n', comb_call_result_25105)
    # SSA join for if statement (line 845)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'kind' (line 857)
    kind_25106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 7), 'kind')
    str_25107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, 15), 'str', 'lower')
    # Applying the binary operator '==' (line 857)
    result_eq_25108 = python_operator(stypy.reporting.localization.Localization(__file__, 857, 7), '==', kind_25106, str_25107)
    
    # Testing the type of an if condition (line 857)
    if_condition_25109 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 857, 4), result_eq_25108)
    # Assigning a type to the variable 'if_condition_25109' (line 857)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 4), 'if_condition_25109', if_condition_25109)
    # SSA begins for if statement (line 857)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 858):
    
    # Assigning a Name to a Name (line 858):
    # Getting the type of 'L_n' (line 858)
    L_n_25110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 12), 'L_n')
    # Assigning a type to the variable 'p' (line 858)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 858, 8), 'p', L_n_25110)
    # SSA branch for the else part of an if statement (line 857)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'kind' (line 859)
    kind_25111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 9), 'kind')
    str_25112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 17), 'str', 'upper')
    # Applying the binary operator '==' (line 859)
    result_eq_25113 = python_operator(stypy.reporting.localization.Localization(__file__, 859, 9), '==', kind_25111, str_25112)
    
    # Testing the type of an if condition (line 859)
    if_condition_25114 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 859, 9), result_eq_25113)
    # Assigning a type to the variable 'if_condition_25114' (line 859)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 9), 'if_condition_25114', if_condition_25114)
    # SSA begins for if statement (line 859)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 860):
    
    # Assigning a Attribute to a Name (line 860):
    # Getting the type of 'L_n' (line 860)
    L_n_25115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 12), 'L_n')
    # Obtaining the member 'T' of a type (line 860)
    T_25116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 12), L_n_25115, 'T')
    # Assigning a type to the variable 'p' (line 860)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 8), 'p', T_25116)
    # SSA branch for the else part of an if statement (line 859)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 862):
    
    # Assigning a Call to a Name (line 862):
    
    # Call to dot(...): (line 862)
    # Processing the call arguments (line 862)
    # Getting the type of 'L_n' (line 862)
    L_n_25119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 19), 'L_n', False)
    # Getting the type of 'L_n' (line 862)
    L_n_25120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 24), 'L_n', False)
    # Obtaining the member 'T' of a type (line 862)
    T_25121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 24), L_n_25120, 'T')
    # Processing the call keyword arguments (line 862)
    kwargs_25122 = {}
    # Getting the type of 'np' (line 862)
    np_25117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 12), 'np', False)
    # Obtaining the member 'dot' of a type (line 862)
    dot_25118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 12), np_25117, 'dot')
    # Calling dot(args, kwargs) (line 862)
    dot_call_result_25123 = invoke(stypy.reporting.localization.Localization(__file__, 862, 12), dot_25118, *[L_n_25119, T_25121], **kwargs_25122)
    
    # Assigning a type to the variable 'p' (line 862)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 8), 'p', dot_call_result_25123)
    # SSA join for if statement (line 859)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 857)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'p' (line 864)
    p_25124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 11), 'p')
    # Assigning a type to the variable 'stypy_return_type' (line 864)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 4), 'stypy_return_type', p_25124)
    
    # ################# End of 'pascal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pascal' in the type store
    # Getting the type of 'stypy_return_type' (line 781)
    stypy_return_type_25125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25125)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pascal'
    return stypy_return_type_25125

# Assigning a type to the variable 'pascal' (line 781)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 0), 'pascal', pascal)

@norecursion
def invpascal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_25126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, 22), 'str', 'symmetric')
    # Getting the type of 'True' (line 867)
    True_25127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 41), 'True')
    defaults = [str_25126, True_25127]
    # Create a new context for function 'invpascal'
    module_type_store = module_type_store.open_function_context('invpascal', 867, 0, False)
    
    # Passed parameters checking function
    invpascal.stypy_localization = localization
    invpascal.stypy_type_of_self = None
    invpascal.stypy_type_store = module_type_store
    invpascal.stypy_function_name = 'invpascal'
    invpascal.stypy_param_names_list = ['n', 'kind', 'exact']
    invpascal.stypy_varargs_param_name = None
    invpascal.stypy_kwargs_param_name = None
    invpascal.stypy_call_defaults = defaults
    invpascal.stypy_call_varargs = varargs
    invpascal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'invpascal', ['n', 'kind', 'exact'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'invpascal', localization, ['n', 'kind', 'exact'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'invpascal(...)' code ##################

    str_25128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, (-1)), 'str', '\n    Returns the inverse of the n x n Pascal matrix.\n\n    The Pascal matrix is a matrix containing the binomial coefficients as\n    its elements.\n\n    Parameters\n    ----------\n    n : int\n        The size of the matrix to create; that is, the result is an n x n\n        matrix.\n    kind : str, optional\n        Must be one of \'symmetric\', \'lower\', or \'upper\'.\n        Default is \'symmetric\'.\n    exact : bool, optional\n        If `exact` is True, the result is either an array of type\n        `numpy.int64` (if `n` <= 35) or an object array of Python integers.\n        If `exact` is False, the coefficients in the matrix are computed using\n        `scipy.special.comb` with `exact=False`.  The result will be a floating\n        point array, and for large `n`, the values in the array will not be the\n        exact coefficients.\n\n    Returns\n    -------\n    invp : (n, n) ndarray\n        The inverse of the Pascal matrix.\n\n    See Also\n    --------\n    pascal\n\n    Notes\n    -----\n\n    .. versionadded:: 0.16.0\n\n    References\n    ----------\n    .. [1] "Pascal matrix",  http://en.wikipedia.org/wiki/Pascal_matrix\n    .. [2] Cohen, A. M., "The inverse of a Pascal matrix", Mathematical\n           Gazette, 59(408), pp. 111-112, 1975.\n\n    Examples\n    --------\n    >>> from scipy.linalg import invpascal, pascal\n    >>> invp = invpascal(5)\n    >>> invp\n    array([[  5, -10,  10,  -5,   1],\n           [-10,  30, -35,  19,  -4],\n           [ 10, -35,  46, -27,   6],\n           [ -5,  19, -27,  17,  -4],\n           [  1,  -4,   6,  -4,   1]])\n\n    >>> p = pascal(5)\n    >>> p.dot(invp)\n    array([[ 1.,  0.,  0.,  0.,  0.],\n           [ 0.,  1.,  0.,  0.,  0.],\n           [ 0.,  0.,  1.,  0.,  0.],\n           [ 0.,  0.,  0.,  1.,  0.],\n           [ 0.,  0.,  0.,  0.,  1.]])\n\n    An example of the use of `kind` and `exact`:\n\n    >>> invpascal(5, kind=\'lower\', exact=False)\n    array([[ 1., -0.,  0., -0.,  0.],\n           [-1.,  1., -0.,  0., -0.],\n           [ 1., -2.,  1., -0.,  0.],\n           [-1.,  3., -3.,  1., -0.],\n           [ 1., -4.,  6., -4.,  1.]])\n\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 939, 4))
    
    # 'from scipy.special import comb' statement (line 939)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_25129 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 939, 4), 'scipy.special')

    if (type(import_25129) is not StypyTypeError):

        if (import_25129 != 'pyd_module'):
            __import__(import_25129)
            sys_modules_25130 = sys.modules[import_25129]
            import_from_module(stypy.reporting.localization.Localization(__file__, 939, 4), 'scipy.special', sys_modules_25130.module_type_store, module_type_store, ['comb'])
            nest_module(stypy.reporting.localization.Localization(__file__, 939, 4), __file__, sys_modules_25130, sys_modules_25130.module_type_store, module_type_store)
        else:
            from scipy.special import comb

            import_from_module(stypy.reporting.localization.Localization(__file__, 939, 4), 'scipy.special', None, module_type_store, ['comb'], [comb])

    else:
        # Assigning a type to the variable 'scipy.special' (line 939)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 4), 'scipy.special', import_25129)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    
    # Getting the type of 'kind' (line 941)
    kind_25131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 7), 'kind')
    
    # Obtaining an instance of the builtin type 'list' (line 941)
    list_25132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 941)
    # Adding element type (line 941)
    str_25133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 20), 'str', 'symmetric')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 941, 19), list_25132, str_25133)
    # Adding element type (line 941)
    str_25134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 33), 'str', 'lower')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 941, 19), list_25132, str_25134)
    # Adding element type (line 941)
    str_25135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 42), 'str', 'upper')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 941, 19), list_25132, str_25135)
    
    # Applying the binary operator 'notin' (line 941)
    result_contains_25136 = python_operator(stypy.reporting.localization.Localization(__file__, 941, 7), 'notin', kind_25131, list_25132)
    
    # Testing the type of an if condition (line 941)
    if_condition_25137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 941, 4), result_contains_25136)
    # Assigning a type to the variable 'if_condition_25137' (line 941)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 941, 4), 'if_condition_25137', if_condition_25137)
    # SSA begins for if statement (line 941)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 942)
    # Processing the call arguments (line 942)
    str_25139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 25), 'str', "'kind' must be 'symmetric', 'lower' or 'upper'.")
    # Processing the call keyword arguments (line 942)
    kwargs_25140 = {}
    # Getting the type of 'ValueError' (line 942)
    ValueError_25138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 942)
    ValueError_call_result_25141 = invoke(stypy.reporting.localization.Localization(__file__, 942, 14), ValueError_25138, *[str_25139], **kwargs_25140)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 942, 8), ValueError_call_result_25141, 'raise parameter', BaseException)
    # SSA join for if statement (line 941)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'kind' (line 944)
    kind_25142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 7), 'kind')
    str_25143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 15), 'str', 'symmetric')
    # Applying the binary operator '==' (line 944)
    result_eq_25144 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 7), '==', kind_25142, str_25143)
    
    # Testing the type of an if condition (line 944)
    if_condition_25145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 944, 4), result_eq_25144)
    # Assigning a type to the variable 'if_condition_25145' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 4), 'if_condition_25145', if_condition_25145)
    # SSA begins for if statement (line 944)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'exact' (line 945)
    exact_25146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 11), 'exact')
    # Testing the type of an if condition (line 945)
    if_condition_25147 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 945, 8), exact_25146)
    # Assigning a type to the variable 'if_condition_25147' (line 945)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 945, 8), 'if_condition_25147', if_condition_25147)
    # SSA begins for if statement (line 945)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'n' (line 946)
    n_25148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 15), 'n')
    int_25149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 19), 'int')
    # Applying the binary operator '>' (line 946)
    result_gt_25150 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 15), '>', n_25148, int_25149)
    
    # Testing the type of an if condition (line 946)
    if_condition_25151 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 946, 12), result_gt_25150)
    # Assigning a type to the variable 'if_condition_25151' (line 946)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 12), 'if_condition_25151', if_condition_25151)
    # SSA begins for if statement (line 946)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 947):
    
    # Assigning a Name to a Name (line 947):
    # Getting the type of 'object' (line 947)
    object_25152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 21), 'object')
    # Assigning a type to the variable 'dt' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 16), 'dt', object_25152)
    # SSA branch for the else part of an if statement (line 946)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 949):
    
    # Assigning a Attribute to a Name (line 949):
    # Getting the type of 'np' (line 949)
    np_25153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 21), 'np')
    # Obtaining the member 'int64' of a type (line 949)
    int64_25154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 949, 21), np_25153, 'int64')
    # Assigning a type to the variable 'dt' (line 949)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 949, 16), 'dt', int64_25154)
    # SSA join for if statement (line 946)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 945)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 951):
    
    # Assigning a Attribute to a Name (line 951):
    # Getting the type of 'np' (line 951)
    np_25155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 17), 'np')
    # Obtaining the member 'float64' of a type (line 951)
    float64_25156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 17), np_25155, 'float64')
    # Assigning a type to the variable 'dt' (line 951)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 12), 'dt', float64_25156)
    # SSA join for if statement (line 945)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 952):
    
    # Assigning a Call to a Name (line 952):
    
    # Call to empty(...): (line 952)
    # Processing the call arguments (line 952)
    
    # Obtaining an instance of the builtin type 'tuple' (line 952)
    tuple_25159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 952)
    # Adding element type (line 952)
    # Getting the type of 'n' (line 952)
    n_25160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 25), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 952, 25), tuple_25159, n_25160)
    # Adding element type (line 952)
    # Getting the type of 'n' (line 952)
    n_25161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 952, 25), tuple_25159, n_25161)
    
    # Processing the call keyword arguments (line 952)
    # Getting the type of 'dt' (line 952)
    dt_25162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 38), 'dt', False)
    keyword_25163 = dt_25162
    kwargs_25164 = {'dtype': keyword_25163}
    # Getting the type of 'np' (line 952)
    np_25157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 15), 'np', False)
    # Obtaining the member 'empty' of a type (line 952)
    empty_25158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 15), np_25157, 'empty')
    # Calling empty(args, kwargs) (line 952)
    empty_call_result_25165 = invoke(stypy.reporting.localization.Localization(__file__, 952, 15), empty_25158, *[tuple_25159], **kwargs_25164)
    
    # Assigning a type to the variable 'invp' (line 952)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'invp', empty_call_result_25165)
    
    
    # Call to range(...): (line 953)
    # Processing the call arguments (line 953)
    # Getting the type of 'n' (line 953)
    n_25167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 23), 'n', False)
    # Processing the call keyword arguments (line 953)
    kwargs_25168 = {}
    # Getting the type of 'range' (line 953)
    range_25166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 17), 'range', False)
    # Calling range(args, kwargs) (line 953)
    range_call_result_25169 = invoke(stypy.reporting.localization.Localization(__file__, 953, 17), range_25166, *[n_25167], **kwargs_25168)
    
    # Testing the type of a for loop iterable (line 953)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 953, 8), range_call_result_25169)
    # Getting the type of the for loop variable (line 953)
    for_loop_var_25170 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 953, 8), range_call_result_25169)
    # Assigning a type to the variable 'i' (line 953)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 8), 'i', for_loop_var_25170)
    # SSA begins for a for statement (line 953)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 954)
    # Processing the call arguments (line 954)
    int_25172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 27), 'int')
    # Getting the type of 'i' (line 954)
    i_25173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 30), 'i', False)
    int_25174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 34), 'int')
    # Applying the binary operator '+' (line 954)
    result_add_25175 = python_operator(stypy.reporting.localization.Localization(__file__, 954, 30), '+', i_25173, int_25174)
    
    # Processing the call keyword arguments (line 954)
    kwargs_25176 = {}
    # Getting the type of 'range' (line 954)
    range_25171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 21), 'range', False)
    # Calling range(args, kwargs) (line 954)
    range_call_result_25177 = invoke(stypy.reporting.localization.Localization(__file__, 954, 21), range_25171, *[int_25172, result_add_25175], **kwargs_25176)
    
    # Testing the type of a for loop iterable (line 954)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 954, 12), range_call_result_25177)
    # Getting the type of the for loop variable (line 954)
    for_loop_var_25178 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 954, 12), range_call_result_25177)
    # Assigning a type to the variable 'j' (line 954)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 12), 'j', for_loop_var_25178)
    # SSA begins for a for statement (line 954)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Num to a Name (line 955):
    
    # Assigning a Num to a Name (line 955):
    int_25179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 20), 'int')
    # Assigning a type to the variable 'v' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 16), 'v', int_25179)
    
    
    # Call to range(...): (line 956)
    # Processing the call arguments (line 956)
    # Getting the type of 'n' (line 956)
    n_25181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 31), 'n', False)
    # Getting the type of 'i' (line 956)
    i_25182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 35), 'i', False)
    # Applying the binary operator '-' (line 956)
    result_sub_25183 = python_operator(stypy.reporting.localization.Localization(__file__, 956, 31), '-', n_25181, i_25182)
    
    # Processing the call keyword arguments (line 956)
    kwargs_25184 = {}
    # Getting the type of 'range' (line 956)
    range_25180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 25), 'range', False)
    # Calling range(args, kwargs) (line 956)
    range_call_result_25185 = invoke(stypy.reporting.localization.Localization(__file__, 956, 25), range_25180, *[result_sub_25183], **kwargs_25184)
    
    # Testing the type of a for loop iterable (line 956)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 956, 16), range_call_result_25185)
    # Getting the type of the for loop variable (line 956)
    for_loop_var_25186 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 956, 16), range_call_result_25185)
    # Assigning a type to the variable 'k' (line 956)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 956, 16), 'k', for_loop_var_25186)
    # SSA begins for a for statement (line 956)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'v' (line 957)
    v_25187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 20), 'v')
    
    # Call to comb(...): (line 957)
    # Processing the call arguments (line 957)
    # Getting the type of 'i' (line 957)
    i_25189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 30), 'i', False)
    # Getting the type of 'k' (line 957)
    k_25190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 34), 'k', False)
    # Applying the binary operator '+' (line 957)
    result_add_25191 = python_operator(stypy.reporting.localization.Localization(__file__, 957, 30), '+', i_25189, k_25190)
    
    # Getting the type of 'k' (line 957)
    k_25192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 37), 'k', False)
    # Processing the call keyword arguments (line 957)
    # Getting the type of 'exact' (line 957)
    exact_25193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 46), 'exact', False)
    keyword_25194 = exact_25193
    kwargs_25195 = {'exact': keyword_25194}
    # Getting the type of 'comb' (line 957)
    comb_25188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 25), 'comb', False)
    # Calling comb(args, kwargs) (line 957)
    comb_call_result_25196 = invoke(stypy.reporting.localization.Localization(__file__, 957, 25), comb_25188, *[result_add_25191, k_25192], **kwargs_25195)
    
    
    # Call to comb(...): (line 957)
    # Processing the call arguments (line 957)
    # Getting the type of 'i' (line 957)
    i_25198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 60), 'i', False)
    # Getting the type of 'k' (line 957)
    k_25199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 64), 'k', False)
    # Applying the binary operator '+' (line 957)
    result_add_25200 = python_operator(stypy.reporting.localization.Localization(__file__, 957, 60), '+', i_25198, k_25199)
    
    # Getting the type of 'i' (line 957)
    i_25201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 67), 'i', False)
    # Getting the type of 'k' (line 957)
    k_25202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 71), 'k', False)
    # Applying the binary operator '+' (line 957)
    result_add_25203 = python_operator(stypy.reporting.localization.Localization(__file__, 957, 67), '+', i_25201, k_25202)
    
    # Getting the type of 'j' (line 957)
    j_25204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 75), 'j', False)
    # Applying the binary operator '-' (line 957)
    result_sub_25205 = python_operator(stypy.reporting.localization.Localization(__file__, 957, 73), '-', result_add_25203, j_25204)
    
    # Processing the call keyword arguments (line 957)
    # Getting the type of 'exact' (line 958)
    exact_25206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 66), 'exact', False)
    keyword_25207 = exact_25206
    kwargs_25208 = {'exact': keyword_25207}
    # Getting the type of 'comb' (line 957)
    comb_25197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 55), 'comb', False)
    # Calling comb(args, kwargs) (line 957)
    comb_call_result_25209 = invoke(stypy.reporting.localization.Localization(__file__, 957, 55), comb_25197, *[result_add_25200, result_sub_25205], **kwargs_25208)
    
    # Applying the binary operator '*' (line 957)
    result_mul_25210 = python_operator(stypy.reporting.localization.Localization(__file__, 957, 25), '*', comb_call_result_25196, comb_call_result_25209)
    
    # Applying the binary operator '+=' (line 957)
    result_iadd_25211 = python_operator(stypy.reporting.localization.Localization(__file__, 957, 20), '+=', v_25187, result_mul_25210)
    # Assigning a type to the variable 'v' (line 957)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 957, 20), 'v', result_iadd_25211)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 959):
    
    # Assigning a BinOp to a Subscript (line 959):
    int_25212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 30), 'int')
    # Getting the type of 'i' (line 959)
    i_25213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 36), 'i')
    # Getting the type of 'j' (line 959)
    j_25214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 40), 'j')
    # Applying the binary operator '-' (line 959)
    result_sub_25215 = python_operator(stypy.reporting.localization.Localization(__file__, 959, 36), '-', i_25213, j_25214)
    
    # Applying the binary operator '**' (line 959)
    result_pow_25216 = python_operator(stypy.reporting.localization.Localization(__file__, 959, 29), '**', int_25212, result_sub_25215)
    
    # Getting the type of 'v' (line 959)
    v_25217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 45), 'v')
    # Applying the binary operator '*' (line 959)
    result_mul_25218 = python_operator(stypy.reporting.localization.Localization(__file__, 959, 29), '*', result_pow_25216, v_25217)
    
    # Getting the type of 'invp' (line 959)
    invp_25219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 16), 'invp')
    
    # Obtaining an instance of the builtin type 'tuple' (line 959)
    tuple_25220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 959)
    # Adding element type (line 959)
    # Getting the type of 'i' (line 959)
    i_25221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 21), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 959, 21), tuple_25220, i_25221)
    # Adding element type (line 959)
    # Getting the type of 'j' (line 959)
    j_25222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 24), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 959, 21), tuple_25220, j_25222)
    
    # Storing an element on a container (line 959)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 959, 16), invp_25219, (tuple_25220, result_mul_25218))
    
    
    # Getting the type of 'i' (line 960)
    i_25223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 19), 'i')
    # Getting the type of 'j' (line 960)
    j_25224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 24), 'j')
    # Applying the binary operator '!=' (line 960)
    result_ne_25225 = python_operator(stypy.reporting.localization.Localization(__file__, 960, 19), '!=', i_25223, j_25224)
    
    # Testing the type of an if condition (line 960)
    if_condition_25226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 960, 16), result_ne_25225)
    # Assigning a type to the variable 'if_condition_25226' (line 960)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 16), 'if_condition_25226', if_condition_25226)
    # SSA begins for if statement (line 960)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 961):
    
    # Assigning a Subscript to a Subscript (line 961):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 961)
    tuple_25227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 961)
    # Adding element type (line 961)
    # Getting the type of 'i' (line 961)
    i_25228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 38), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 961, 38), tuple_25227, i_25228)
    # Adding element type (line 961)
    # Getting the type of 'j' (line 961)
    j_25229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 41), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 961, 38), tuple_25227, j_25229)
    
    # Getting the type of 'invp' (line 961)
    invp_25230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 33), 'invp')
    # Obtaining the member '__getitem__' of a type (line 961)
    getitem___25231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 33), invp_25230, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 961)
    subscript_call_result_25232 = invoke(stypy.reporting.localization.Localization(__file__, 961, 33), getitem___25231, tuple_25227)
    
    # Getting the type of 'invp' (line 961)
    invp_25233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 20), 'invp')
    
    # Obtaining an instance of the builtin type 'tuple' (line 961)
    tuple_25234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 961)
    # Adding element type (line 961)
    # Getting the type of 'j' (line 961)
    j_25235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 25), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 961, 25), tuple_25234, j_25235)
    # Adding element type (line 961)
    # Getting the type of 'i' (line 961)
    i_25236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 28), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 961, 25), tuple_25234, i_25236)
    
    # Storing an element on a container (line 961)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 961, 20), invp_25233, (tuple_25234, subscript_call_result_25232))
    # SSA join for if statement (line 960)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 944)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 965):
    
    # Assigning a Call to a Name (line 965):
    
    # Call to pascal(...): (line 965)
    # Processing the call arguments (line 965)
    # Getting the type of 'n' (line 965)
    n_25238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 22), 'n', False)
    # Processing the call keyword arguments (line 965)
    # Getting the type of 'kind' (line 965)
    kind_25239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 30), 'kind', False)
    keyword_25240 = kind_25239
    # Getting the type of 'exact' (line 965)
    exact_25241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 42), 'exact', False)
    keyword_25242 = exact_25241
    kwargs_25243 = {'kind': keyword_25240, 'exact': keyword_25242}
    # Getting the type of 'pascal' (line 965)
    pascal_25237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 15), 'pascal', False)
    # Calling pascal(args, kwargs) (line 965)
    pascal_call_result_25244 = invoke(stypy.reporting.localization.Localization(__file__, 965, 15), pascal_25237, *[n_25238], **kwargs_25243)
    
    # Assigning a type to the variable 'invp' (line 965)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 8), 'invp', pascal_call_result_25244)
    
    
    # Getting the type of 'invp' (line 966)
    invp_25245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 11), 'invp')
    # Obtaining the member 'dtype' of a type (line 966)
    dtype_25246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 11), invp_25245, 'dtype')
    # Getting the type of 'np' (line 966)
    np_25247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 25), 'np')
    # Obtaining the member 'uint64' of a type (line 966)
    uint64_25248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 25), np_25247, 'uint64')
    # Applying the binary operator '==' (line 966)
    result_eq_25249 = python_operator(stypy.reporting.localization.Localization(__file__, 966, 11), '==', dtype_25246, uint64_25248)
    
    # Testing the type of an if condition (line 966)
    if_condition_25250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 966, 8), result_eq_25249)
    # Assigning a type to the variable 'if_condition_25250' (line 966)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 8), 'if_condition_25250', if_condition_25250)
    # SSA begins for if statement (line 966)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 969):
    
    # Assigning a Call to a Name (line 969):
    
    # Call to view(...): (line 969)
    # Processing the call arguments (line 969)
    # Getting the type of 'np' (line 969)
    np_25253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 29), 'np', False)
    # Obtaining the member 'int64' of a type (line 969)
    int64_25254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 969, 29), np_25253, 'int64')
    # Processing the call keyword arguments (line 969)
    kwargs_25255 = {}
    # Getting the type of 'invp' (line 969)
    invp_25251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 19), 'invp', False)
    # Obtaining the member 'view' of a type (line 969)
    view_25252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 969, 19), invp_25251, 'view')
    # Calling view(args, kwargs) (line 969)
    view_call_result_25256 = invoke(stypy.reporting.localization.Localization(__file__, 969, 19), view_25252, *[int64_25254], **kwargs_25255)
    
    # Assigning a type to the variable 'invp' (line 969)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 12), 'invp', view_call_result_25256)
    # SSA join for if statement (line 966)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'invp' (line 972)
    invp_25257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 8), 'invp')
    
    # Call to astype(...): (line 972)
    # Processing the call arguments (line 972)
    # Getting the type of 'invp' (line 972)
    invp_25269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 52), 'invp', False)
    # Obtaining the member 'dtype' of a type (line 972)
    dtype_25270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 52), invp_25269, 'dtype')
    # Processing the call keyword arguments (line 972)
    kwargs_25271 = {}
    
    # Call to toeplitz(...): (line 972)
    # Processing the call arguments (line 972)
    int_25259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 26), 'int')
    
    # Call to arange(...): (line 972)
    # Processing the call arguments (line 972)
    # Getting the type of 'n' (line 972)
    n_25262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 41), 'n', False)
    # Processing the call keyword arguments (line 972)
    kwargs_25263 = {}
    # Getting the type of 'np' (line 972)
    np_25260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 31), 'np', False)
    # Obtaining the member 'arange' of a type (line 972)
    arange_25261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 31), np_25260, 'arange')
    # Calling arange(args, kwargs) (line 972)
    arange_call_result_25264 = invoke(stypy.reporting.localization.Localization(__file__, 972, 31), arange_25261, *[n_25262], **kwargs_25263)
    
    # Applying the binary operator '**' (line 972)
    result_pow_25265 = python_operator(stypy.reporting.localization.Localization(__file__, 972, 25), '**', int_25259, arange_call_result_25264)
    
    # Processing the call keyword arguments (line 972)
    kwargs_25266 = {}
    # Getting the type of 'toeplitz' (line 972)
    toeplitz_25258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 16), 'toeplitz', False)
    # Calling toeplitz(args, kwargs) (line 972)
    toeplitz_call_result_25267 = invoke(stypy.reporting.localization.Localization(__file__, 972, 16), toeplitz_25258, *[result_pow_25265], **kwargs_25266)
    
    # Obtaining the member 'astype' of a type (line 972)
    astype_25268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 16), toeplitz_call_result_25267, 'astype')
    # Calling astype(args, kwargs) (line 972)
    astype_call_result_25272 = invoke(stypy.reporting.localization.Localization(__file__, 972, 16), astype_25268, *[dtype_25270], **kwargs_25271)
    
    # Applying the binary operator '*=' (line 972)
    result_imul_25273 = python_operator(stypy.reporting.localization.Localization(__file__, 972, 8), '*=', invp_25257, astype_call_result_25272)
    # Assigning a type to the variable 'invp' (line 972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 8), 'invp', result_imul_25273)
    
    # SSA join for if statement (line 944)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'invp' (line 974)
    invp_25274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 11), 'invp')
    # Assigning a type to the variable 'stypy_return_type' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 4), 'stypy_return_type', invp_25274)
    
    # ################# End of 'invpascal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'invpascal' in the type store
    # Getting the type of 'stypy_return_type' (line 867)
    stypy_return_type_25275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25275)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'invpascal'
    return stypy_return_type_25275

# Assigning a type to the variable 'invpascal' (line 867)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 0), 'invpascal', invpascal)

@norecursion
def dft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 977)
    None_25276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 17), 'None')
    defaults = [None_25276]
    # Create a new context for function 'dft'
    module_type_store = module_type_store.open_function_context('dft', 977, 0, False)
    
    # Passed parameters checking function
    dft.stypy_localization = localization
    dft.stypy_type_of_self = None
    dft.stypy_type_store = module_type_store
    dft.stypy_function_name = 'dft'
    dft.stypy_param_names_list = ['n', 'scale']
    dft.stypy_varargs_param_name = None
    dft.stypy_kwargs_param_name = None
    dft.stypy_call_defaults = defaults
    dft.stypy_call_varargs = varargs
    dft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dft', ['n', 'scale'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dft', localization, ['n', 'scale'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dft(...)' code ##################

    str_25277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, (-1)), 'str', '\n    Discrete Fourier transform matrix.\n\n    Create the matrix that computes the discrete Fourier transform of a\n    sequence [1]_.  The n-th primitive root of unity used to generate the\n    matrix is exp(-2*pi*i/n), where i = sqrt(-1).\n\n    Parameters\n    ----------\n    n : int\n        Size the matrix to create.\n    scale : str, optional\n        Must be None, \'sqrtn\', or \'n\'.\n        If `scale` is \'sqrtn\', the matrix is divided by `sqrt(n)`.\n        If `scale` is \'n\', the matrix is divided by `n`.\n        If `scale` is None (the default), the matrix is not normalized, and the\n        return value is simply the Vandermonde matrix of the roots of unity.\n\n    Returns\n    -------\n    m : (n, n) ndarray\n        The DFT matrix.\n\n    Notes\n    -----\n    When `scale` is None, multiplying a vector by the matrix returned by\n    `dft` is mathematically equivalent to (but much less efficient than)\n    the calculation performed by `scipy.fftpack.fft`.\n\n    .. versionadded:: 0.14.0\n\n    References\n    ----------\n    .. [1] "DFT matrix", http://en.wikipedia.org/wiki/DFT_matrix\n\n    Examples\n    --------\n    >>> from scipy.linalg import dft\n    >>> np.set_printoptions(precision=5, suppress=True)\n    >>> x = np.array([1, 2, 3, 0, 3, 2, 1, 0])\n    >>> m = dft(8)\n    >>> m.dot(x)   # Compute the DFT of x\n    array([ 12.+0.j,  -2.-2.j,   0.-4.j,  -2.+2.j,   4.+0.j,  -2.-2.j,\n            -0.+4.j,  -2.+2.j])\n\n    Verify that ``m.dot(x)`` is the same as ``fft(x)``.\n\n    >>> from scipy.fftpack import fft\n    >>> fft(x)     # Same result as m.dot(x)\n    array([ 12.+0.j,  -2.-2.j,   0.-4.j,  -2.+2.j,   4.+0.j,  -2.-2.j,\n             0.+4.j,  -2.+2.j])\n    ')
    
    
    # Getting the type of 'scale' (line 1030)
    scale_25278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 7), 'scale')
    
    # Obtaining an instance of the builtin type 'list' (line 1030)
    list_25279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1030)
    # Adding element type (line 1030)
    # Getting the type of 'None' (line 1030)
    None_25280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 21), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1030, 20), list_25279, None_25280)
    # Adding element type (line 1030)
    str_25281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 27), 'str', 'sqrtn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1030, 20), list_25279, str_25281)
    # Adding element type (line 1030)
    str_25282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 36), 'str', 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1030, 20), list_25279, str_25282)
    
    # Applying the binary operator 'notin' (line 1030)
    result_contains_25283 = python_operator(stypy.reporting.localization.Localization(__file__, 1030, 7), 'notin', scale_25278, list_25279)
    
    # Testing the type of an if condition (line 1030)
    if_condition_25284 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1030, 4), result_contains_25283)
    # Assigning a type to the variable 'if_condition_25284' (line 1030)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1030, 4), 'if_condition_25284', if_condition_25284)
    # SSA begins for if statement (line 1030)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1031)
    # Processing the call arguments (line 1031)
    str_25286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 25), 'str', "scale must be None, 'sqrtn', or 'n'; %r is not valid.")
    
    # Obtaining an instance of the builtin type 'tuple' (line 1032)
    tuple_25287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1032)
    # Adding element type (line 1032)
    # Getting the type of 'scale' (line 1032)
    scale_25288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 47), 'scale', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1032, 47), tuple_25287, scale_25288)
    
    # Applying the binary operator '%' (line 1031)
    result_mod_25289 = python_operator(stypy.reporting.localization.Localization(__file__, 1031, 25), '%', str_25286, tuple_25287)
    
    # Processing the call keyword arguments (line 1031)
    kwargs_25290 = {}
    # Getting the type of 'ValueError' (line 1031)
    ValueError_25285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1031)
    ValueError_call_result_25291 = invoke(stypy.reporting.localization.Localization(__file__, 1031, 14), ValueError_25285, *[result_mod_25289], **kwargs_25290)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1031, 8), ValueError_call_result_25291, 'raise parameter', BaseException)
    # SSA join for if statement (line 1030)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1034):
    
    # Assigning a Call to a Name (line 1034):
    
    # Call to reshape(...): (line 1034)
    # Processing the call arguments (line 1034)
    int_25309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 60), 'int')
    int_25310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 64), 'int')
    # Processing the call keyword arguments (line 1034)
    kwargs_25311 = {}
    
    # Call to exp(...): (line 1034)
    # Processing the call arguments (line 1034)
    complex_25294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 20), 'complex')
    # Getting the type of 'np' (line 1034)
    np_25295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 26), 'np', False)
    # Obtaining the member 'pi' of a type (line 1034)
    pi_25296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1034, 26), np_25295, 'pi')
    # Applying the binary operator '*' (line 1034)
    result_mul_25297 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 20), '*', complex_25294, pi_25296)
    
    
    # Call to arange(...): (line 1034)
    # Processing the call arguments (line 1034)
    # Getting the type of 'n' (line 1034)
    n_25300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 44), 'n', False)
    # Processing the call keyword arguments (line 1034)
    kwargs_25301 = {}
    # Getting the type of 'np' (line 1034)
    np_25298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 34), 'np', False)
    # Obtaining the member 'arange' of a type (line 1034)
    arange_25299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1034, 34), np_25298, 'arange')
    # Calling arange(args, kwargs) (line 1034)
    arange_call_result_25302 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 34), arange_25299, *[n_25300], **kwargs_25301)
    
    # Applying the binary operator '*' (line 1034)
    result_mul_25303 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 32), '*', result_mul_25297, arange_call_result_25302)
    
    # Getting the type of 'n' (line 1034)
    n_25304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 49), 'n', False)
    # Applying the binary operator 'div' (line 1034)
    result_div_25305 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 47), 'div', result_mul_25303, n_25304)
    
    # Processing the call keyword arguments (line 1034)
    kwargs_25306 = {}
    # Getting the type of 'np' (line 1034)
    np_25292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 13), 'np', False)
    # Obtaining the member 'exp' of a type (line 1034)
    exp_25293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1034, 13), np_25292, 'exp')
    # Calling exp(args, kwargs) (line 1034)
    exp_call_result_25307 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 13), exp_25293, *[result_div_25305], **kwargs_25306)
    
    # Obtaining the member 'reshape' of a type (line 1034)
    reshape_25308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1034, 13), exp_call_result_25307, 'reshape')
    # Calling reshape(args, kwargs) (line 1034)
    reshape_call_result_25312 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 13), reshape_25308, *[int_25309, int_25310], **kwargs_25311)
    
    # Assigning a type to the variable 'omegas' (line 1034)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 4), 'omegas', reshape_call_result_25312)
    
    # Assigning a BinOp to a Name (line 1035):
    
    # Assigning a BinOp to a Name (line 1035):
    # Getting the type of 'omegas' (line 1035)
    omegas_25313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 8), 'omegas')
    
    # Call to arange(...): (line 1035)
    # Processing the call arguments (line 1035)
    # Getting the type of 'n' (line 1035)
    n_25316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 28), 'n', False)
    # Processing the call keyword arguments (line 1035)
    kwargs_25317 = {}
    # Getting the type of 'np' (line 1035)
    np_25314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 18), 'np', False)
    # Obtaining the member 'arange' of a type (line 1035)
    arange_25315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1035, 18), np_25314, 'arange')
    # Calling arange(args, kwargs) (line 1035)
    arange_call_result_25318 = invoke(stypy.reporting.localization.Localization(__file__, 1035, 18), arange_25315, *[n_25316], **kwargs_25317)
    
    # Applying the binary operator '**' (line 1035)
    result_pow_25319 = python_operator(stypy.reporting.localization.Localization(__file__, 1035, 8), '**', omegas_25313, arange_call_result_25318)
    
    # Assigning a type to the variable 'm' (line 1035)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1035, 4), 'm', result_pow_25319)
    
    
    # Getting the type of 'scale' (line 1036)
    scale_25320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 7), 'scale')
    str_25321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 16), 'str', 'sqrtn')
    # Applying the binary operator '==' (line 1036)
    result_eq_25322 = python_operator(stypy.reporting.localization.Localization(__file__, 1036, 7), '==', scale_25320, str_25321)
    
    # Testing the type of an if condition (line 1036)
    if_condition_25323 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1036, 4), result_eq_25322)
    # Assigning a type to the variable 'if_condition_25323' (line 1036)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1036, 4), 'if_condition_25323', if_condition_25323)
    # SSA begins for if statement (line 1036)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'm' (line 1037)
    m_25324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 8), 'm')
    
    # Call to sqrt(...): (line 1037)
    # Processing the call arguments (line 1037)
    # Getting the type of 'n' (line 1037)
    n_25327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 23), 'n', False)
    # Processing the call keyword arguments (line 1037)
    kwargs_25328 = {}
    # Getting the type of 'math' (line 1037)
    math_25325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 13), 'math', False)
    # Obtaining the member 'sqrt' of a type (line 1037)
    sqrt_25326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 13), math_25325, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1037)
    sqrt_call_result_25329 = invoke(stypy.reporting.localization.Localization(__file__, 1037, 13), sqrt_25326, *[n_25327], **kwargs_25328)
    
    # Applying the binary operator 'div=' (line 1037)
    result_div_25330 = python_operator(stypy.reporting.localization.Localization(__file__, 1037, 8), 'div=', m_25324, sqrt_call_result_25329)
    # Assigning a type to the variable 'm' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 8), 'm', result_div_25330)
    
    # SSA branch for the else part of an if statement (line 1036)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'scale' (line 1038)
    scale_25331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 9), 'scale')
    str_25332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 18), 'str', 'n')
    # Applying the binary operator '==' (line 1038)
    result_eq_25333 = python_operator(stypy.reporting.localization.Localization(__file__, 1038, 9), '==', scale_25331, str_25332)
    
    # Testing the type of an if condition (line 1038)
    if_condition_25334 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1038, 9), result_eq_25333)
    # Assigning a type to the variable 'if_condition_25334' (line 1038)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1038, 9), 'if_condition_25334', if_condition_25334)
    # SSA begins for if statement (line 1038)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'm' (line 1039)
    m_25335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'm')
    # Getting the type of 'n' (line 1039)
    n_25336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 13), 'n')
    # Applying the binary operator 'div=' (line 1039)
    result_div_25337 = python_operator(stypy.reporting.localization.Localization(__file__, 1039, 8), 'div=', m_25335, n_25336)
    # Assigning a type to the variable 'm' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'm', result_div_25337)
    
    # SSA join for if statement (line 1038)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1036)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'm' (line 1040)
    m_25338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 11), 'm')
    # Assigning a type to the variable 'stypy_return_type' (line 1040)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1040, 4), 'stypy_return_type', m_25338)
    
    # ################# End of 'dft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dft' in the type store
    # Getting the type of 'stypy_return_type' (line 977)
    stypy_return_type_25339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25339)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dft'
    return stypy_return_type_25339

# Assigning a type to the variable 'dft' (line 977)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 977, 0), 'dft', dft)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
