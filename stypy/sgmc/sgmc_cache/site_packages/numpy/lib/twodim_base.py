
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Basic functions for manipulating 2d arrays
2: 
3: '''
4: from __future__ import division, absolute_import, print_function
5: 
6: from numpy.core.numeric import (
7:     asanyarray, arange, zeros, greater_equal, multiply, ones, asarray,
8:     where, int8, int16, int32, int64, empty, promote_types, diagonal,
9:     )
10: from numpy.core import iinfo
11: 
12: 
13: __all__ = [
14:     'diag', 'diagflat', 'eye', 'fliplr', 'flipud', 'rot90', 'tri', 'triu',
15:     'tril', 'vander', 'histogram2d', 'mask_indices', 'tril_indices',
16:     'tril_indices_from', 'triu_indices', 'triu_indices_from', ]
17: 
18: 
19: i1 = iinfo(int8)
20: i2 = iinfo(int16)
21: i4 = iinfo(int32)
22: 
23: 
24: def _min_int(low, high):
25:     ''' get small int that fits the range '''
26:     if high <= i1.max and low >= i1.min:
27:         return int8
28:     if high <= i2.max and low >= i2.min:
29:         return int16
30:     if high <= i4.max and low >= i4.min:
31:         return int32
32:     return int64
33: 
34: 
35: def fliplr(m):
36:     '''
37:     Flip array in the left/right direction.
38: 
39:     Flip the entries in each row in the left/right direction.
40:     Columns are preserved, but appear in a different order than before.
41: 
42:     Parameters
43:     ----------
44:     m : array_like
45:         Input array, must be at least 2-D.
46: 
47:     Returns
48:     -------
49:     f : ndarray
50:         A view of `m` with the columns reversed.  Since a view
51:         is returned, this operation is :math:`\\mathcal O(1)`.
52: 
53:     See Also
54:     --------
55:     flipud : Flip array in the up/down direction.
56:     rot90 : Rotate array counterclockwise.
57: 
58:     Notes
59:     -----
60:     Equivalent to A[:,::-1]. Requires the array to be at least 2-D.
61: 
62:     Examples
63:     --------
64:     >>> A = np.diag([1.,2.,3.])
65:     >>> A
66:     array([[ 1.,  0.,  0.],
67:            [ 0.,  2.,  0.],
68:            [ 0.,  0.,  3.]])
69:     >>> np.fliplr(A)
70:     array([[ 0.,  0.,  1.],
71:            [ 0.,  2.,  0.],
72:            [ 3.,  0.,  0.]])
73: 
74:     >>> A = np.random.randn(2,3,5)
75:     >>> np.all(np.fliplr(A)==A[:,::-1,...])
76:     True
77: 
78:     '''
79:     m = asanyarray(m)
80:     if m.ndim < 2:
81:         raise ValueError("Input must be >= 2-d.")
82:     return m[:, ::-1]
83: 
84: 
85: def flipud(m):
86:     '''
87:     Flip array in the up/down direction.
88: 
89:     Flip the entries in each column in the up/down direction.
90:     Rows are preserved, but appear in a different order than before.
91: 
92:     Parameters
93:     ----------
94:     m : array_like
95:         Input array.
96: 
97:     Returns
98:     -------
99:     out : array_like
100:         A view of `m` with the rows reversed.  Since a view is
101:         returned, this operation is :math:`\\mathcal O(1)`.
102: 
103:     See Also
104:     --------
105:     fliplr : Flip array in the left/right direction.
106:     rot90 : Rotate array counterclockwise.
107: 
108:     Notes
109:     -----
110:     Equivalent to ``A[::-1,...]``.
111:     Does not require the array to be two-dimensional.
112: 
113:     Examples
114:     --------
115:     >>> A = np.diag([1.0, 2, 3])
116:     >>> A
117:     array([[ 1.,  0.,  0.],
118:            [ 0.,  2.,  0.],
119:            [ 0.,  0.,  3.]])
120:     >>> np.flipud(A)
121:     array([[ 0.,  0.,  3.],
122:            [ 0.,  2.,  0.],
123:            [ 1.,  0.,  0.]])
124: 
125:     >>> A = np.random.randn(2,3,5)
126:     >>> np.all(np.flipud(A)==A[::-1,...])
127:     True
128: 
129:     >>> np.flipud([1,2])
130:     array([2, 1])
131: 
132:     '''
133:     m = asanyarray(m)
134:     if m.ndim < 1:
135:         raise ValueError("Input must be >= 1-d.")
136:     return m[::-1, ...]
137: 
138: 
139: def rot90(m, k=1):
140:     '''
141:     Rotate an array by 90 degrees in the counter-clockwise direction.
142: 
143:     The first two dimensions are rotated; therefore, the array must be at
144:     least 2-D.
145: 
146:     Parameters
147:     ----------
148:     m : array_like
149:         Array of two or more dimensions.
150:     k : integer
151:         Number of times the array is rotated by 90 degrees.
152: 
153:     Returns
154:     -------
155:     y : ndarray
156:         Rotated array.
157: 
158:     See Also
159:     --------
160:     fliplr : Flip an array horizontally.
161:     flipud : Flip an array vertically.
162: 
163:     Examples
164:     --------
165:     >>> m = np.array([[1,2],[3,4]], int)
166:     >>> m
167:     array([[1, 2],
168:            [3, 4]])
169:     >>> np.rot90(m)
170:     array([[2, 4],
171:            [1, 3]])
172:     >>> np.rot90(m, 2)
173:     array([[4, 3],
174:            [2, 1]])
175: 
176:     '''
177:     m = asanyarray(m)
178:     if m.ndim < 2:
179:         raise ValueError("Input must >= 2-d.")
180:     k = k % 4
181:     if k == 0:
182:         return m
183:     elif k == 1:
184:         return fliplr(m).swapaxes(0, 1)
185:     elif k == 2:
186:         return fliplr(flipud(m))
187:     else:
188:         # k == 3
189:         return fliplr(m.swapaxes(0, 1))
190: 
191: 
192: def eye(N, M=None, k=0, dtype=float):
193:     '''
194:     Return a 2-D array with ones on the diagonal and zeros elsewhere.
195: 
196:     Parameters
197:     ----------
198:     N : int
199:       Number of rows in the output.
200:     M : int, optional
201:       Number of columns in the output. If None, defaults to `N`.
202:     k : int, optional
203:       Index of the diagonal: 0 (the default) refers to the main diagonal,
204:       a positive value refers to an upper diagonal, and a negative value
205:       to a lower diagonal.
206:     dtype : data-type, optional
207:       Data-type of the returned array.
208: 
209:     Returns
210:     -------
211:     I : ndarray of shape (N,M)
212:       An array where all elements are equal to zero, except for the `k`-th
213:       diagonal, whose values are equal to one.
214: 
215:     See Also
216:     --------
217:     identity : (almost) equivalent function
218:     diag : diagonal 2-D array from a 1-D array specified by the user.
219: 
220:     Examples
221:     --------
222:     >>> np.eye(2, dtype=int)
223:     array([[1, 0],
224:            [0, 1]])
225:     >>> np.eye(3, k=1)
226:     array([[ 0.,  1.,  0.],
227:            [ 0.,  0.,  1.],
228:            [ 0.,  0.,  0.]])
229: 
230:     '''
231:     if M is None:
232:         M = N
233:     m = zeros((N, M), dtype=dtype)
234:     if k >= M:
235:         return m
236:     if k >= 0:
237:         i = k
238:     else:
239:         i = (-k) * M
240:     m[:M-k].flat[i::M+1] = 1
241:     return m
242: 
243: 
244: def diag(v, k=0):
245:     '''
246:     Extract a diagonal or construct a diagonal array.
247: 
248:     See the more detailed documentation for ``numpy.diagonal`` if you use this
249:     function to extract a diagonal and wish to write to the resulting array;
250:     whether it returns a copy or a view depends on what version of numpy you
251:     are using.
252: 
253:     Parameters
254:     ----------
255:     v : array_like
256:         If `v` is a 2-D array, return a copy of its `k`-th diagonal.
257:         If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
258:         diagonal.
259:     k : int, optional
260:         Diagonal in question. The default is 0. Use `k>0` for diagonals
261:         above the main diagonal, and `k<0` for diagonals below the main
262:         diagonal.
263: 
264:     Returns
265:     -------
266:     out : ndarray
267:         The extracted diagonal or constructed diagonal array.
268: 
269:     See Also
270:     --------
271:     diagonal : Return specified diagonals.
272:     diagflat : Create a 2-D array with the flattened input as a diagonal.
273:     trace : Sum along diagonals.
274:     triu : Upper triangle of an array.
275:     tril : Lower triangle of an array.
276: 
277:     Examples
278:     --------
279:     >>> x = np.arange(9).reshape((3,3))
280:     >>> x
281:     array([[0, 1, 2],
282:            [3, 4, 5],
283:            [6, 7, 8]])
284: 
285:     >>> np.diag(x)
286:     array([0, 4, 8])
287:     >>> np.diag(x, k=1)
288:     array([1, 5])
289:     >>> np.diag(x, k=-1)
290:     array([3, 7])
291: 
292:     >>> np.diag(np.diag(x))
293:     array([[0, 0, 0],
294:            [0, 4, 0],
295:            [0, 0, 8]])
296: 
297:     '''
298:     v = asanyarray(v)
299:     s = v.shape
300:     if len(s) == 1:
301:         n = s[0]+abs(k)
302:         res = zeros((n, n), v.dtype)
303:         if k >= 0:
304:             i = k
305:         else:
306:             i = (-k) * n
307:         res[:n-k].flat[i::n+1] = v
308:         return res
309:     elif len(s) == 2:
310:         return diagonal(v, k)
311:     else:
312:         raise ValueError("Input must be 1- or 2-d.")
313: 
314: 
315: def diagflat(v, k=0):
316:     '''
317:     Create a two-dimensional array with the flattened input as a diagonal.
318: 
319:     Parameters
320:     ----------
321:     v : array_like
322:         Input data, which is flattened and set as the `k`-th
323:         diagonal of the output.
324:     k : int, optional
325:         Diagonal to set; 0, the default, corresponds to the "main" diagonal,
326:         a positive (negative) `k` giving the number of the diagonal above
327:         (below) the main.
328: 
329:     Returns
330:     -------
331:     out : ndarray
332:         The 2-D output array.
333: 
334:     See Also
335:     --------
336:     diag : MATLAB work-alike for 1-D and 2-D arrays.
337:     diagonal : Return specified diagonals.
338:     trace : Sum along diagonals.
339: 
340:     Examples
341:     --------
342:     >>> np.diagflat([[1,2], [3,4]])
343:     array([[1, 0, 0, 0],
344:            [0, 2, 0, 0],
345:            [0, 0, 3, 0],
346:            [0, 0, 0, 4]])
347: 
348:     >>> np.diagflat([1,2], 1)
349:     array([[0, 1, 0],
350:            [0, 0, 2],
351:            [0, 0, 0]])
352: 
353:     '''
354:     try:
355:         wrap = v.__array_wrap__
356:     except AttributeError:
357:         wrap = None
358:     v = asarray(v).ravel()
359:     s = len(v)
360:     n = s + abs(k)
361:     res = zeros((n, n), v.dtype)
362:     if (k >= 0):
363:         i = arange(0, n-k)
364:         fi = i+k+i*n
365:     else:
366:         i = arange(0, n+k)
367:         fi = i+(i-k)*n
368:     res.flat[fi] = v
369:     if not wrap:
370:         return res
371:     return wrap(res)
372: 
373: 
374: def tri(N, M=None, k=0, dtype=float):
375:     '''
376:     An array with ones at and below the given diagonal and zeros elsewhere.
377: 
378:     Parameters
379:     ----------
380:     N : int
381:         Number of rows in the array.
382:     M : int, optional
383:         Number of columns in the array.
384:         By default, `M` is taken equal to `N`.
385:     k : int, optional
386:         The sub-diagonal at and below which the array is filled.
387:         `k` = 0 is the main diagonal, while `k` < 0 is below it,
388:         and `k` > 0 is above.  The default is 0.
389:     dtype : dtype, optional
390:         Data type of the returned array.  The default is float.
391: 
392:     Returns
393:     -------
394:     tri : ndarray of shape (N, M)
395:         Array with its lower triangle filled with ones and zero elsewhere;
396:         in other words ``T[i,j] == 1`` for ``i <= j + k``, 0 otherwise.
397: 
398:     Examples
399:     --------
400:     >>> np.tri(3, 5, 2, dtype=int)
401:     array([[1, 1, 1, 0, 0],
402:            [1, 1, 1, 1, 0],
403:            [1, 1, 1, 1, 1]])
404: 
405:     >>> np.tri(3, 5, -1)
406:     array([[ 0.,  0.,  0.,  0.,  0.],
407:            [ 1.,  0.,  0.,  0.,  0.],
408:            [ 1.,  1.,  0.,  0.,  0.]])
409: 
410:     '''
411:     if M is None:
412:         M = N
413: 
414:     m = greater_equal.outer(arange(N, dtype=_min_int(0, N)),
415:                             arange(-k, M-k, dtype=_min_int(-k, M - k)))
416: 
417:     # Avoid making a copy if the requested type is already bool
418:     m = m.astype(dtype, copy=False)
419: 
420:     return m
421: 
422: 
423: def tril(m, k=0):
424:     '''
425:     Lower triangle of an array.
426: 
427:     Return a copy of an array with elements above the `k`-th diagonal zeroed.
428: 
429:     Parameters
430:     ----------
431:     m : array_like, shape (M, N)
432:         Input array.
433:     k : int, optional
434:         Diagonal above which to zero elements.  `k = 0` (the default) is the
435:         main diagonal, `k < 0` is below it and `k > 0` is above.
436: 
437:     Returns
438:     -------
439:     tril : ndarray, shape (M, N)
440:         Lower triangle of `m`, of same shape and data-type as `m`.
441: 
442:     See Also
443:     --------
444:     triu : same thing, only for the upper triangle
445: 
446:     Examples
447:     --------
448:     >>> np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
449:     array([[ 0,  0,  0],
450:            [ 4,  0,  0],
451:            [ 7,  8,  0],
452:            [10, 11, 12]])
453: 
454:     '''
455:     m = asanyarray(m)
456:     mask = tri(*m.shape[-2:], k=k, dtype=bool)
457: 
458:     return where(mask, m, zeros(1, m.dtype))
459: 
460: 
461: def triu(m, k=0):
462:     '''
463:     Upper triangle of an array.
464: 
465:     Return a copy of a matrix with the elements below the `k`-th diagonal
466:     zeroed.
467: 
468:     Please refer to the documentation for `tril` for further details.
469: 
470:     See Also
471:     --------
472:     tril : lower triangle of an array
473: 
474:     Examples
475:     --------
476:     >>> np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
477:     array([[ 1,  2,  3],
478:            [ 4,  5,  6],
479:            [ 0,  8,  9],
480:            [ 0,  0, 12]])
481: 
482:     '''
483:     m = asanyarray(m)
484:     mask = tri(*m.shape[-2:], k=k-1, dtype=bool)
485: 
486:     return where(mask, zeros(1, m.dtype), m)
487: 
488: 
489: # Originally borrowed from John Hunter and matplotlib
490: def vander(x, N=None, increasing=False):
491:     '''
492:     Generate a Vandermonde matrix.
493: 
494:     The columns of the output matrix are powers of the input vector. The
495:     order of the powers is determined by the `increasing` boolean argument.
496:     Specifically, when `increasing` is False, the `i`-th output column is
497:     the input vector raised element-wise to the power of ``N - i - 1``. Such
498:     a matrix with a geometric progression in each row is named for Alexandre-
499:     Theophile Vandermonde.
500: 
501:     Parameters
502:     ----------
503:     x : array_like
504:         1-D input array.
505:     N : int, optional
506:         Number of columns in the output.  If `N` is not specified, a square
507:         array is returned (``N = len(x)``).
508:     increasing : bool, optional
509:         Order of the powers of the columns.  If True, the powers increase
510:         from left to right, if False (the default) they are reversed.
511: 
512:         .. versionadded:: 1.9.0
513: 
514:     Returns
515:     -------
516:     out : ndarray
517:         Vandermonde matrix.  If `increasing` is False, the first column is
518:         ``x^(N-1)``, the second ``x^(N-2)`` and so forth. If `increasing` is
519:         True, the columns are ``x^0, x^1, ..., x^(N-1)``.
520: 
521:     See Also
522:     --------
523:     polynomial.polynomial.polyvander
524: 
525:     Examples
526:     --------
527:     >>> x = np.array([1, 2, 3, 5])
528:     >>> N = 3
529:     >>> np.vander(x, N)
530:     array([[ 1,  1,  1],
531:            [ 4,  2,  1],
532:            [ 9,  3,  1],
533:            [25,  5,  1]])
534: 
535:     >>> np.column_stack([x**(N-1-i) for i in range(N)])
536:     array([[ 1,  1,  1],
537:            [ 4,  2,  1],
538:            [ 9,  3,  1],
539:            [25,  5,  1]])
540: 
541:     >>> x = np.array([1, 2, 3, 5])
542:     >>> np.vander(x)
543:     array([[  1,   1,   1,   1],
544:            [  8,   4,   2,   1],
545:            [ 27,   9,   3,   1],
546:            [125,  25,   5,   1]])
547:     >>> np.vander(x, increasing=True)
548:     array([[  1,   1,   1,   1],
549:            [  1,   2,   4,   8],
550:            [  1,   3,   9,  27],
551:            [  1,   5,  25, 125]])
552: 
553:     The determinant of a square Vandermonde matrix is the product
554:     of the differences between the values of the input vector:
555: 
556:     >>> np.linalg.det(np.vander(x))
557:     48.000000000000043
558:     >>> (5-3)*(5-2)*(5-1)*(3-2)*(3-1)*(2-1)
559:     48
560: 
561:     '''
562:     x = asarray(x)
563:     if x.ndim != 1:
564:         raise ValueError("x must be a one-dimensional array or sequence.")
565:     if N is None:
566:         N = len(x)
567: 
568:     v = empty((len(x), N), dtype=promote_types(x.dtype, int))
569:     tmp = v[:, ::-1] if not increasing else v
570: 
571:     if N > 0:
572:         tmp[:, 0] = 1
573:     if N > 1:
574:         tmp[:, 1:] = x[:, None]
575:         multiply.accumulate(tmp[:, 1:], out=tmp[:, 1:], axis=1)
576: 
577:     return v
578: 
579: 
580: def histogram2d(x, y, bins=10, range=None, normed=False, weights=None):
581:     '''
582:     Compute the bi-dimensional histogram of two data samples.
583: 
584:     Parameters
585:     ----------
586:     x : array_like, shape (N,)
587:         An array containing the x coordinates of the points to be
588:         histogrammed.
589:     y : array_like, shape (N,)
590:         An array containing the y coordinates of the points to be
591:         histogrammed.
592:     bins : int or array_like or [int, int] or [array, array], optional
593:         The bin specification:
594: 
595:           * If int, the number of bins for the two dimensions (nx=ny=bins).
596:           * If array_like, the bin edges for the two dimensions
597:             (x_edges=y_edges=bins).
598:           * If [int, int], the number of bins in each dimension
599:             (nx, ny = bins).
600:           * If [array, array], the bin edges in each dimension
601:             (x_edges, y_edges = bins).
602:           * A combination [int, array] or [array, int], where int
603:             is the number of bins and array is the bin edges.
604: 
605:     range : array_like, shape(2,2), optional
606:         The leftmost and rightmost edges of the bins along each dimension
607:         (if not specified explicitly in the `bins` parameters):
608:         ``[[xmin, xmax], [ymin, ymax]]``. All values outside of this range
609:         will be considered outliers and not tallied in the histogram.
610:     normed : bool, optional
611:         If False, returns the number of samples in each bin. If True,
612:         returns the bin density ``bin_count / sample_count / bin_area``.
613:     weights : array_like, shape(N,), optional
614:         An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.
615:         Weights are normalized to 1 if `normed` is True. If `normed` is
616:         False, the values of the returned histogram are equal to the sum of
617:         the weights belonging to the samples falling into each bin.
618: 
619:     Returns
620:     -------
621:     H : ndarray, shape(nx, ny)
622:         The bi-dimensional histogram of samples `x` and `y`. Values in `x`
623:         are histogrammed along the first dimension and values in `y` are
624:         histogrammed along the second dimension.
625:     xedges : ndarray, shape(nx,)
626:         The bin edges along the first dimension.
627:     yedges : ndarray, shape(ny,)
628:         The bin edges along the second dimension.
629: 
630:     See Also
631:     --------
632:     histogram : 1D histogram
633:     histogramdd : Multidimensional histogram
634: 
635:     Notes
636:     -----
637:     When `normed` is True, then the returned histogram is the sample
638:     density, defined such that the sum over bins of the product
639:     ``bin_value * bin_area`` is 1.
640: 
641:     Please note that the histogram does not follow the Cartesian convention
642:     where `x` values are on the abscissa and `y` values on the ordinate
643:     axis.  Rather, `x` is histogrammed along the first dimension of the
644:     array (vertical), and `y` along the second dimension of the array
645:     (horizontal).  This ensures compatibility with `histogramdd`.
646: 
647:     Examples
648:     --------
649:     >>> import matplotlib as mpl
650:     >>> import matplotlib.pyplot as plt
651: 
652:     Construct a 2D-histogram with variable bin width. First define the bin
653:     edges:
654: 
655:     >>> xedges = [0, 1, 1.5, 3, 5]
656:     >>> yedges = [0, 2, 3, 4, 6]
657: 
658:     Next we create a histogram H with random bin content:
659: 
660:     >>> x = np.random.normal(3, 1, 100)
661:     >>> y = np.random.normal(1, 1, 100)
662:     >>> H, xedges, yedges = np.histogram2d(y, x, bins=(xedges, yedges))
663: 
664:     Or we fill the histogram H with a determined bin content:
665: 
666:     >>> H = np.ones((4, 4)).cumsum().reshape(4, 4)
667:     >>> print(H[::-1])  # This shows the bin content in the order as plotted
668:     [[ 13.  14.  15.  16.]
669:      [  9.  10.  11.  12.]
670:      [  5.   6.   7.   8.]
671:      [  1.   2.   3.   4.]]
672: 
673:     Imshow can only do an equidistant representation of bins:
674: 
675:     >>> fig = plt.figure(figsize=(7, 3))
676:     >>> ax = fig.add_subplot(131)
677:     >>> ax.set_title('imshow: equidistant')
678:     >>> im = plt.imshow(H, interpolation='nearest', origin='low',
679:                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
680: 
681:     pcolormesh can display exact bin edges:
682: 
683:     >>> ax = fig.add_subplot(132)
684:     >>> ax.set_title('pcolormesh: exact bin edges')
685:     >>> X, Y = np.meshgrid(xedges, yedges)
686:     >>> ax.pcolormesh(X, Y, H)
687:     >>> ax.set_aspect('equal')
688: 
689:     NonUniformImage displays exact bin edges with interpolation:
690: 
691:     >>> ax = fig.add_subplot(133)
692:     >>> ax.set_title('NonUniformImage: interpolated')
693:     >>> im = mpl.image.NonUniformImage(ax, interpolation='bilinear')
694:     >>> xcenters = xedges[:-1] + 0.5 * (xedges[1:] - xedges[:-1])
695:     >>> ycenters = yedges[:-1] + 0.5 * (yedges[1:] - yedges[:-1])
696:     >>> im.set_data(xcenters, ycenters, H)
697:     >>> ax.images.append(im)
698:     >>> ax.set_xlim(xedges[0], xedges[-1])
699:     >>> ax.set_ylim(yedges[0], yedges[-1])
700:     >>> ax.set_aspect('equal')
701:     >>> plt.show()
702: 
703:     '''
704:     from numpy import histogramdd
705: 
706:     try:
707:         N = len(bins)
708:     except TypeError:
709:         N = 1
710: 
711:     if N != 1 and N != 2:
712:         xedges = yedges = asarray(bins, float)
713:         bins = [xedges, yedges]
714:     hist, edges = histogramdd([x, y], bins, range, normed, weights)
715:     return hist, edges[0], edges[1]
716: 
717: 
718: def mask_indices(n, mask_func, k=0):
719:     '''
720:     Return the indices to access (n, n) arrays, given a masking function.
721: 
722:     Assume `mask_func` is a function that, for a square array a of size
723:     ``(n, n)`` with a possible offset argument `k`, when called as
724:     ``mask_func(a, k)`` returns a new array with zeros in certain locations
725:     (functions like `triu` or `tril` do precisely this). Then this function
726:     returns the indices where the non-zero values would be located.
727: 
728:     Parameters
729:     ----------
730:     n : int
731:         The returned indices will be valid to access arrays of shape (n, n).
732:     mask_func : callable
733:         A function whose call signature is similar to that of `triu`, `tril`.
734:         That is, ``mask_func(x, k)`` returns a boolean array, shaped like `x`.
735:         `k` is an optional argument to the function.
736:     k : scalar
737:         An optional argument which is passed through to `mask_func`. Functions
738:         like `triu`, `tril` take a second argument that is interpreted as an
739:         offset.
740: 
741:     Returns
742:     -------
743:     indices : tuple of arrays.
744:         The `n` arrays of indices corresponding to the locations where
745:         ``mask_func(np.ones((n, n)), k)`` is True.
746: 
747:     See Also
748:     --------
749:     triu, tril, triu_indices, tril_indices
750: 
751:     Notes
752:     -----
753:     .. versionadded:: 1.4.0
754: 
755:     Examples
756:     --------
757:     These are the indices that would allow you to access the upper triangular
758:     part of any 3x3 array:
759: 
760:     >>> iu = np.mask_indices(3, np.triu)
761: 
762:     For example, if `a` is a 3x3 array:
763: 
764:     >>> a = np.arange(9).reshape(3, 3)
765:     >>> a
766:     array([[0, 1, 2],
767:            [3, 4, 5],
768:            [6, 7, 8]])
769:     >>> a[iu]
770:     array([0, 1, 2, 4, 5, 8])
771: 
772:     An offset can be passed also to the masking function.  This gets us the
773:     indices starting on the first diagonal right of the main one:
774: 
775:     >>> iu1 = np.mask_indices(3, np.triu, 1)
776: 
777:     with which we now extract only three elements:
778: 
779:     >>> a[iu1]
780:     array([1, 2, 5])
781: 
782:     '''
783:     m = ones((n, n), int)
784:     a = mask_func(m, k)
785:     return where(a != 0)
786: 
787: 
788: def tril_indices(n, k=0, m=None):
789:     '''
790:     Return the indices for the lower-triangle of an (n, m) array.
791: 
792:     Parameters
793:     ----------
794:     n : int
795:         The row dimension of the arrays for which the returned
796:         indices will be valid.
797:     k : int, optional
798:         Diagonal offset (see `tril` for details).
799:     m : int, optional
800:         .. versionadded:: 1.9.0
801: 
802:         The column dimension of the arrays for which the returned
803:         arrays will be valid.
804:         By default `m` is taken equal to `n`.
805: 
806: 
807:     Returns
808:     -------
809:     inds : tuple of arrays
810:         The indices for the triangle. The returned tuple contains two arrays,
811:         each with the indices along one dimension of the array.
812: 
813:     See also
814:     --------
815:     triu_indices : similar function, for upper-triangular.
816:     mask_indices : generic function accepting an arbitrary mask function.
817:     tril, triu
818: 
819:     Notes
820:     -----
821:     .. versionadded:: 1.4.0
822: 
823:     Examples
824:     --------
825:     Compute two different sets of indices to access 4x4 arrays, one for the
826:     lower triangular part starting at the main diagonal, and one starting two
827:     diagonals further right:
828: 
829:     >>> il1 = np.tril_indices(4)
830:     >>> il2 = np.tril_indices(4, 2)
831: 
832:     Here is how they can be used with a sample array:
833: 
834:     >>> a = np.arange(16).reshape(4, 4)
835:     >>> a
836:     array([[ 0,  1,  2,  3],
837:            [ 4,  5,  6,  7],
838:            [ 8,  9, 10, 11],
839:            [12, 13, 14, 15]])
840: 
841:     Both for indexing:
842: 
843:     >>> a[il1]
844:     array([ 0,  4,  5,  8,  9, 10, 12, 13, 14, 15])
845: 
846:     And for assigning values:
847: 
848:     >>> a[il1] = -1
849:     >>> a
850:     array([[-1,  1,  2,  3],
851:            [-1, -1,  6,  7],
852:            [-1, -1, -1, 11],
853:            [-1, -1, -1, -1]])
854: 
855:     These cover almost the whole array (two diagonals right of the main one):
856: 
857:     >>> a[il2] = -10
858:     >>> a
859:     array([[-10, -10, -10,   3],
860:            [-10, -10, -10, -10],
861:            [-10, -10, -10, -10],
862:            [-10, -10, -10, -10]])
863: 
864:     '''
865:     return where(tri(n, m, k=k, dtype=bool))
866: 
867: 
868: def tril_indices_from(arr, k=0):
869:     '''
870:     Return the indices for the lower-triangle of arr.
871: 
872:     See `tril_indices` for full details.
873: 
874:     Parameters
875:     ----------
876:     arr : array_like
877:         The indices will be valid for square arrays whose dimensions are
878:         the same as arr.
879:     k : int, optional
880:         Diagonal offset (see `tril` for details).
881: 
882:     See Also
883:     --------
884:     tril_indices, tril
885: 
886:     Notes
887:     -----
888:     .. versionadded:: 1.4.0
889: 
890:     '''
891:     if arr.ndim != 2:
892:         raise ValueError("input array must be 2-d")
893:     return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1])
894: 
895: 
896: def triu_indices(n, k=0, m=None):
897:     '''
898:     Return the indices for the upper-triangle of an (n, m) array.
899: 
900:     Parameters
901:     ----------
902:     n : int
903:         The size of the arrays for which the returned indices will
904:         be valid.
905:     k : int, optional
906:         Diagonal offset (see `triu` for details).
907:     m : int, optional
908:         .. versionadded:: 1.9.0
909: 
910:         The column dimension of the arrays for which the returned
911:         arrays will be valid.
912:         By default `m` is taken equal to `n`.
913: 
914: 
915:     Returns
916:     -------
917:     inds : tuple, shape(2) of ndarrays, shape(`n`)
918:         The indices for the triangle. The returned tuple contains two arrays,
919:         each with the indices along one dimension of the array.  Can be used
920:         to slice a ndarray of shape(`n`, `n`).
921: 
922:     See also
923:     --------
924:     tril_indices : similar function, for lower-triangular.
925:     mask_indices : generic function accepting an arbitrary mask function.
926:     triu, tril
927: 
928:     Notes
929:     -----
930:     .. versionadded:: 1.4.0
931: 
932:     Examples
933:     --------
934:     Compute two different sets of indices to access 4x4 arrays, one for the
935:     upper triangular part starting at the main diagonal, and one starting two
936:     diagonals further right:
937: 
938:     >>> iu1 = np.triu_indices(4)
939:     >>> iu2 = np.triu_indices(4, 2)
940: 
941:     Here is how they can be used with a sample array:
942: 
943:     >>> a = np.arange(16).reshape(4, 4)
944:     >>> a
945:     array([[ 0,  1,  2,  3],
946:            [ 4,  5,  6,  7],
947:            [ 8,  9, 10, 11],
948:            [12, 13, 14, 15]])
949: 
950:     Both for indexing:
951: 
952:     >>> a[iu1]
953:     array([ 0,  1,  2,  3,  5,  6,  7, 10, 11, 15])
954: 
955:     And for assigning values:
956: 
957:     >>> a[iu1] = -1
958:     >>> a
959:     array([[-1, -1, -1, -1],
960:            [ 4, -1, -1, -1],
961:            [ 8,  9, -1, -1],
962:            [12, 13, 14, -1]])
963: 
964:     These cover only a small part of the whole array (two diagonals right
965:     of the main one):
966: 
967:     >>> a[iu2] = -10
968:     >>> a
969:     array([[ -1,  -1, -10, -10],
970:            [  4,  -1,  -1, -10],
971:            [  8,   9,  -1,  -1],
972:            [ 12,  13,  14,  -1]])
973: 
974:     '''
975:     return where(~tri(n, m, k=k-1, dtype=bool))
976: 
977: 
978: def triu_indices_from(arr, k=0):
979:     '''
980:     Return the indices for the upper-triangle of arr.
981: 
982:     See `triu_indices` for full details.
983: 
984:     Parameters
985:     ----------
986:     arr : ndarray, shape(N, N)
987:         The indices will be valid for square arrays.
988:     k : int, optional
989:         Diagonal offset (see `triu` for details).
990: 
991:     Returns
992:     -------
993:     triu_indices_from : tuple, shape(2) of ndarray, shape(N)
994:         Indices for the upper-triangle of `arr`.
995: 
996:     See Also
997:     --------
998:     triu_indices, triu
999: 
1000:     Notes
1001:     -----
1002:     .. versionadded:: 1.4.0
1003: 
1004:     '''
1005:     if arr.ndim != 2:
1006:         raise ValueError("input array must be 2-d")
1007:     return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1])
1008: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_126630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', ' Basic functions for manipulating 2d arrays\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.core.numeric import asanyarray, arange, zeros, greater_equal, multiply, ones, asarray, where, int8, int16, int32, int64, empty, promote_types, diagonal' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_126631 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric')

if (type(import_126631) is not StypyTypeError):

    if (import_126631 != 'pyd_module'):
        __import__(import_126631)
        sys_modules_126632 = sys.modules[import_126631]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric', sys_modules_126632.module_type_store, module_type_store, ['asanyarray', 'arange', 'zeros', 'greater_equal', 'multiply', 'ones', 'asarray', 'where', 'int8', 'int16', 'int32', 'int64', 'empty', 'promote_types', 'diagonal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_126632, sys_modules_126632.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import asanyarray, arange, zeros, greater_equal, multiply, ones, asarray, where, int8, int16, int32, int64, empty, promote_types, diagonal

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric', None, module_type_store, ['asanyarray', 'arange', 'zeros', 'greater_equal', 'multiply', 'ones', 'asarray', 'where', 'int8', 'int16', 'int32', 'int64', 'empty', 'promote_types', 'diagonal'], [asanyarray, arange, zeros, greater_equal, multiply, ones, asarray, where, int8, int16, int32, int64, empty, promote_types, diagonal])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric', import_126631)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.core import iinfo' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_126633 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core')

if (type(import_126633) is not StypyTypeError):

    if (import_126633 != 'pyd_module'):
        __import__(import_126633)
        sys_modules_126634 = sys.modules[import_126633]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', sys_modules_126634.module_type_store, module_type_store, ['iinfo'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_126634, sys_modules_126634.module_type_store, module_type_store)
    else:
        from numpy.core import iinfo

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', None, module_type_store, ['iinfo'], [iinfo])

else:
    # Assigning a type to the variable 'numpy.core' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', import_126633)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a List to a Name (line 13):

# Assigning a List to a Name (line 13):
__all__ = ['diag', 'diagflat', 'eye', 'fliplr', 'flipud', 'rot90', 'tri', 'triu', 'tril', 'vander', 'histogram2d', 'mask_indices', 'tril_indices', 'tril_indices_from', 'triu_indices', 'triu_indices_from']
module_type_store.set_exportable_members(['diag', 'diagflat', 'eye', 'fliplr', 'flipud', 'rot90', 'tri', 'triu', 'tril', 'vander', 'histogram2d', 'mask_indices', 'tril_indices', 'tril_indices_from', 'triu_indices', 'triu_indices_from'])

# Obtaining an instance of the builtin type 'list' (line 13)
list_126635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
str_126636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', 'diag')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126636)
# Adding element type (line 13)
str_126637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'str', 'diagflat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126637)
# Adding element type (line 13)
str_126638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'str', 'eye')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126638)
# Adding element type (line 13)
str_126639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 31), 'str', 'fliplr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126639)
# Adding element type (line 13)
str_126640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 41), 'str', 'flipud')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126640)
# Adding element type (line 13)
str_126641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 51), 'str', 'rot90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126641)
# Adding element type (line 13)
str_126642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 60), 'str', 'tri')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126642)
# Adding element type (line 13)
str_126643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 67), 'str', 'triu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126643)
# Adding element type (line 13)
str_126644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'str', 'tril')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126644)
# Adding element type (line 13)
str_126645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'str', 'vander')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126645)
# Adding element type (line 13)
str_126646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 22), 'str', 'histogram2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126646)
# Adding element type (line 13)
str_126647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 37), 'str', 'mask_indices')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126647)
# Adding element type (line 13)
str_126648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 53), 'str', 'tril_indices')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126648)
# Adding element type (line 13)
str_126649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'str', 'tril_indices_from')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126649)
# Adding element type (line 13)
str_126650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'str', 'triu_indices')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126650)
# Adding element type (line 13)
str_126651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 41), 'str', 'triu_indices_from')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_126635, str_126651)

# Assigning a type to the variable '__all__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__all__', list_126635)

# Assigning a Call to a Name (line 19):

# Assigning a Call to a Name (line 19):

# Call to iinfo(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'int8' (line 19)
int8_126653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'int8', False)
# Processing the call keyword arguments (line 19)
kwargs_126654 = {}
# Getting the type of 'iinfo' (line 19)
iinfo_126652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'iinfo', False)
# Calling iinfo(args, kwargs) (line 19)
iinfo_call_result_126655 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), iinfo_126652, *[int8_126653], **kwargs_126654)

# Assigning a type to the variable 'i1' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'i1', iinfo_call_result_126655)

# Assigning a Call to a Name (line 20):

# Assigning a Call to a Name (line 20):

# Call to iinfo(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of 'int16' (line 20)
int16_126657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'int16', False)
# Processing the call keyword arguments (line 20)
kwargs_126658 = {}
# Getting the type of 'iinfo' (line 20)
iinfo_126656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 5), 'iinfo', False)
# Calling iinfo(args, kwargs) (line 20)
iinfo_call_result_126659 = invoke(stypy.reporting.localization.Localization(__file__, 20, 5), iinfo_126656, *[int16_126657], **kwargs_126658)

# Assigning a type to the variable 'i2' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'i2', iinfo_call_result_126659)

# Assigning a Call to a Name (line 21):

# Assigning a Call to a Name (line 21):

# Call to iinfo(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of 'int32' (line 21)
int32_126661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'int32', False)
# Processing the call keyword arguments (line 21)
kwargs_126662 = {}
# Getting the type of 'iinfo' (line 21)
iinfo_126660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 5), 'iinfo', False)
# Calling iinfo(args, kwargs) (line 21)
iinfo_call_result_126663 = invoke(stypy.reporting.localization.Localization(__file__, 21, 5), iinfo_126660, *[int32_126661], **kwargs_126662)

# Assigning a type to the variable 'i4' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'i4', iinfo_call_result_126663)

@norecursion
def _min_int(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_min_int'
    module_type_store = module_type_store.open_function_context('_min_int', 24, 0, False)
    
    # Passed parameters checking function
    _min_int.stypy_localization = localization
    _min_int.stypy_type_of_self = None
    _min_int.stypy_type_store = module_type_store
    _min_int.stypy_function_name = '_min_int'
    _min_int.stypy_param_names_list = ['low', 'high']
    _min_int.stypy_varargs_param_name = None
    _min_int.stypy_kwargs_param_name = None
    _min_int.stypy_call_defaults = defaults
    _min_int.stypy_call_varargs = varargs
    _min_int.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_min_int', ['low', 'high'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_min_int', localization, ['low', 'high'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_min_int(...)' code ##################

    str_126664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'str', ' get small int that fits the range ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'high' (line 26)
    high_126665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 7), 'high')
    # Getting the type of 'i1' (line 26)
    i1_126666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'i1')
    # Obtaining the member 'max' of a type (line 26)
    max_126667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 15), i1_126666, 'max')
    # Applying the binary operator '<=' (line 26)
    result_le_126668 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 7), '<=', high_126665, max_126667)
    
    
    # Getting the type of 'low' (line 26)
    low_126669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 26), 'low')
    # Getting the type of 'i1' (line 26)
    i1_126670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 33), 'i1')
    # Obtaining the member 'min' of a type (line 26)
    min_126671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 33), i1_126670, 'min')
    # Applying the binary operator '>=' (line 26)
    result_ge_126672 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 26), '>=', low_126669, min_126671)
    
    # Applying the binary operator 'and' (line 26)
    result_and_keyword_126673 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 7), 'and', result_le_126668, result_ge_126672)
    
    # Testing the type of an if condition (line 26)
    if_condition_126674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 4), result_and_keyword_126673)
    # Assigning a type to the variable 'if_condition_126674' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'if_condition_126674', if_condition_126674)
    # SSA begins for if statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'int8' (line 27)
    int8_126675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'int8')
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', int8_126675)
    # SSA join for if statement (line 26)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'high' (line 28)
    high_126676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 7), 'high')
    # Getting the type of 'i2' (line 28)
    i2_126677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'i2')
    # Obtaining the member 'max' of a type (line 28)
    max_126678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 15), i2_126677, 'max')
    # Applying the binary operator '<=' (line 28)
    result_le_126679 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 7), '<=', high_126676, max_126678)
    
    
    # Getting the type of 'low' (line 28)
    low_126680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'low')
    # Getting the type of 'i2' (line 28)
    i2_126681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 33), 'i2')
    # Obtaining the member 'min' of a type (line 28)
    min_126682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 33), i2_126681, 'min')
    # Applying the binary operator '>=' (line 28)
    result_ge_126683 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 26), '>=', low_126680, min_126682)
    
    # Applying the binary operator 'and' (line 28)
    result_and_keyword_126684 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 7), 'and', result_le_126679, result_ge_126683)
    
    # Testing the type of an if condition (line 28)
    if_condition_126685 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 4), result_and_keyword_126684)
    # Assigning a type to the variable 'if_condition_126685' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'if_condition_126685', if_condition_126685)
    # SSA begins for if statement (line 28)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'int16' (line 29)
    int16_126686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'int16')
    # Assigning a type to the variable 'stypy_return_type' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', int16_126686)
    # SSA join for if statement (line 28)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'high' (line 30)
    high_126687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 7), 'high')
    # Getting the type of 'i4' (line 30)
    i4_126688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'i4')
    # Obtaining the member 'max' of a type (line 30)
    max_126689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 15), i4_126688, 'max')
    # Applying the binary operator '<=' (line 30)
    result_le_126690 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 7), '<=', high_126687, max_126689)
    
    
    # Getting the type of 'low' (line 30)
    low_126691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'low')
    # Getting the type of 'i4' (line 30)
    i4_126692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 33), 'i4')
    # Obtaining the member 'min' of a type (line 30)
    min_126693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 33), i4_126692, 'min')
    # Applying the binary operator '>=' (line 30)
    result_ge_126694 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 26), '>=', low_126691, min_126693)
    
    # Applying the binary operator 'and' (line 30)
    result_and_keyword_126695 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 7), 'and', result_le_126690, result_ge_126694)
    
    # Testing the type of an if condition (line 30)
    if_condition_126696 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 4), result_and_keyword_126695)
    # Assigning a type to the variable 'if_condition_126696' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'if_condition_126696', if_condition_126696)
    # SSA begins for if statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'int32' (line 31)
    int32_126697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'int32')
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', int32_126697)
    # SSA join for if statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'int64' (line 32)
    int64_126698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'int64')
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type', int64_126698)
    
    # ################# End of '_min_int(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_min_int' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_126699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126699)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_min_int'
    return stypy_return_type_126699

# Assigning a type to the variable '_min_int' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), '_min_int', _min_int)

@norecursion
def fliplr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fliplr'
    module_type_store = module_type_store.open_function_context('fliplr', 35, 0, False)
    
    # Passed parameters checking function
    fliplr.stypy_localization = localization
    fliplr.stypy_type_of_self = None
    fliplr.stypy_type_store = module_type_store
    fliplr.stypy_function_name = 'fliplr'
    fliplr.stypy_param_names_list = ['m']
    fliplr.stypy_varargs_param_name = None
    fliplr.stypy_kwargs_param_name = None
    fliplr.stypy_call_defaults = defaults
    fliplr.stypy_call_varargs = varargs
    fliplr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fliplr', ['m'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fliplr', localization, ['m'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fliplr(...)' code ##################

    str_126700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'str', '\n    Flip array in the left/right direction.\n\n    Flip the entries in each row in the left/right direction.\n    Columns are preserved, but appear in a different order than before.\n\n    Parameters\n    ----------\n    m : array_like\n        Input array, must be at least 2-D.\n\n    Returns\n    -------\n    f : ndarray\n        A view of `m` with the columns reversed.  Since a view\n        is returned, this operation is :math:`\\mathcal O(1)`.\n\n    See Also\n    --------\n    flipud : Flip array in the up/down direction.\n    rot90 : Rotate array counterclockwise.\n\n    Notes\n    -----\n    Equivalent to A[:,::-1]. Requires the array to be at least 2-D.\n\n    Examples\n    --------\n    >>> A = np.diag([1.,2.,3.])\n    >>> A\n    array([[ 1.,  0.,  0.],\n           [ 0.,  2.,  0.],\n           [ 0.,  0.,  3.]])\n    >>> np.fliplr(A)\n    array([[ 0.,  0.,  1.],\n           [ 0.,  2.,  0.],\n           [ 3.,  0.,  0.]])\n\n    >>> A = np.random.randn(2,3,5)\n    >>> np.all(np.fliplr(A)==A[:,::-1,...])\n    True\n\n    ')
    
    # Assigning a Call to a Name (line 79):
    
    # Assigning a Call to a Name (line 79):
    
    # Call to asanyarray(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'm' (line 79)
    m_126702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'm', False)
    # Processing the call keyword arguments (line 79)
    kwargs_126703 = {}
    # Getting the type of 'asanyarray' (line 79)
    asanyarray_126701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 79)
    asanyarray_call_result_126704 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), asanyarray_126701, *[m_126702], **kwargs_126703)
    
    # Assigning a type to the variable 'm' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'm', asanyarray_call_result_126704)
    
    
    # Getting the type of 'm' (line 80)
    m_126705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 7), 'm')
    # Obtaining the member 'ndim' of a type (line 80)
    ndim_126706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 7), m_126705, 'ndim')
    int_126707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 16), 'int')
    # Applying the binary operator '<' (line 80)
    result_lt_126708 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 7), '<', ndim_126706, int_126707)
    
    # Testing the type of an if condition (line 80)
    if_condition_126709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 4), result_lt_126708)
    # Assigning a type to the variable 'if_condition_126709' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'if_condition_126709', if_condition_126709)
    # SSA begins for if statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 81)
    # Processing the call arguments (line 81)
    str_126711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'str', 'Input must be >= 2-d.')
    # Processing the call keyword arguments (line 81)
    kwargs_126712 = {}
    # Getting the type of 'ValueError' (line 81)
    ValueError_126710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 81)
    ValueError_call_result_126713 = invoke(stypy.reporting.localization.Localization(__file__, 81, 14), ValueError_126710, *[str_126711], **kwargs_126712)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 81, 8), ValueError_call_result_126713, 'raise parameter', BaseException)
    # SSA join for if statement (line 80)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    slice_126714 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 82, 11), None, None, None)
    int_126715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'int')
    slice_126716 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 82, 11), None, None, int_126715)
    # Getting the type of 'm' (line 82)
    m_126717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'm')
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___126718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 11), m_126717, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_126719 = invoke(stypy.reporting.localization.Localization(__file__, 82, 11), getitem___126718, (slice_126714, slice_126716))
    
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type', subscript_call_result_126719)
    
    # ################# End of 'fliplr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fliplr' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_126720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126720)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fliplr'
    return stypy_return_type_126720

# Assigning a type to the variable 'fliplr' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'fliplr', fliplr)

@norecursion
def flipud(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'flipud'
    module_type_store = module_type_store.open_function_context('flipud', 85, 0, False)
    
    # Passed parameters checking function
    flipud.stypy_localization = localization
    flipud.stypy_type_of_self = None
    flipud.stypy_type_store = module_type_store
    flipud.stypy_function_name = 'flipud'
    flipud.stypy_param_names_list = ['m']
    flipud.stypy_varargs_param_name = None
    flipud.stypy_kwargs_param_name = None
    flipud.stypy_call_defaults = defaults
    flipud.stypy_call_varargs = varargs
    flipud.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'flipud', ['m'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'flipud', localization, ['m'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'flipud(...)' code ##################

    str_126721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'str', '\n    Flip array in the up/down direction.\n\n    Flip the entries in each column in the up/down direction.\n    Rows are preserved, but appear in a different order than before.\n\n    Parameters\n    ----------\n    m : array_like\n        Input array.\n\n    Returns\n    -------\n    out : array_like\n        A view of `m` with the rows reversed.  Since a view is\n        returned, this operation is :math:`\\mathcal O(1)`.\n\n    See Also\n    --------\n    fliplr : Flip array in the left/right direction.\n    rot90 : Rotate array counterclockwise.\n\n    Notes\n    -----\n    Equivalent to ``A[::-1,...]``.\n    Does not require the array to be two-dimensional.\n\n    Examples\n    --------\n    >>> A = np.diag([1.0, 2, 3])\n    >>> A\n    array([[ 1.,  0.,  0.],\n           [ 0.,  2.,  0.],\n           [ 0.,  0.,  3.]])\n    >>> np.flipud(A)\n    array([[ 0.,  0.,  3.],\n           [ 0.,  2.,  0.],\n           [ 1.,  0.,  0.]])\n\n    >>> A = np.random.randn(2,3,5)\n    >>> np.all(np.flipud(A)==A[::-1,...])\n    True\n\n    >>> np.flipud([1,2])\n    array([2, 1])\n\n    ')
    
    # Assigning a Call to a Name (line 133):
    
    # Assigning a Call to a Name (line 133):
    
    # Call to asanyarray(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'm' (line 133)
    m_126723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'm', False)
    # Processing the call keyword arguments (line 133)
    kwargs_126724 = {}
    # Getting the type of 'asanyarray' (line 133)
    asanyarray_126722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 133)
    asanyarray_call_result_126725 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), asanyarray_126722, *[m_126723], **kwargs_126724)
    
    # Assigning a type to the variable 'm' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'm', asanyarray_call_result_126725)
    
    
    # Getting the type of 'm' (line 134)
    m_126726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 7), 'm')
    # Obtaining the member 'ndim' of a type (line 134)
    ndim_126727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 7), m_126726, 'ndim')
    int_126728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 16), 'int')
    # Applying the binary operator '<' (line 134)
    result_lt_126729 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 7), '<', ndim_126727, int_126728)
    
    # Testing the type of an if condition (line 134)
    if_condition_126730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 4), result_lt_126729)
    # Assigning a type to the variable 'if_condition_126730' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'if_condition_126730', if_condition_126730)
    # SSA begins for if statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 135)
    # Processing the call arguments (line 135)
    str_126732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 25), 'str', 'Input must be >= 1-d.')
    # Processing the call keyword arguments (line 135)
    kwargs_126733 = {}
    # Getting the type of 'ValueError' (line 135)
    ValueError_126731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 135)
    ValueError_call_result_126734 = invoke(stypy.reporting.localization.Localization(__file__, 135, 14), ValueError_126731, *[str_126732], **kwargs_126733)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 135, 8), ValueError_call_result_126734, 'raise parameter', BaseException)
    # SSA join for if statement (line 134)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_126735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 15), 'int')
    slice_126736 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 136, 11), None, None, int_126735)
    Ellipsis_126737 = Ellipsis
    # Getting the type of 'm' (line 136)
    m_126738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 'm')
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___126739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 11), m_126738, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_126740 = invoke(stypy.reporting.localization.Localization(__file__, 136, 11), getitem___126739, (slice_126736, Ellipsis_126737))
    
    # Assigning a type to the variable 'stypy_return_type' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type', subscript_call_result_126740)
    
    # ################# End of 'flipud(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'flipud' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_126741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126741)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'flipud'
    return stypy_return_type_126741

# Assigning a type to the variable 'flipud' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'flipud', flipud)

@norecursion
def rot90(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_126742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 15), 'int')
    defaults = [int_126742]
    # Create a new context for function 'rot90'
    module_type_store = module_type_store.open_function_context('rot90', 139, 0, False)
    
    # Passed parameters checking function
    rot90.stypy_localization = localization
    rot90.stypy_type_of_self = None
    rot90.stypy_type_store = module_type_store
    rot90.stypy_function_name = 'rot90'
    rot90.stypy_param_names_list = ['m', 'k']
    rot90.stypy_varargs_param_name = None
    rot90.stypy_kwargs_param_name = None
    rot90.stypy_call_defaults = defaults
    rot90.stypy_call_varargs = varargs
    rot90.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rot90', ['m', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rot90', localization, ['m', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rot90(...)' code ##################

    str_126743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, (-1)), 'str', '\n    Rotate an array by 90 degrees in the counter-clockwise direction.\n\n    The first two dimensions are rotated; therefore, the array must be at\n    least 2-D.\n\n    Parameters\n    ----------\n    m : array_like\n        Array of two or more dimensions.\n    k : integer\n        Number of times the array is rotated by 90 degrees.\n\n    Returns\n    -------\n    y : ndarray\n        Rotated array.\n\n    See Also\n    --------\n    fliplr : Flip an array horizontally.\n    flipud : Flip an array vertically.\n\n    Examples\n    --------\n    >>> m = np.array([[1,2],[3,4]], int)\n    >>> m\n    array([[1, 2],\n           [3, 4]])\n    >>> np.rot90(m)\n    array([[2, 4],\n           [1, 3]])\n    >>> np.rot90(m, 2)\n    array([[4, 3],\n           [2, 1]])\n\n    ')
    
    # Assigning a Call to a Name (line 177):
    
    # Assigning a Call to a Name (line 177):
    
    # Call to asanyarray(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'm' (line 177)
    m_126745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'm', False)
    # Processing the call keyword arguments (line 177)
    kwargs_126746 = {}
    # Getting the type of 'asanyarray' (line 177)
    asanyarray_126744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 177)
    asanyarray_call_result_126747 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), asanyarray_126744, *[m_126745], **kwargs_126746)
    
    # Assigning a type to the variable 'm' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'm', asanyarray_call_result_126747)
    
    
    # Getting the type of 'm' (line 178)
    m_126748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 7), 'm')
    # Obtaining the member 'ndim' of a type (line 178)
    ndim_126749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 7), m_126748, 'ndim')
    int_126750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 16), 'int')
    # Applying the binary operator '<' (line 178)
    result_lt_126751 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 7), '<', ndim_126749, int_126750)
    
    # Testing the type of an if condition (line 178)
    if_condition_126752 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 4), result_lt_126751)
    # Assigning a type to the variable 'if_condition_126752' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'if_condition_126752', if_condition_126752)
    # SSA begins for if statement (line 178)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 179)
    # Processing the call arguments (line 179)
    str_126754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 25), 'str', 'Input must >= 2-d.')
    # Processing the call keyword arguments (line 179)
    kwargs_126755 = {}
    # Getting the type of 'ValueError' (line 179)
    ValueError_126753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 179)
    ValueError_call_result_126756 = invoke(stypy.reporting.localization.Localization(__file__, 179, 14), ValueError_126753, *[str_126754], **kwargs_126755)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 179, 8), ValueError_call_result_126756, 'raise parameter', BaseException)
    # SSA join for if statement (line 178)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 180):
    
    # Assigning a BinOp to a Name (line 180):
    # Getting the type of 'k' (line 180)
    k_126757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'k')
    int_126758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 12), 'int')
    # Applying the binary operator '%' (line 180)
    result_mod_126759 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 8), '%', k_126757, int_126758)
    
    # Assigning a type to the variable 'k' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'k', result_mod_126759)
    
    
    # Getting the type of 'k' (line 181)
    k_126760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 7), 'k')
    int_126761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 12), 'int')
    # Applying the binary operator '==' (line 181)
    result_eq_126762 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 7), '==', k_126760, int_126761)
    
    # Testing the type of an if condition (line 181)
    if_condition_126763 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 4), result_eq_126762)
    # Assigning a type to the variable 'if_condition_126763' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'if_condition_126763', if_condition_126763)
    # SSA begins for if statement (line 181)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'm' (line 182)
    m_126764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'm')
    # Assigning a type to the variable 'stypy_return_type' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'stypy_return_type', m_126764)
    # SSA branch for the else part of an if statement (line 181)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'k' (line 183)
    k_126765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 9), 'k')
    int_126766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 14), 'int')
    # Applying the binary operator '==' (line 183)
    result_eq_126767 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 9), '==', k_126765, int_126766)
    
    # Testing the type of an if condition (line 183)
    if_condition_126768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 9), result_eq_126767)
    # Assigning a type to the variable 'if_condition_126768' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 9), 'if_condition_126768', if_condition_126768)
    # SSA begins for if statement (line 183)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to swapaxes(...): (line 184)
    # Processing the call arguments (line 184)
    int_126774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 34), 'int')
    int_126775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 37), 'int')
    # Processing the call keyword arguments (line 184)
    kwargs_126776 = {}
    
    # Call to fliplr(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'm' (line 184)
    m_126770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 22), 'm', False)
    # Processing the call keyword arguments (line 184)
    kwargs_126771 = {}
    # Getting the type of 'fliplr' (line 184)
    fliplr_126769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'fliplr', False)
    # Calling fliplr(args, kwargs) (line 184)
    fliplr_call_result_126772 = invoke(stypy.reporting.localization.Localization(__file__, 184, 15), fliplr_126769, *[m_126770], **kwargs_126771)
    
    # Obtaining the member 'swapaxes' of a type (line 184)
    swapaxes_126773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 15), fliplr_call_result_126772, 'swapaxes')
    # Calling swapaxes(args, kwargs) (line 184)
    swapaxes_call_result_126777 = invoke(stypy.reporting.localization.Localization(__file__, 184, 15), swapaxes_126773, *[int_126774, int_126775], **kwargs_126776)
    
    # Assigning a type to the variable 'stypy_return_type' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'stypy_return_type', swapaxes_call_result_126777)
    # SSA branch for the else part of an if statement (line 183)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'k' (line 185)
    k_126778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 9), 'k')
    int_126779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 14), 'int')
    # Applying the binary operator '==' (line 185)
    result_eq_126780 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 9), '==', k_126778, int_126779)
    
    # Testing the type of an if condition (line 185)
    if_condition_126781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 9), result_eq_126780)
    # Assigning a type to the variable 'if_condition_126781' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 9), 'if_condition_126781', if_condition_126781)
    # SSA begins for if statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to fliplr(...): (line 186)
    # Processing the call arguments (line 186)
    
    # Call to flipud(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'm' (line 186)
    m_126784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 29), 'm', False)
    # Processing the call keyword arguments (line 186)
    kwargs_126785 = {}
    # Getting the type of 'flipud' (line 186)
    flipud_126783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 22), 'flipud', False)
    # Calling flipud(args, kwargs) (line 186)
    flipud_call_result_126786 = invoke(stypy.reporting.localization.Localization(__file__, 186, 22), flipud_126783, *[m_126784], **kwargs_126785)
    
    # Processing the call keyword arguments (line 186)
    kwargs_126787 = {}
    # Getting the type of 'fliplr' (line 186)
    fliplr_126782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 15), 'fliplr', False)
    # Calling fliplr(args, kwargs) (line 186)
    fliplr_call_result_126788 = invoke(stypy.reporting.localization.Localization(__file__, 186, 15), fliplr_126782, *[flipud_call_result_126786], **kwargs_126787)
    
    # Assigning a type to the variable 'stypy_return_type' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'stypy_return_type', fliplr_call_result_126788)
    # SSA branch for the else part of an if statement (line 185)
    module_type_store.open_ssa_branch('else')
    
    # Call to fliplr(...): (line 189)
    # Processing the call arguments (line 189)
    
    # Call to swapaxes(...): (line 189)
    # Processing the call arguments (line 189)
    int_126792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 33), 'int')
    int_126793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 36), 'int')
    # Processing the call keyword arguments (line 189)
    kwargs_126794 = {}
    # Getting the type of 'm' (line 189)
    m_126790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 22), 'm', False)
    # Obtaining the member 'swapaxes' of a type (line 189)
    swapaxes_126791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 22), m_126790, 'swapaxes')
    # Calling swapaxes(args, kwargs) (line 189)
    swapaxes_call_result_126795 = invoke(stypy.reporting.localization.Localization(__file__, 189, 22), swapaxes_126791, *[int_126792, int_126793], **kwargs_126794)
    
    # Processing the call keyword arguments (line 189)
    kwargs_126796 = {}
    # Getting the type of 'fliplr' (line 189)
    fliplr_126789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'fliplr', False)
    # Calling fliplr(args, kwargs) (line 189)
    fliplr_call_result_126797 = invoke(stypy.reporting.localization.Localization(__file__, 189, 15), fliplr_126789, *[swapaxes_call_result_126795], **kwargs_126796)
    
    # Assigning a type to the variable 'stypy_return_type' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'stypy_return_type', fliplr_call_result_126797)
    # SSA join for if statement (line 185)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 183)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 181)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'rot90(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rot90' in the type store
    # Getting the type of 'stypy_return_type' (line 139)
    stypy_return_type_126798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126798)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rot90'
    return stypy_return_type_126798

# Assigning a type to the variable 'rot90' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'rot90', rot90)

@norecursion
def eye(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 192)
    None_126799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 13), 'None')
    int_126800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 21), 'int')
    # Getting the type of 'float' (line 192)
    float_126801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 30), 'float')
    defaults = [None_126799, int_126800, float_126801]
    # Create a new context for function 'eye'
    module_type_store = module_type_store.open_function_context('eye', 192, 0, False)
    
    # Passed parameters checking function
    eye.stypy_localization = localization
    eye.stypy_type_of_self = None
    eye.stypy_type_store = module_type_store
    eye.stypy_function_name = 'eye'
    eye.stypy_param_names_list = ['N', 'M', 'k', 'dtype']
    eye.stypy_varargs_param_name = None
    eye.stypy_kwargs_param_name = None
    eye.stypy_call_defaults = defaults
    eye.stypy_call_varargs = varargs
    eye.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eye', ['N', 'M', 'k', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eye', localization, ['N', 'M', 'k', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eye(...)' code ##################

    str_126802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, (-1)), 'str', '\n    Return a 2-D array with ones on the diagonal and zeros elsewhere.\n\n    Parameters\n    ----------\n    N : int\n      Number of rows in the output.\n    M : int, optional\n      Number of columns in the output. If None, defaults to `N`.\n    k : int, optional\n      Index of the diagonal: 0 (the default) refers to the main diagonal,\n      a positive value refers to an upper diagonal, and a negative value\n      to a lower diagonal.\n    dtype : data-type, optional\n      Data-type of the returned array.\n\n    Returns\n    -------\n    I : ndarray of shape (N,M)\n      An array where all elements are equal to zero, except for the `k`-th\n      diagonal, whose values are equal to one.\n\n    See Also\n    --------\n    identity : (almost) equivalent function\n    diag : diagonal 2-D array from a 1-D array specified by the user.\n\n    Examples\n    --------\n    >>> np.eye(2, dtype=int)\n    array([[1, 0],\n           [0, 1]])\n    >>> np.eye(3, k=1)\n    array([[ 0.,  1.,  0.],\n           [ 0.,  0.,  1.],\n           [ 0.,  0.,  0.]])\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 231)
    # Getting the type of 'M' (line 231)
    M_126803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 7), 'M')
    # Getting the type of 'None' (line 231)
    None_126804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'None')
    
    (may_be_126805, more_types_in_union_126806) = may_be_none(M_126803, None_126804)

    if may_be_126805:

        if more_types_in_union_126806:
            # Runtime conditional SSA (line 231)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 232):
        
        # Assigning a Name to a Name (line 232):
        # Getting the type of 'N' (line 232)
        N_126807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'N')
        # Assigning a type to the variable 'M' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'M', N_126807)

        if more_types_in_union_126806:
            # SSA join for if statement (line 231)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 233):
    
    # Assigning a Call to a Name (line 233):
    
    # Call to zeros(...): (line 233)
    # Processing the call arguments (line 233)
    
    # Obtaining an instance of the builtin type 'tuple' (line 233)
    tuple_126809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 233)
    # Adding element type (line 233)
    # Getting the type of 'N' (line 233)
    N_126810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 15), tuple_126809, N_126810)
    # Adding element type (line 233)
    # Getting the type of 'M' (line 233)
    M_126811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 18), 'M', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 15), tuple_126809, M_126811)
    
    # Processing the call keyword arguments (line 233)
    # Getting the type of 'dtype' (line 233)
    dtype_126812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 28), 'dtype', False)
    keyword_126813 = dtype_126812
    kwargs_126814 = {'dtype': keyword_126813}
    # Getting the type of 'zeros' (line 233)
    zeros_126808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'zeros', False)
    # Calling zeros(args, kwargs) (line 233)
    zeros_call_result_126815 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), zeros_126808, *[tuple_126809], **kwargs_126814)
    
    # Assigning a type to the variable 'm' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'm', zeros_call_result_126815)
    
    
    # Getting the type of 'k' (line 234)
    k_126816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 7), 'k')
    # Getting the type of 'M' (line 234)
    M_126817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'M')
    # Applying the binary operator '>=' (line 234)
    result_ge_126818 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 7), '>=', k_126816, M_126817)
    
    # Testing the type of an if condition (line 234)
    if_condition_126819 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 4), result_ge_126818)
    # Assigning a type to the variable 'if_condition_126819' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'if_condition_126819', if_condition_126819)
    # SSA begins for if statement (line 234)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'm' (line 235)
    m_126820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), 'm')
    # Assigning a type to the variable 'stypy_return_type' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'stypy_return_type', m_126820)
    # SSA join for if statement (line 234)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'k' (line 236)
    k_126821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 7), 'k')
    int_126822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 12), 'int')
    # Applying the binary operator '>=' (line 236)
    result_ge_126823 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 7), '>=', k_126821, int_126822)
    
    # Testing the type of an if condition (line 236)
    if_condition_126824 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 4), result_ge_126823)
    # Assigning a type to the variable 'if_condition_126824' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'if_condition_126824', if_condition_126824)
    # SSA begins for if statement (line 236)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 237):
    
    # Assigning a Name to a Name (line 237):
    # Getting the type of 'k' (line 237)
    k_126825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'k')
    # Assigning a type to the variable 'i' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'i', k_126825)
    # SSA branch for the else part of an if statement (line 236)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 239):
    
    # Assigning a BinOp to a Name (line 239):
    
    # Getting the type of 'k' (line 239)
    k_126826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 14), 'k')
    # Applying the 'usub' unary operator (line 239)
    result___neg___126827 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 13), 'usub', k_126826)
    
    # Getting the type of 'M' (line 239)
    M_126828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 19), 'M')
    # Applying the binary operator '*' (line 239)
    result_mul_126829 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 12), '*', result___neg___126827, M_126828)
    
    # Assigning a type to the variable 'i' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'i', result_mul_126829)
    # SSA join for if statement (line 236)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Subscript (line 240):
    
    # Assigning a Num to a Subscript (line 240):
    int_126830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 27), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'M' (line 240)
    M_126831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 7), 'M')
    # Getting the type of 'k' (line 240)
    k_126832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 9), 'k')
    # Applying the binary operator '-' (line 240)
    result_sub_126833 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 7), '-', M_126831, k_126832)
    
    slice_126834 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 240, 4), None, result_sub_126833, None)
    # Getting the type of 'm' (line 240)
    m_126835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'm')
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___126836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 4), m_126835, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_126837 = invoke(stypy.reporting.localization.Localization(__file__, 240, 4), getitem___126836, slice_126834)
    
    # Obtaining the member 'flat' of a type (line 240)
    flat_126838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 4), subscript_call_result_126837, 'flat')
    # Getting the type of 'i' (line 240)
    i_126839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 17), 'i')
    # Getting the type of 'M' (line 240)
    M_126840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'M')
    int_126841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 22), 'int')
    # Applying the binary operator '+' (line 240)
    result_add_126842 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 20), '+', M_126840, int_126841)
    
    slice_126843 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 240, 4), i_126839, None, result_add_126842)
    # Storing an element on a container (line 240)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 4), flat_126838, (slice_126843, int_126830))
    # Getting the type of 'm' (line 241)
    m_126844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 'm')
    # Assigning a type to the variable 'stypy_return_type' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'stypy_return_type', m_126844)
    
    # ################# End of 'eye(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eye' in the type store
    # Getting the type of 'stypy_return_type' (line 192)
    stypy_return_type_126845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126845)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eye'
    return stypy_return_type_126845

# Assigning a type to the variable 'eye' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'eye', eye)

@norecursion
def diag(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_126846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 14), 'int')
    defaults = [int_126846]
    # Create a new context for function 'diag'
    module_type_store = module_type_store.open_function_context('diag', 244, 0, False)
    
    # Passed parameters checking function
    diag.stypy_localization = localization
    diag.stypy_type_of_self = None
    diag.stypy_type_store = module_type_store
    diag.stypy_function_name = 'diag'
    diag.stypy_param_names_list = ['v', 'k']
    diag.stypy_varargs_param_name = None
    diag.stypy_kwargs_param_name = None
    diag.stypy_call_defaults = defaults
    diag.stypy_call_varargs = varargs
    diag.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'diag', ['v', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'diag', localization, ['v', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'diag(...)' code ##################

    str_126847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, (-1)), 'str', '\n    Extract a diagonal or construct a diagonal array.\n\n    See the more detailed documentation for ``numpy.diagonal`` if you use this\n    function to extract a diagonal and wish to write to the resulting array;\n    whether it returns a copy or a view depends on what version of numpy you\n    are using.\n\n    Parameters\n    ----------\n    v : array_like\n        If `v` is a 2-D array, return a copy of its `k`-th diagonal.\n        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th\n        diagonal.\n    k : int, optional\n        Diagonal in question. The default is 0. Use `k>0` for diagonals\n        above the main diagonal, and `k<0` for diagonals below the main\n        diagonal.\n\n    Returns\n    -------\n    out : ndarray\n        The extracted diagonal or constructed diagonal array.\n\n    See Also\n    --------\n    diagonal : Return specified diagonals.\n    diagflat : Create a 2-D array with the flattened input as a diagonal.\n    trace : Sum along diagonals.\n    triu : Upper triangle of an array.\n    tril : Lower triangle of an array.\n\n    Examples\n    --------\n    >>> x = np.arange(9).reshape((3,3))\n    >>> x\n    array([[0, 1, 2],\n           [3, 4, 5],\n           [6, 7, 8]])\n\n    >>> np.diag(x)\n    array([0, 4, 8])\n    >>> np.diag(x, k=1)\n    array([1, 5])\n    >>> np.diag(x, k=-1)\n    array([3, 7])\n\n    >>> np.diag(np.diag(x))\n    array([[0, 0, 0],\n           [0, 4, 0],\n           [0, 0, 8]])\n\n    ')
    
    # Assigning a Call to a Name (line 298):
    
    # Assigning a Call to a Name (line 298):
    
    # Call to asanyarray(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'v' (line 298)
    v_126849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 19), 'v', False)
    # Processing the call keyword arguments (line 298)
    kwargs_126850 = {}
    # Getting the type of 'asanyarray' (line 298)
    asanyarray_126848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 298)
    asanyarray_call_result_126851 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), asanyarray_126848, *[v_126849], **kwargs_126850)
    
    # Assigning a type to the variable 'v' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'v', asanyarray_call_result_126851)
    
    # Assigning a Attribute to a Name (line 299):
    
    # Assigning a Attribute to a Name (line 299):
    # Getting the type of 'v' (line 299)
    v_126852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'v')
    # Obtaining the member 'shape' of a type (line 299)
    shape_126853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), v_126852, 'shape')
    # Assigning a type to the variable 's' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 's', shape_126853)
    
    
    
    # Call to len(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 's' (line 300)
    s_126855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 11), 's', False)
    # Processing the call keyword arguments (line 300)
    kwargs_126856 = {}
    # Getting the type of 'len' (line 300)
    len_126854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 7), 'len', False)
    # Calling len(args, kwargs) (line 300)
    len_call_result_126857 = invoke(stypy.reporting.localization.Localization(__file__, 300, 7), len_126854, *[s_126855], **kwargs_126856)
    
    int_126858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 17), 'int')
    # Applying the binary operator '==' (line 300)
    result_eq_126859 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 7), '==', len_call_result_126857, int_126858)
    
    # Testing the type of an if condition (line 300)
    if_condition_126860 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 4), result_eq_126859)
    # Assigning a type to the variable 'if_condition_126860' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'if_condition_126860', if_condition_126860)
    # SSA begins for if statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 301):
    
    # Assigning a BinOp to a Name (line 301):
    
    # Obtaining the type of the subscript
    int_126861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 14), 'int')
    # Getting the type of 's' (line 301)
    s_126862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 's')
    # Obtaining the member '__getitem__' of a type (line 301)
    getitem___126863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), s_126862, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 301)
    subscript_call_result_126864 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), getitem___126863, int_126861)
    
    
    # Call to abs(...): (line 301)
    # Processing the call arguments (line 301)
    # Getting the type of 'k' (line 301)
    k_126866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 21), 'k', False)
    # Processing the call keyword arguments (line 301)
    kwargs_126867 = {}
    # Getting the type of 'abs' (line 301)
    abs_126865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 17), 'abs', False)
    # Calling abs(args, kwargs) (line 301)
    abs_call_result_126868 = invoke(stypy.reporting.localization.Localization(__file__, 301, 17), abs_126865, *[k_126866], **kwargs_126867)
    
    # Applying the binary operator '+' (line 301)
    result_add_126869 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 12), '+', subscript_call_result_126864, abs_call_result_126868)
    
    # Assigning a type to the variable 'n' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'n', result_add_126869)
    
    # Assigning a Call to a Name (line 302):
    
    # Assigning a Call to a Name (line 302):
    
    # Call to zeros(...): (line 302)
    # Processing the call arguments (line 302)
    
    # Obtaining an instance of the builtin type 'tuple' (line 302)
    tuple_126871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 302)
    # Adding element type (line 302)
    # Getting the type of 'n' (line 302)
    n_126872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 21), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 21), tuple_126871, n_126872)
    # Adding element type (line 302)
    # Getting the type of 'n' (line 302)
    n_126873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 21), tuple_126871, n_126873)
    
    # Getting the type of 'v' (line 302)
    v_126874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 28), 'v', False)
    # Obtaining the member 'dtype' of a type (line 302)
    dtype_126875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 28), v_126874, 'dtype')
    # Processing the call keyword arguments (line 302)
    kwargs_126876 = {}
    # Getting the type of 'zeros' (line 302)
    zeros_126870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 14), 'zeros', False)
    # Calling zeros(args, kwargs) (line 302)
    zeros_call_result_126877 = invoke(stypy.reporting.localization.Localization(__file__, 302, 14), zeros_126870, *[tuple_126871, dtype_126875], **kwargs_126876)
    
    # Assigning a type to the variable 'res' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'res', zeros_call_result_126877)
    
    
    # Getting the type of 'k' (line 303)
    k_126878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 11), 'k')
    int_126879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 16), 'int')
    # Applying the binary operator '>=' (line 303)
    result_ge_126880 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 11), '>=', k_126878, int_126879)
    
    # Testing the type of an if condition (line 303)
    if_condition_126881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 8), result_ge_126880)
    # Assigning a type to the variable 'if_condition_126881' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'if_condition_126881', if_condition_126881)
    # SSA begins for if statement (line 303)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 304):
    
    # Assigning a Name to a Name (line 304):
    # Getting the type of 'k' (line 304)
    k_126882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'k')
    # Assigning a type to the variable 'i' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'i', k_126882)
    # SSA branch for the else part of an if statement (line 303)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 306):
    
    # Assigning a BinOp to a Name (line 306):
    
    # Getting the type of 'k' (line 306)
    k_126883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 18), 'k')
    # Applying the 'usub' unary operator (line 306)
    result___neg___126884 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 17), 'usub', k_126883)
    
    # Getting the type of 'n' (line 306)
    n_126885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 23), 'n')
    # Applying the binary operator '*' (line 306)
    result_mul_126886 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 16), '*', result___neg___126884, n_126885)
    
    # Assigning a type to the variable 'i' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'i', result_mul_126886)
    # SSA join for if statement (line 303)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 307):
    
    # Assigning a Name to a Subscript (line 307):
    # Getting the type of 'v' (line 307)
    v_126887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 33), 'v')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 307)
    n_126888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 13), 'n')
    # Getting the type of 'k' (line 307)
    k_126889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), 'k')
    # Applying the binary operator '-' (line 307)
    result_sub_126890 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 13), '-', n_126888, k_126889)
    
    slice_126891 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 307, 8), None, result_sub_126890, None)
    # Getting the type of 'res' (line 307)
    res_126892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'res')
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___126893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), res_126892, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_126894 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), getitem___126893, slice_126891)
    
    # Obtaining the member 'flat' of a type (line 307)
    flat_126895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), subscript_call_result_126894, 'flat')
    # Getting the type of 'i' (line 307)
    i_126896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 23), 'i')
    # Getting the type of 'n' (line 307)
    n_126897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 26), 'n')
    int_126898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 28), 'int')
    # Applying the binary operator '+' (line 307)
    result_add_126899 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 26), '+', n_126897, int_126898)
    
    slice_126900 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 307, 8), i_126896, None, result_add_126899)
    # Storing an element on a container (line 307)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 8), flat_126895, (slice_126900, v_126887))
    # Getting the type of 'res' (line 308)
    res_126901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 15), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'stypy_return_type', res_126901)
    # SSA branch for the else part of an if statement (line 300)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 's' (line 309)
    s_126903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 13), 's', False)
    # Processing the call keyword arguments (line 309)
    kwargs_126904 = {}
    # Getting the type of 'len' (line 309)
    len_126902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 9), 'len', False)
    # Calling len(args, kwargs) (line 309)
    len_call_result_126905 = invoke(stypy.reporting.localization.Localization(__file__, 309, 9), len_126902, *[s_126903], **kwargs_126904)
    
    int_126906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 19), 'int')
    # Applying the binary operator '==' (line 309)
    result_eq_126907 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 9), '==', len_call_result_126905, int_126906)
    
    # Testing the type of an if condition (line 309)
    if_condition_126908 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 309, 9), result_eq_126907)
    # Assigning a type to the variable 'if_condition_126908' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 9), 'if_condition_126908', if_condition_126908)
    # SSA begins for if statement (line 309)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to diagonal(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'v' (line 310)
    v_126910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 24), 'v', False)
    # Getting the type of 'k' (line 310)
    k_126911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 27), 'k', False)
    # Processing the call keyword arguments (line 310)
    kwargs_126912 = {}
    # Getting the type of 'diagonal' (line 310)
    diagonal_126909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'diagonal', False)
    # Calling diagonal(args, kwargs) (line 310)
    diagonal_call_result_126913 = invoke(stypy.reporting.localization.Localization(__file__, 310, 15), diagonal_126909, *[v_126910, k_126911], **kwargs_126912)
    
    # Assigning a type to the variable 'stypy_return_type' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'stypy_return_type', diagonal_call_result_126913)
    # SSA branch for the else part of an if statement (line 309)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 312)
    # Processing the call arguments (line 312)
    str_126915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 25), 'str', 'Input must be 1- or 2-d.')
    # Processing the call keyword arguments (line 312)
    kwargs_126916 = {}
    # Getting the type of 'ValueError' (line 312)
    ValueError_126914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 312)
    ValueError_call_result_126917 = invoke(stypy.reporting.localization.Localization(__file__, 312, 14), ValueError_126914, *[str_126915], **kwargs_126916)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 312, 8), ValueError_call_result_126917, 'raise parameter', BaseException)
    # SSA join for if statement (line 309)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 300)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'diag(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'diag' in the type store
    # Getting the type of 'stypy_return_type' (line 244)
    stypy_return_type_126918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126918)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'diag'
    return stypy_return_type_126918

# Assigning a type to the variable 'diag' (line 244)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'diag', diag)

@norecursion
def diagflat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_126919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 18), 'int')
    defaults = [int_126919]
    # Create a new context for function 'diagflat'
    module_type_store = module_type_store.open_function_context('diagflat', 315, 0, False)
    
    # Passed parameters checking function
    diagflat.stypy_localization = localization
    diagflat.stypy_type_of_self = None
    diagflat.stypy_type_store = module_type_store
    diagflat.stypy_function_name = 'diagflat'
    diagflat.stypy_param_names_list = ['v', 'k']
    diagflat.stypy_varargs_param_name = None
    diagflat.stypy_kwargs_param_name = None
    diagflat.stypy_call_defaults = defaults
    diagflat.stypy_call_varargs = varargs
    diagflat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'diagflat', ['v', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'diagflat', localization, ['v', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'diagflat(...)' code ##################

    str_126920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, (-1)), 'str', '\n    Create a two-dimensional array with the flattened input as a diagonal.\n\n    Parameters\n    ----------\n    v : array_like\n        Input data, which is flattened and set as the `k`-th\n        diagonal of the output.\n    k : int, optional\n        Diagonal to set; 0, the default, corresponds to the "main" diagonal,\n        a positive (negative) `k` giving the number of the diagonal above\n        (below) the main.\n\n    Returns\n    -------\n    out : ndarray\n        The 2-D output array.\n\n    See Also\n    --------\n    diag : MATLAB work-alike for 1-D and 2-D arrays.\n    diagonal : Return specified diagonals.\n    trace : Sum along diagonals.\n\n    Examples\n    --------\n    >>> np.diagflat([[1,2], [3,4]])\n    array([[1, 0, 0, 0],\n           [0, 2, 0, 0],\n           [0, 0, 3, 0],\n           [0, 0, 0, 4]])\n\n    >>> np.diagflat([1,2], 1)\n    array([[0, 1, 0],\n           [0, 0, 2],\n           [0, 0, 0]])\n\n    ')
    
    
    # SSA begins for try-except statement (line 354)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 355):
    
    # Assigning a Attribute to a Name (line 355):
    # Getting the type of 'v' (line 355)
    v_126921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 15), 'v')
    # Obtaining the member '__array_wrap__' of a type (line 355)
    array_wrap___126922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 15), v_126921, '__array_wrap__')
    # Assigning a type to the variable 'wrap' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'wrap', array_wrap___126922)
    # SSA branch for the except part of a try statement (line 354)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 354)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 357):
    
    # Assigning a Name to a Name (line 357):
    # Getting the type of 'None' (line 357)
    None_126923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 15), 'None')
    # Assigning a type to the variable 'wrap' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'wrap', None_126923)
    # SSA join for try-except statement (line 354)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 358):
    
    # Assigning a Call to a Name (line 358):
    
    # Call to ravel(...): (line 358)
    # Processing the call keyword arguments (line 358)
    kwargs_126929 = {}
    
    # Call to asarray(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'v' (line 358)
    v_126925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'v', False)
    # Processing the call keyword arguments (line 358)
    kwargs_126926 = {}
    # Getting the type of 'asarray' (line 358)
    asarray_126924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 358)
    asarray_call_result_126927 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), asarray_126924, *[v_126925], **kwargs_126926)
    
    # Obtaining the member 'ravel' of a type (line 358)
    ravel_126928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), asarray_call_result_126927, 'ravel')
    # Calling ravel(args, kwargs) (line 358)
    ravel_call_result_126930 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), ravel_126928, *[], **kwargs_126929)
    
    # Assigning a type to the variable 'v' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'v', ravel_call_result_126930)
    
    # Assigning a Call to a Name (line 359):
    
    # Assigning a Call to a Name (line 359):
    
    # Call to len(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of 'v' (line 359)
    v_126932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'v', False)
    # Processing the call keyword arguments (line 359)
    kwargs_126933 = {}
    # Getting the type of 'len' (line 359)
    len_126931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'len', False)
    # Calling len(args, kwargs) (line 359)
    len_call_result_126934 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), len_126931, *[v_126932], **kwargs_126933)
    
    # Assigning a type to the variable 's' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 's', len_call_result_126934)
    
    # Assigning a BinOp to a Name (line 360):
    
    # Assigning a BinOp to a Name (line 360):
    # Getting the type of 's' (line 360)
    s_126935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 's')
    
    # Call to abs(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'k' (line 360)
    k_126937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 16), 'k', False)
    # Processing the call keyword arguments (line 360)
    kwargs_126938 = {}
    # Getting the type of 'abs' (line 360)
    abs_126936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'abs', False)
    # Calling abs(args, kwargs) (line 360)
    abs_call_result_126939 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), abs_126936, *[k_126937], **kwargs_126938)
    
    # Applying the binary operator '+' (line 360)
    result_add_126940 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 8), '+', s_126935, abs_call_result_126939)
    
    # Assigning a type to the variable 'n' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'n', result_add_126940)
    
    # Assigning a Call to a Name (line 361):
    
    # Assigning a Call to a Name (line 361):
    
    # Call to zeros(...): (line 361)
    # Processing the call arguments (line 361)
    
    # Obtaining an instance of the builtin type 'tuple' (line 361)
    tuple_126942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 361)
    # Adding element type (line 361)
    # Getting the type of 'n' (line 361)
    n_126943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 17), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 17), tuple_126942, n_126943)
    # Adding element type (line 361)
    # Getting the type of 'n' (line 361)
    n_126944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 17), tuple_126942, n_126944)
    
    # Getting the type of 'v' (line 361)
    v_126945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 24), 'v', False)
    # Obtaining the member 'dtype' of a type (line 361)
    dtype_126946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 24), v_126945, 'dtype')
    # Processing the call keyword arguments (line 361)
    kwargs_126947 = {}
    # Getting the type of 'zeros' (line 361)
    zeros_126941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 10), 'zeros', False)
    # Calling zeros(args, kwargs) (line 361)
    zeros_call_result_126948 = invoke(stypy.reporting.localization.Localization(__file__, 361, 10), zeros_126941, *[tuple_126942, dtype_126946], **kwargs_126947)
    
    # Assigning a type to the variable 'res' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'res', zeros_call_result_126948)
    
    
    # Getting the type of 'k' (line 362)
    k_126949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'k')
    int_126950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 13), 'int')
    # Applying the binary operator '>=' (line 362)
    result_ge_126951 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 8), '>=', k_126949, int_126950)
    
    # Testing the type of an if condition (line 362)
    if_condition_126952 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 4), result_ge_126951)
    # Assigning a type to the variable 'if_condition_126952' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'if_condition_126952', if_condition_126952)
    # SSA begins for if statement (line 362)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 363):
    
    # Assigning a Call to a Name (line 363):
    
    # Call to arange(...): (line 363)
    # Processing the call arguments (line 363)
    int_126954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 19), 'int')
    # Getting the type of 'n' (line 363)
    n_126955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 22), 'n', False)
    # Getting the type of 'k' (line 363)
    k_126956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 24), 'k', False)
    # Applying the binary operator '-' (line 363)
    result_sub_126957 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 22), '-', n_126955, k_126956)
    
    # Processing the call keyword arguments (line 363)
    kwargs_126958 = {}
    # Getting the type of 'arange' (line 363)
    arange_126953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'arange', False)
    # Calling arange(args, kwargs) (line 363)
    arange_call_result_126959 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), arange_126953, *[int_126954, result_sub_126957], **kwargs_126958)
    
    # Assigning a type to the variable 'i' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'i', arange_call_result_126959)
    
    # Assigning a BinOp to a Name (line 364):
    
    # Assigning a BinOp to a Name (line 364):
    # Getting the type of 'i' (line 364)
    i_126960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 13), 'i')
    # Getting the type of 'k' (line 364)
    k_126961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 15), 'k')
    # Applying the binary operator '+' (line 364)
    result_add_126962 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 13), '+', i_126960, k_126961)
    
    # Getting the type of 'i' (line 364)
    i_126963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 17), 'i')
    # Getting the type of 'n' (line 364)
    n_126964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 19), 'n')
    # Applying the binary operator '*' (line 364)
    result_mul_126965 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 17), '*', i_126963, n_126964)
    
    # Applying the binary operator '+' (line 364)
    result_add_126966 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 16), '+', result_add_126962, result_mul_126965)
    
    # Assigning a type to the variable 'fi' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'fi', result_add_126966)
    # SSA branch for the else part of an if statement (line 362)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 366):
    
    # Assigning a Call to a Name (line 366):
    
    # Call to arange(...): (line 366)
    # Processing the call arguments (line 366)
    int_126968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 19), 'int')
    # Getting the type of 'n' (line 366)
    n_126969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'n', False)
    # Getting the type of 'k' (line 366)
    k_126970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 24), 'k', False)
    # Applying the binary operator '+' (line 366)
    result_add_126971 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 22), '+', n_126969, k_126970)
    
    # Processing the call keyword arguments (line 366)
    kwargs_126972 = {}
    # Getting the type of 'arange' (line 366)
    arange_126967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'arange', False)
    # Calling arange(args, kwargs) (line 366)
    arange_call_result_126973 = invoke(stypy.reporting.localization.Localization(__file__, 366, 12), arange_126967, *[int_126968, result_add_126971], **kwargs_126972)
    
    # Assigning a type to the variable 'i' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'i', arange_call_result_126973)
    
    # Assigning a BinOp to a Name (line 367):
    
    # Assigning a BinOp to a Name (line 367):
    # Getting the type of 'i' (line 367)
    i_126974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 13), 'i')
    # Getting the type of 'i' (line 367)
    i_126975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'i')
    # Getting the type of 'k' (line 367)
    k_126976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 18), 'k')
    # Applying the binary operator '-' (line 367)
    result_sub_126977 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 16), '-', i_126975, k_126976)
    
    # Getting the type of 'n' (line 367)
    n_126978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 21), 'n')
    # Applying the binary operator '*' (line 367)
    result_mul_126979 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 15), '*', result_sub_126977, n_126978)
    
    # Applying the binary operator '+' (line 367)
    result_add_126980 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 13), '+', i_126974, result_mul_126979)
    
    # Assigning a type to the variable 'fi' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'fi', result_add_126980)
    # SSA join for if statement (line 362)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 368):
    
    # Assigning a Name to a Subscript (line 368):
    # Getting the type of 'v' (line 368)
    v_126981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 19), 'v')
    # Getting the type of 'res' (line 368)
    res_126982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'res')
    # Obtaining the member 'flat' of a type (line 368)
    flat_126983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 4), res_126982, 'flat')
    # Getting the type of 'fi' (line 368)
    fi_126984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 13), 'fi')
    # Storing an element on a container (line 368)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 4), flat_126983, (fi_126984, v_126981))
    
    
    # Getting the type of 'wrap' (line 369)
    wrap_126985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 11), 'wrap')
    # Applying the 'not' unary operator (line 369)
    result_not__126986 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 7), 'not', wrap_126985)
    
    # Testing the type of an if condition (line 369)
    if_condition_126987 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 4), result_not__126986)
    # Assigning a type to the variable 'if_condition_126987' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'if_condition_126987', if_condition_126987)
    # SSA begins for if statement (line 369)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'res' (line 370)
    res_126988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 15), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'stypy_return_type', res_126988)
    # SSA join for if statement (line 369)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to wrap(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of 'res' (line 371)
    res_126990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'res', False)
    # Processing the call keyword arguments (line 371)
    kwargs_126991 = {}
    # Getting the type of 'wrap' (line 371)
    wrap_126989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 11), 'wrap', False)
    # Calling wrap(args, kwargs) (line 371)
    wrap_call_result_126992 = invoke(stypy.reporting.localization.Localization(__file__, 371, 11), wrap_126989, *[res_126990], **kwargs_126991)
    
    # Assigning a type to the variable 'stypy_return_type' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type', wrap_call_result_126992)
    
    # ################# End of 'diagflat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'diagflat' in the type store
    # Getting the type of 'stypy_return_type' (line 315)
    stypy_return_type_126993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126993)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'diagflat'
    return stypy_return_type_126993

# Assigning a type to the variable 'diagflat' (line 315)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 0), 'diagflat', diagflat)

@norecursion
def tri(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 374)
    None_126994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 13), 'None')
    int_126995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 21), 'int')
    # Getting the type of 'float' (line 374)
    float_126996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 30), 'float')
    defaults = [None_126994, int_126995, float_126996]
    # Create a new context for function 'tri'
    module_type_store = module_type_store.open_function_context('tri', 374, 0, False)
    
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

    str_126997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, (-1)), 'str', '\n    An array with ones at and below the given diagonal and zeros elsewhere.\n\n    Parameters\n    ----------\n    N : int\n        Number of rows in the array.\n    M : int, optional\n        Number of columns in the array.\n        By default, `M` is taken equal to `N`.\n    k : int, optional\n        The sub-diagonal at and below which the array is filled.\n        `k` = 0 is the main diagonal, while `k` < 0 is below it,\n        and `k` > 0 is above.  The default is 0.\n    dtype : dtype, optional\n        Data type of the returned array.  The default is float.\n\n    Returns\n    -------\n    tri : ndarray of shape (N, M)\n        Array with its lower triangle filled with ones and zero elsewhere;\n        in other words ``T[i,j] == 1`` for ``i <= j + k``, 0 otherwise.\n\n    Examples\n    --------\n    >>> np.tri(3, 5, 2, dtype=int)\n    array([[1, 1, 1, 0, 0],\n           [1, 1, 1, 1, 0],\n           [1, 1, 1, 1, 1]])\n\n    >>> np.tri(3, 5, -1)\n    array([[ 0.,  0.,  0.,  0.,  0.],\n           [ 1.,  0.,  0.,  0.,  0.],\n           [ 1.,  1.,  0.,  0.,  0.]])\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 411)
    # Getting the type of 'M' (line 411)
    M_126998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 7), 'M')
    # Getting the type of 'None' (line 411)
    None_126999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'None')
    
    (may_be_127000, more_types_in_union_127001) = may_be_none(M_126998, None_126999)

    if may_be_127000:

        if more_types_in_union_127001:
            # Runtime conditional SSA (line 411)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 412):
        
        # Assigning a Name to a Name (line 412):
        # Getting the type of 'N' (line 412)
        N_127002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'N')
        # Assigning a type to the variable 'M' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'M', N_127002)

        if more_types_in_union_127001:
            # SSA join for if statement (line 411)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 414):
    
    # Assigning a Call to a Name (line 414):
    
    # Call to outer(...): (line 414)
    # Processing the call arguments (line 414)
    
    # Call to arange(...): (line 414)
    # Processing the call arguments (line 414)
    # Getting the type of 'N' (line 414)
    N_127006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 35), 'N', False)
    # Processing the call keyword arguments (line 414)
    
    # Call to _min_int(...): (line 414)
    # Processing the call arguments (line 414)
    int_127008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 53), 'int')
    # Getting the type of 'N' (line 414)
    N_127009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 56), 'N', False)
    # Processing the call keyword arguments (line 414)
    kwargs_127010 = {}
    # Getting the type of '_min_int' (line 414)
    _min_int_127007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 44), '_min_int', False)
    # Calling _min_int(args, kwargs) (line 414)
    _min_int_call_result_127011 = invoke(stypy.reporting.localization.Localization(__file__, 414, 44), _min_int_127007, *[int_127008, N_127009], **kwargs_127010)
    
    keyword_127012 = _min_int_call_result_127011
    kwargs_127013 = {'dtype': keyword_127012}
    # Getting the type of 'arange' (line 414)
    arange_127005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 28), 'arange', False)
    # Calling arange(args, kwargs) (line 414)
    arange_call_result_127014 = invoke(stypy.reporting.localization.Localization(__file__, 414, 28), arange_127005, *[N_127006], **kwargs_127013)
    
    
    # Call to arange(...): (line 415)
    # Processing the call arguments (line 415)
    
    # Getting the type of 'k' (line 415)
    k_127016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 36), 'k', False)
    # Applying the 'usub' unary operator (line 415)
    result___neg___127017 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 35), 'usub', k_127016)
    
    # Getting the type of 'M' (line 415)
    M_127018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 39), 'M', False)
    # Getting the type of 'k' (line 415)
    k_127019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 41), 'k', False)
    # Applying the binary operator '-' (line 415)
    result_sub_127020 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 39), '-', M_127018, k_127019)
    
    # Processing the call keyword arguments (line 415)
    
    # Call to _min_int(...): (line 415)
    # Processing the call arguments (line 415)
    
    # Getting the type of 'k' (line 415)
    k_127022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 60), 'k', False)
    # Applying the 'usub' unary operator (line 415)
    result___neg___127023 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 59), 'usub', k_127022)
    
    # Getting the type of 'M' (line 415)
    M_127024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 63), 'M', False)
    # Getting the type of 'k' (line 415)
    k_127025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 67), 'k', False)
    # Applying the binary operator '-' (line 415)
    result_sub_127026 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 63), '-', M_127024, k_127025)
    
    # Processing the call keyword arguments (line 415)
    kwargs_127027 = {}
    # Getting the type of '_min_int' (line 415)
    _min_int_127021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 50), '_min_int', False)
    # Calling _min_int(args, kwargs) (line 415)
    _min_int_call_result_127028 = invoke(stypy.reporting.localization.Localization(__file__, 415, 50), _min_int_127021, *[result___neg___127023, result_sub_127026], **kwargs_127027)
    
    keyword_127029 = _min_int_call_result_127028
    kwargs_127030 = {'dtype': keyword_127029}
    # Getting the type of 'arange' (line 415)
    arange_127015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 28), 'arange', False)
    # Calling arange(args, kwargs) (line 415)
    arange_call_result_127031 = invoke(stypy.reporting.localization.Localization(__file__, 415, 28), arange_127015, *[result___neg___127017, result_sub_127020], **kwargs_127030)
    
    # Processing the call keyword arguments (line 414)
    kwargs_127032 = {}
    # Getting the type of 'greater_equal' (line 414)
    greater_equal_127003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'greater_equal', False)
    # Obtaining the member 'outer' of a type (line 414)
    outer_127004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 8), greater_equal_127003, 'outer')
    # Calling outer(args, kwargs) (line 414)
    outer_call_result_127033 = invoke(stypy.reporting.localization.Localization(__file__, 414, 8), outer_127004, *[arange_call_result_127014, arange_call_result_127031], **kwargs_127032)
    
    # Assigning a type to the variable 'm' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'm', outer_call_result_127033)
    
    # Assigning a Call to a Name (line 418):
    
    # Assigning a Call to a Name (line 418):
    
    # Call to astype(...): (line 418)
    # Processing the call arguments (line 418)
    # Getting the type of 'dtype' (line 418)
    dtype_127036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 17), 'dtype', False)
    # Processing the call keyword arguments (line 418)
    # Getting the type of 'False' (line 418)
    False_127037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 29), 'False', False)
    keyword_127038 = False_127037
    kwargs_127039 = {'copy': keyword_127038}
    # Getting the type of 'm' (line 418)
    m_127034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'm', False)
    # Obtaining the member 'astype' of a type (line 418)
    astype_127035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), m_127034, 'astype')
    # Calling astype(args, kwargs) (line 418)
    astype_call_result_127040 = invoke(stypy.reporting.localization.Localization(__file__, 418, 8), astype_127035, *[dtype_127036], **kwargs_127039)
    
    # Assigning a type to the variable 'm' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'm', astype_call_result_127040)
    # Getting the type of 'm' (line 420)
    m_127041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 11), 'm')
    # Assigning a type to the variable 'stypy_return_type' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'stypy_return_type', m_127041)
    
    # ################# End of 'tri(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tri' in the type store
    # Getting the type of 'stypy_return_type' (line 374)
    stypy_return_type_127042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127042)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tri'
    return stypy_return_type_127042

# Assigning a type to the variable 'tri' (line 374)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 0), 'tri', tri)

@norecursion
def tril(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_127043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 14), 'int')
    defaults = [int_127043]
    # Create a new context for function 'tril'
    module_type_store = module_type_store.open_function_context('tril', 423, 0, False)
    
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

    str_127044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, (-1)), 'str', '\n    Lower triangle of an array.\n\n    Return a copy of an array with elements above the `k`-th diagonal zeroed.\n\n    Parameters\n    ----------\n    m : array_like, shape (M, N)\n        Input array.\n    k : int, optional\n        Diagonal above which to zero elements.  `k = 0` (the default) is the\n        main diagonal, `k < 0` is below it and `k > 0` is above.\n\n    Returns\n    -------\n    tril : ndarray, shape (M, N)\n        Lower triangle of `m`, of same shape and data-type as `m`.\n\n    See Also\n    --------\n    triu : same thing, only for the upper triangle\n\n    Examples\n    --------\n    >>> np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)\n    array([[ 0,  0,  0],\n           [ 4,  0,  0],\n           [ 7,  8,  0],\n           [10, 11, 12]])\n\n    ')
    
    # Assigning a Call to a Name (line 455):
    
    # Assigning a Call to a Name (line 455):
    
    # Call to asanyarray(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'm' (line 455)
    m_127046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 19), 'm', False)
    # Processing the call keyword arguments (line 455)
    kwargs_127047 = {}
    # Getting the type of 'asanyarray' (line 455)
    asanyarray_127045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 455)
    asanyarray_call_result_127048 = invoke(stypy.reporting.localization.Localization(__file__, 455, 8), asanyarray_127045, *[m_127046], **kwargs_127047)
    
    # Assigning a type to the variable 'm' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'm', asanyarray_call_result_127048)
    
    # Assigning a Call to a Name (line 456):
    
    # Assigning a Call to a Name (line 456):
    
    # Call to tri(...): (line 456)
    
    # Obtaining the type of the subscript
    int_127050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 24), 'int')
    slice_127051 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 456, 16), int_127050, None, None)
    # Getting the type of 'm' (line 456)
    m_127052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'm', False)
    # Obtaining the member 'shape' of a type (line 456)
    shape_127053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 16), m_127052, 'shape')
    # Obtaining the member '__getitem__' of a type (line 456)
    getitem___127054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 16), shape_127053, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 456)
    subscript_call_result_127055 = invoke(stypy.reporting.localization.Localization(__file__, 456, 16), getitem___127054, slice_127051)
    
    # Processing the call keyword arguments (line 456)
    # Getting the type of 'k' (line 456)
    k_127056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 32), 'k', False)
    keyword_127057 = k_127056
    # Getting the type of 'bool' (line 456)
    bool_127058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 41), 'bool', False)
    keyword_127059 = bool_127058
    kwargs_127060 = {'dtype': keyword_127059, 'k': keyword_127057}
    # Getting the type of 'tri' (line 456)
    tri_127049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 11), 'tri', False)
    # Calling tri(args, kwargs) (line 456)
    tri_call_result_127061 = invoke(stypy.reporting.localization.Localization(__file__, 456, 11), tri_127049, *[subscript_call_result_127055], **kwargs_127060)
    
    # Assigning a type to the variable 'mask' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'mask', tri_call_result_127061)
    
    # Call to where(...): (line 458)
    # Processing the call arguments (line 458)
    # Getting the type of 'mask' (line 458)
    mask_127063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 17), 'mask', False)
    # Getting the type of 'm' (line 458)
    m_127064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 23), 'm', False)
    
    # Call to zeros(...): (line 458)
    # Processing the call arguments (line 458)
    int_127066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 32), 'int')
    # Getting the type of 'm' (line 458)
    m_127067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 35), 'm', False)
    # Obtaining the member 'dtype' of a type (line 458)
    dtype_127068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 35), m_127067, 'dtype')
    # Processing the call keyword arguments (line 458)
    kwargs_127069 = {}
    # Getting the type of 'zeros' (line 458)
    zeros_127065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 26), 'zeros', False)
    # Calling zeros(args, kwargs) (line 458)
    zeros_call_result_127070 = invoke(stypy.reporting.localization.Localization(__file__, 458, 26), zeros_127065, *[int_127066, dtype_127068], **kwargs_127069)
    
    # Processing the call keyword arguments (line 458)
    kwargs_127071 = {}
    # Getting the type of 'where' (line 458)
    where_127062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 11), 'where', False)
    # Calling where(args, kwargs) (line 458)
    where_call_result_127072 = invoke(stypy.reporting.localization.Localization(__file__, 458, 11), where_127062, *[mask_127063, m_127064, zeros_call_result_127070], **kwargs_127071)
    
    # Assigning a type to the variable 'stypy_return_type' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'stypy_return_type', where_call_result_127072)
    
    # ################# End of 'tril(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tril' in the type store
    # Getting the type of 'stypy_return_type' (line 423)
    stypy_return_type_127073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127073)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tril'
    return stypy_return_type_127073

# Assigning a type to the variable 'tril' (line 423)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 0), 'tril', tril)

@norecursion
def triu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_127074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 14), 'int')
    defaults = [int_127074]
    # Create a new context for function 'triu'
    module_type_store = module_type_store.open_function_context('triu', 461, 0, False)
    
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

    str_127075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, (-1)), 'str', '\n    Upper triangle of an array.\n\n    Return a copy of a matrix with the elements below the `k`-th diagonal\n    zeroed.\n\n    Please refer to the documentation for `tril` for further details.\n\n    See Also\n    --------\n    tril : lower triangle of an array\n\n    Examples\n    --------\n    >>> np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)\n    array([[ 1,  2,  3],\n           [ 4,  5,  6],\n           [ 0,  8,  9],\n           [ 0,  0, 12]])\n\n    ')
    
    # Assigning a Call to a Name (line 483):
    
    # Assigning a Call to a Name (line 483):
    
    # Call to asanyarray(...): (line 483)
    # Processing the call arguments (line 483)
    # Getting the type of 'm' (line 483)
    m_127077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 19), 'm', False)
    # Processing the call keyword arguments (line 483)
    kwargs_127078 = {}
    # Getting the type of 'asanyarray' (line 483)
    asanyarray_127076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 483)
    asanyarray_call_result_127079 = invoke(stypy.reporting.localization.Localization(__file__, 483, 8), asanyarray_127076, *[m_127077], **kwargs_127078)
    
    # Assigning a type to the variable 'm' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'm', asanyarray_call_result_127079)
    
    # Assigning a Call to a Name (line 484):
    
    # Assigning a Call to a Name (line 484):
    
    # Call to tri(...): (line 484)
    
    # Obtaining the type of the subscript
    int_127081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 24), 'int')
    slice_127082 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 484, 16), int_127081, None, None)
    # Getting the type of 'm' (line 484)
    m_127083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 16), 'm', False)
    # Obtaining the member 'shape' of a type (line 484)
    shape_127084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 16), m_127083, 'shape')
    # Obtaining the member '__getitem__' of a type (line 484)
    getitem___127085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 16), shape_127084, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 484)
    subscript_call_result_127086 = invoke(stypy.reporting.localization.Localization(__file__, 484, 16), getitem___127085, slice_127082)
    
    # Processing the call keyword arguments (line 484)
    # Getting the type of 'k' (line 484)
    k_127087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 32), 'k', False)
    int_127088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 34), 'int')
    # Applying the binary operator '-' (line 484)
    result_sub_127089 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 32), '-', k_127087, int_127088)
    
    keyword_127090 = result_sub_127089
    # Getting the type of 'bool' (line 484)
    bool_127091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 43), 'bool', False)
    keyword_127092 = bool_127091
    kwargs_127093 = {'dtype': keyword_127092, 'k': keyword_127090}
    # Getting the type of 'tri' (line 484)
    tri_127080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 11), 'tri', False)
    # Calling tri(args, kwargs) (line 484)
    tri_call_result_127094 = invoke(stypy.reporting.localization.Localization(__file__, 484, 11), tri_127080, *[subscript_call_result_127086], **kwargs_127093)
    
    # Assigning a type to the variable 'mask' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'mask', tri_call_result_127094)
    
    # Call to where(...): (line 486)
    # Processing the call arguments (line 486)
    # Getting the type of 'mask' (line 486)
    mask_127096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 17), 'mask', False)
    
    # Call to zeros(...): (line 486)
    # Processing the call arguments (line 486)
    int_127098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 29), 'int')
    # Getting the type of 'm' (line 486)
    m_127099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 32), 'm', False)
    # Obtaining the member 'dtype' of a type (line 486)
    dtype_127100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 32), m_127099, 'dtype')
    # Processing the call keyword arguments (line 486)
    kwargs_127101 = {}
    # Getting the type of 'zeros' (line 486)
    zeros_127097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 23), 'zeros', False)
    # Calling zeros(args, kwargs) (line 486)
    zeros_call_result_127102 = invoke(stypy.reporting.localization.Localization(__file__, 486, 23), zeros_127097, *[int_127098, dtype_127100], **kwargs_127101)
    
    # Getting the type of 'm' (line 486)
    m_127103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 42), 'm', False)
    # Processing the call keyword arguments (line 486)
    kwargs_127104 = {}
    # Getting the type of 'where' (line 486)
    where_127095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 11), 'where', False)
    # Calling where(args, kwargs) (line 486)
    where_call_result_127105 = invoke(stypy.reporting.localization.Localization(__file__, 486, 11), where_127095, *[mask_127096, zeros_call_result_127102, m_127103], **kwargs_127104)
    
    # Assigning a type to the variable 'stypy_return_type' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'stypy_return_type', where_call_result_127105)
    
    # ################# End of 'triu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'triu' in the type store
    # Getting the type of 'stypy_return_type' (line 461)
    stypy_return_type_127106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127106)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'triu'
    return stypy_return_type_127106

# Assigning a type to the variable 'triu' (line 461)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 0), 'triu', triu)

@norecursion
def vander(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 490)
    None_127107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 16), 'None')
    # Getting the type of 'False' (line 490)
    False_127108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 33), 'False')
    defaults = [None_127107, False_127108]
    # Create a new context for function 'vander'
    module_type_store = module_type_store.open_function_context('vander', 490, 0, False)
    
    # Passed parameters checking function
    vander.stypy_localization = localization
    vander.stypy_type_of_self = None
    vander.stypy_type_store = module_type_store
    vander.stypy_function_name = 'vander'
    vander.stypy_param_names_list = ['x', 'N', 'increasing']
    vander.stypy_varargs_param_name = None
    vander.stypy_kwargs_param_name = None
    vander.stypy_call_defaults = defaults
    vander.stypy_call_varargs = varargs
    vander.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'vander', ['x', 'N', 'increasing'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'vander', localization, ['x', 'N', 'increasing'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'vander(...)' code ##################

    str_127109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, (-1)), 'str', '\n    Generate a Vandermonde matrix.\n\n    The columns of the output matrix are powers of the input vector. The\n    order of the powers is determined by the `increasing` boolean argument.\n    Specifically, when `increasing` is False, the `i`-th output column is\n    the input vector raised element-wise to the power of ``N - i - 1``. Such\n    a matrix with a geometric progression in each row is named for Alexandre-\n    Theophile Vandermonde.\n\n    Parameters\n    ----------\n    x : array_like\n        1-D input array.\n    N : int, optional\n        Number of columns in the output.  If `N` is not specified, a square\n        array is returned (``N = len(x)``).\n    increasing : bool, optional\n        Order of the powers of the columns.  If True, the powers increase\n        from left to right, if False (the default) they are reversed.\n\n        .. versionadded:: 1.9.0\n\n    Returns\n    -------\n    out : ndarray\n        Vandermonde matrix.  If `increasing` is False, the first column is\n        ``x^(N-1)``, the second ``x^(N-2)`` and so forth. If `increasing` is\n        True, the columns are ``x^0, x^1, ..., x^(N-1)``.\n\n    See Also\n    --------\n    polynomial.polynomial.polyvander\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3, 5])\n    >>> N = 3\n    >>> np.vander(x, N)\n    array([[ 1,  1,  1],\n           [ 4,  2,  1],\n           [ 9,  3,  1],\n           [25,  5,  1]])\n\n    >>> np.column_stack([x**(N-1-i) for i in range(N)])\n    array([[ 1,  1,  1],\n           [ 4,  2,  1],\n           [ 9,  3,  1],\n           [25,  5,  1]])\n\n    >>> x = np.array([1, 2, 3, 5])\n    >>> np.vander(x)\n    array([[  1,   1,   1,   1],\n           [  8,   4,   2,   1],\n           [ 27,   9,   3,   1],\n           [125,  25,   5,   1]])\n    >>> np.vander(x, increasing=True)\n    array([[  1,   1,   1,   1],\n           [  1,   2,   4,   8],\n           [  1,   3,   9,  27],\n           [  1,   5,  25, 125]])\n\n    The determinant of a square Vandermonde matrix is the product\n    of the differences between the values of the input vector:\n\n    >>> np.linalg.det(np.vander(x))\n    48.000000000000043\n    >>> (5-3)*(5-2)*(5-1)*(3-2)*(3-1)*(2-1)\n    48\n\n    ')
    
    # Assigning a Call to a Name (line 562):
    
    # Assigning a Call to a Name (line 562):
    
    # Call to asarray(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'x' (line 562)
    x_127111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 16), 'x', False)
    # Processing the call keyword arguments (line 562)
    kwargs_127112 = {}
    # Getting the type of 'asarray' (line 562)
    asarray_127110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 562)
    asarray_call_result_127113 = invoke(stypy.reporting.localization.Localization(__file__, 562, 8), asarray_127110, *[x_127111], **kwargs_127112)
    
    # Assigning a type to the variable 'x' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'x', asarray_call_result_127113)
    
    
    # Getting the type of 'x' (line 563)
    x_127114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 7), 'x')
    # Obtaining the member 'ndim' of a type (line 563)
    ndim_127115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 7), x_127114, 'ndim')
    int_127116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 17), 'int')
    # Applying the binary operator '!=' (line 563)
    result_ne_127117 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 7), '!=', ndim_127115, int_127116)
    
    # Testing the type of an if condition (line 563)
    if_condition_127118 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 563, 4), result_ne_127117)
    # Assigning a type to the variable 'if_condition_127118' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'if_condition_127118', if_condition_127118)
    # SSA begins for if statement (line 563)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 564)
    # Processing the call arguments (line 564)
    str_127120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 25), 'str', 'x must be a one-dimensional array or sequence.')
    # Processing the call keyword arguments (line 564)
    kwargs_127121 = {}
    # Getting the type of 'ValueError' (line 564)
    ValueError_127119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 564)
    ValueError_call_result_127122 = invoke(stypy.reporting.localization.Localization(__file__, 564, 14), ValueError_127119, *[str_127120], **kwargs_127121)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 564, 8), ValueError_call_result_127122, 'raise parameter', BaseException)
    # SSA join for if statement (line 563)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 565)
    # Getting the type of 'N' (line 565)
    N_127123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 7), 'N')
    # Getting the type of 'None' (line 565)
    None_127124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'None')
    
    (may_be_127125, more_types_in_union_127126) = may_be_none(N_127123, None_127124)

    if may_be_127125:

        if more_types_in_union_127126:
            # Runtime conditional SSA (line 565)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 566):
        
        # Assigning a Call to a Name (line 566):
        
        # Call to len(...): (line 566)
        # Processing the call arguments (line 566)
        # Getting the type of 'x' (line 566)
        x_127128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 16), 'x', False)
        # Processing the call keyword arguments (line 566)
        kwargs_127129 = {}
        # Getting the type of 'len' (line 566)
        len_127127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'len', False)
        # Calling len(args, kwargs) (line 566)
        len_call_result_127130 = invoke(stypy.reporting.localization.Localization(__file__, 566, 12), len_127127, *[x_127128], **kwargs_127129)
        
        # Assigning a type to the variable 'N' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'N', len_call_result_127130)

        if more_types_in_union_127126:
            # SSA join for if statement (line 565)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 568):
    
    # Assigning a Call to a Name (line 568):
    
    # Call to empty(...): (line 568)
    # Processing the call arguments (line 568)
    
    # Obtaining an instance of the builtin type 'tuple' (line 568)
    tuple_127132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 568)
    # Adding element type (line 568)
    
    # Call to len(...): (line 568)
    # Processing the call arguments (line 568)
    # Getting the type of 'x' (line 568)
    x_127134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 19), 'x', False)
    # Processing the call keyword arguments (line 568)
    kwargs_127135 = {}
    # Getting the type of 'len' (line 568)
    len_127133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 15), 'len', False)
    # Calling len(args, kwargs) (line 568)
    len_call_result_127136 = invoke(stypy.reporting.localization.Localization(__file__, 568, 15), len_127133, *[x_127134], **kwargs_127135)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 15), tuple_127132, len_call_result_127136)
    # Adding element type (line 568)
    # Getting the type of 'N' (line 568)
    N_127137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 23), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 15), tuple_127132, N_127137)
    
    # Processing the call keyword arguments (line 568)
    
    # Call to promote_types(...): (line 568)
    # Processing the call arguments (line 568)
    # Getting the type of 'x' (line 568)
    x_127139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 47), 'x', False)
    # Obtaining the member 'dtype' of a type (line 568)
    dtype_127140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 47), x_127139, 'dtype')
    # Getting the type of 'int' (line 568)
    int_127141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 56), 'int', False)
    # Processing the call keyword arguments (line 568)
    kwargs_127142 = {}
    # Getting the type of 'promote_types' (line 568)
    promote_types_127138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 33), 'promote_types', False)
    # Calling promote_types(args, kwargs) (line 568)
    promote_types_call_result_127143 = invoke(stypy.reporting.localization.Localization(__file__, 568, 33), promote_types_127138, *[dtype_127140, int_127141], **kwargs_127142)
    
    keyword_127144 = promote_types_call_result_127143
    kwargs_127145 = {'dtype': keyword_127144}
    # Getting the type of 'empty' (line 568)
    empty_127131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'empty', False)
    # Calling empty(args, kwargs) (line 568)
    empty_call_result_127146 = invoke(stypy.reporting.localization.Localization(__file__, 568, 8), empty_127131, *[tuple_127132], **kwargs_127145)
    
    # Assigning a type to the variable 'v' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'v', empty_call_result_127146)
    
    # Assigning a IfExp to a Name (line 569):
    
    # Assigning a IfExp to a Name (line 569):
    
    
    # Getting the type of 'increasing' (line 569)
    increasing_127147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 28), 'increasing')
    # Applying the 'not' unary operator (line 569)
    result_not__127148 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 24), 'not', increasing_127147)
    
    # Testing the type of an if expression (line 569)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 569, 10), result_not__127148)
    # SSA begins for if expression (line 569)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    slice_127149 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 569, 10), None, None, None)
    int_127150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 17), 'int')
    slice_127151 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 569, 10), None, None, int_127150)
    # Getting the type of 'v' (line 569)
    v_127152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 10), 'v')
    # Obtaining the member '__getitem__' of a type (line 569)
    getitem___127153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 10), v_127152, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 569)
    subscript_call_result_127154 = invoke(stypy.reporting.localization.Localization(__file__, 569, 10), getitem___127153, (slice_127149, slice_127151))
    
    # SSA branch for the else part of an if expression (line 569)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'v' (line 569)
    v_127155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 44), 'v')
    # SSA join for if expression (line 569)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_127156 = union_type.UnionType.add(subscript_call_result_127154, v_127155)
    
    # Assigning a type to the variable 'tmp' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'tmp', if_exp_127156)
    
    
    # Getting the type of 'N' (line 571)
    N_127157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 7), 'N')
    int_127158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 11), 'int')
    # Applying the binary operator '>' (line 571)
    result_gt_127159 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 7), '>', N_127157, int_127158)
    
    # Testing the type of an if condition (line 571)
    if_condition_127160 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 571, 4), result_gt_127159)
    # Assigning a type to the variable 'if_condition_127160' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 4), 'if_condition_127160', if_condition_127160)
    # SSA begins for if statement (line 571)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 572):
    
    # Assigning a Num to a Subscript (line 572):
    int_127161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 20), 'int')
    # Getting the type of 'tmp' (line 572)
    tmp_127162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'tmp')
    slice_127163 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 572, 8), None, None, None)
    int_127164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 15), 'int')
    # Storing an element on a container (line 572)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 8), tmp_127162, ((slice_127163, int_127164), int_127161))
    # SSA join for if statement (line 571)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'N' (line 573)
    N_127165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 7), 'N')
    int_127166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 11), 'int')
    # Applying the binary operator '>' (line 573)
    result_gt_127167 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 7), '>', N_127165, int_127166)
    
    # Testing the type of an if condition (line 573)
    if_condition_127168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 573, 4), result_gt_127167)
    # Assigning a type to the variable 'if_condition_127168' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'if_condition_127168', if_condition_127168)
    # SSA begins for if statement (line 573)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 574):
    
    # Assigning a Subscript to a Subscript (line 574):
    
    # Obtaining the type of the subscript
    slice_127169 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 574, 21), None, None, None)
    # Getting the type of 'None' (line 574)
    None_127170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 26), 'None')
    # Getting the type of 'x' (line 574)
    x_127171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 21), 'x')
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___127172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 21), x_127171, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_127173 = invoke(stypy.reporting.localization.Localization(__file__, 574, 21), getitem___127172, (slice_127169, None_127170))
    
    # Getting the type of 'tmp' (line 574)
    tmp_127174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'tmp')
    slice_127175 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 574, 8), None, None, None)
    int_127176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 15), 'int')
    slice_127177 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 574, 8), int_127176, None, None)
    # Storing an element on a container (line 574)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 8), tmp_127174, ((slice_127175, slice_127177), subscript_call_result_127173))
    
    # Call to accumulate(...): (line 575)
    # Processing the call arguments (line 575)
    
    # Obtaining the type of the subscript
    slice_127180 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 575, 28), None, None, None)
    int_127181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 35), 'int')
    slice_127182 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 575, 28), int_127181, None, None)
    # Getting the type of 'tmp' (line 575)
    tmp_127183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 28), 'tmp', False)
    # Obtaining the member '__getitem__' of a type (line 575)
    getitem___127184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 28), tmp_127183, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 575)
    subscript_call_result_127185 = invoke(stypy.reporting.localization.Localization(__file__, 575, 28), getitem___127184, (slice_127180, slice_127182))
    
    # Processing the call keyword arguments (line 575)
    
    # Obtaining the type of the subscript
    slice_127186 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 575, 44), None, None, None)
    int_127187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 51), 'int')
    slice_127188 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 575, 44), int_127187, None, None)
    # Getting the type of 'tmp' (line 575)
    tmp_127189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 44), 'tmp', False)
    # Obtaining the member '__getitem__' of a type (line 575)
    getitem___127190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 44), tmp_127189, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 575)
    subscript_call_result_127191 = invoke(stypy.reporting.localization.Localization(__file__, 575, 44), getitem___127190, (slice_127186, slice_127188))
    
    keyword_127192 = subscript_call_result_127191
    int_127193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 61), 'int')
    keyword_127194 = int_127193
    kwargs_127195 = {'axis': keyword_127194, 'out': keyword_127192}
    # Getting the type of 'multiply' (line 575)
    multiply_127178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'multiply', False)
    # Obtaining the member 'accumulate' of a type (line 575)
    accumulate_127179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 8), multiply_127178, 'accumulate')
    # Calling accumulate(args, kwargs) (line 575)
    accumulate_call_result_127196 = invoke(stypy.reporting.localization.Localization(__file__, 575, 8), accumulate_127179, *[subscript_call_result_127185], **kwargs_127195)
    
    # SSA join for if statement (line 573)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'v' (line 577)
    v_127197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 11), 'v')
    # Assigning a type to the variable 'stypy_return_type' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'stypy_return_type', v_127197)
    
    # ################# End of 'vander(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'vander' in the type store
    # Getting the type of 'stypy_return_type' (line 490)
    stypy_return_type_127198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127198)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'vander'
    return stypy_return_type_127198

# Assigning a type to the variable 'vander' (line 490)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 0), 'vander', vander)

@norecursion
def histogram2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_127199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 27), 'int')
    # Getting the type of 'None' (line 580)
    None_127200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 37), 'None')
    # Getting the type of 'False' (line 580)
    False_127201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 50), 'False')
    # Getting the type of 'None' (line 580)
    None_127202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 65), 'None')
    defaults = [int_127199, None_127200, False_127201, None_127202]
    # Create a new context for function 'histogram2d'
    module_type_store = module_type_store.open_function_context('histogram2d', 580, 0, False)
    
    # Passed parameters checking function
    histogram2d.stypy_localization = localization
    histogram2d.stypy_type_of_self = None
    histogram2d.stypy_type_store = module_type_store
    histogram2d.stypy_function_name = 'histogram2d'
    histogram2d.stypy_param_names_list = ['x', 'y', 'bins', 'range', 'normed', 'weights']
    histogram2d.stypy_varargs_param_name = None
    histogram2d.stypy_kwargs_param_name = None
    histogram2d.stypy_call_defaults = defaults
    histogram2d.stypy_call_varargs = varargs
    histogram2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'histogram2d', ['x', 'y', 'bins', 'range', 'normed', 'weights'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'histogram2d', localization, ['x', 'y', 'bins', 'range', 'normed', 'weights'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'histogram2d(...)' code ##################

    str_127203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, (-1)), 'str', "\n    Compute the bi-dimensional histogram of two data samples.\n\n    Parameters\n    ----------\n    x : array_like, shape (N,)\n        An array containing the x coordinates of the points to be\n        histogrammed.\n    y : array_like, shape (N,)\n        An array containing the y coordinates of the points to be\n        histogrammed.\n    bins : int or array_like or [int, int] or [array, array], optional\n        The bin specification:\n\n          * If int, the number of bins for the two dimensions (nx=ny=bins).\n          * If array_like, the bin edges for the two dimensions\n            (x_edges=y_edges=bins).\n          * If [int, int], the number of bins in each dimension\n            (nx, ny = bins).\n          * If [array, array], the bin edges in each dimension\n            (x_edges, y_edges = bins).\n          * A combination [int, array] or [array, int], where int\n            is the number of bins and array is the bin edges.\n\n    range : array_like, shape(2,2), optional\n        The leftmost and rightmost edges of the bins along each dimension\n        (if not specified explicitly in the `bins` parameters):\n        ``[[xmin, xmax], [ymin, ymax]]``. All values outside of this range\n        will be considered outliers and not tallied in the histogram.\n    normed : bool, optional\n        If False, returns the number of samples in each bin. If True,\n        returns the bin density ``bin_count / sample_count / bin_area``.\n    weights : array_like, shape(N,), optional\n        An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.\n        Weights are normalized to 1 if `normed` is True. If `normed` is\n        False, the values of the returned histogram are equal to the sum of\n        the weights belonging to the samples falling into each bin.\n\n    Returns\n    -------\n    H : ndarray, shape(nx, ny)\n        The bi-dimensional histogram of samples `x` and `y`. Values in `x`\n        are histogrammed along the first dimension and values in `y` are\n        histogrammed along the second dimension.\n    xedges : ndarray, shape(nx,)\n        The bin edges along the first dimension.\n    yedges : ndarray, shape(ny,)\n        The bin edges along the second dimension.\n\n    See Also\n    --------\n    histogram : 1D histogram\n    histogramdd : Multidimensional histogram\n\n    Notes\n    -----\n    When `normed` is True, then the returned histogram is the sample\n    density, defined such that the sum over bins of the product\n    ``bin_value * bin_area`` is 1.\n\n    Please note that the histogram does not follow the Cartesian convention\n    where `x` values are on the abscissa and `y` values on the ordinate\n    axis.  Rather, `x` is histogrammed along the first dimension of the\n    array (vertical), and `y` along the second dimension of the array\n    (horizontal).  This ensures compatibility with `histogramdd`.\n\n    Examples\n    --------\n    >>> import matplotlib as mpl\n    >>> import matplotlib.pyplot as plt\n\n    Construct a 2D-histogram with variable bin width. First define the bin\n    edges:\n\n    >>> xedges = [0, 1, 1.5, 3, 5]\n    >>> yedges = [0, 2, 3, 4, 6]\n\n    Next we create a histogram H with random bin content:\n\n    >>> x = np.random.normal(3, 1, 100)\n    >>> y = np.random.normal(1, 1, 100)\n    >>> H, xedges, yedges = np.histogram2d(y, x, bins=(xedges, yedges))\n\n    Or we fill the histogram H with a determined bin content:\n\n    >>> H = np.ones((4, 4)).cumsum().reshape(4, 4)\n    >>> print(H[::-1])  # This shows the bin content in the order as plotted\n    [[ 13.  14.  15.  16.]\n     [  9.  10.  11.  12.]\n     [  5.   6.   7.   8.]\n     [  1.   2.   3.   4.]]\n\n    Imshow can only do an equidistant representation of bins:\n\n    >>> fig = plt.figure(figsize=(7, 3))\n    >>> ax = fig.add_subplot(131)\n    >>> ax.set_title('imshow: equidistant')\n    >>> im = plt.imshow(H, interpolation='nearest', origin='low',\n                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])\n\n    pcolormesh can display exact bin edges:\n\n    >>> ax = fig.add_subplot(132)\n    >>> ax.set_title('pcolormesh: exact bin edges')\n    >>> X, Y = np.meshgrid(xedges, yedges)\n    >>> ax.pcolormesh(X, Y, H)\n    >>> ax.set_aspect('equal')\n\n    NonUniformImage displays exact bin edges with interpolation:\n\n    >>> ax = fig.add_subplot(133)\n    >>> ax.set_title('NonUniformImage: interpolated')\n    >>> im = mpl.image.NonUniformImage(ax, interpolation='bilinear')\n    >>> xcenters = xedges[:-1] + 0.5 * (xedges[1:] - xedges[:-1])\n    >>> ycenters = yedges[:-1] + 0.5 * (yedges[1:] - yedges[:-1])\n    >>> im.set_data(xcenters, ycenters, H)\n    >>> ax.images.append(im)\n    >>> ax.set_xlim(xedges[0], xedges[-1])\n    >>> ax.set_ylim(yedges[0], yedges[-1])\n    >>> ax.set_aspect('equal')\n    >>> plt.show()\n\n    ")
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 704, 4))
    
    # 'from numpy import histogramdd' statement (line 704)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
    import_127204 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 704, 4), 'numpy')

    if (type(import_127204) is not StypyTypeError):

        if (import_127204 != 'pyd_module'):
            __import__(import_127204)
            sys_modules_127205 = sys.modules[import_127204]
            import_from_module(stypy.reporting.localization.Localization(__file__, 704, 4), 'numpy', sys_modules_127205.module_type_store, module_type_store, ['histogramdd'])
            nest_module(stypy.reporting.localization.Localization(__file__, 704, 4), __file__, sys_modules_127205, sys_modules_127205.module_type_store, module_type_store)
        else:
            from numpy import histogramdd

            import_from_module(stypy.reporting.localization.Localization(__file__, 704, 4), 'numpy', None, module_type_store, ['histogramdd'], [histogramdd])

    else:
        # Assigning a type to the variable 'numpy' (line 704)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'numpy', import_127204)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')
    
    
    
    # SSA begins for try-except statement (line 706)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 707):
    
    # Assigning a Call to a Name (line 707):
    
    # Call to len(...): (line 707)
    # Processing the call arguments (line 707)
    # Getting the type of 'bins' (line 707)
    bins_127207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 16), 'bins', False)
    # Processing the call keyword arguments (line 707)
    kwargs_127208 = {}
    # Getting the type of 'len' (line 707)
    len_127206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 12), 'len', False)
    # Calling len(args, kwargs) (line 707)
    len_call_result_127209 = invoke(stypy.reporting.localization.Localization(__file__, 707, 12), len_127206, *[bins_127207], **kwargs_127208)
    
    # Assigning a type to the variable 'N' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'N', len_call_result_127209)
    # SSA branch for the except part of a try statement (line 706)
    # SSA branch for the except 'TypeError' branch of a try statement (line 706)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Num to a Name (line 709):
    
    # Assigning a Num to a Name (line 709):
    int_127210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 12), 'int')
    # Assigning a type to the variable 'N' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), 'N', int_127210)
    # SSA join for try-except statement (line 706)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'N' (line 711)
    N_127211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 7), 'N')
    int_127212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 12), 'int')
    # Applying the binary operator '!=' (line 711)
    result_ne_127213 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 7), '!=', N_127211, int_127212)
    
    
    # Getting the type of 'N' (line 711)
    N_127214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 18), 'N')
    int_127215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 23), 'int')
    # Applying the binary operator '!=' (line 711)
    result_ne_127216 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 18), '!=', N_127214, int_127215)
    
    # Applying the binary operator 'and' (line 711)
    result_and_keyword_127217 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 7), 'and', result_ne_127213, result_ne_127216)
    
    # Testing the type of an if condition (line 711)
    if_condition_127218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 711, 4), result_and_keyword_127217)
    # Assigning a type to the variable 'if_condition_127218' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'if_condition_127218', if_condition_127218)
    # SSA begins for if statement (line 711)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Call to a Name (line 712):
    
    # Call to asarray(...): (line 712)
    # Processing the call arguments (line 712)
    # Getting the type of 'bins' (line 712)
    bins_127220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 34), 'bins', False)
    # Getting the type of 'float' (line 712)
    float_127221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 40), 'float', False)
    # Processing the call keyword arguments (line 712)
    kwargs_127222 = {}
    # Getting the type of 'asarray' (line 712)
    asarray_127219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 26), 'asarray', False)
    # Calling asarray(args, kwargs) (line 712)
    asarray_call_result_127223 = invoke(stypy.reporting.localization.Localization(__file__, 712, 26), asarray_127219, *[bins_127220, float_127221], **kwargs_127222)
    
    # Assigning a type to the variable 'yedges' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 17), 'yedges', asarray_call_result_127223)
    
    # Assigning a Name to a Name (line 712):
    # Getting the type of 'yedges' (line 712)
    yedges_127224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 17), 'yedges')
    # Assigning a type to the variable 'xedges' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'xedges', yedges_127224)
    
    # Assigning a List to a Name (line 713):
    
    # Assigning a List to a Name (line 713):
    
    # Obtaining an instance of the builtin type 'list' (line 713)
    list_127225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 713)
    # Adding element type (line 713)
    # Getting the type of 'xedges' (line 713)
    xedges_127226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 16), 'xedges')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 15), list_127225, xedges_127226)
    # Adding element type (line 713)
    # Getting the type of 'yedges' (line 713)
    yedges_127227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 24), 'yedges')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 15), list_127225, yedges_127227)
    
    # Assigning a type to the variable 'bins' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'bins', list_127225)
    # SSA join for if statement (line 711)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 714):
    
    # Assigning a Call to a Name:
    
    # Call to histogramdd(...): (line 714)
    # Processing the call arguments (line 714)
    
    # Obtaining an instance of the builtin type 'list' (line 714)
    list_127229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 714)
    # Adding element type (line 714)
    # Getting the type of 'x' (line 714)
    x_127230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 31), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 30), list_127229, x_127230)
    # Adding element type (line 714)
    # Getting the type of 'y' (line 714)
    y_127231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 34), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 30), list_127229, y_127231)
    
    # Getting the type of 'bins' (line 714)
    bins_127232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 38), 'bins', False)
    # Getting the type of 'range' (line 714)
    range_127233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 44), 'range', False)
    # Getting the type of 'normed' (line 714)
    normed_127234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 51), 'normed', False)
    # Getting the type of 'weights' (line 714)
    weights_127235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 59), 'weights', False)
    # Processing the call keyword arguments (line 714)
    kwargs_127236 = {}
    # Getting the type of 'histogramdd' (line 714)
    histogramdd_127228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 18), 'histogramdd', False)
    # Calling histogramdd(args, kwargs) (line 714)
    histogramdd_call_result_127237 = invoke(stypy.reporting.localization.Localization(__file__, 714, 18), histogramdd_127228, *[list_127229, bins_127232, range_127233, normed_127234, weights_127235], **kwargs_127236)
    
    # Assigning a type to the variable 'call_assignment_126627' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'call_assignment_126627', histogramdd_call_result_127237)
    
    # Assigning a Call to a Name (line 714):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_127240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 4), 'int')
    # Processing the call keyword arguments
    kwargs_127241 = {}
    # Getting the type of 'call_assignment_126627' (line 714)
    call_assignment_126627_127238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'call_assignment_126627', False)
    # Obtaining the member '__getitem__' of a type (line 714)
    getitem___127239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 4), call_assignment_126627_127238, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_127242 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___127239, *[int_127240], **kwargs_127241)
    
    # Assigning a type to the variable 'call_assignment_126628' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'call_assignment_126628', getitem___call_result_127242)
    
    # Assigning a Name to a Name (line 714):
    # Getting the type of 'call_assignment_126628' (line 714)
    call_assignment_126628_127243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'call_assignment_126628')
    # Assigning a type to the variable 'hist' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'hist', call_assignment_126628_127243)
    
    # Assigning a Call to a Name (line 714):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_127246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 4), 'int')
    # Processing the call keyword arguments
    kwargs_127247 = {}
    # Getting the type of 'call_assignment_126627' (line 714)
    call_assignment_126627_127244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'call_assignment_126627', False)
    # Obtaining the member '__getitem__' of a type (line 714)
    getitem___127245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 4), call_assignment_126627_127244, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_127248 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___127245, *[int_127246], **kwargs_127247)
    
    # Assigning a type to the variable 'call_assignment_126629' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'call_assignment_126629', getitem___call_result_127248)
    
    # Assigning a Name to a Name (line 714):
    # Getting the type of 'call_assignment_126629' (line 714)
    call_assignment_126629_127249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'call_assignment_126629')
    # Assigning a type to the variable 'edges' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 10), 'edges', call_assignment_126629_127249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 715)
    tuple_127250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 715)
    # Adding element type (line 715)
    # Getting the type of 'hist' (line 715)
    hist_127251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 11), 'hist')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 11), tuple_127250, hist_127251)
    # Adding element type (line 715)
    
    # Obtaining the type of the subscript
    int_127252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 23), 'int')
    # Getting the type of 'edges' (line 715)
    edges_127253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 17), 'edges')
    # Obtaining the member '__getitem__' of a type (line 715)
    getitem___127254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 17), edges_127253, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 715)
    subscript_call_result_127255 = invoke(stypy.reporting.localization.Localization(__file__, 715, 17), getitem___127254, int_127252)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 11), tuple_127250, subscript_call_result_127255)
    # Adding element type (line 715)
    
    # Obtaining the type of the subscript
    int_127256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 33), 'int')
    # Getting the type of 'edges' (line 715)
    edges_127257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 27), 'edges')
    # Obtaining the member '__getitem__' of a type (line 715)
    getitem___127258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 27), edges_127257, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 715)
    subscript_call_result_127259 = invoke(stypy.reporting.localization.Localization(__file__, 715, 27), getitem___127258, int_127256)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 11), tuple_127250, subscript_call_result_127259)
    
    # Assigning a type to the variable 'stypy_return_type' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 4), 'stypy_return_type', tuple_127250)
    
    # ################# End of 'histogram2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'histogram2d' in the type store
    # Getting the type of 'stypy_return_type' (line 580)
    stypy_return_type_127260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127260)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'histogram2d'
    return stypy_return_type_127260

# Assigning a type to the variable 'histogram2d' (line 580)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 0), 'histogram2d', histogram2d)

@norecursion
def mask_indices(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_127261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 33), 'int')
    defaults = [int_127261]
    # Create a new context for function 'mask_indices'
    module_type_store = module_type_store.open_function_context('mask_indices', 718, 0, False)
    
    # Passed parameters checking function
    mask_indices.stypy_localization = localization
    mask_indices.stypy_type_of_self = None
    mask_indices.stypy_type_store = module_type_store
    mask_indices.stypy_function_name = 'mask_indices'
    mask_indices.stypy_param_names_list = ['n', 'mask_func', 'k']
    mask_indices.stypy_varargs_param_name = None
    mask_indices.stypy_kwargs_param_name = None
    mask_indices.stypy_call_defaults = defaults
    mask_indices.stypy_call_varargs = varargs
    mask_indices.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mask_indices', ['n', 'mask_func', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mask_indices', localization, ['n', 'mask_func', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mask_indices(...)' code ##################

    str_127262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, (-1)), 'str', '\n    Return the indices to access (n, n) arrays, given a masking function.\n\n    Assume `mask_func` is a function that, for a square array a of size\n    ``(n, n)`` with a possible offset argument `k`, when called as\n    ``mask_func(a, k)`` returns a new array with zeros in certain locations\n    (functions like `triu` or `tril` do precisely this). Then this function\n    returns the indices where the non-zero values would be located.\n\n    Parameters\n    ----------\n    n : int\n        The returned indices will be valid to access arrays of shape (n, n).\n    mask_func : callable\n        A function whose call signature is similar to that of `triu`, `tril`.\n        That is, ``mask_func(x, k)`` returns a boolean array, shaped like `x`.\n        `k` is an optional argument to the function.\n    k : scalar\n        An optional argument which is passed through to `mask_func`. Functions\n        like `triu`, `tril` take a second argument that is interpreted as an\n        offset.\n\n    Returns\n    -------\n    indices : tuple of arrays.\n        The `n` arrays of indices corresponding to the locations where\n        ``mask_func(np.ones((n, n)), k)`` is True.\n\n    See Also\n    --------\n    triu, tril, triu_indices, tril_indices\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    Examples\n    --------\n    These are the indices that would allow you to access the upper triangular\n    part of any 3x3 array:\n\n    >>> iu = np.mask_indices(3, np.triu)\n\n    For example, if `a` is a 3x3 array:\n\n    >>> a = np.arange(9).reshape(3, 3)\n    >>> a\n    array([[0, 1, 2],\n           [3, 4, 5],\n           [6, 7, 8]])\n    >>> a[iu]\n    array([0, 1, 2, 4, 5, 8])\n\n    An offset can be passed also to the masking function.  This gets us the\n    indices starting on the first diagonal right of the main one:\n\n    >>> iu1 = np.mask_indices(3, np.triu, 1)\n\n    with which we now extract only three elements:\n\n    >>> a[iu1]\n    array([1, 2, 5])\n\n    ')
    
    # Assigning a Call to a Name (line 783):
    
    # Assigning a Call to a Name (line 783):
    
    # Call to ones(...): (line 783)
    # Processing the call arguments (line 783)
    
    # Obtaining an instance of the builtin type 'tuple' (line 783)
    tuple_127264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 783)
    # Adding element type (line 783)
    # Getting the type of 'n' (line 783)
    n_127265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 14), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 783, 14), tuple_127264, n_127265)
    # Adding element type (line 783)
    # Getting the type of 'n' (line 783)
    n_127266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 17), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 783, 14), tuple_127264, n_127266)
    
    # Getting the type of 'int' (line 783)
    int_127267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 21), 'int', False)
    # Processing the call keyword arguments (line 783)
    kwargs_127268 = {}
    # Getting the type of 'ones' (line 783)
    ones_127263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 8), 'ones', False)
    # Calling ones(args, kwargs) (line 783)
    ones_call_result_127269 = invoke(stypy.reporting.localization.Localization(__file__, 783, 8), ones_127263, *[tuple_127264, int_127267], **kwargs_127268)
    
    # Assigning a type to the variable 'm' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'm', ones_call_result_127269)
    
    # Assigning a Call to a Name (line 784):
    
    # Assigning a Call to a Name (line 784):
    
    # Call to mask_func(...): (line 784)
    # Processing the call arguments (line 784)
    # Getting the type of 'm' (line 784)
    m_127271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 18), 'm', False)
    # Getting the type of 'k' (line 784)
    k_127272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 21), 'k', False)
    # Processing the call keyword arguments (line 784)
    kwargs_127273 = {}
    # Getting the type of 'mask_func' (line 784)
    mask_func_127270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 8), 'mask_func', False)
    # Calling mask_func(args, kwargs) (line 784)
    mask_func_call_result_127274 = invoke(stypy.reporting.localization.Localization(__file__, 784, 8), mask_func_127270, *[m_127271, k_127272], **kwargs_127273)
    
    # Assigning a type to the variable 'a' (line 784)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 4), 'a', mask_func_call_result_127274)
    
    # Call to where(...): (line 785)
    # Processing the call arguments (line 785)
    
    # Getting the type of 'a' (line 785)
    a_127276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 17), 'a', False)
    int_127277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 22), 'int')
    # Applying the binary operator '!=' (line 785)
    result_ne_127278 = python_operator(stypy.reporting.localization.Localization(__file__, 785, 17), '!=', a_127276, int_127277)
    
    # Processing the call keyword arguments (line 785)
    kwargs_127279 = {}
    # Getting the type of 'where' (line 785)
    where_127275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 11), 'where', False)
    # Calling where(args, kwargs) (line 785)
    where_call_result_127280 = invoke(stypy.reporting.localization.Localization(__file__, 785, 11), where_127275, *[result_ne_127278], **kwargs_127279)
    
    # Assigning a type to the variable 'stypy_return_type' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 4), 'stypy_return_type', where_call_result_127280)
    
    # ################# End of 'mask_indices(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mask_indices' in the type store
    # Getting the type of 'stypy_return_type' (line 718)
    stypy_return_type_127281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127281)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mask_indices'
    return stypy_return_type_127281

# Assigning a type to the variable 'mask_indices' (line 718)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 0), 'mask_indices', mask_indices)

@norecursion
def tril_indices(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_127282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 22), 'int')
    # Getting the type of 'None' (line 788)
    None_127283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 27), 'None')
    defaults = [int_127282, None_127283]
    # Create a new context for function 'tril_indices'
    module_type_store = module_type_store.open_function_context('tril_indices', 788, 0, False)
    
    # Passed parameters checking function
    tril_indices.stypy_localization = localization
    tril_indices.stypy_type_of_self = None
    tril_indices.stypy_type_store = module_type_store
    tril_indices.stypy_function_name = 'tril_indices'
    tril_indices.stypy_param_names_list = ['n', 'k', 'm']
    tril_indices.stypy_varargs_param_name = None
    tril_indices.stypy_kwargs_param_name = None
    tril_indices.stypy_call_defaults = defaults
    tril_indices.stypy_call_varargs = varargs
    tril_indices.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tril_indices', ['n', 'k', 'm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tril_indices', localization, ['n', 'k', 'm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tril_indices(...)' code ##################

    str_127284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, (-1)), 'str', '\n    Return the indices for the lower-triangle of an (n, m) array.\n\n    Parameters\n    ----------\n    n : int\n        The row dimension of the arrays for which the returned\n        indices will be valid.\n    k : int, optional\n        Diagonal offset (see `tril` for details).\n    m : int, optional\n        .. versionadded:: 1.9.0\n\n        The column dimension of the arrays for which the returned\n        arrays will be valid.\n        By default `m` is taken equal to `n`.\n\n\n    Returns\n    -------\n    inds : tuple of arrays\n        The indices for the triangle. The returned tuple contains two arrays,\n        each with the indices along one dimension of the array.\n\n    See also\n    --------\n    triu_indices : similar function, for upper-triangular.\n    mask_indices : generic function accepting an arbitrary mask function.\n    tril, triu\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    Examples\n    --------\n    Compute two different sets of indices to access 4x4 arrays, one for the\n    lower triangular part starting at the main diagonal, and one starting two\n    diagonals further right:\n\n    >>> il1 = np.tril_indices(4)\n    >>> il2 = np.tril_indices(4, 2)\n\n    Here is how they can be used with a sample array:\n\n    >>> a = np.arange(16).reshape(4, 4)\n    >>> a\n    array([[ 0,  1,  2,  3],\n           [ 4,  5,  6,  7],\n           [ 8,  9, 10, 11],\n           [12, 13, 14, 15]])\n\n    Both for indexing:\n\n    >>> a[il1]\n    array([ 0,  4,  5,  8,  9, 10, 12, 13, 14, 15])\n\n    And for assigning values:\n\n    >>> a[il1] = -1\n    >>> a\n    array([[-1,  1,  2,  3],\n           [-1, -1,  6,  7],\n           [-1, -1, -1, 11],\n           [-1, -1, -1, -1]])\n\n    These cover almost the whole array (two diagonals right of the main one):\n\n    >>> a[il2] = -10\n    >>> a\n    array([[-10, -10, -10,   3],\n           [-10, -10, -10, -10],\n           [-10, -10, -10, -10],\n           [-10, -10, -10, -10]])\n\n    ')
    
    # Call to where(...): (line 865)
    # Processing the call arguments (line 865)
    
    # Call to tri(...): (line 865)
    # Processing the call arguments (line 865)
    # Getting the type of 'n' (line 865)
    n_127287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 21), 'n', False)
    # Getting the type of 'm' (line 865)
    m_127288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 24), 'm', False)
    # Processing the call keyword arguments (line 865)
    # Getting the type of 'k' (line 865)
    k_127289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 29), 'k', False)
    keyword_127290 = k_127289
    # Getting the type of 'bool' (line 865)
    bool_127291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 38), 'bool', False)
    keyword_127292 = bool_127291
    kwargs_127293 = {'dtype': keyword_127292, 'k': keyword_127290}
    # Getting the type of 'tri' (line 865)
    tri_127286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 17), 'tri', False)
    # Calling tri(args, kwargs) (line 865)
    tri_call_result_127294 = invoke(stypy.reporting.localization.Localization(__file__, 865, 17), tri_127286, *[n_127287, m_127288], **kwargs_127293)
    
    # Processing the call keyword arguments (line 865)
    kwargs_127295 = {}
    # Getting the type of 'where' (line 865)
    where_127285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 11), 'where', False)
    # Calling where(args, kwargs) (line 865)
    where_call_result_127296 = invoke(stypy.reporting.localization.Localization(__file__, 865, 11), where_127285, *[tri_call_result_127294], **kwargs_127295)
    
    # Assigning a type to the variable 'stypy_return_type' (line 865)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 4), 'stypy_return_type', where_call_result_127296)
    
    # ################# End of 'tril_indices(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tril_indices' in the type store
    # Getting the type of 'stypy_return_type' (line 788)
    stypy_return_type_127297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127297)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tril_indices'
    return stypy_return_type_127297

# Assigning a type to the variable 'tril_indices' (line 788)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 0), 'tril_indices', tril_indices)

@norecursion
def tril_indices_from(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_127298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 29), 'int')
    defaults = [int_127298]
    # Create a new context for function 'tril_indices_from'
    module_type_store = module_type_store.open_function_context('tril_indices_from', 868, 0, False)
    
    # Passed parameters checking function
    tril_indices_from.stypy_localization = localization
    tril_indices_from.stypy_type_of_self = None
    tril_indices_from.stypy_type_store = module_type_store
    tril_indices_from.stypy_function_name = 'tril_indices_from'
    tril_indices_from.stypy_param_names_list = ['arr', 'k']
    tril_indices_from.stypy_varargs_param_name = None
    tril_indices_from.stypy_kwargs_param_name = None
    tril_indices_from.stypy_call_defaults = defaults
    tril_indices_from.stypy_call_varargs = varargs
    tril_indices_from.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tril_indices_from', ['arr', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tril_indices_from', localization, ['arr', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tril_indices_from(...)' code ##################

    str_127299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, (-1)), 'str', '\n    Return the indices for the lower-triangle of arr.\n\n    See `tril_indices` for full details.\n\n    Parameters\n    ----------\n    arr : array_like\n        The indices will be valid for square arrays whose dimensions are\n        the same as arr.\n    k : int, optional\n        Diagonal offset (see `tril` for details).\n\n    See Also\n    --------\n    tril_indices, tril\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    ')
    
    
    # Getting the type of 'arr' (line 891)
    arr_127300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 7), 'arr')
    # Obtaining the member 'ndim' of a type (line 891)
    ndim_127301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 891, 7), arr_127300, 'ndim')
    int_127302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 19), 'int')
    # Applying the binary operator '!=' (line 891)
    result_ne_127303 = python_operator(stypy.reporting.localization.Localization(__file__, 891, 7), '!=', ndim_127301, int_127302)
    
    # Testing the type of an if condition (line 891)
    if_condition_127304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 891, 4), result_ne_127303)
    # Assigning a type to the variable 'if_condition_127304' (line 891)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 891, 4), 'if_condition_127304', if_condition_127304)
    # SSA begins for if statement (line 891)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 892)
    # Processing the call arguments (line 892)
    str_127306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 25), 'str', 'input array must be 2-d')
    # Processing the call keyword arguments (line 892)
    kwargs_127307 = {}
    # Getting the type of 'ValueError' (line 892)
    ValueError_127305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 892)
    ValueError_call_result_127308 = invoke(stypy.reporting.localization.Localization(__file__, 892, 14), ValueError_127305, *[str_127306], **kwargs_127307)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 892, 8), ValueError_call_result_127308, 'raise parameter', BaseException)
    # SSA join for if statement (line 891)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to tril_indices(...): (line 893)
    # Processing the call arguments (line 893)
    
    # Obtaining the type of the subscript
    int_127310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 34), 'int')
    # Getting the type of 'arr' (line 893)
    arr_127311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 24), 'arr', False)
    # Obtaining the member 'shape' of a type (line 893)
    shape_127312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 893, 24), arr_127311, 'shape')
    # Obtaining the member '__getitem__' of a type (line 893)
    getitem___127313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 893, 24), shape_127312, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 893)
    subscript_call_result_127314 = invoke(stypy.reporting.localization.Localization(__file__, 893, 24), getitem___127313, int_127310)
    
    # Processing the call keyword arguments (line 893)
    # Getting the type of 'k' (line 893)
    k_127315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 41), 'k', False)
    keyword_127316 = k_127315
    
    # Obtaining the type of the subscript
    int_127317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 56), 'int')
    # Getting the type of 'arr' (line 893)
    arr_127318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 46), 'arr', False)
    # Obtaining the member 'shape' of a type (line 893)
    shape_127319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 893, 46), arr_127318, 'shape')
    # Obtaining the member '__getitem__' of a type (line 893)
    getitem___127320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 893, 46), shape_127319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 893)
    subscript_call_result_127321 = invoke(stypy.reporting.localization.Localization(__file__, 893, 46), getitem___127320, int_127317)
    
    keyword_127322 = subscript_call_result_127321
    kwargs_127323 = {'k': keyword_127316, 'm': keyword_127322}
    # Getting the type of 'tril_indices' (line 893)
    tril_indices_127309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 11), 'tril_indices', False)
    # Calling tril_indices(args, kwargs) (line 893)
    tril_indices_call_result_127324 = invoke(stypy.reporting.localization.Localization(__file__, 893, 11), tril_indices_127309, *[subscript_call_result_127314], **kwargs_127323)
    
    # Assigning a type to the variable 'stypy_return_type' (line 893)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 893, 4), 'stypy_return_type', tril_indices_call_result_127324)
    
    # ################# End of 'tril_indices_from(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tril_indices_from' in the type store
    # Getting the type of 'stypy_return_type' (line 868)
    stypy_return_type_127325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127325)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tril_indices_from'
    return stypy_return_type_127325

# Assigning a type to the variable 'tril_indices_from' (line 868)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 0), 'tril_indices_from', tril_indices_from)

@norecursion
def triu_indices(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_127326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 896, 22), 'int')
    # Getting the type of 'None' (line 896)
    None_127327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 27), 'None')
    defaults = [int_127326, None_127327]
    # Create a new context for function 'triu_indices'
    module_type_store = module_type_store.open_function_context('triu_indices', 896, 0, False)
    
    # Passed parameters checking function
    triu_indices.stypy_localization = localization
    triu_indices.stypy_type_of_self = None
    triu_indices.stypy_type_store = module_type_store
    triu_indices.stypy_function_name = 'triu_indices'
    triu_indices.stypy_param_names_list = ['n', 'k', 'm']
    triu_indices.stypy_varargs_param_name = None
    triu_indices.stypy_kwargs_param_name = None
    triu_indices.stypy_call_defaults = defaults
    triu_indices.stypy_call_varargs = varargs
    triu_indices.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'triu_indices', ['n', 'k', 'm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'triu_indices', localization, ['n', 'k', 'm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'triu_indices(...)' code ##################

    str_127328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, (-1)), 'str', '\n    Return the indices for the upper-triangle of an (n, m) array.\n\n    Parameters\n    ----------\n    n : int\n        The size of the arrays for which the returned indices will\n        be valid.\n    k : int, optional\n        Diagonal offset (see `triu` for details).\n    m : int, optional\n        .. versionadded:: 1.9.0\n\n        The column dimension of the arrays for which the returned\n        arrays will be valid.\n        By default `m` is taken equal to `n`.\n\n\n    Returns\n    -------\n    inds : tuple, shape(2) of ndarrays, shape(`n`)\n        The indices for the triangle. The returned tuple contains two arrays,\n        each with the indices along one dimension of the array.  Can be used\n        to slice a ndarray of shape(`n`, `n`).\n\n    See also\n    --------\n    tril_indices : similar function, for lower-triangular.\n    mask_indices : generic function accepting an arbitrary mask function.\n    triu, tril\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    Examples\n    --------\n    Compute two different sets of indices to access 4x4 arrays, one for the\n    upper triangular part starting at the main diagonal, and one starting two\n    diagonals further right:\n\n    >>> iu1 = np.triu_indices(4)\n    >>> iu2 = np.triu_indices(4, 2)\n\n    Here is how they can be used with a sample array:\n\n    >>> a = np.arange(16).reshape(4, 4)\n    >>> a\n    array([[ 0,  1,  2,  3],\n           [ 4,  5,  6,  7],\n           [ 8,  9, 10, 11],\n           [12, 13, 14, 15]])\n\n    Both for indexing:\n\n    >>> a[iu1]\n    array([ 0,  1,  2,  3,  5,  6,  7, 10, 11, 15])\n\n    And for assigning values:\n\n    >>> a[iu1] = -1\n    >>> a\n    array([[-1, -1, -1, -1],\n           [ 4, -1, -1, -1],\n           [ 8,  9, -1, -1],\n           [12, 13, 14, -1]])\n\n    These cover only a small part of the whole array (two diagonals right\n    of the main one):\n\n    >>> a[iu2] = -10\n    >>> a\n    array([[ -1,  -1, -10, -10],\n           [  4,  -1,  -1, -10],\n           [  8,   9,  -1,  -1],\n           [ 12,  13,  14,  -1]])\n\n    ')
    
    # Call to where(...): (line 975)
    # Processing the call arguments (line 975)
    
    
    # Call to tri(...): (line 975)
    # Processing the call arguments (line 975)
    # Getting the type of 'n' (line 975)
    n_127331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 22), 'n', False)
    # Getting the type of 'm' (line 975)
    m_127332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 25), 'm', False)
    # Processing the call keyword arguments (line 975)
    # Getting the type of 'k' (line 975)
    k_127333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 30), 'k', False)
    int_127334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 32), 'int')
    # Applying the binary operator '-' (line 975)
    result_sub_127335 = python_operator(stypy.reporting.localization.Localization(__file__, 975, 30), '-', k_127333, int_127334)
    
    keyword_127336 = result_sub_127335
    # Getting the type of 'bool' (line 975)
    bool_127337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 41), 'bool', False)
    keyword_127338 = bool_127337
    kwargs_127339 = {'dtype': keyword_127338, 'k': keyword_127336}
    # Getting the type of 'tri' (line 975)
    tri_127330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 18), 'tri', False)
    # Calling tri(args, kwargs) (line 975)
    tri_call_result_127340 = invoke(stypy.reporting.localization.Localization(__file__, 975, 18), tri_127330, *[n_127331, m_127332], **kwargs_127339)
    
    # Applying the '~' unary operator (line 975)
    result_inv_127341 = python_operator(stypy.reporting.localization.Localization(__file__, 975, 17), '~', tri_call_result_127340)
    
    # Processing the call keyword arguments (line 975)
    kwargs_127342 = {}
    # Getting the type of 'where' (line 975)
    where_127329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 11), 'where', False)
    # Calling where(args, kwargs) (line 975)
    where_call_result_127343 = invoke(stypy.reporting.localization.Localization(__file__, 975, 11), where_127329, *[result_inv_127341], **kwargs_127342)
    
    # Assigning a type to the variable 'stypy_return_type' (line 975)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 4), 'stypy_return_type', where_call_result_127343)
    
    # ################# End of 'triu_indices(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'triu_indices' in the type store
    # Getting the type of 'stypy_return_type' (line 896)
    stypy_return_type_127344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127344)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'triu_indices'
    return stypy_return_type_127344

# Assigning a type to the variable 'triu_indices' (line 896)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 0), 'triu_indices', triu_indices)

@norecursion
def triu_indices_from(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_127345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 29), 'int')
    defaults = [int_127345]
    # Create a new context for function 'triu_indices_from'
    module_type_store = module_type_store.open_function_context('triu_indices_from', 978, 0, False)
    
    # Passed parameters checking function
    triu_indices_from.stypy_localization = localization
    triu_indices_from.stypy_type_of_self = None
    triu_indices_from.stypy_type_store = module_type_store
    triu_indices_from.stypy_function_name = 'triu_indices_from'
    triu_indices_from.stypy_param_names_list = ['arr', 'k']
    triu_indices_from.stypy_varargs_param_name = None
    triu_indices_from.stypy_kwargs_param_name = None
    triu_indices_from.stypy_call_defaults = defaults
    triu_indices_from.stypy_call_varargs = varargs
    triu_indices_from.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'triu_indices_from', ['arr', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'triu_indices_from', localization, ['arr', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'triu_indices_from(...)' code ##################

    str_127346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1004, (-1)), 'str', '\n    Return the indices for the upper-triangle of arr.\n\n    See `triu_indices` for full details.\n\n    Parameters\n    ----------\n    arr : ndarray, shape(N, N)\n        The indices will be valid for square arrays.\n    k : int, optional\n        Diagonal offset (see `triu` for details).\n\n    Returns\n    -------\n    triu_indices_from : tuple, shape(2) of ndarray, shape(N)\n        Indices for the upper-triangle of `arr`.\n\n    See Also\n    --------\n    triu_indices, triu\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    ')
    
    
    # Getting the type of 'arr' (line 1005)
    arr_127347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 7), 'arr')
    # Obtaining the member 'ndim' of a type (line 1005)
    ndim_127348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1005, 7), arr_127347, 'ndim')
    int_127349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1005, 19), 'int')
    # Applying the binary operator '!=' (line 1005)
    result_ne_127350 = python_operator(stypy.reporting.localization.Localization(__file__, 1005, 7), '!=', ndim_127348, int_127349)
    
    # Testing the type of an if condition (line 1005)
    if_condition_127351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1005, 4), result_ne_127350)
    # Assigning a type to the variable 'if_condition_127351' (line 1005)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1005, 4), 'if_condition_127351', if_condition_127351)
    # SSA begins for if statement (line 1005)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1006)
    # Processing the call arguments (line 1006)
    str_127353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1006, 25), 'str', 'input array must be 2-d')
    # Processing the call keyword arguments (line 1006)
    kwargs_127354 = {}
    # Getting the type of 'ValueError' (line 1006)
    ValueError_127352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1006)
    ValueError_call_result_127355 = invoke(stypy.reporting.localization.Localization(__file__, 1006, 14), ValueError_127352, *[str_127353], **kwargs_127354)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1006, 8), ValueError_call_result_127355, 'raise parameter', BaseException)
    # SSA join for if statement (line 1005)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to triu_indices(...): (line 1007)
    # Processing the call arguments (line 1007)
    
    # Obtaining the type of the subscript
    int_127357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 34), 'int')
    # Getting the type of 'arr' (line 1007)
    arr_127358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 24), 'arr', False)
    # Obtaining the member 'shape' of a type (line 1007)
    shape_127359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 24), arr_127358, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1007)
    getitem___127360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 24), shape_127359, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1007)
    subscript_call_result_127361 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 24), getitem___127360, int_127357)
    
    # Processing the call keyword arguments (line 1007)
    # Getting the type of 'k' (line 1007)
    k_127362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 41), 'k', False)
    keyword_127363 = k_127362
    
    # Obtaining the type of the subscript
    int_127364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 56), 'int')
    # Getting the type of 'arr' (line 1007)
    arr_127365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 46), 'arr', False)
    # Obtaining the member 'shape' of a type (line 1007)
    shape_127366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 46), arr_127365, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1007)
    getitem___127367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 46), shape_127366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1007)
    subscript_call_result_127368 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 46), getitem___127367, int_127364)
    
    keyword_127369 = subscript_call_result_127368
    kwargs_127370 = {'k': keyword_127363, 'm': keyword_127369}
    # Getting the type of 'triu_indices' (line 1007)
    triu_indices_127356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 11), 'triu_indices', False)
    # Calling triu_indices(args, kwargs) (line 1007)
    triu_indices_call_result_127371 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 11), triu_indices_127356, *[subscript_call_result_127361], **kwargs_127370)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'stypy_return_type', triu_indices_call_result_127371)
    
    # ################# End of 'triu_indices_from(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'triu_indices_from' in the type store
    # Getting the type of 'stypy_return_type' (line 978)
    stypy_return_type_127372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127372)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'triu_indices_from'
    return stypy_return_type_127372

# Assigning a type to the variable 'triu_indices_from' (line 978)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 0), 'triu_indices_from', triu_indices_from)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
