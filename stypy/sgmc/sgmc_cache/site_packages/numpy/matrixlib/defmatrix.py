
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: __all__ = ['matrix', 'bmat', 'mat', 'asmatrix']
4: 
5: import sys
6: import numpy.core.numeric as N
7: from numpy.core.numeric import concatenate, isscalar, binary_repr, identity, asanyarray
8: from numpy.core.numerictypes import issubdtype
9: 
10: # make translation table
11: _numchars = '0123456789.-+jeEL'
12: 
13: if sys.version_info[0] >= 3:
14:     class _NumCharTable:
15:         def __getitem__(self, i):
16:             if chr(i) in _numchars:
17:                 return chr(i)
18:             else:
19:                 return None
20:     _table = _NumCharTable()
21:     def _eval(astr):
22:         str_ = astr.translate(_table)
23:         if not str_:
24:             raise TypeError("Invalid data string supplied: " + astr)
25:         else:
26:             return eval(str_)
27: 
28: else:
29:     _table = [None]*256
30:     for k in range(256):
31:         _table[k] = chr(k)
32:     _table = ''.join(_table)
33: 
34:     _todelete = []
35:     for k in _table:
36:         if k not in _numchars:
37:             _todelete.append(k)
38:     _todelete = ''.join(_todelete)
39:     del k
40: 
41:     def _eval(astr):
42:         str_ = astr.translate(_table, _todelete)
43:         if not str_:
44:             raise TypeError("Invalid data string supplied: " + astr)
45:         else:
46:             return eval(str_)
47: 
48: def _convert_from_string(data):
49:     rows = data.split(';')
50:     newdata = []
51:     count = 0
52:     for row in rows:
53:         trow = row.split(',')
54:         newrow = []
55:         for col in trow:
56:             temp = col.split()
57:             newrow.extend(map(_eval, temp))
58:         if count == 0:
59:             Ncols = len(newrow)
60:         elif len(newrow) != Ncols:
61:             raise ValueError("Rows not the same size.")
62:         count += 1
63:         newdata.append(newrow)
64:     return newdata
65: 
66: def asmatrix(data, dtype=None):
67:     '''
68:     Interpret the input as a matrix.
69: 
70:     Unlike `matrix`, `asmatrix` does not make a copy if the input is already
71:     a matrix or an ndarray.  Equivalent to ``matrix(data, copy=False)``.
72: 
73:     Parameters
74:     ----------
75:     data : array_like
76:         Input data.
77:     dtype : data-type
78:        Data-type of the output matrix.
79: 
80:     Returns
81:     -------
82:     mat : matrix
83:         `data` interpreted as a matrix.
84: 
85:     Examples
86:     --------
87:     >>> x = np.array([[1, 2], [3, 4]])
88: 
89:     >>> m = np.asmatrix(x)
90: 
91:     >>> x[0,0] = 5
92: 
93:     >>> m
94:     matrix([[5, 2],
95:             [3, 4]])
96: 
97:     '''
98:     return matrix(data, dtype=dtype, copy=False)
99: 
100: def matrix_power(M, n):
101:     '''
102:     Raise a square matrix to the (integer) power `n`.
103: 
104:     For positive integers `n`, the power is computed by repeated matrix
105:     squarings and matrix multiplications. If ``n == 0``, the identity matrix
106:     of the same shape as M is returned. If ``n < 0``, the inverse
107:     is computed and then raised to the ``abs(n)``.
108: 
109:     Parameters
110:     ----------
111:     M : ndarray or matrix object
112:         Matrix to be "powered."  Must be square, i.e. ``M.shape == (m, m)``,
113:         with `m` a positive integer.
114:     n : int
115:         The exponent can be any integer or long integer, positive,
116:         negative, or zero.
117: 
118:     Returns
119:     -------
120:     M**n : ndarray or matrix object
121:         The return value is the same shape and type as `M`;
122:         if the exponent is positive or zero then the type of the
123:         elements is the same as those of `M`. If the exponent is
124:         negative the elements are floating-point.
125: 
126:     Raises
127:     ------
128:     LinAlgError
129:         If the matrix is not numerically invertible.
130: 
131:     See Also
132:     --------
133:     matrix
134:         Provides an equivalent function as the exponentiation operator
135:         (``**``, not ``^``).
136: 
137:     Examples
138:     --------
139:     >>> from numpy import linalg as LA
140:     >>> i = np.array([[0, 1], [-1, 0]]) # matrix equiv. of the imaginary unit
141:     >>> LA.matrix_power(i, 3) # should = -i
142:     array([[ 0, -1],
143:            [ 1,  0]])
144:     >>> LA.matrix_power(np.matrix(i), 3) # matrix arg returns matrix
145:     matrix([[ 0, -1],
146:             [ 1,  0]])
147:     >>> LA.matrix_power(i, 0)
148:     array([[1, 0],
149:            [0, 1]])
150:     >>> LA.matrix_power(i, -3) # should = 1/(-i) = i, but w/ f.p. elements
151:     array([[ 0.,  1.],
152:            [-1.,  0.]])
153: 
154:     Somewhat more sophisticated example
155: 
156:     >>> q = np.zeros((4, 4))
157:     >>> q[0:2, 0:2] = -i
158:     >>> q[2:4, 2:4] = i
159:     >>> q # one of the three quarternion units not equal to 1
160:     array([[ 0., -1.,  0.,  0.],
161:            [ 1.,  0.,  0.,  0.],
162:            [ 0.,  0.,  0.,  1.],
163:            [ 0.,  0., -1.,  0.]])
164:     >>> LA.matrix_power(q, 2) # = -np.eye(4)
165:     array([[-1.,  0.,  0.,  0.],
166:            [ 0., -1.,  0.,  0.],
167:            [ 0.,  0., -1.,  0.],
168:            [ 0.,  0.,  0., -1.]])
169: 
170:     '''
171:     M = asanyarray(M)
172:     if len(M.shape) != 2 or M.shape[0] != M.shape[1]:
173:         raise ValueError("input must be a square array")
174:     if not issubdtype(type(n), int):
175:         raise TypeError("exponent must be an integer")
176: 
177:     from numpy.linalg import inv
178: 
179:     if n==0:
180:         M = M.copy()
181:         M[:] = identity(M.shape[0])
182:         return M
183:     elif n<0:
184:         M = inv(M)
185:         n *= -1
186: 
187:     result = M
188:     if n <= 3:
189:         for _ in range(n-1):
190:             result=N.dot(result, M)
191:         return result
192: 
193:     # binary decomposition to reduce the number of Matrix
194:     # multiplications for n > 3.
195:     beta = binary_repr(n)
196:     Z, q, t = M, 0, len(beta)
197:     while beta[t-q-1] == '0':
198:         Z = N.dot(Z, Z)
199:         q += 1
200:     result = Z
201:     for k in range(q+1, t):
202:         Z = N.dot(Z, Z)
203:         if beta[t-k-1] == '1':
204:             result = N.dot(result, Z)
205:     return result
206: 
207: 
208: class matrix(N.ndarray):
209:     '''
210:     matrix(data, dtype=None, copy=True)
211: 
212:     Returns a matrix from an array-like object, or from a string of data.
213:     A matrix is a specialized 2-D array that retains its 2-D nature
214:     through operations.  It has certain special operators, such as ``*``
215:     (matrix multiplication) and ``**`` (matrix power).
216: 
217:     Parameters
218:     ----------
219:     data : array_like or string
220:        If `data` is a string, it is interpreted as a matrix with commas
221:        or spaces separating columns, and semicolons separating rows.
222:     dtype : data-type
223:        Data-type of the output matrix.
224:     copy : bool
225:        If `data` is already an `ndarray`, then this flag determines
226:        whether the data is copied (the default), or whether a view is
227:        constructed.
228: 
229:     See Also
230:     --------
231:     array
232: 
233:     Examples
234:     --------
235:     >>> a = np.matrix('1 2; 3 4')
236:     >>> print(a)
237:     [[1 2]
238:      [3 4]]
239: 
240:     >>> np.matrix([[1, 2], [3, 4]])
241:     matrix([[1, 2],
242:             [3, 4]])
243: 
244:     '''
245:     __array_priority__ = 10.0
246:     def __new__(subtype, data, dtype=None, copy=True):
247:         if isinstance(data, matrix):
248:             dtype2 = data.dtype
249:             if (dtype is None):
250:                 dtype = dtype2
251:             if (dtype2 == dtype) and (not copy):
252:                 return data
253:             return data.astype(dtype)
254: 
255:         if isinstance(data, N.ndarray):
256:             if dtype is None:
257:                 intype = data.dtype
258:             else:
259:                 intype = N.dtype(dtype)
260:             new = data.view(subtype)
261:             if intype != data.dtype:
262:                 return new.astype(intype)
263:             if copy: return new.copy()
264:             else: return new
265: 
266:         if isinstance(data, str):
267:             data = _convert_from_string(data)
268: 
269:         # now convert data to an array
270:         arr = N.array(data, dtype=dtype, copy=copy)
271:         ndim = arr.ndim
272:         shape = arr.shape
273:         if (ndim > 2):
274:             raise ValueError("matrix must be 2-dimensional")
275:         elif ndim == 0:
276:             shape = (1, 1)
277:         elif ndim == 1:
278:             shape = (1, shape[0])
279: 
280:         order = 'C'
281:         if (ndim == 2) and arr.flags.fortran:
282:             order = 'F'
283: 
284:         if not (order or arr.flags.contiguous):
285:             arr = arr.copy()
286: 
287:         ret = N.ndarray.__new__(subtype, shape, arr.dtype,
288:                                 buffer=arr,
289:                                 order=order)
290:         return ret
291: 
292:     def __array_finalize__(self, obj):
293:         self._getitem = False
294:         if (isinstance(obj, matrix) and obj._getitem): return
295:         ndim = self.ndim
296:         if (ndim == 2):
297:             return
298:         if (ndim > 2):
299:             newshape = tuple([x for x in self.shape if x > 1])
300:             ndim = len(newshape)
301:             if ndim == 2:
302:                 self.shape = newshape
303:                 return
304:             elif (ndim > 2):
305:                 raise ValueError("shape too large to be a matrix.")
306:         else:
307:             newshape = self.shape
308:         if ndim == 0:
309:             self.shape = (1, 1)
310:         elif ndim == 1:
311:             self.shape = (1, newshape[0])
312:         return
313: 
314:     def __getitem__(self, index):
315:         self._getitem = True
316: 
317:         try:
318:             out = N.ndarray.__getitem__(self, index)
319:         finally:
320:             self._getitem = False
321: 
322:         if not isinstance(out, N.ndarray):
323:             return out
324: 
325:         if out.ndim == 0:
326:             return out[()]
327:         if out.ndim == 1:
328:             sh = out.shape[0]
329:             # Determine when we should have a column array
330:             try:
331:                 n = len(index)
332:             except:
333:                 n = 0
334:             if n > 1 and isscalar(index[1]):
335:                 out.shape = (sh, 1)
336:             else:
337:                 out.shape = (1, sh)
338:         return out
339: 
340:     def __mul__(self, other):
341:         if isinstance(other, (N.ndarray, list, tuple)) :
342:             # This promotes 1-D vectors to row vectors
343:             return N.dot(self, asmatrix(other))
344:         if isscalar(other) or not hasattr(other, '__rmul__') :
345:             return N.dot(self, other)
346:         return NotImplemented
347: 
348:     def __rmul__(self, other):
349:         return N.dot(other, self)
350: 
351:     def __imul__(self, other):
352:         self[:] = self * other
353:         return self
354: 
355:     def __pow__(self, other):
356:         return matrix_power(self, other)
357: 
358:     def __ipow__(self, other):
359:         self[:] = self ** other
360:         return self
361: 
362:     def __rpow__(self, other):
363:         return NotImplemented
364: 
365:     def __repr__(self):
366:         s = repr(self.__array__()).replace('array', 'matrix')
367:         # now, 'matrix' has 6 letters, and 'array' 5, so the columns don't
368:         # line up anymore. We need to add a space.
369:         l = s.splitlines()
370:         for i in range(1, len(l)):
371:             if l[i]:
372:                 l[i] = ' ' + l[i]
373:         return '\n'.join(l)
374: 
375:     def __str__(self):
376:         return str(self.__array__())
377: 
378:     def _align(self, axis):
379:         '''A convenience function for operations that need to preserve axis
380:         orientation.
381:         '''
382:         if axis is None:
383:             return self[0, 0]
384:         elif axis==0:
385:             return self
386:         elif axis==1:
387:             return self.transpose()
388:         else:
389:             raise ValueError("unsupported axis")
390: 
391:     def _collapse(self, axis):
392:         '''A convenience function for operations that want to collapse
393:         to a scalar like _align, but are using keepdims=True
394:         '''
395:         if axis is None:
396:             return self[0, 0]
397:         else:
398:             return self
399: 
400:     # Necessary because base-class tolist expects dimension
401:     #  reduction by x[0]
402:     def tolist(self):
403:         '''
404:         Return the matrix as a (possibly nested) list.
405: 
406:         See `ndarray.tolist` for full documentation.
407: 
408:         See Also
409:         --------
410:         ndarray.tolist
411: 
412:         Examples
413:         --------
414:         >>> x = np.matrix(np.arange(12).reshape((3,4))); x
415:         matrix([[ 0,  1,  2,  3],
416:                 [ 4,  5,  6,  7],
417:                 [ 8,  9, 10, 11]])
418:         >>> x.tolist()
419:         [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
420: 
421:         '''
422:         return self.__array__().tolist()
423: 
424:     # To preserve orientation of result...
425:     def sum(self, axis=None, dtype=None, out=None):
426:         '''
427:         Returns the sum of the matrix elements, along the given axis.
428: 
429:         Refer to `numpy.sum` for full documentation.
430: 
431:         See Also
432:         --------
433:         numpy.sum
434: 
435:         Notes
436:         -----
437:         This is the same as `ndarray.sum`, except that where an `ndarray` would
438:         be returned, a `matrix` object is returned instead.
439: 
440:         Examples
441:         --------
442:         >>> x = np.matrix([[1, 2], [4, 3]])
443:         >>> x.sum()
444:         10
445:         >>> x.sum(axis=1)
446:         matrix([[3],
447:                 [7]])
448:         >>> x.sum(axis=1, dtype='float')
449:         matrix([[ 3.],
450:                 [ 7.]])
451:         >>> out = np.zeros((1, 2), dtype='float')
452:         >>> x.sum(axis=1, dtype='float', out=out)
453:         matrix([[ 3.],
454:                 [ 7.]])
455: 
456:         '''
457:         return N.ndarray.sum(self, axis, dtype, out, keepdims=True)._collapse(axis)
458: 
459: 
460:     # To update docstring from array to matrix...
461:     def squeeze(self, axis=None):
462:         '''
463:         Return a possibly reshaped matrix.
464: 
465:         Refer to `numpy.squeeze` for more documentation.
466: 
467:         Parameters
468:         ----------
469:         axis : None or int or tuple of ints, optional
470:             Selects a subset of the single-dimensional entries in the shape.
471:             If an axis is selected with shape entry greater than one,
472:             an error is raised.
473: 
474:         Returns
475:         -------
476:         squeezed : matrix
477:             The matrix, but as a (1, N) matrix if it had shape (N, 1).
478: 
479:         See Also
480:         --------
481:         numpy.squeeze : related function
482: 
483:         Notes
484:         -----
485:         If `m` has a single column then that column is returned
486:         as the single row of a matrix.  Otherwise `m` is returned.
487:         The returned matrix is always either `m` itself or a view into `m`.
488:         Supplying an axis keyword argument will not affect the returned matrix
489:         but it may cause an error to be raised.
490: 
491:         Examples
492:         --------
493:         >>> c = np.matrix([[1], [2]])
494:         >>> c
495:         matrix([[1],
496:                 [2]])
497:         >>> c.squeeze()
498:         matrix([[1, 2]])
499:         >>> r = c.T
500:         >>> r
501:         matrix([[1, 2]])
502:         >>> r.squeeze()
503:         matrix([[1, 2]])
504:         >>> m = np.matrix([[1, 2], [3, 4]])
505:         >>> m.squeeze()
506:         matrix([[1, 2],
507:                 [3, 4]])
508: 
509:         '''
510:         return N.ndarray.squeeze(self, axis=axis)
511: 
512: 
513:     # To update docstring from array to matrix...
514:     def flatten(self, order='C'):
515:         '''
516:         Return a flattened copy of the matrix.
517: 
518:         All `N` elements of the matrix are placed into a single row.
519: 
520:         Parameters
521:         ----------
522:         order : {'C', 'F', 'A', 'K'}, optional
523:             'C' means to flatten in row-major (C-style) order. 'F' means to
524:             flatten in column-major (Fortran-style) order. 'A' means to
525:             flatten in column-major order if `m` is Fortran *contiguous* in
526:             memory, row-major order otherwise. 'K' means to flatten `m` in
527:             the order the elements occur in memory. The default is 'C'.
528: 
529:         Returns
530:         -------
531:         y : matrix
532:             A copy of the matrix, flattened to a `(1, N)` matrix where `N`
533:             is the number of elements in the original matrix.
534: 
535:         See Also
536:         --------
537:         ravel : Return a flattened array.
538:         flat : A 1-D flat iterator over the matrix.
539: 
540:         Examples
541:         --------
542:         >>> m = np.matrix([[1,2], [3,4]])
543:         >>> m.flatten()
544:         matrix([[1, 2, 3, 4]])
545:         >>> m.flatten('F')
546:         matrix([[1, 3, 2, 4]])
547: 
548:         '''
549:         return N.ndarray.flatten(self, order=order)
550: 
551:     def mean(self, axis=None, dtype=None, out=None):
552:         '''
553:         Returns the average of the matrix elements along the given axis.
554: 
555:         Refer to `numpy.mean` for full documentation.
556: 
557:         See Also
558:         --------
559:         numpy.mean
560: 
561:         Notes
562:         -----
563:         Same as `ndarray.mean` except that, where that returns an `ndarray`,
564:         this returns a `matrix` object.
565: 
566:         Examples
567:         --------
568:         >>> x = np.matrix(np.arange(12).reshape((3, 4)))
569:         >>> x
570:         matrix([[ 0,  1,  2,  3],
571:                 [ 4,  5,  6,  7],
572:                 [ 8,  9, 10, 11]])
573:         >>> x.mean()
574:         5.5
575:         >>> x.mean(0)
576:         matrix([[ 4.,  5.,  6.,  7.]])
577:         >>> x.mean(1)
578:         matrix([[ 1.5],
579:                 [ 5.5],
580:                 [ 9.5]])
581: 
582:         '''
583:         return N.ndarray.mean(self, axis, dtype, out, keepdims=True)._collapse(axis)
584: 
585:     def std(self, axis=None, dtype=None, out=None, ddof=0):
586:         '''
587:         Return the standard deviation of the array elements along the given axis.
588: 
589:         Refer to `numpy.std` for full documentation.
590: 
591:         See Also
592:         --------
593:         numpy.std
594: 
595:         Notes
596:         -----
597:         This is the same as `ndarray.std`, except that where an `ndarray` would
598:         be returned, a `matrix` object is returned instead.
599: 
600:         Examples
601:         --------
602:         >>> x = np.matrix(np.arange(12).reshape((3, 4)))
603:         >>> x
604:         matrix([[ 0,  1,  2,  3],
605:                 [ 4,  5,  6,  7],
606:                 [ 8,  9, 10, 11]])
607:         >>> x.std()
608:         3.4520525295346629
609:         >>> x.std(0)
610:         matrix([[ 3.26598632,  3.26598632,  3.26598632,  3.26598632]])
611:         >>> x.std(1)
612:         matrix([[ 1.11803399],
613:                 [ 1.11803399],
614:                 [ 1.11803399]])
615: 
616:         '''
617:         return N.ndarray.std(self, axis, dtype, out, ddof, keepdims=True)._collapse(axis)
618: 
619:     def var(self, axis=None, dtype=None, out=None, ddof=0):
620:         '''
621:         Returns the variance of the matrix elements, along the given axis.
622: 
623:         Refer to `numpy.var` for full documentation.
624: 
625:         See Also
626:         --------
627:         numpy.var
628: 
629:         Notes
630:         -----
631:         This is the same as `ndarray.var`, except that where an `ndarray` would
632:         be returned, a `matrix` object is returned instead.
633: 
634:         Examples
635:         --------
636:         >>> x = np.matrix(np.arange(12).reshape((3, 4)))
637:         >>> x
638:         matrix([[ 0,  1,  2,  3],
639:                 [ 4,  5,  6,  7],
640:                 [ 8,  9, 10, 11]])
641:         >>> x.var()
642:         11.916666666666666
643:         >>> x.var(0)
644:         matrix([[ 10.66666667,  10.66666667,  10.66666667,  10.66666667]])
645:         >>> x.var(1)
646:         matrix([[ 1.25],
647:                 [ 1.25],
648:                 [ 1.25]])
649: 
650:         '''
651:         return N.ndarray.var(self, axis, dtype, out, ddof, keepdims=True)._collapse(axis)
652: 
653:     def prod(self, axis=None, dtype=None, out=None):
654:         '''
655:         Return the product of the array elements over the given axis.
656: 
657:         Refer to `prod` for full documentation.
658: 
659:         See Also
660:         --------
661:         prod, ndarray.prod
662: 
663:         Notes
664:         -----
665:         Same as `ndarray.prod`, except, where that returns an `ndarray`, this
666:         returns a `matrix` object instead.
667: 
668:         Examples
669:         --------
670:         >>> x = np.matrix(np.arange(12).reshape((3,4))); x
671:         matrix([[ 0,  1,  2,  3],
672:                 [ 4,  5,  6,  7],
673:                 [ 8,  9, 10, 11]])
674:         >>> x.prod()
675:         0
676:         >>> x.prod(0)
677:         matrix([[  0,  45, 120, 231]])
678:         >>> x.prod(1)
679:         matrix([[   0],
680:                 [ 840],
681:                 [7920]])
682: 
683:         '''
684:         return N.ndarray.prod(self, axis, dtype, out, keepdims=True)._collapse(axis)
685: 
686:     def any(self, axis=None, out=None):
687:         '''
688:         Test whether any array element along a given axis evaluates to True.
689: 
690:         Refer to `numpy.any` for full documentation.
691: 
692:         Parameters
693:         ----------
694:         axis : int, optional
695:             Axis along which logical OR is performed
696:         out : ndarray, optional
697:             Output to existing array instead of creating new one, must have
698:             same shape as expected output
699: 
700:         Returns
701:         -------
702:             any : bool, ndarray
703:                 Returns a single bool if `axis` is ``None``; otherwise,
704:                 returns `ndarray`
705: 
706:         '''
707:         return N.ndarray.any(self, axis, out, keepdims=True)._collapse(axis)
708: 
709:     def all(self, axis=None, out=None):
710:         '''
711:         Test whether all matrix elements along a given axis evaluate to True.
712: 
713:         Parameters
714:         ----------
715:         See `numpy.all` for complete descriptions
716: 
717:         See Also
718:         --------
719:         numpy.all
720: 
721:         Notes
722:         -----
723:         This is the same as `ndarray.all`, but it returns a `matrix` object.
724: 
725:         Examples
726:         --------
727:         >>> x = np.matrix(np.arange(12).reshape((3,4))); x
728:         matrix([[ 0,  1,  2,  3],
729:                 [ 4,  5,  6,  7],
730:                 [ 8,  9, 10, 11]])
731:         >>> y = x[0]; y
732:         matrix([[0, 1, 2, 3]])
733:         >>> (x == y)
734:         matrix([[ True,  True,  True,  True],
735:                 [False, False, False, False],
736:                 [False, False, False, False]], dtype=bool)
737:         >>> (x == y).all()
738:         False
739:         >>> (x == y).all(0)
740:         matrix([[False, False, False, False]], dtype=bool)
741:         >>> (x == y).all(1)
742:         matrix([[ True],
743:                 [False],
744:                 [False]], dtype=bool)
745: 
746:         '''
747:         return N.ndarray.all(self, axis, out, keepdims=True)._collapse(axis)
748: 
749:     def max(self, axis=None, out=None):
750:         '''
751:         Return the maximum value along an axis.
752: 
753:         Parameters
754:         ----------
755:         See `amax` for complete descriptions
756: 
757:         See Also
758:         --------
759:         amax, ndarray.max
760: 
761:         Notes
762:         -----
763:         This is the same as `ndarray.max`, but returns a `matrix` object
764:         where `ndarray.max` would return an ndarray.
765: 
766:         Examples
767:         --------
768:         >>> x = np.matrix(np.arange(12).reshape((3,4))); x
769:         matrix([[ 0,  1,  2,  3],
770:                 [ 4,  5,  6,  7],
771:                 [ 8,  9, 10, 11]])
772:         >>> x.max()
773:         11
774:         >>> x.max(0)
775:         matrix([[ 8,  9, 10, 11]])
776:         >>> x.max(1)
777:         matrix([[ 3],
778:                 [ 7],
779:                 [11]])
780: 
781:         '''
782:         return N.ndarray.max(self, axis, out, keepdims=True)._collapse(axis)
783: 
784:     def argmax(self, axis=None, out=None):
785:         '''
786:         Indexes of the maximum values along an axis.
787: 
788:         Return the indexes of the first occurrences of the maximum values
789:         along the specified axis.  If axis is None, the index is for the
790:         flattened matrix.
791: 
792:         Parameters
793:         ----------
794:         See `numpy.argmax` for complete descriptions
795: 
796:         See Also
797:         --------
798:         numpy.argmax
799: 
800:         Notes
801:         -----
802:         This is the same as `ndarray.argmax`, but returns a `matrix` object
803:         where `ndarray.argmax` would return an `ndarray`.
804: 
805:         Examples
806:         --------
807:         >>> x = np.matrix(np.arange(12).reshape((3,4))); x
808:         matrix([[ 0,  1,  2,  3],
809:                 [ 4,  5,  6,  7],
810:                 [ 8,  9, 10, 11]])
811:         >>> x.argmax()
812:         11
813:         >>> x.argmax(0)
814:         matrix([[2, 2, 2, 2]])
815:         >>> x.argmax(1)
816:         matrix([[3],
817:                 [3],
818:                 [3]])
819: 
820:         '''
821:         return N.ndarray.argmax(self, axis, out)._align(axis)
822: 
823:     def min(self, axis=None, out=None):
824:         '''
825:         Return the minimum value along an axis.
826: 
827:         Parameters
828:         ----------
829:         See `amin` for complete descriptions.
830: 
831:         See Also
832:         --------
833:         amin, ndarray.min
834: 
835:         Notes
836:         -----
837:         This is the same as `ndarray.min`, but returns a `matrix` object
838:         where `ndarray.min` would return an ndarray.
839: 
840:         Examples
841:         --------
842:         >>> x = -np.matrix(np.arange(12).reshape((3,4))); x
843:         matrix([[  0,  -1,  -2,  -3],
844:                 [ -4,  -5,  -6,  -7],
845:                 [ -8,  -9, -10, -11]])
846:         >>> x.min()
847:         -11
848:         >>> x.min(0)
849:         matrix([[ -8,  -9, -10, -11]])
850:         >>> x.min(1)
851:         matrix([[ -3],
852:                 [ -7],
853:                 [-11]])
854: 
855:         '''
856:         return N.ndarray.min(self, axis, out, keepdims=True)._collapse(axis)
857: 
858:     def argmin(self, axis=None, out=None):
859:         '''
860:         Indexes of the minimum values along an axis.
861: 
862:         Return the indexes of the first occurrences of the minimum values
863:         along the specified axis.  If axis is None, the index is for the
864:         flattened matrix.
865: 
866:         Parameters
867:         ----------
868:         See `numpy.argmin` for complete descriptions.
869: 
870:         See Also
871:         --------
872:         numpy.argmin
873: 
874:         Notes
875:         -----
876:         This is the same as `ndarray.argmin`, but returns a `matrix` object
877:         where `ndarray.argmin` would return an `ndarray`.
878: 
879:         Examples
880:         --------
881:         >>> x = -np.matrix(np.arange(12).reshape((3,4))); x
882:         matrix([[  0,  -1,  -2,  -3],
883:                 [ -4,  -5,  -6,  -7],
884:                 [ -8,  -9, -10, -11]])
885:         >>> x.argmin()
886:         11
887:         >>> x.argmin(0)
888:         matrix([[2, 2, 2, 2]])
889:         >>> x.argmin(1)
890:         matrix([[3],
891:                 [3],
892:                 [3]])
893: 
894:         '''
895:         return N.ndarray.argmin(self, axis, out)._align(axis)
896: 
897:     def ptp(self, axis=None, out=None):
898:         '''
899:         Peak-to-peak (maximum - minimum) value along the given axis.
900: 
901:         Refer to `numpy.ptp` for full documentation.
902: 
903:         See Also
904:         --------
905:         numpy.ptp
906: 
907:         Notes
908:         -----
909:         Same as `ndarray.ptp`, except, where that would return an `ndarray` object,
910:         this returns a `matrix` object.
911: 
912:         Examples
913:         --------
914:         >>> x = np.matrix(np.arange(12).reshape((3,4))); x
915:         matrix([[ 0,  1,  2,  3],
916:                 [ 4,  5,  6,  7],
917:                 [ 8,  9, 10, 11]])
918:         >>> x.ptp()
919:         11
920:         >>> x.ptp(0)
921:         matrix([[8, 8, 8, 8]])
922:         >>> x.ptp(1)
923:         matrix([[3],
924:                 [3],
925:                 [3]])
926: 
927:         '''
928:         return N.ndarray.ptp(self, axis, out)._align(axis)
929: 
930:     def getI(self):
931:         '''
932:         Returns the (multiplicative) inverse of invertible `self`.
933: 
934:         Parameters
935:         ----------
936:         None
937: 
938:         Returns
939:         -------
940:         ret : matrix object
941:             If `self` is non-singular, `ret` is such that ``ret * self`` ==
942:             ``self * ret`` == ``np.matrix(np.eye(self[0,:].size)`` all return
943:             ``True``.
944: 
945:         Raises
946:         ------
947:         numpy.linalg.LinAlgError: Singular matrix
948:             If `self` is singular.
949: 
950:         See Also
951:         --------
952:         linalg.inv
953: 
954:         Examples
955:         --------
956:         >>> m = np.matrix('[1, 2; 3, 4]'); m
957:         matrix([[1, 2],
958:                 [3, 4]])
959:         >>> m.getI()
960:         matrix([[-2. ,  1. ],
961:                 [ 1.5, -0.5]])
962:         >>> m.getI() * m
963:         matrix([[ 1.,  0.],
964:                 [ 0.,  1.]])
965: 
966:         '''
967:         M, N = self.shape
968:         if M == N:
969:             from numpy.dual import inv as func
970:         else:
971:             from numpy.dual import pinv as func
972:         return asmatrix(func(self))
973: 
974:     def getA(self):
975:         '''
976:         Return `self` as an `ndarray` object.
977: 
978:         Equivalent to ``np.asarray(self)``.
979: 
980:         Parameters
981:         ----------
982:         None
983: 
984:         Returns
985:         -------
986:         ret : ndarray
987:             `self` as an `ndarray`
988: 
989:         Examples
990:         --------
991:         >>> x = np.matrix(np.arange(12).reshape((3,4))); x
992:         matrix([[ 0,  1,  2,  3],
993:                 [ 4,  5,  6,  7],
994:                 [ 8,  9, 10, 11]])
995:         >>> x.getA()
996:         array([[ 0,  1,  2,  3],
997:                [ 4,  5,  6,  7],
998:                [ 8,  9, 10, 11]])
999: 
1000:         '''
1001:         return self.__array__()
1002: 
1003:     def getA1(self):
1004:         '''
1005:         Return `self` as a flattened `ndarray`.
1006: 
1007:         Equivalent to ``np.asarray(x).ravel()``
1008: 
1009:         Parameters
1010:         ----------
1011:         None
1012: 
1013:         Returns
1014:         -------
1015:         ret : ndarray
1016:             `self`, 1-D, as an `ndarray`
1017: 
1018:         Examples
1019:         --------
1020:         >>> x = np.matrix(np.arange(12).reshape((3,4))); x
1021:         matrix([[ 0,  1,  2,  3],
1022:                 [ 4,  5,  6,  7],
1023:                 [ 8,  9, 10, 11]])
1024:         >>> x.getA1()
1025:         array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
1026: 
1027:         '''
1028:         return self.__array__().ravel()
1029: 
1030: 
1031:     def ravel(self, order='C'):
1032:         '''
1033:         Return a flattened matrix.
1034: 
1035:         Refer to `numpy.ravel` for more documentation.
1036: 
1037:         Parameters
1038:         ----------
1039:         order : {'C', 'F', 'A', 'K'}, optional
1040:             The elements of `m` are read using this index order. 'C' means to
1041:             index the elements in C-like order, with the last axis index
1042:             changing fastest, back to the first axis index changing slowest.
1043:             'F' means to index the elements in Fortran-like index order, with
1044:             the first index changing fastest, and the last index changing
1045:             slowest. Note that the 'C' and 'F' options take no account of the
1046:             memory layout of the underlying array, and only refer to the order
1047:             of axis indexing.  'A' means to read the elements in Fortran-like
1048:             index order if `m` is Fortran *contiguous* in memory, C-like order
1049:             otherwise.  'K' means to read the elements in the order they occur
1050:             in memory, except for reversing the data when strides are negative.
1051:             By default, 'C' index order is used.
1052: 
1053:         Returns
1054:         -------
1055:         ret : matrix
1056:             Return the matrix flattened to shape `(1, N)` where `N`
1057:             is the number of elements in the original matrix.
1058:             A copy is made only if necessary.
1059: 
1060:         See Also
1061:         --------
1062:         matrix.flatten : returns a similar output matrix but always a copy
1063:         matrix.flat : a flat iterator on the array.
1064:         numpy.ravel : related function which returns an ndarray
1065: 
1066:         '''
1067:         return N.ndarray.ravel(self, order=order)
1068: 
1069: 
1070:     def getT(self):
1071:         '''
1072:         Returns the transpose of the matrix.
1073: 
1074:         Does *not* conjugate!  For the complex conjugate transpose, use ``.H``.
1075: 
1076:         Parameters
1077:         ----------
1078:         None
1079: 
1080:         Returns
1081:         -------
1082:         ret : matrix object
1083:             The (non-conjugated) transpose of the matrix.
1084: 
1085:         See Also
1086:         --------
1087:         transpose, getH
1088: 
1089:         Examples
1090:         --------
1091:         >>> m = np.matrix('[1, 2; 3, 4]')
1092:         >>> m
1093:         matrix([[1, 2],
1094:                 [3, 4]])
1095:         >>> m.getT()
1096:         matrix([[1, 3],
1097:                 [2, 4]])
1098: 
1099:         '''
1100:         return self.transpose()
1101: 
1102:     def getH(self):
1103:         '''
1104:         Returns the (complex) conjugate transpose of `self`.
1105: 
1106:         Equivalent to ``np.transpose(self)`` if `self` is real-valued.
1107: 
1108:         Parameters
1109:         ----------
1110:         None
1111: 
1112:         Returns
1113:         -------
1114:         ret : matrix object
1115:             complex conjugate transpose of `self`
1116: 
1117:         Examples
1118:         --------
1119:         >>> x = np.matrix(np.arange(12).reshape((3,4)))
1120:         >>> z = x - 1j*x; z
1121:         matrix([[  0. +0.j,   1. -1.j,   2. -2.j,   3. -3.j],
1122:                 [  4. -4.j,   5. -5.j,   6. -6.j,   7. -7.j],
1123:                 [  8. -8.j,   9. -9.j,  10.-10.j,  11.-11.j]])
1124:         >>> z.getH()
1125:         matrix([[  0. +0.j,   4. +4.j,   8. +8.j],
1126:                 [  1. +1.j,   5. +5.j,   9. +9.j],
1127:                 [  2. +2.j,   6. +6.j,  10.+10.j],
1128:                 [  3. +3.j,   7. +7.j,  11.+11.j]])
1129: 
1130:         '''
1131:         if issubclass(self.dtype.type, N.complexfloating):
1132:             return self.transpose().conjugate()
1133:         else:
1134:             return self.transpose()
1135: 
1136:     T = property(getT, None)
1137:     A = property(getA, None)
1138:     A1 = property(getA1, None)
1139:     H = property(getH, None)
1140:     I = property(getI, None)
1141: 
1142: def _from_string(str, gdict, ldict):
1143:     rows = str.split(';')
1144:     rowtup = []
1145:     for row in rows:
1146:         trow = row.split(',')
1147:         newrow = []
1148:         for x in trow:
1149:             newrow.extend(x.split())
1150:         trow = newrow
1151:         coltup = []
1152:         for col in trow:
1153:             col = col.strip()
1154:             try:
1155:                 thismat = ldict[col]
1156:             except KeyError:
1157:                 try:
1158:                     thismat = gdict[col]
1159:                 except KeyError:
1160:                     raise KeyError("%s not found" % (col,))
1161: 
1162:             coltup.append(thismat)
1163:         rowtup.append(concatenate(coltup, axis=-1))
1164:     return concatenate(rowtup, axis=0)
1165: 
1166: 
1167: def bmat(obj, ldict=None, gdict=None):
1168:     '''
1169:     Build a matrix object from a string, nested sequence, or array.
1170: 
1171:     Parameters
1172:     ----------
1173:     obj : str or array_like
1174:         Input data.  Names of variables in the current scope may be
1175:         referenced, even if `obj` is a string.
1176:     ldict : dict, optional
1177:         A dictionary that replaces local operands in current frame.
1178:         Ignored if `obj` is not a string or `gdict` is `None`.
1179:     gdict : dict, optional
1180:         A dictionary that replaces global operands in current frame.
1181:         Ignored if `obj` is not a string.
1182: 
1183:     Returns
1184:     -------
1185:     out : matrix
1186:         Returns a matrix object, which is a specialized 2-D array.
1187: 
1188:     See Also
1189:     --------
1190:     matrix
1191: 
1192:     Examples
1193:     --------
1194:     >>> A = np.mat('1 1; 1 1')
1195:     >>> B = np.mat('2 2; 2 2')
1196:     >>> C = np.mat('3 4; 5 6')
1197:     >>> D = np.mat('7 8; 9 0')
1198: 
1199:     All the following expressions construct the same block matrix:
1200: 
1201:     >>> np.bmat([[A, B], [C, D]])
1202:     matrix([[1, 1, 2, 2],
1203:             [1, 1, 2, 2],
1204:             [3, 4, 7, 8],
1205:             [5, 6, 9, 0]])
1206:     >>> np.bmat(np.r_[np.c_[A, B], np.c_[C, D]])
1207:     matrix([[1, 1, 2, 2],
1208:             [1, 1, 2, 2],
1209:             [3, 4, 7, 8],
1210:             [5, 6, 9, 0]])
1211:     >>> np.bmat('A,B; C,D')
1212:     matrix([[1, 1, 2, 2],
1213:             [1, 1, 2, 2],
1214:             [3, 4, 7, 8],
1215:             [5, 6, 9, 0]])
1216: 
1217:     '''
1218:     if isinstance(obj, str):
1219:         if gdict is None:
1220:             # get previous frame
1221:             frame = sys._getframe().f_back
1222:             glob_dict = frame.f_globals
1223:             loc_dict = frame.f_locals
1224:         else:
1225:             glob_dict = gdict
1226:             loc_dict = ldict
1227: 
1228:         return matrix(_from_string(obj, glob_dict, loc_dict))
1229: 
1230:     if isinstance(obj, (tuple, list)):
1231:         # [[A,B],[C,D]]
1232:         arr_rows = []
1233:         for row in obj:
1234:             if isinstance(row, N.ndarray):  # not 2-d
1235:                 return matrix(concatenate(obj, axis=-1))
1236:             else:
1237:                 arr_rows.append(concatenate(row, axis=-1))
1238:         return matrix(concatenate(arr_rows, axis=0))
1239:     if isinstance(obj, N.ndarray):
1240:         return matrix(obj)
1241: 
1242: mat = asmatrix
1243: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 3):

# Assigning a List to a Name (line 3):
__all__ = ['matrix', 'bmat', 'mat', 'asmatrix']
module_type_store.set_exportable_members(['matrix', 'bmat', 'mat', 'asmatrix'])

# Obtaining an instance of the builtin type 'list' (line 3)
list_160498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 3)
# Adding element type (line 3)
str_160499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', 'matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_160498, str_160499)
# Adding element type (line 3)
str_160500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 21), 'str', 'bmat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_160498, str_160500)
# Adding element type (line 3)
str_160501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 29), 'str', 'mat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_160498, str_160501)
# Adding element type (line 3)
str_160502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 36), 'str', 'asmatrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_160498, str_160502)

# Assigning a type to the variable '__all__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__all__', list_160498)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy.core.numeric' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/matrixlib/')
import_160503 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric')

if (type(import_160503) is not StypyTypeError):

    if (import_160503 != 'pyd_module'):
        __import__(import_160503)
        sys_modules_160504 = sys.modules[import_160503]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'N', sys_modules_160504.module_type_store, module_type_store)
    else:
        import numpy.core.numeric as N

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'N', numpy.core.numeric, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric', import_160503)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/matrixlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.core.numeric import concatenate, isscalar, binary_repr, identity, asanyarray' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/matrixlib/')
import_160505 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.numeric')

if (type(import_160505) is not StypyTypeError):

    if (import_160505 != 'pyd_module'):
        __import__(import_160505)
        sys_modules_160506 = sys.modules[import_160505]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.numeric', sys_modules_160506.module_type_store, module_type_store, ['concatenate', 'isscalar', 'binary_repr', 'identity', 'asanyarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_160506, sys_modules_160506.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import concatenate, isscalar, binary_repr, identity, asanyarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.numeric', None, module_type_store, ['concatenate', 'isscalar', 'binary_repr', 'identity', 'asanyarray'], [concatenate, isscalar, binary_repr, identity, asanyarray])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.numeric', import_160505)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/matrixlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.core.numerictypes import issubdtype' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/matrixlib/')
import_160507 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.numerictypes')

if (type(import_160507) is not StypyTypeError):

    if (import_160507 != 'pyd_module'):
        __import__(import_160507)
        sys_modules_160508 = sys.modules[import_160507]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.numerictypes', sys_modules_160508.module_type_store, module_type_store, ['issubdtype'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_160508, sys_modules_160508.module_type_store, module_type_store)
    else:
        from numpy.core.numerictypes import issubdtype

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.numerictypes', None, module_type_store, ['issubdtype'], [issubdtype])

else:
    # Assigning a type to the variable 'numpy.core.numerictypes' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.numerictypes', import_160507)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/matrixlib/')


# Assigning a Str to a Name (line 11):

# Assigning a Str to a Name (line 11):
str_160509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 12), 'str', '0123456789.-+jeEL')
# Assigning a type to the variable '_numchars' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '_numchars', str_160509)



# Obtaining the type of the subscript
int_160510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'int')
# Getting the type of 'sys' (line 13)
sys_160511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 13)
version_info_160512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 3), sys_160511, 'version_info')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___160513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 3), version_info_160512, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_160514 = invoke(stypy.reporting.localization.Localization(__file__, 13, 3), getitem___160513, int_160510)

int_160515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 26), 'int')
# Applying the binary operator '>=' (line 13)
result_ge_160516 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 3), '>=', subscript_call_result_160514, int_160515)

# Testing the type of an if condition (line 13)
if_condition_160517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 13, 0), result_ge_160516)
# Assigning a type to the variable 'if_condition_160517' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'if_condition_160517', if_condition_160517)
# SSA begins for if statement (line 13)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
# Declaration of the '_NumCharTable' class

class _NumCharTable:

    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 15, 8, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        _NumCharTable.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        _NumCharTable.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _NumCharTable.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _NumCharTable.__getitem__.__dict__.__setitem__('stypy_function_name', '_NumCharTable.__getitem__')
        _NumCharTable.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['i'])
        _NumCharTable.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _NumCharTable.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _NumCharTable.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _NumCharTable.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _NumCharTable.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _NumCharTable.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_NumCharTable.__getitem__', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        
        
        # Call to chr(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'i' (line 16)
        i_160519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 19), 'i', False)
        # Processing the call keyword arguments (line 16)
        kwargs_160520 = {}
        # Getting the type of 'chr' (line 16)
        chr_160518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'chr', False)
        # Calling chr(args, kwargs) (line 16)
        chr_call_result_160521 = invoke(stypy.reporting.localization.Localization(__file__, 16, 15), chr_160518, *[i_160519], **kwargs_160520)
        
        # Getting the type of '_numchars' (line 16)
        _numchars_160522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 25), '_numchars')
        # Applying the binary operator 'in' (line 16)
        result_contains_160523 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 15), 'in', chr_call_result_160521, _numchars_160522)
        
        # Testing the type of an if condition (line 16)
        if_condition_160524 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 16, 12), result_contains_160523)
        # Assigning a type to the variable 'if_condition_160524' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'if_condition_160524', if_condition_160524)
        # SSA begins for if statement (line 16)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to chr(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'i' (line 17)
        i_160526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 27), 'i', False)
        # Processing the call keyword arguments (line 17)
        kwargs_160527 = {}
        # Getting the type of 'chr' (line 17)
        chr_160525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 'chr', False)
        # Calling chr(args, kwargs) (line 17)
        chr_call_result_160528 = invoke(stypy.reporting.localization.Localization(__file__, 17, 23), chr_160525, *[i_160526], **kwargs_160527)
        
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'stypy_return_type', chr_call_result_160528)
        # SSA branch for the else part of an if statement (line 16)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'None' (line 19)
        None_160529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'stypy_return_type', None_160529)
        # SSA join for if statement (line 16)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_160530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_160530)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_160530


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_NumCharTable.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_NumCharTable' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), '_NumCharTable', _NumCharTable)

# Assigning a Call to a Name (line 20):

# Assigning a Call to a Name (line 20):

# Call to _NumCharTable(...): (line 20)
# Processing the call keyword arguments (line 20)
kwargs_160532 = {}
# Getting the type of '_NumCharTable' (line 20)
_NumCharTable_160531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 13), '_NumCharTable', False)
# Calling _NumCharTable(args, kwargs) (line 20)
_NumCharTable_call_result_160533 = invoke(stypy.reporting.localization.Localization(__file__, 20, 13), _NumCharTable_160531, *[], **kwargs_160532)

# Assigning a type to the variable '_table' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), '_table', _NumCharTable_call_result_160533)

@norecursion
def _eval(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_eval'
    module_type_store = module_type_store.open_function_context('_eval', 21, 4, False)
    
    # Passed parameters checking function
    _eval.stypy_localization = localization
    _eval.stypy_type_of_self = None
    _eval.stypy_type_store = module_type_store
    _eval.stypy_function_name = '_eval'
    _eval.stypy_param_names_list = ['astr']
    _eval.stypy_varargs_param_name = None
    _eval.stypy_kwargs_param_name = None
    _eval.stypy_call_defaults = defaults
    _eval.stypy_call_varargs = varargs
    _eval.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_eval', ['astr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_eval', localization, ['astr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_eval(...)' code ##################

    
    # Assigning a Call to a Name (line 22):
    
    # Assigning a Call to a Name (line 22):
    
    # Call to translate(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of '_table' (line 22)
    _table_160536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 30), '_table', False)
    # Processing the call keyword arguments (line 22)
    kwargs_160537 = {}
    # Getting the type of 'astr' (line 22)
    astr_160534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'astr', False)
    # Obtaining the member 'translate' of a type (line 22)
    translate_160535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 15), astr_160534, 'translate')
    # Calling translate(args, kwargs) (line 22)
    translate_call_result_160538 = invoke(stypy.reporting.localization.Localization(__file__, 22, 15), translate_160535, *[_table_160536], **kwargs_160537)
    
    # Assigning a type to the variable 'str_' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'str_', translate_call_result_160538)
    
    
    # Getting the type of 'str_' (line 23)
    str__160539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'str_')
    # Applying the 'not' unary operator (line 23)
    result_not__160540 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 11), 'not', str__160539)
    
    # Testing the type of an if condition (line 23)
    if_condition_160541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 8), result_not__160540)
    # Assigning a type to the variable 'if_condition_160541' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'if_condition_160541', if_condition_160541)
    # SSA begins for if statement (line 23)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 24)
    # Processing the call arguments (line 24)
    str_160543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 28), 'str', 'Invalid data string supplied: ')
    # Getting the type of 'astr' (line 24)
    astr_160544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 63), 'astr', False)
    # Applying the binary operator '+' (line 24)
    result_add_160545 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 28), '+', str_160543, astr_160544)
    
    # Processing the call keyword arguments (line 24)
    kwargs_160546 = {}
    # Getting the type of 'TypeError' (line 24)
    TypeError_160542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 24)
    TypeError_call_result_160547 = invoke(stypy.reporting.localization.Localization(__file__, 24, 18), TypeError_160542, *[result_add_160545], **kwargs_160546)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 24, 12), TypeError_call_result_160547, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 23)
    module_type_store.open_ssa_branch('else')
    
    # Call to eval(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'str_' (line 26)
    str__160549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 24), 'str_', False)
    # Processing the call keyword arguments (line 26)
    kwargs_160550 = {}
    # Getting the type of 'eval' (line 26)
    eval_160548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 19), 'eval', False)
    # Calling eval(args, kwargs) (line 26)
    eval_call_result_160551 = invoke(stypy.reporting.localization.Localization(__file__, 26, 19), eval_160548, *[str__160549], **kwargs_160550)
    
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'stypy_return_type', eval_call_result_160551)
    # SSA join for if statement (line 23)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_eval(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_eval' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_160552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_160552)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_eval'
    return stypy_return_type_160552

# Assigning a type to the variable '_eval' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), '_eval', _eval)
# SSA branch for the else part of an if statement (line 13)
module_type_store.open_ssa_branch('else')

# Assigning a BinOp to a Name (line 29):

# Assigning a BinOp to a Name (line 29):

# Obtaining an instance of the builtin type 'list' (line 29)
list_160553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)
# Getting the type of 'None' (line 29)
None_160554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 13), list_160553, None_160554)

int_160555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 20), 'int')
# Applying the binary operator '*' (line 29)
result_mul_160556 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 13), '*', list_160553, int_160555)

# Assigning a type to the variable '_table' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), '_table', result_mul_160556)


# Call to range(...): (line 30)
# Processing the call arguments (line 30)
int_160558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 19), 'int')
# Processing the call keyword arguments (line 30)
kwargs_160559 = {}
# Getting the type of 'range' (line 30)
range_160557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 13), 'range', False)
# Calling range(args, kwargs) (line 30)
range_call_result_160560 = invoke(stypy.reporting.localization.Localization(__file__, 30, 13), range_160557, *[int_160558], **kwargs_160559)

# Testing the type of a for loop iterable (line 30)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 30, 4), range_call_result_160560)
# Getting the type of the for loop variable (line 30)
for_loop_var_160561 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 30, 4), range_call_result_160560)
# Assigning a type to the variable 'k' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'k', for_loop_var_160561)
# SSA begins for a for statement (line 30)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Call to a Subscript (line 31):

# Assigning a Call to a Subscript (line 31):

# Call to chr(...): (line 31)
# Processing the call arguments (line 31)
# Getting the type of 'k' (line 31)
k_160563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'k', False)
# Processing the call keyword arguments (line 31)
kwargs_160564 = {}
# Getting the type of 'chr' (line 31)
chr_160562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'chr', False)
# Calling chr(args, kwargs) (line 31)
chr_call_result_160565 = invoke(stypy.reporting.localization.Localization(__file__, 31, 20), chr_160562, *[k_160563], **kwargs_160564)

# Getting the type of '_table' (line 31)
_table_160566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), '_table')
# Getting the type of 'k' (line 31)
k_160567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'k')
# Storing an element on a container (line 31)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 8), _table_160566, (k_160567, chr_call_result_160565))
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 32):

# Assigning a Call to a Name (line 32):

# Call to join(...): (line 32)
# Processing the call arguments (line 32)
# Getting the type of '_table' (line 32)
_table_160570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 21), '_table', False)
# Processing the call keyword arguments (line 32)
kwargs_160571 = {}
str_160568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 13), 'str', '')
# Obtaining the member 'join' of a type (line 32)
join_160569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 13), str_160568, 'join')
# Calling join(args, kwargs) (line 32)
join_call_result_160572 = invoke(stypy.reporting.localization.Localization(__file__, 32, 13), join_160569, *[_table_160570], **kwargs_160571)

# Assigning a type to the variable '_table' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), '_table', join_call_result_160572)

# Assigning a List to a Name (line 34):

# Assigning a List to a Name (line 34):

# Obtaining an instance of the builtin type 'list' (line 34)
list_160573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 34)

# Assigning a type to the variable '_todelete' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), '_todelete', list_160573)

# Getting the type of '_table' (line 35)
_table_160574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), '_table')
# Testing the type of a for loop iterable (line 35)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 4), _table_160574)
# Getting the type of the for loop variable (line 35)
for_loop_var_160575 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 4), _table_160574)
# Assigning a type to the variable 'k' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'k', for_loop_var_160575)
# SSA begins for a for statement (line 35)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')


# Getting the type of 'k' (line 36)
k_160576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'k')
# Getting the type of '_numchars' (line 36)
_numchars_160577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), '_numchars')
# Applying the binary operator 'notin' (line 36)
result_contains_160578 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 11), 'notin', k_160576, _numchars_160577)

# Testing the type of an if condition (line 36)
if_condition_160579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 8), result_contains_160578)
# Assigning a type to the variable 'if_condition_160579' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'if_condition_160579', if_condition_160579)
# SSA begins for if statement (line 36)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to append(...): (line 37)
# Processing the call arguments (line 37)
# Getting the type of 'k' (line 37)
k_160582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'k', False)
# Processing the call keyword arguments (line 37)
kwargs_160583 = {}
# Getting the type of '_todelete' (line 37)
_todelete_160580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), '_todelete', False)
# Obtaining the member 'append' of a type (line 37)
append_160581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), _todelete_160580, 'append')
# Calling append(args, kwargs) (line 37)
append_call_result_160584 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), append_160581, *[k_160582], **kwargs_160583)

# SSA join for if statement (line 36)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 38):

# Assigning a Call to a Name (line 38):

# Call to join(...): (line 38)
# Processing the call arguments (line 38)
# Getting the type of '_todelete' (line 38)
_todelete_160587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), '_todelete', False)
# Processing the call keyword arguments (line 38)
kwargs_160588 = {}
str_160585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 16), 'str', '')
# Obtaining the member 'join' of a type (line 38)
join_160586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 16), str_160585, 'join')
# Calling join(args, kwargs) (line 38)
join_call_result_160589 = invoke(stypy.reporting.localization.Localization(__file__, 38, 16), join_160586, *[_todelete_160587], **kwargs_160588)

# Assigning a type to the variable '_todelete' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), '_todelete', join_call_result_160589)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 39, 4), module_type_store, 'k')

@norecursion
def _eval(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_eval'
    module_type_store = module_type_store.open_function_context('_eval', 41, 4, False)
    
    # Passed parameters checking function
    _eval.stypy_localization = localization
    _eval.stypy_type_of_self = None
    _eval.stypy_type_store = module_type_store
    _eval.stypy_function_name = '_eval'
    _eval.stypy_param_names_list = ['astr']
    _eval.stypy_varargs_param_name = None
    _eval.stypy_kwargs_param_name = None
    _eval.stypy_call_defaults = defaults
    _eval.stypy_call_varargs = varargs
    _eval.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_eval', ['astr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_eval', localization, ['astr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_eval(...)' code ##################

    
    # Assigning a Call to a Name (line 42):
    
    # Assigning a Call to a Name (line 42):
    
    # Call to translate(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of '_table' (line 42)
    _table_160592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 30), '_table', False)
    # Getting the type of '_todelete' (line 42)
    _todelete_160593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 38), '_todelete', False)
    # Processing the call keyword arguments (line 42)
    kwargs_160594 = {}
    # Getting the type of 'astr' (line 42)
    astr_160590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'astr', False)
    # Obtaining the member 'translate' of a type (line 42)
    translate_160591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 15), astr_160590, 'translate')
    # Calling translate(args, kwargs) (line 42)
    translate_call_result_160595 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), translate_160591, *[_table_160592, _todelete_160593], **kwargs_160594)
    
    # Assigning a type to the variable 'str_' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'str_', translate_call_result_160595)
    
    
    # Getting the type of 'str_' (line 43)
    str__160596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'str_')
    # Applying the 'not' unary operator (line 43)
    result_not__160597 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 11), 'not', str__160596)
    
    # Testing the type of an if condition (line 43)
    if_condition_160598 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 8), result_not__160597)
    # Assigning a type to the variable 'if_condition_160598' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'if_condition_160598', if_condition_160598)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 44)
    # Processing the call arguments (line 44)
    str_160600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 28), 'str', 'Invalid data string supplied: ')
    # Getting the type of 'astr' (line 44)
    astr_160601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 63), 'astr', False)
    # Applying the binary operator '+' (line 44)
    result_add_160602 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 28), '+', str_160600, astr_160601)
    
    # Processing the call keyword arguments (line 44)
    kwargs_160603 = {}
    # Getting the type of 'TypeError' (line 44)
    TypeError_160599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 44)
    TypeError_call_result_160604 = invoke(stypy.reporting.localization.Localization(__file__, 44, 18), TypeError_160599, *[result_add_160602], **kwargs_160603)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 44, 12), TypeError_call_result_160604, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 43)
    module_type_store.open_ssa_branch('else')
    
    # Call to eval(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'str_' (line 46)
    str__160606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'str_', False)
    # Processing the call keyword arguments (line 46)
    kwargs_160607 = {}
    # Getting the type of 'eval' (line 46)
    eval_160605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'eval', False)
    # Calling eval(args, kwargs) (line 46)
    eval_call_result_160608 = invoke(stypy.reporting.localization.Localization(__file__, 46, 19), eval_160605, *[str__160606], **kwargs_160607)
    
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type', eval_call_result_160608)
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_eval(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_eval' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_160609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_160609)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_eval'
    return stypy_return_type_160609

# Assigning a type to the variable '_eval' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), '_eval', _eval)
# SSA join for if statement (line 13)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def _convert_from_string(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_convert_from_string'
    module_type_store = module_type_store.open_function_context('_convert_from_string', 48, 0, False)
    
    # Passed parameters checking function
    _convert_from_string.stypy_localization = localization
    _convert_from_string.stypy_type_of_self = None
    _convert_from_string.stypy_type_store = module_type_store
    _convert_from_string.stypy_function_name = '_convert_from_string'
    _convert_from_string.stypy_param_names_list = ['data']
    _convert_from_string.stypy_varargs_param_name = None
    _convert_from_string.stypy_kwargs_param_name = None
    _convert_from_string.stypy_call_defaults = defaults
    _convert_from_string.stypy_call_varargs = varargs
    _convert_from_string.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_convert_from_string', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_convert_from_string', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_convert_from_string(...)' code ##################

    
    # Assigning a Call to a Name (line 49):
    
    # Assigning a Call to a Name (line 49):
    
    # Call to split(...): (line 49)
    # Processing the call arguments (line 49)
    str_160612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 22), 'str', ';')
    # Processing the call keyword arguments (line 49)
    kwargs_160613 = {}
    # Getting the type of 'data' (line 49)
    data_160610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'data', False)
    # Obtaining the member 'split' of a type (line 49)
    split_160611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 11), data_160610, 'split')
    # Calling split(args, kwargs) (line 49)
    split_call_result_160614 = invoke(stypy.reporting.localization.Localization(__file__, 49, 11), split_160611, *[str_160612], **kwargs_160613)
    
    # Assigning a type to the variable 'rows' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'rows', split_call_result_160614)
    
    # Assigning a List to a Name (line 50):
    
    # Assigning a List to a Name (line 50):
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_160615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    
    # Assigning a type to the variable 'newdata' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'newdata', list_160615)
    
    # Assigning a Num to a Name (line 51):
    
    # Assigning a Num to a Name (line 51):
    int_160616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 12), 'int')
    # Assigning a type to the variable 'count' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'count', int_160616)
    
    # Getting the type of 'rows' (line 52)
    rows_160617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'rows')
    # Testing the type of a for loop iterable (line 52)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 4), rows_160617)
    # Getting the type of the for loop variable (line 52)
    for_loop_var_160618 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 4), rows_160617)
    # Assigning a type to the variable 'row' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'row', for_loop_var_160618)
    # SSA begins for a for statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 53):
    
    # Assigning a Call to a Name (line 53):
    
    # Call to split(...): (line 53)
    # Processing the call arguments (line 53)
    str_160621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'str', ',')
    # Processing the call keyword arguments (line 53)
    kwargs_160622 = {}
    # Getting the type of 'row' (line 53)
    row_160619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'row', False)
    # Obtaining the member 'split' of a type (line 53)
    split_160620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 15), row_160619, 'split')
    # Calling split(args, kwargs) (line 53)
    split_call_result_160623 = invoke(stypy.reporting.localization.Localization(__file__, 53, 15), split_160620, *[str_160621], **kwargs_160622)
    
    # Assigning a type to the variable 'trow' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'trow', split_call_result_160623)
    
    # Assigning a List to a Name (line 54):
    
    # Assigning a List to a Name (line 54):
    
    # Obtaining an instance of the builtin type 'list' (line 54)
    list_160624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 54)
    
    # Assigning a type to the variable 'newrow' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'newrow', list_160624)
    
    # Getting the type of 'trow' (line 55)
    trow_160625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'trow')
    # Testing the type of a for loop iterable (line 55)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 8), trow_160625)
    # Getting the type of the for loop variable (line 55)
    for_loop_var_160626 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 8), trow_160625)
    # Assigning a type to the variable 'col' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'col', for_loop_var_160626)
    # SSA begins for a for statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 56):
    
    # Assigning a Call to a Name (line 56):
    
    # Call to split(...): (line 56)
    # Processing the call keyword arguments (line 56)
    kwargs_160629 = {}
    # Getting the type of 'col' (line 56)
    col_160627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'col', False)
    # Obtaining the member 'split' of a type (line 56)
    split_160628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 19), col_160627, 'split')
    # Calling split(args, kwargs) (line 56)
    split_call_result_160630 = invoke(stypy.reporting.localization.Localization(__file__, 56, 19), split_160628, *[], **kwargs_160629)
    
    # Assigning a type to the variable 'temp' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'temp', split_call_result_160630)
    
    # Call to extend(...): (line 57)
    # Processing the call arguments (line 57)
    
    # Call to map(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of '_eval' (line 57)
    _eval_160634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), '_eval', False)
    # Getting the type of 'temp' (line 57)
    temp_160635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 37), 'temp', False)
    # Processing the call keyword arguments (line 57)
    kwargs_160636 = {}
    # Getting the type of 'map' (line 57)
    map_160633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'map', False)
    # Calling map(args, kwargs) (line 57)
    map_call_result_160637 = invoke(stypy.reporting.localization.Localization(__file__, 57, 26), map_160633, *[_eval_160634, temp_160635], **kwargs_160636)
    
    # Processing the call keyword arguments (line 57)
    kwargs_160638 = {}
    # Getting the type of 'newrow' (line 57)
    newrow_160631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'newrow', False)
    # Obtaining the member 'extend' of a type (line 57)
    extend_160632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), newrow_160631, 'extend')
    # Calling extend(args, kwargs) (line 57)
    extend_call_result_160639 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), extend_160632, *[map_call_result_160637], **kwargs_160638)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'count' (line 58)
    count_160640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'count')
    int_160641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 20), 'int')
    # Applying the binary operator '==' (line 58)
    result_eq_160642 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 11), '==', count_160640, int_160641)
    
    # Testing the type of an if condition (line 58)
    if_condition_160643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), result_eq_160642)
    # Assigning a type to the variable 'if_condition_160643' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_160643', if_condition_160643)
    # SSA begins for if statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 59):
    
    # Assigning a Call to a Name (line 59):
    
    # Call to len(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'newrow' (line 59)
    newrow_160645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'newrow', False)
    # Processing the call keyword arguments (line 59)
    kwargs_160646 = {}
    # Getting the type of 'len' (line 59)
    len_160644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'len', False)
    # Calling len(args, kwargs) (line 59)
    len_call_result_160647 = invoke(stypy.reporting.localization.Localization(__file__, 59, 20), len_160644, *[newrow_160645], **kwargs_160646)
    
    # Assigning a type to the variable 'Ncols' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'Ncols', len_call_result_160647)
    # SSA branch for the else part of an if statement (line 58)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'newrow' (line 60)
    newrow_160649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'newrow', False)
    # Processing the call keyword arguments (line 60)
    kwargs_160650 = {}
    # Getting the type of 'len' (line 60)
    len_160648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 13), 'len', False)
    # Calling len(args, kwargs) (line 60)
    len_call_result_160651 = invoke(stypy.reporting.localization.Localization(__file__, 60, 13), len_160648, *[newrow_160649], **kwargs_160650)
    
    # Getting the type of 'Ncols' (line 60)
    Ncols_160652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 28), 'Ncols')
    # Applying the binary operator '!=' (line 60)
    result_ne_160653 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 13), '!=', len_call_result_160651, Ncols_160652)
    
    # Testing the type of an if condition (line 60)
    if_condition_160654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 13), result_ne_160653)
    # Assigning a type to the variable 'if_condition_160654' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 13), 'if_condition_160654', if_condition_160654)
    # SSA begins for if statement (line 60)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 61)
    # Processing the call arguments (line 61)
    str_160656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 29), 'str', 'Rows not the same size.')
    # Processing the call keyword arguments (line 61)
    kwargs_160657 = {}
    # Getting the type of 'ValueError' (line 61)
    ValueError_160655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 61)
    ValueError_call_result_160658 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), ValueError_160655, *[str_160656], **kwargs_160657)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 61, 12), ValueError_call_result_160658, 'raise parameter', BaseException)
    # SSA join for if statement (line 60)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'count' (line 62)
    count_160659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'count')
    int_160660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 17), 'int')
    # Applying the binary operator '+=' (line 62)
    result_iadd_160661 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 8), '+=', count_160659, int_160660)
    # Assigning a type to the variable 'count' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'count', result_iadd_160661)
    
    
    # Call to append(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'newrow' (line 63)
    newrow_160664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 23), 'newrow', False)
    # Processing the call keyword arguments (line 63)
    kwargs_160665 = {}
    # Getting the type of 'newdata' (line 63)
    newdata_160662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'newdata', False)
    # Obtaining the member 'append' of a type (line 63)
    append_160663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), newdata_160662, 'append')
    # Calling append(args, kwargs) (line 63)
    append_call_result_160666 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), append_160663, *[newrow_160664], **kwargs_160665)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'newdata' (line 64)
    newdata_160667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'newdata')
    # Assigning a type to the variable 'stypy_return_type' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type', newdata_160667)
    
    # ################# End of '_convert_from_string(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_convert_from_string' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_160668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_160668)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_convert_from_string'
    return stypy_return_type_160668

# Assigning a type to the variable '_convert_from_string' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), '_convert_from_string', _convert_from_string)

@norecursion
def asmatrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 66)
    None_160669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'None')
    defaults = [None_160669]
    # Create a new context for function 'asmatrix'
    module_type_store = module_type_store.open_function_context('asmatrix', 66, 0, False)
    
    # Passed parameters checking function
    asmatrix.stypy_localization = localization
    asmatrix.stypy_type_of_self = None
    asmatrix.stypy_type_store = module_type_store
    asmatrix.stypy_function_name = 'asmatrix'
    asmatrix.stypy_param_names_list = ['data', 'dtype']
    asmatrix.stypy_varargs_param_name = None
    asmatrix.stypy_kwargs_param_name = None
    asmatrix.stypy_call_defaults = defaults
    asmatrix.stypy_call_varargs = varargs
    asmatrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'asmatrix', ['data', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'asmatrix', localization, ['data', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'asmatrix(...)' code ##################

    str_160670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, (-1)), 'str', '\n    Interpret the input as a matrix.\n\n    Unlike `matrix`, `asmatrix` does not make a copy if the input is already\n    a matrix or an ndarray.  Equivalent to ``matrix(data, copy=False)``.\n\n    Parameters\n    ----------\n    data : array_like\n        Input data.\n    dtype : data-type\n       Data-type of the output matrix.\n\n    Returns\n    -------\n    mat : matrix\n        `data` interpreted as a matrix.\n\n    Examples\n    --------\n    >>> x = np.array([[1, 2], [3, 4]])\n\n    >>> m = np.asmatrix(x)\n\n    >>> x[0,0] = 5\n\n    >>> m\n    matrix([[5, 2],\n            [3, 4]])\n\n    ')
    
    # Call to matrix(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'data' (line 98)
    data_160672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'data', False)
    # Processing the call keyword arguments (line 98)
    # Getting the type of 'dtype' (line 98)
    dtype_160673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), 'dtype', False)
    keyword_160674 = dtype_160673
    # Getting the type of 'False' (line 98)
    False_160675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'False', False)
    keyword_160676 = False_160675
    kwargs_160677 = {'dtype': keyword_160674, 'copy': keyword_160676}
    # Getting the type of 'matrix' (line 98)
    matrix_160671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'matrix', False)
    # Calling matrix(args, kwargs) (line 98)
    matrix_call_result_160678 = invoke(stypy.reporting.localization.Localization(__file__, 98, 11), matrix_160671, *[data_160672], **kwargs_160677)
    
    # Assigning a type to the variable 'stypy_return_type' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type', matrix_call_result_160678)
    
    # ################# End of 'asmatrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'asmatrix' in the type store
    # Getting the type of 'stypy_return_type' (line 66)
    stypy_return_type_160679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_160679)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'asmatrix'
    return stypy_return_type_160679

# Assigning a type to the variable 'asmatrix' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'asmatrix', asmatrix)

@norecursion
def matrix_power(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'matrix_power'
    module_type_store = module_type_store.open_function_context('matrix_power', 100, 0, False)
    
    # Passed parameters checking function
    matrix_power.stypy_localization = localization
    matrix_power.stypy_type_of_self = None
    matrix_power.stypy_type_store = module_type_store
    matrix_power.stypy_function_name = 'matrix_power'
    matrix_power.stypy_param_names_list = ['M', 'n']
    matrix_power.stypy_varargs_param_name = None
    matrix_power.stypy_kwargs_param_name = None
    matrix_power.stypy_call_defaults = defaults
    matrix_power.stypy_call_varargs = varargs
    matrix_power.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'matrix_power', ['M', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'matrix_power', localization, ['M', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'matrix_power(...)' code ##################

    str_160680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, (-1)), 'str', '\n    Raise a square matrix to the (integer) power `n`.\n\n    For positive integers `n`, the power is computed by repeated matrix\n    squarings and matrix multiplications. If ``n == 0``, the identity matrix\n    of the same shape as M is returned. If ``n < 0``, the inverse\n    is computed and then raised to the ``abs(n)``.\n\n    Parameters\n    ----------\n    M : ndarray or matrix object\n        Matrix to be "powered."  Must be square, i.e. ``M.shape == (m, m)``,\n        with `m` a positive integer.\n    n : int\n        The exponent can be any integer or long integer, positive,\n        negative, or zero.\n\n    Returns\n    -------\n    M**n : ndarray or matrix object\n        The return value is the same shape and type as `M`;\n        if the exponent is positive or zero then the type of the\n        elements is the same as those of `M`. If the exponent is\n        negative the elements are floating-point.\n\n    Raises\n    ------\n    LinAlgError\n        If the matrix is not numerically invertible.\n\n    See Also\n    --------\n    matrix\n        Provides an equivalent function as the exponentiation operator\n        (``**``, not ``^``).\n\n    Examples\n    --------\n    >>> from numpy import linalg as LA\n    >>> i = np.array([[0, 1], [-1, 0]]) # matrix equiv. of the imaginary unit\n    >>> LA.matrix_power(i, 3) # should = -i\n    array([[ 0, -1],\n           [ 1,  0]])\n    >>> LA.matrix_power(np.matrix(i), 3) # matrix arg returns matrix\n    matrix([[ 0, -1],\n            [ 1,  0]])\n    >>> LA.matrix_power(i, 0)\n    array([[1, 0],\n           [0, 1]])\n    >>> LA.matrix_power(i, -3) # should = 1/(-i) = i, but w/ f.p. elements\n    array([[ 0.,  1.],\n           [-1.,  0.]])\n\n    Somewhat more sophisticated example\n\n    >>> q = np.zeros((4, 4))\n    >>> q[0:2, 0:2] = -i\n    >>> q[2:4, 2:4] = i\n    >>> q # one of the three quarternion units not equal to 1\n    array([[ 0., -1.,  0.,  0.],\n           [ 1.,  0.,  0.,  0.],\n           [ 0.,  0.,  0.,  1.],\n           [ 0.,  0., -1.,  0.]])\n    >>> LA.matrix_power(q, 2) # = -np.eye(4)\n    array([[-1.,  0.,  0.,  0.],\n           [ 0., -1.,  0.,  0.],\n           [ 0.,  0., -1.,  0.],\n           [ 0.,  0.,  0., -1.]])\n\n    ')
    
    # Assigning a Call to a Name (line 171):
    
    # Assigning a Call to a Name (line 171):
    
    # Call to asanyarray(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'M' (line 171)
    M_160682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 19), 'M', False)
    # Processing the call keyword arguments (line 171)
    kwargs_160683 = {}
    # Getting the type of 'asanyarray' (line 171)
    asanyarray_160681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 171)
    asanyarray_call_result_160684 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), asanyarray_160681, *[M_160682], **kwargs_160683)
    
    # Assigning a type to the variable 'M' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'M', asanyarray_call_result_160684)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'M' (line 172)
    M_160686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'M', False)
    # Obtaining the member 'shape' of a type (line 172)
    shape_160687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 11), M_160686, 'shape')
    # Processing the call keyword arguments (line 172)
    kwargs_160688 = {}
    # Getting the type of 'len' (line 172)
    len_160685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 7), 'len', False)
    # Calling len(args, kwargs) (line 172)
    len_call_result_160689 = invoke(stypy.reporting.localization.Localization(__file__, 172, 7), len_160685, *[shape_160687], **kwargs_160688)
    
    int_160690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 23), 'int')
    # Applying the binary operator '!=' (line 172)
    result_ne_160691 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 7), '!=', len_call_result_160689, int_160690)
    
    
    
    # Obtaining the type of the subscript
    int_160692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 36), 'int')
    # Getting the type of 'M' (line 172)
    M_160693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'M')
    # Obtaining the member 'shape' of a type (line 172)
    shape_160694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 28), M_160693, 'shape')
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___160695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 28), shape_160694, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_160696 = invoke(stypy.reporting.localization.Localization(__file__, 172, 28), getitem___160695, int_160692)
    
    
    # Obtaining the type of the subscript
    int_160697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 50), 'int')
    # Getting the type of 'M' (line 172)
    M_160698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 42), 'M')
    # Obtaining the member 'shape' of a type (line 172)
    shape_160699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 42), M_160698, 'shape')
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___160700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 42), shape_160699, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_160701 = invoke(stypy.reporting.localization.Localization(__file__, 172, 42), getitem___160700, int_160697)
    
    # Applying the binary operator '!=' (line 172)
    result_ne_160702 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 28), '!=', subscript_call_result_160696, subscript_call_result_160701)
    
    # Applying the binary operator 'or' (line 172)
    result_or_keyword_160703 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 7), 'or', result_ne_160691, result_ne_160702)
    
    # Testing the type of an if condition (line 172)
    if_condition_160704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 4), result_or_keyword_160703)
    # Assigning a type to the variable 'if_condition_160704' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'if_condition_160704', if_condition_160704)
    # SSA begins for if statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 173)
    # Processing the call arguments (line 173)
    str_160706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 25), 'str', 'input must be a square array')
    # Processing the call keyword arguments (line 173)
    kwargs_160707 = {}
    # Getting the type of 'ValueError' (line 173)
    ValueError_160705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 173)
    ValueError_call_result_160708 = invoke(stypy.reporting.localization.Localization(__file__, 173, 14), ValueError_160705, *[str_160706], **kwargs_160707)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 173, 8), ValueError_call_result_160708, 'raise parameter', BaseException)
    # SSA join for if statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to issubdtype(...): (line 174)
    # Processing the call arguments (line 174)
    
    # Call to type(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'n' (line 174)
    n_160711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'n', False)
    # Processing the call keyword arguments (line 174)
    kwargs_160712 = {}
    # Getting the type of 'type' (line 174)
    type_160710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 22), 'type', False)
    # Calling type(args, kwargs) (line 174)
    type_call_result_160713 = invoke(stypy.reporting.localization.Localization(__file__, 174, 22), type_160710, *[n_160711], **kwargs_160712)
    
    # Getting the type of 'int' (line 174)
    int_160714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 31), 'int', False)
    # Processing the call keyword arguments (line 174)
    kwargs_160715 = {}
    # Getting the type of 'issubdtype' (line 174)
    issubdtype_160709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'issubdtype', False)
    # Calling issubdtype(args, kwargs) (line 174)
    issubdtype_call_result_160716 = invoke(stypy.reporting.localization.Localization(__file__, 174, 11), issubdtype_160709, *[type_call_result_160713, int_160714], **kwargs_160715)
    
    # Applying the 'not' unary operator (line 174)
    result_not__160717 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 7), 'not', issubdtype_call_result_160716)
    
    # Testing the type of an if condition (line 174)
    if_condition_160718 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 4), result_not__160717)
    # Assigning a type to the variable 'if_condition_160718' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'if_condition_160718', if_condition_160718)
    # SSA begins for if statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 175)
    # Processing the call arguments (line 175)
    str_160720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 24), 'str', 'exponent must be an integer')
    # Processing the call keyword arguments (line 175)
    kwargs_160721 = {}
    # Getting the type of 'TypeError' (line 175)
    TypeError_160719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 175)
    TypeError_call_result_160722 = invoke(stypy.reporting.localization.Localization(__file__, 175, 14), TypeError_160719, *[str_160720], **kwargs_160721)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 175, 8), TypeError_call_result_160722, 'raise parameter', BaseException)
    # SSA join for if statement (line 174)
    module_type_store = module_type_store.join_ssa_context()
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 177, 4))
    
    # 'from numpy.linalg import inv' statement (line 177)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/matrixlib/')
    import_160723 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 177, 4), 'numpy.linalg')

    if (type(import_160723) is not StypyTypeError):

        if (import_160723 != 'pyd_module'):
            __import__(import_160723)
            sys_modules_160724 = sys.modules[import_160723]
            import_from_module(stypy.reporting.localization.Localization(__file__, 177, 4), 'numpy.linalg', sys_modules_160724.module_type_store, module_type_store, ['inv'])
            nest_module(stypy.reporting.localization.Localization(__file__, 177, 4), __file__, sys_modules_160724, sys_modules_160724.module_type_store, module_type_store)
        else:
            from numpy.linalg import inv

            import_from_module(stypy.reporting.localization.Localization(__file__, 177, 4), 'numpy.linalg', None, module_type_store, ['inv'], [inv])

    else:
        # Assigning a type to the variable 'numpy.linalg' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'numpy.linalg', import_160723)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/matrixlib/')
    
    
    
    # Getting the type of 'n' (line 179)
    n_160725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 7), 'n')
    int_160726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 10), 'int')
    # Applying the binary operator '==' (line 179)
    result_eq_160727 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 7), '==', n_160725, int_160726)
    
    # Testing the type of an if condition (line 179)
    if_condition_160728 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 4), result_eq_160727)
    # Assigning a type to the variable 'if_condition_160728' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'if_condition_160728', if_condition_160728)
    # SSA begins for if statement (line 179)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to copy(...): (line 180)
    # Processing the call keyword arguments (line 180)
    kwargs_160731 = {}
    # Getting the type of 'M' (line 180)
    M_160729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'M', False)
    # Obtaining the member 'copy' of a type (line 180)
    copy_160730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 12), M_160729, 'copy')
    # Calling copy(args, kwargs) (line 180)
    copy_call_result_160732 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), copy_160730, *[], **kwargs_160731)
    
    # Assigning a type to the variable 'M' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'M', copy_call_result_160732)
    
    # Assigning a Call to a Subscript (line 181):
    
    # Assigning a Call to a Subscript (line 181):
    
    # Call to identity(...): (line 181)
    # Processing the call arguments (line 181)
    
    # Obtaining the type of the subscript
    int_160734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 32), 'int')
    # Getting the type of 'M' (line 181)
    M_160735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'M', False)
    # Obtaining the member 'shape' of a type (line 181)
    shape_160736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 24), M_160735, 'shape')
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___160737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 24), shape_160736, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_160738 = invoke(stypy.reporting.localization.Localization(__file__, 181, 24), getitem___160737, int_160734)
    
    # Processing the call keyword arguments (line 181)
    kwargs_160739 = {}
    # Getting the type of 'identity' (line 181)
    identity_160733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'identity', False)
    # Calling identity(args, kwargs) (line 181)
    identity_call_result_160740 = invoke(stypy.reporting.localization.Localization(__file__, 181, 15), identity_160733, *[subscript_call_result_160738], **kwargs_160739)
    
    # Getting the type of 'M' (line 181)
    M_160741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'M')
    slice_160742 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 181, 8), None, None, None)
    # Storing an element on a container (line 181)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 8), M_160741, (slice_160742, identity_call_result_160740))
    # Getting the type of 'M' (line 182)
    M_160743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'M')
    # Assigning a type to the variable 'stypy_return_type' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'stypy_return_type', M_160743)
    # SSA branch for the else part of an if statement (line 179)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'n' (line 183)
    n_160744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 9), 'n')
    int_160745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 11), 'int')
    # Applying the binary operator '<' (line 183)
    result_lt_160746 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 9), '<', n_160744, int_160745)
    
    # Testing the type of an if condition (line 183)
    if_condition_160747 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 9), result_lt_160746)
    # Assigning a type to the variable 'if_condition_160747' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 9), 'if_condition_160747', if_condition_160747)
    # SSA begins for if statement (line 183)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 184):
    
    # Assigning a Call to a Name (line 184):
    
    # Call to inv(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'M' (line 184)
    M_160749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'M', False)
    # Processing the call keyword arguments (line 184)
    kwargs_160750 = {}
    # Getting the type of 'inv' (line 184)
    inv_160748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'inv', False)
    # Calling inv(args, kwargs) (line 184)
    inv_call_result_160751 = invoke(stypy.reporting.localization.Localization(__file__, 184, 12), inv_160748, *[M_160749], **kwargs_160750)
    
    # Assigning a type to the variable 'M' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'M', inv_call_result_160751)
    
    # Getting the type of 'n' (line 185)
    n_160752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'n')
    int_160753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 13), 'int')
    # Applying the binary operator '*=' (line 185)
    result_imul_160754 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 8), '*=', n_160752, int_160753)
    # Assigning a type to the variable 'n' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'n', result_imul_160754)
    
    # SSA join for if statement (line 183)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 179)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 187):
    
    # Assigning a Name to a Name (line 187):
    # Getting the type of 'M' (line 187)
    M_160755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 13), 'M')
    # Assigning a type to the variable 'result' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'result', M_160755)
    
    
    # Getting the type of 'n' (line 188)
    n_160756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 7), 'n')
    int_160757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 12), 'int')
    # Applying the binary operator '<=' (line 188)
    result_le_160758 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 7), '<=', n_160756, int_160757)
    
    # Testing the type of an if condition (line 188)
    if_condition_160759 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 4), result_le_160758)
    # Assigning a type to the variable 'if_condition_160759' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'if_condition_160759', if_condition_160759)
    # SSA begins for if statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to range(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'n' (line 189)
    n_160761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 23), 'n', False)
    int_160762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 25), 'int')
    # Applying the binary operator '-' (line 189)
    result_sub_160763 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 23), '-', n_160761, int_160762)
    
    # Processing the call keyword arguments (line 189)
    kwargs_160764 = {}
    # Getting the type of 'range' (line 189)
    range_160760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 17), 'range', False)
    # Calling range(args, kwargs) (line 189)
    range_call_result_160765 = invoke(stypy.reporting.localization.Localization(__file__, 189, 17), range_160760, *[result_sub_160763], **kwargs_160764)
    
    # Testing the type of a for loop iterable (line 189)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 189, 8), range_call_result_160765)
    # Getting the type of the for loop variable (line 189)
    for_loop_var_160766 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 189, 8), range_call_result_160765)
    # Assigning a type to the variable '_' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), '_', for_loop_var_160766)
    # SSA begins for a for statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 190):
    
    # Assigning a Call to a Name (line 190):
    
    # Call to dot(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'result' (line 190)
    result_160769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), 'result', False)
    # Getting the type of 'M' (line 190)
    M_160770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 33), 'M', False)
    # Processing the call keyword arguments (line 190)
    kwargs_160771 = {}
    # Getting the type of 'N' (line 190)
    N_160767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'N', False)
    # Obtaining the member 'dot' of a type (line 190)
    dot_160768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 19), N_160767, 'dot')
    # Calling dot(args, kwargs) (line 190)
    dot_call_result_160772 = invoke(stypy.reporting.localization.Localization(__file__, 190, 19), dot_160768, *[result_160769, M_160770], **kwargs_160771)
    
    # Assigning a type to the variable 'result' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'result', dot_call_result_160772)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 191)
    result_160773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'stypy_return_type', result_160773)
    # SSA join for if statement (line 188)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 195):
    
    # Assigning a Call to a Name (line 195):
    
    # Call to binary_repr(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'n' (line 195)
    n_160775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 23), 'n', False)
    # Processing the call keyword arguments (line 195)
    kwargs_160776 = {}
    # Getting the type of 'binary_repr' (line 195)
    binary_repr_160774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 11), 'binary_repr', False)
    # Calling binary_repr(args, kwargs) (line 195)
    binary_repr_call_result_160777 = invoke(stypy.reporting.localization.Localization(__file__, 195, 11), binary_repr_160774, *[n_160775], **kwargs_160776)
    
    # Assigning a type to the variable 'beta' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'beta', binary_repr_call_result_160777)
    
    # Assigning a Tuple to a Tuple (line 196):
    
    # Assigning a Name to a Name (line 196):
    # Getting the type of 'M' (line 196)
    M_160778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 14), 'M')
    # Assigning a type to the variable 'tuple_assignment_160493' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'tuple_assignment_160493', M_160778)
    
    # Assigning a Num to a Name (line 196):
    int_160779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 17), 'int')
    # Assigning a type to the variable 'tuple_assignment_160494' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'tuple_assignment_160494', int_160779)
    
    # Assigning a Call to a Name (line 196):
    
    # Call to len(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'beta' (line 196)
    beta_160781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'beta', False)
    # Processing the call keyword arguments (line 196)
    kwargs_160782 = {}
    # Getting the type of 'len' (line 196)
    len_160780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'len', False)
    # Calling len(args, kwargs) (line 196)
    len_call_result_160783 = invoke(stypy.reporting.localization.Localization(__file__, 196, 20), len_160780, *[beta_160781], **kwargs_160782)
    
    # Assigning a type to the variable 'tuple_assignment_160495' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'tuple_assignment_160495', len_call_result_160783)
    
    # Assigning a Name to a Name (line 196):
    # Getting the type of 'tuple_assignment_160493' (line 196)
    tuple_assignment_160493_160784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'tuple_assignment_160493')
    # Assigning a type to the variable 'Z' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'Z', tuple_assignment_160493_160784)
    
    # Assigning a Name to a Name (line 196):
    # Getting the type of 'tuple_assignment_160494' (line 196)
    tuple_assignment_160494_160785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'tuple_assignment_160494')
    # Assigning a type to the variable 'q' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 7), 'q', tuple_assignment_160494_160785)
    
    # Assigning a Name to a Name (line 196):
    # Getting the type of 'tuple_assignment_160495' (line 196)
    tuple_assignment_160495_160786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'tuple_assignment_160495')
    # Assigning a type to the variable 't' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 10), 't', tuple_assignment_160495_160786)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 197)
    t_160787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 't')
    # Getting the type of 'q' (line 197)
    q_160788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 17), 'q')
    # Applying the binary operator '-' (line 197)
    result_sub_160789 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 15), '-', t_160787, q_160788)
    
    int_160790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 19), 'int')
    # Applying the binary operator '-' (line 197)
    result_sub_160791 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 18), '-', result_sub_160789, int_160790)
    
    # Getting the type of 'beta' (line 197)
    beta_160792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 10), 'beta')
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___160793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 10), beta_160792, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_160794 = invoke(stypy.reporting.localization.Localization(__file__, 197, 10), getitem___160793, result_sub_160791)
    
    str_160795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 25), 'str', '0')
    # Applying the binary operator '==' (line 197)
    result_eq_160796 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 10), '==', subscript_call_result_160794, str_160795)
    
    # Testing the type of an if condition (line 197)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 4), result_eq_160796)
    # SSA begins for while statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to dot(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'Z' (line 198)
    Z_160799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 18), 'Z', False)
    # Getting the type of 'Z' (line 198)
    Z_160800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 21), 'Z', False)
    # Processing the call keyword arguments (line 198)
    kwargs_160801 = {}
    # Getting the type of 'N' (line 198)
    N_160797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'N', False)
    # Obtaining the member 'dot' of a type (line 198)
    dot_160798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 12), N_160797, 'dot')
    # Calling dot(args, kwargs) (line 198)
    dot_call_result_160802 = invoke(stypy.reporting.localization.Localization(__file__, 198, 12), dot_160798, *[Z_160799, Z_160800], **kwargs_160801)
    
    # Assigning a type to the variable 'Z' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'Z', dot_call_result_160802)
    
    # Getting the type of 'q' (line 199)
    q_160803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'q')
    int_160804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 13), 'int')
    # Applying the binary operator '+=' (line 199)
    result_iadd_160805 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 8), '+=', q_160803, int_160804)
    # Assigning a type to the variable 'q' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'q', result_iadd_160805)
    
    # SSA join for while statement (line 197)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 200):
    
    # Assigning a Name to a Name (line 200):
    # Getting the type of 'Z' (line 200)
    Z_160806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'Z')
    # Assigning a type to the variable 'result' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'result', Z_160806)
    
    
    # Call to range(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'q' (line 201)
    q_160808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'q', False)
    int_160809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 21), 'int')
    # Applying the binary operator '+' (line 201)
    result_add_160810 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 19), '+', q_160808, int_160809)
    
    # Getting the type of 't' (line 201)
    t_160811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 24), 't', False)
    # Processing the call keyword arguments (line 201)
    kwargs_160812 = {}
    # Getting the type of 'range' (line 201)
    range_160807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 13), 'range', False)
    # Calling range(args, kwargs) (line 201)
    range_call_result_160813 = invoke(stypy.reporting.localization.Localization(__file__, 201, 13), range_160807, *[result_add_160810, t_160811], **kwargs_160812)
    
    # Testing the type of a for loop iterable (line 201)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 201, 4), range_call_result_160813)
    # Getting the type of the for loop variable (line 201)
    for_loop_var_160814 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 201, 4), range_call_result_160813)
    # Assigning a type to the variable 'k' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'k', for_loop_var_160814)
    # SSA begins for a for statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Call to a Name (line 202):
    
    # Call to dot(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'Z' (line 202)
    Z_160817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'Z', False)
    # Getting the type of 'Z' (line 202)
    Z_160818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 21), 'Z', False)
    # Processing the call keyword arguments (line 202)
    kwargs_160819 = {}
    # Getting the type of 'N' (line 202)
    N_160815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'N', False)
    # Obtaining the member 'dot' of a type (line 202)
    dot_160816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 12), N_160815, 'dot')
    # Calling dot(args, kwargs) (line 202)
    dot_call_result_160820 = invoke(stypy.reporting.localization.Localization(__file__, 202, 12), dot_160816, *[Z_160817, Z_160818], **kwargs_160819)
    
    # Assigning a type to the variable 'Z' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'Z', dot_call_result_160820)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 203)
    t_160821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 't')
    # Getting the type of 'k' (line 203)
    k_160822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 18), 'k')
    # Applying the binary operator '-' (line 203)
    result_sub_160823 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 16), '-', t_160821, k_160822)
    
    int_160824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 20), 'int')
    # Applying the binary operator '-' (line 203)
    result_sub_160825 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 19), '-', result_sub_160823, int_160824)
    
    # Getting the type of 'beta' (line 203)
    beta_160826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'beta')
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___160827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 11), beta_160826, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_160828 = invoke(stypy.reporting.localization.Localization(__file__, 203, 11), getitem___160827, result_sub_160825)
    
    str_160829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 26), 'str', '1')
    # Applying the binary operator '==' (line 203)
    result_eq_160830 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 11), '==', subscript_call_result_160828, str_160829)
    
    # Testing the type of an if condition (line 203)
    if_condition_160831 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 8), result_eq_160830)
    # Assigning a type to the variable 'if_condition_160831' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'if_condition_160831', if_condition_160831)
    # SSA begins for if statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 204):
    
    # Assigning a Call to a Name (line 204):
    
    # Call to dot(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'result' (line 204)
    result_160834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 27), 'result', False)
    # Getting the type of 'Z' (line 204)
    Z_160835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 35), 'Z', False)
    # Processing the call keyword arguments (line 204)
    kwargs_160836 = {}
    # Getting the type of 'N' (line 204)
    N_160832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'N', False)
    # Obtaining the member 'dot' of a type (line 204)
    dot_160833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 21), N_160832, 'dot')
    # Calling dot(args, kwargs) (line 204)
    dot_call_result_160837 = invoke(stypy.reporting.localization.Localization(__file__, 204, 21), dot_160833, *[result_160834, Z_160835], **kwargs_160836)
    
    # Assigning a type to the variable 'result' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'result', dot_call_result_160837)
    # SSA join for if statement (line 203)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 205)
    result_160838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'stypy_return_type', result_160838)
    
    # ################# End of 'matrix_power(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'matrix_power' in the type store
    # Getting the type of 'stypy_return_type' (line 100)
    stypy_return_type_160839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_160839)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'matrix_power'
    return stypy_return_type_160839

# Assigning a type to the variable 'matrix_power' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'matrix_power', matrix_power)
# Declaration of the 'matrix' class
# Getting the type of 'N' (line 208)
N_160840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 13), 'N')
# Obtaining the member 'ndarray' of a type (line 208)
ndarray_160841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 13), N_160840, 'ndarray')

class matrix(ndarray_160841, ):
    str_160842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, (-1)), 'str', "\n    matrix(data, dtype=None, copy=True)\n\n    Returns a matrix from an array-like object, or from a string of data.\n    A matrix is a specialized 2-D array that retains its 2-D nature\n    through operations.  It has certain special operators, such as ``*``\n    (matrix multiplication) and ``**`` (matrix power).\n\n    Parameters\n    ----------\n    data : array_like or string\n       If `data` is a string, it is interpreted as a matrix with commas\n       or spaces separating columns, and semicolons separating rows.\n    dtype : data-type\n       Data-type of the output matrix.\n    copy : bool\n       If `data` is already an `ndarray`, then this flag determines\n       whether the data is copied (the default), or whether a view is\n       constructed.\n\n    See Also\n    --------\n    array\n\n    Examples\n    --------\n    >>> a = np.matrix('1 2; 3 4')\n    >>> print(a)\n    [[1 2]\n     [3 4]]\n\n    >>> np.matrix([[1, 2], [3, 4]])\n    matrix([[1, 2],\n            [3, 4]])\n\n    ")
    
    # Assigning a Num to a Name (line 245):

    @norecursion
    def __new__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 246)
        None_160843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 37), 'None')
        # Getting the type of 'True' (line 246)
        True_160844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 48), 'True')
        defaults = [None_160843, True_160844]
        # Create a new context for function '__new__'
        module_type_store = module_type_store.open_function_context('__new__', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.__new__.__dict__.__setitem__('stypy_localization', localization)
        matrix.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.__new__.__dict__.__setitem__('stypy_function_name', 'matrix.__new__')
        matrix.__new__.__dict__.__setitem__('stypy_param_names_list', ['data', 'dtype', 'copy'])
        matrix.__new__.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.__new__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.__new__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.__new__', ['data', 'dtype', 'copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__new__', localization, ['data', 'dtype', 'copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__new__(...)' code ##################

        
        
        # Call to isinstance(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'data' (line 247)
        data_160846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 22), 'data', False)
        # Getting the type of 'matrix' (line 247)
        matrix_160847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 28), 'matrix', False)
        # Processing the call keyword arguments (line 247)
        kwargs_160848 = {}
        # Getting the type of 'isinstance' (line 247)
        isinstance_160845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 247)
        isinstance_call_result_160849 = invoke(stypy.reporting.localization.Localization(__file__, 247, 11), isinstance_160845, *[data_160846, matrix_160847], **kwargs_160848)
        
        # Testing the type of an if condition (line 247)
        if_condition_160850 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 8), isinstance_call_result_160849)
        # Assigning a type to the variable 'if_condition_160850' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'if_condition_160850', if_condition_160850)
        # SSA begins for if statement (line 247)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 248):
        
        # Assigning a Attribute to a Name (line 248):
        # Getting the type of 'data' (line 248)
        data_160851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 21), 'data')
        # Obtaining the member 'dtype' of a type (line 248)
        dtype_160852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 21), data_160851, 'dtype')
        # Assigning a type to the variable 'dtype2' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'dtype2', dtype_160852)
        
        # Type idiom detected: calculating its left and rigth part (line 249)
        # Getting the type of 'dtype' (line 249)
        dtype_160853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 16), 'dtype')
        # Getting the type of 'None' (line 249)
        None_160854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 25), 'None')
        
        (may_be_160855, more_types_in_union_160856) = may_be_none(dtype_160853, None_160854)

        if may_be_160855:

            if more_types_in_union_160856:
                # Runtime conditional SSA (line 249)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 250):
            
            # Assigning a Name to a Name (line 250):
            # Getting the type of 'dtype2' (line 250)
            dtype2_160857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 24), 'dtype2')
            # Assigning a type to the variable 'dtype' (line 250)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'dtype', dtype2_160857)

            if more_types_in_union_160856:
                # SSA join for if statement (line 249)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'dtype2' (line 251)
        dtype2_160858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'dtype2')
        # Getting the type of 'dtype' (line 251)
        dtype_160859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 26), 'dtype')
        # Applying the binary operator '==' (line 251)
        result_eq_160860 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 16), '==', dtype2_160858, dtype_160859)
        
        
        # Getting the type of 'copy' (line 251)
        copy_160861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 42), 'copy')
        # Applying the 'not' unary operator (line 251)
        result_not__160862 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 38), 'not', copy_160861)
        
        # Applying the binary operator 'and' (line 251)
        result_and_keyword_160863 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 15), 'and', result_eq_160860, result_not__160862)
        
        # Testing the type of an if condition (line 251)
        if_condition_160864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 12), result_and_keyword_160863)
        # Assigning a type to the variable 'if_condition_160864' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'if_condition_160864', if_condition_160864)
        # SSA begins for if statement (line 251)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'data' (line 252)
        data_160865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 23), 'data')
        # Assigning a type to the variable 'stypy_return_type' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'stypy_return_type', data_160865)
        # SSA join for if statement (line 251)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to astype(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'dtype' (line 253)
        dtype_160868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 31), 'dtype', False)
        # Processing the call keyword arguments (line 253)
        kwargs_160869 = {}
        # Getting the type of 'data' (line 253)
        data_160866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 19), 'data', False)
        # Obtaining the member 'astype' of a type (line 253)
        astype_160867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 19), data_160866, 'astype')
        # Calling astype(args, kwargs) (line 253)
        astype_call_result_160870 = invoke(stypy.reporting.localization.Localization(__file__, 253, 19), astype_160867, *[dtype_160868], **kwargs_160869)
        
        # Assigning a type to the variable 'stypy_return_type' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'stypy_return_type', astype_call_result_160870)
        # SSA join for if statement (line 247)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isinstance(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'data' (line 255)
        data_160872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 22), 'data', False)
        # Getting the type of 'N' (line 255)
        N_160873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 28), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 255)
        ndarray_160874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 28), N_160873, 'ndarray')
        # Processing the call keyword arguments (line 255)
        kwargs_160875 = {}
        # Getting the type of 'isinstance' (line 255)
        isinstance_160871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 255)
        isinstance_call_result_160876 = invoke(stypy.reporting.localization.Localization(__file__, 255, 11), isinstance_160871, *[data_160872, ndarray_160874], **kwargs_160875)
        
        # Testing the type of an if condition (line 255)
        if_condition_160877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 8), isinstance_call_result_160876)
        # Assigning a type to the variable 'if_condition_160877' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'if_condition_160877', if_condition_160877)
        # SSA begins for if statement (line 255)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 256)
        # Getting the type of 'dtype' (line 256)
        dtype_160878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 15), 'dtype')
        # Getting the type of 'None' (line 256)
        None_160879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 24), 'None')
        
        (may_be_160880, more_types_in_union_160881) = may_be_none(dtype_160878, None_160879)

        if may_be_160880:

            if more_types_in_union_160881:
                # Runtime conditional SSA (line 256)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 257):
            
            # Assigning a Attribute to a Name (line 257):
            # Getting the type of 'data' (line 257)
            data_160882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 25), 'data')
            # Obtaining the member 'dtype' of a type (line 257)
            dtype_160883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 25), data_160882, 'dtype')
            # Assigning a type to the variable 'intype' (line 257)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'intype', dtype_160883)

            if more_types_in_union_160881:
                # Runtime conditional SSA for else branch (line 256)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_160880) or more_types_in_union_160881):
            
            # Assigning a Call to a Name (line 259):
            
            # Assigning a Call to a Name (line 259):
            
            # Call to dtype(...): (line 259)
            # Processing the call arguments (line 259)
            # Getting the type of 'dtype' (line 259)
            dtype_160886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 33), 'dtype', False)
            # Processing the call keyword arguments (line 259)
            kwargs_160887 = {}
            # Getting the type of 'N' (line 259)
            N_160884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 25), 'N', False)
            # Obtaining the member 'dtype' of a type (line 259)
            dtype_160885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 25), N_160884, 'dtype')
            # Calling dtype(args, kwargs) (line 259)
            dtype_call_result_160888 = invoke(stypy.reporting.localization.Localization(__file__, 259, 25), dtype_160885, *[dtype_160886], **kwargs_160887)
            
            # Assigning a type to the variable 'intype' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'intype', dtype_call_result_160888)

            if (may_be_160880 and more_types_in_union_160881):
                # SSA join for if statement (line 256)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 260):
        
        # Assigning a Call to a Name (line 260):
        
        # Call to view(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'subtype' (line 260)
        subtype_160891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 28), 'subtype', False)
        # Processing the call keyword arguments (line 260)
        kwargs_160892 = {}
        # Getting the type of 'data' (line 260)
        data_160889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 18), 'data', False)
        # Obtaining the member 'view' of a type (line 260)
        view_160890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 18), data_160889, 'view')
        # Calling view(args, kwargs) (line 260)
        view_call_result_160893 = invoke(stypy.reporting.localization.Localization(__file__, 260, 18), view_160890, *[subtype_160891], **kwargs_160892)
        
        # Assigning a type to the variable 'new' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'new', view_call_result_160893)
        
        
        # Getting the type of 'intype' (line 261)
        intype_160894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'intype')
        # Getting the type of 'data' (line 261)
        data_160895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 25), 'data')
        # Obtaining the member 'dtype' of a type (line 261)
        dtype_160896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 25), data_160895, 'dtype')
        # Applying the binary operator '!=' (line 261)
        result_ne_160897 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 15), '!=', intype_160894, dtype_160896)
        
        # Testing the type of an if condition (line 261)
        if_condition_160898 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 12), result_ne_160897)
        # Assigning a type to the variable 'if_condition_160898' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'if_condition_160898', if_condition_160898)
        # SSA begins for if statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to astype(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'intype' (line 262)
        intype_160901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 34), 'intype', False)
        # Processing the call keyword arguments (line 262)
        kwargs_160902 = {}
        # Getting the type of 'new' (line 262)
        new_160899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 23), 'new', False)
        # Obtaining the member 'astype' of a type (line 262)
        astype_160900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 23), new_160899, 'astype')
        # Calling astype(args, kwargs) (line 262)
        astype_call_result_160903 = invoke(stypy.reporting.localization.Localization(__file__, 262, 23), astype_160900, *[intype_160901], **kwargs_160902)
        
        # Assigning a type to the variable 'stypy_return_type' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'stypy_return_type', astype_call_result_160903)
        # SSA join for if statement (line 261)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'copy' (line 263)
        copy_160904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'copy')
        # Testing the type of an if condition (line 263)
        if_condition_160905 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 12), copy_160904)
        # Assigning a type to the variable 'if_condition_160905' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'if_condition_160905', if_condition_160905)
        # SSA begins for if statement (line 263)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_160908 = {}
        # Getting the type of 'new' (line 263)
        new_160906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 28), 'new', False)
        # Obtaining the member 'copy' of a type (line 263)
        copy_160907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 28), new_160906, 'copy')
        # Calling copy(args, kwargs) (line 263)
        copy_call_result_160909 = invoke(stypy.reporting.localization.Localization(__file__, 263, 28), copy_160907, *[], **kwargs_160908)
        
        # Assigning a type to the variable 'stypy_return_type' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 21), 'stypy_return_type', copy_call_result_160909)
        # SSA branch for the else part of an if statement (line 263)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'new' (line 264)
        new_160910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 25), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 18), 'stypy_return_type', new_160910)
        # SSA join for if statement (line 263)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 255)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 266)
        # Getting the type of 'str' (line 266)
        str_160911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 28), 'str')
        # Getting the type of 'data' (line 266)
        data_160912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 22), 'data')
        
        (may_be_160913, more_types_in_union_160914) = may_be_subtype(str_160911, data_160912)

        if may_be_160913:

            if more_types_in_union_160914:
                # Runtime conditional SSA (line 266)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'data' (line 266)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'data', remove_not_subtype_from_union(data_160912, str))
            
            # Assigning a Call to a Name (line 267):
            
            # Assigning a Call to a Name (line 267):
            
            # Call to _convert_from_string(...): (line 267)
            # Processing the call arguments (line 267)
            # Getting the type of 'data' (line 267)
            data_160916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 40), 'data', False)
            # Processing the call keyword arguments (line 267)
            kwargs_160917 = {}
            # Getting the type of '_convert_from_string' (line 267)
            _convert_from_string_160915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 19), '_convert_from_string', False)
            # Calling _convert_from_string(args, kwargs) (line 267)
            _convert_from_string_call_result_160918 = invoke(stypy.reporting.localization.Localization(__file__, 267, 19), _convert_from_string_160915, *[data_160916], **kwargs_160917)
            
            # Assigning a type to the variable 'data' (line 267)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'data', _convert_from_string_call_result_160918)

            if more_types_in_union_160914:
                # SSA join for if statement (line 266)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 270):
        
        # Assigning a Call to a Name (line 270):
        
        # Call to array(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'data' (line 270)
        data_160921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 22), 'data', False)
        # Processing the call keyword arguments (line 270)
        # Getting the type of 'dtype' (line 270)
        dtype_160922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 34), 'dtype', False)
        keyword_160923 = dtype_160922
        # Getting the type of 'copy' (line 270)
        copy_160924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 46), 'copy', False)
        keyword_160925 = copy_160924
        kwargs_160926 = {'dtype': keyword_160923, 'copy': keyword_160925}
        # Getting the type of 'N' (line 270)
        N_160919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 14), 'N', False)
        # Obtaining the member 'array' of a type (line 270)
        array_160920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 14), N_160919, 'array')
        # Calling array(args, kwargs) (line 270)
        array_call_result_160927 = invoke(stypy.reporting.localization.Localization(__file__, 270, 14), array_160920, *[data_160921], **kwargs_160926)
        
        # Assigning a type to the variable 'arr' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'arr', array_call_result_160927)
        
        # Assigning a Attribute to a Name (line 271):
        
        # Assigning a Attribute to a Name (line 271):
        # Getting the type of 'arr' (line 271)
        arr_160928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'arr')
        # Obtaining the member 'ndim' of a type (line 271)
        ndim_160929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 15), arr_160928, 'ndim')
        # Assigning a type to the variable 'ndim' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'ndim', ndim_160929)
        
        # Assigning a Attribute to a Name (line 272):
        
        # Assigning a Attribute to a Name (line 272):
        # Getting the type of 'arr' (line 272)
        arr_160930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'arr')
        # Obtaining the member 'shape' of a type (line 272)
        shape_160931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 16), arr_160930, 'shape')
        # Assigning a type to the variable 'shape' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'shape', shape_160931)
        
        
        # Getting the type of 'ndim' (line 273)
        ndim_160932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'ndim')
        int_160933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 19), 'int')
        # Applying the binary operator '>' (line 273)
        result_gt_160934 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 12), '>', ndim_160932, int_160933)
        
        # Testing the type of an if condition (line 273)
        if_condition_160935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_gt_160934)
        # Assigning a type to the variable 'if_condition_160935' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_160935', if_condition_160935)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 274)
        # Processing the call arguments (line 274)
        str_160937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 29), 'str', 'matrix must be 2-dimensional')
        # Processing the call keyword arguments (line 274)
        kwargs_160938 = {}
        # Getting the type of 'ValueError' (line 274)
        ValueError_160936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 274)
        ValueError_call_result_160939 = invoke(stypy.reporting.localization.Localization(__file__, 274, 18), ValueError_160936, *[str_160937], **kwargs_160938)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 274, 12), ValueError_call_result_160939, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 273)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ndim' (line 275)
        ndim_160940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'ndim')
        int_160941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 21), 'int')
        # Applying the binary operator '==' (line 275)
        result_eq_160942 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 13), '==', ndim_160940, int_160941)
        
        # Testing the type of an if condition (line 275)
        if_condition_160943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 13), result_eq_160942)
        # Assigning a type to the variable 'if_condition_160943' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'if_condition_160943', if_condition_160943)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 276):
        
        # Assigning a Tuple to a Name (line 276):
        
        # Obtaining an instance of the builtin type 'tuple' (line 276)
        tuple_160944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 276)
        # Adding element type (line 276)
        int_160945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 21), tuple_160944, int_160945)
        # Adding element type (line 276)
        int_160946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 21), tuple_160944, int_160946)
        
        # Assigning a type to the variable 'shape' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'shape', tuple_160944)
        # SSA branch for the else part of an if statement (line 275)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ndim' (line 277)
        ndim_160947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 13), 'ndim')
        int_160948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 21), 'int')
        # Applying the binary operator '==' (line 277)
        result_eq_160949 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 13), '==', ndim_160947, int_160948)
        
        # Testing the type of an if condition (line 277)
        if_condition_160950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 13), result_eq_160949)
        # Assigning a type to the variable 'if_condition_160950' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 13), 'if_condition_160950', if_condition_160950)
        # SSA begins for if statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 278):
        
        # Assigning a Tuple to a Name (line 278):
        
        # Obtaining an instance of the builtin type 'tuple' (line 278)
        tuple_160951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 278)
        # Adding element type (line 278)
        int_160952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 21), tuple_160951, int_160952)
        # Adding element type (line 278)
        
        # Obtaining the type of the subscript
        int_160953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 30), 'int')
        # Getting the type of 'shape' (line 278)
        shape_160954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 24), 'shape')
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___160955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 24), shape_160954, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_160956 = invoke(stypy.reporting.localization.Localization(__file__, 278, 24), getitem___160955, int_160953)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 21), tuple_160951, subscript_call_result_160956)
        
        # Assigning a type to the variable 'shape' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'shape', tuple_160951)
        # SSA join for if statement (line 277)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Str to a Name (line 280):
        
        # Assigning a Str to a Name (line 280):
        str_160957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 16), 'str', 'C')
        # Assigning a type to the variable 'order' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'order', str_160957)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ndim' (line 281)
        ndim_160958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'ndim')
        int_160959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 20), 'int')
        # Applying the binary operator '==' (line 281)
        result_eq_160960 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 12), '==', ndim_160958, int_160959)
        
        # Getting the type of 'arr' (line 281)
        arr_160961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 27), 'arr')
        # Obtaining the member 'flags' of a type (line 281)
        flags_160962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 27), arr_160961, 'flags')
        # Obtaining the member 'fortran' of a type (line 281)
        fortran_160963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 27), flags_160962, 'fortran')
        # Applying the binary operator 'and' (line 281)
        result_and_keyword_160964 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 11), 'and', result_eq_160960, fortran_160963)
        
        # Testing the type of an if condition (line 281)
        if_condition_160965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 8), result_and_keyword_160964)
        # Assigning a type to the variable 'if_condition_160965' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'if_condition_160965', if_condition_160965)
        # SSA begins for if statement (line 281)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 282):
        
        # Assigning a Str to a Name (line 282):
        str_160966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 20), 'str', 'F')
        # Assigning a type to the variable 'order' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'order', str_160966)
        # SSA join for if statement (line 281)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'order' (line 284)
        order_160967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'order')
        # Getting the type of 'arr' (line 284)
        arr_160968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 25), 'arr')
        # Obtaining the member 'flags' of a type (line 284)
        flags_160969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 25), arr_160968, 'flags')
        # Obtaining the member 'contiguous' of a type (line 284)
        contiguous_160970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 25), flags_160969, 'contiguous')
        # Applying the binary operator 'or' (line 284)
        result_or_keyword_160971 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 16), 'or', order_160967, contiguous_160970)
        
        # Applying the 'not' unary operator (line 284)
        result_not__160972 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), 'not', result_or_keyword_160971)
        
        # Testing the type of an if condition (line 284)
        if_condition_160973 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 8), result_not__160972)
        # Assigning a type to the variable 'if_condition_160973' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'if_condition_160973', if_condition_160973)
        # SSA begins for if statement (line 284)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 285):
        
        # Assigning a Call to a Name (line 285):
        
        # Call to copy(...): (line 285)
        # Processing the call keyword arguments (line 285)
        kwargs_160976 = {}
        # Getting the type of 'arr' (line 285)
        arr_160974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 18), 'arr', False)
        # Obtaining the member 'copy' of a type (line 285)
        copy_160975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 18), arr_160974, 'copy')
        # Calling copy(args, kwargs) (line 285)
        copy_call_result_160977 = invoke(stypy.reporting.localization.Localization(__file__, 285, 18), copy_160975, *[], **kwargs_160976)
        
        # Assigning a type to the variable 'arr' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'arr', copy_call_result_160977)
        # SSA join for if statement (line 284)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 287):
        
        # Assigning a Call to a Name (line 287):
        
        # Call to __new__(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'subtype' (line 287)
        subtype_160981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 32), 'subtype', False)
        # Getting the type of 'shape' (line 287)
        shape_160982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 41), 'shape', False)
        # Getting the type of 'arr' (line 287)
        arr_160983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 48), 'arr', False)
        # Obtaining the member 'dtype' of a type (line 287)
        dtype_160984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 48), arr_160983, 'dtype')
        # Processing the call keyword arguments (line 287)
        # Getting the type of 'arr' (line 288)
        arr_160985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 39), 'arr', False)
        keyword_160986 = arr_160985
        # Getting the type of 'order' (line 289)
        order_160987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 38), 'order', False)
        keyword_160988 = order_160987
        kwargs_160989 = {'buffer': keyword_160986, 'order': keyword_160988}
        # Getting the type of 'N' (line 287)
        N_160978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 14), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 287)
        ndarray_160979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 14), N_160978, 'ndarray')
        # Obtaining the member '__new__' of a type (line 287)
        new___160980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 14), ndarray_160979, '__new__')
        # Calling __new__(args, kwargs) (line 287)
        new___call_result_160990 = invoke(stypy.reporting.localization.Localization(__file__, 287, 14), new___160980, *[subtype_160981, shape_160982, dtype_160984], **kwargs_160989)
        
        # Assigning a type to the variable 'ret' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'ret', new___call_result_160990)
        # Getting the type of 'ret' (line 290)
        ret_160991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'stypy_return_type', ret_160991)
        
        # ################# End of '__new__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__new__' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_160992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_160992)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__new__'
        return stypy_return_type_160992


    @norecursion
    def __array_finalize__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__array_finalize__'
        module_type_store = module_type_store.open_function_context('__array_finalize__', 292, 4, False)
        # Assigning a type to the variable 'self' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.__array_finalize__.__dict__.__setitem__('stypy_localization', localization)
        matrix.__array_finalize__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.__array_finalize__.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.__array_finalize__.__dict__.__setitem__('stypy_function_name', 'matrix.__array_finalize__')
        matrix.__array_finalize__.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        matrix.__array_finalize__.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.__array_finalize__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.__array_finalize__.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.__array_finalize__.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.__array_finalize__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.__array_finalize__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.__array_finalize__', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__array_finalize__', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__array_finalize__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 293):
        
        # Assigning a Name to a Attribute (line 293):
        # Getting the type of 'False' (line 293)
        False_160993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 24), 'False')
        # Getting the type of 'self' (line 293)
        self_160994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'self')
        # Setting the type of the member '_getitem' of a type (line 293)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), self_160994, '_getitem', False_160993)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'obj' (line 294)
        obj_160996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 23), 'obj', False)
        # Getting the type of 'matrix' (line 294)
        matrix_160997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 28), 'matrix', False)
        # Processing the call keyword arguments (line 294)
        kwargs_160998 = {}
        # Getting the type of 'isinstance' (line 294)
        isinstance_160995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 294)
        isinstance_call_result_160999 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), isinstance_160995, *[obj_160996, matrix_160997], **kwargs_160998)
        
        # Getting the type of 'obj' (line 294)
        obj_161000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 40), 'obj')
        # Obtaining the member '_getitem' of a type (line 294)
        _getitem_161001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 40), obj_161000, '_getitem')
        # Applying the binary operator 'and' (line 294)
        result_and_keyword_161002 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 12), 'and', isinstance_call_result_160999, _getitem_161001)
        
        # Testing the type of an if condition (line 294)
        if_condition_161003 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 8), result_and_keyword_161002)
        # Assigning a type to the variable 'if_condition_161003' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'if_condition_161003', if_condition_161003)
        # SSA begins for if statement (line 294)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 55), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 294)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 295):
        
        # Assigning a Attribute to a Name (line 295):
        # Getting the type of 'self' (line 295)
        self_161004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'self')
        # Obtaining the member 'ndim' of a type (line 295)
        ndim_161005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 15), self_161004, 'ndim')
        # Assigning a type to the variable 'ndim' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'ndim', ndim_161005)
        
        
        # Getting the type of 'ndim' (line 296)
        ndim_161006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'ndim')
        int_161007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 20), 'int')
        # Applying the binary operator '==' (line 296)
        result_eq_161008 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 12), '==', ndim_161006, int_161007)
        
        # Testing the type of an if condition (line 296)
        if_condition_161009 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 296, 8), result_eq_161008)
        # Assigning a type to the variable 'if_condition_161009' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'if_condition_161009', if_condition_161009)
        # SSA begins for if statement (line 296)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 296)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'ndim' (line 298)
        ndim_161010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'ndim')
        int_161011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 19), 'int')
        # Applying the binary operator '>' (line 298)
        result_gt_161012 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 12), '>', ndim_161010, int_161011)
        
        # Testing the type of an if condition (line 298)
        if_condition_161013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 8), result_gt_161012)
        # Assigning a type to the variable 'if_condition_161013' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'if_condition_161013', if_condition_161013)
        # SSA begins for if statement (line 298)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Call to tuple(...): (line 299)
        # Processing the call arguments (line 299)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 299)
        self_161019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 41), 'self', False)
        # Obtaining the member 'shape' of a type (line 299)
        shape_161020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 41), self_161019, 'shape')
        comprehension_161021 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 30), shape_161020)
        # Assigning a type to the variable 'x' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 30), 'x', comprehension_161021)
        
        # Getting the type of 'x' (line 299)
        x_161016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 55), 'x', False)
        int_161017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 59), 'int')
        # Applying the binary operator '>' (line 299)
        result_gt_161018 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 55), '>', x_161016, int_161017)
        
        # Getting the type of 'x' (line 299)
        x_161015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 30), 'x', False)
        list_161022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 30), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 30), list_161022, x_161015)
        # Processing the call keyword arguments (line 299)
        kwargs_161023 = {}
        # Getting the type of 'tuple' (line 299)
        tuple_161014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 23), 'tuple', False)
        # Calling tuple(args, kwargs) (line 299)
        tuple_call_result_161024 = invoke(stypy.reporting.localization.Localization(__file__, 299, 23), tuple_161014, *[list_161022], **kwargs_161023)
        
        # Assigning a type to the variable 'newshape' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'newshape', tuple_call_result_161024)
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Call to len(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'newshape' (line 300)
        newshape_161026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 23), 'newshape', False)
        # Processing the call keyword arguments (line 300)
        kwargs_161027 = {}
        # Getting the type of 'len' (line 300)
        len_161025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 19), 'len', False)
        # Calling len(args, kwargs) (line 300)
        len_call_result_161028 = invoke(stypy.reporting.localization.Localization(__file__, 300, 19), len_161025, *[newshape_161026], **kwargs_161027)
        
        # Assigning a type to the variable 'ndim' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'ndim', len_call_result_161028)
        
        
        # Getting the type of 'ndim' (line 301)
        ndim_161029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 15), 'ndim')
        int_161030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 23), 'int')
        # Applying the binary operator '==' (line 301)
        result_eq_161031 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 15), '==', ndim_161029, int_161030)
        
        # Testing the type of an if condition (line 301)
        if_condition_161032 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 12), result_eq_161031)
        # Assigning a type to the variable 'if_condition_161032' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'if_condition_161032', if_condition_161032)
        # SSA begins for if statement (line 301)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 302):
        
        # Assigning a Name to a Attribute (line 302):
        # Getting the type of 'newshape' (line 302)
        newshape_161033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 29), 'newshape')
        # Getting the type of 'self' (line 302)
        self_161034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'self')
        # Setting the type of the member 'shape' of a type (line 302)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 16), self_161034, 'shape', newshape_161033)
        # Assigning a type to the variable 'stypy_return_type' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'stypy_return_type', types.NoneType)
        # SSA branch for the else part of an if statement (line 301)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ndim' (line 304)
        ndim_161035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 18), 'ndim')
        int_161036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 25), 'int')
        # Applying the binary operator '>' (line 304)
        result_gt_161037 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 18), '>', ndim_161035, int_161036)
        
        # Testing the type of an if condition (line 304)
        if_condition_161038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 17), result_gt_161037)
        # Assigning a type to the variable 'if_condition_161038' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 17), 'if_condition_161038', if_condition_161038)
        # SSA begins for if statement (line 304)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 305)
        # Processing the call arguments (line 305)
        str_161040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 33), 'str', 'shape too large to be a matrix.')
        # Processing the call keyword arguments (line 305)
        kwargs_161041 = {}
        # Getting the type of 'ValueError' (line 305)
        ValueError_161039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 305)
        ValueError_call_result_161042 = invoke(stypy.reporting.localization.Localization(__file__, 305, 22), ValueError_161039, *[str_161040], **kwargs_161041)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 305, 16), ValueError_call_result_161042, 'raise parameter', BaseException)
        # SSA join for if statement (line 304)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 301)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 298)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 307):
        
        # Assigning a Attribute to a Name (line 307):
        # Getting the type of 'self' (line 307)
        self_161043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 23), 'self')
        # Obtaining the member 'shape' of a type (line 307)
        shape_161044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 23), self_161043, 'shape')
        # Assigning a type to the variable 'newshape' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'newshape', shape_161044)
        # SSA join for if statement (line 298)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'ndim' (line 308)
        ndim_161045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 11), 'ndim')
        int_161046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 19), 'int')
        # Applying the binary operator '==' (line 308)
        result_eq_161047 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 11), '==', ndim_161045, int_161046)
        
        # Testing the type of an if condition (line 308)
        if_condition_161048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 8), result_eq_161047)
        # Assigning a type to the variable 'if_condition_161048' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'if_condition_161048', if_condition_161048)
        # SSA begins for if statement (line 308)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Attribute (line 309):
        
        # Assigning a Tuple to a Attribute (line 309):
        
        # Obtaining an instance of the builtin type 'tuple' (line 309)
        tuple_161049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 309)
        # Adding element type (line 309)
        int_161050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 26), tuple_161049, int_161050)
        # Adding element type (line 309)
        int_161051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 26), tuple_161049, int_161051)
        
        # Getting the type of 'self' (line 309)
        self_161052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'self')
        # Setting the type of the member 'shape' of a type (line 309)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 12), self_161052, 'shape', tuple_161049)
        # SSA branch for the else part of an if statement (line 308)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ndim' (line 310)
        ndim_161053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 13), 'ndim')
        int_161054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 21), 'int')
        # Applying the binary operator '==' (line 310)
        result_eq_161055 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 13), '==', ndim_161053, int_161054)
        
        # Testing the type of an if condition (line 310)
        if_condition_161056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 13), result_eq_161055)
        # Assigning a type to the variable 'if_condition_161056' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 13), 'if_condition_161056', if_condition_161056)
        # SSA begins for if statement (line 310)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Attribute (line 311):
        
        # Assigning a Tuple to a Attribute (line 311):
        
        # Obtaining an instance of the builtin type 'tuple' (line 311)
        tuple_161057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 311)
        # Adding element type (line 311)
        int_161058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 26), tuple_161057, int_161058)
        # Adding element type (line 311)
        
        # Obtaining the type of the subscript
        int_161059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 38), 'int')
        # Getting the type of 'newshape' (line 311)
        newshape_161060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 29), 'newshape')
        # Obtaining the member '__getitem__' of a type (line 311)
        getitem___161061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 29), newshape_161060, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 311)
        subscript_call_result_161062 = invoke(stypy.reporting.localization.Localization(__file__, 311, 29), getitem___161061, int_161059)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 26), tuple_161057, subscript_call_result_161062)
        
        # Getting the type of 'self' (line 311)
        self_161063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'self')
        # Setting the type of the member 'shape' of a type (line 311)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), self_161063, 'shape', tuple_161057)
        # SSA join for if statement (line 310)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 308)
        module_type_store = module_type_store.join_ssa_context()
        
        # Assigning a type to the variable 'stypy_return_type' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of '__array_finalize__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__array_finalize__' in the type store
        # Getting the type of 'stypy_return_type' (line 292)
        stypy_return_type_161064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161064)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__array_finalize__'
        return stypy_return_type_161064


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 314, 4, False)
        # Assigning a type to the variable 'self' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        matrix.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.__getitem__.__dict__.__setitem__('stypy_function_name', 'matrix.__getitem__')
        matrix.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['index'])
        matrix.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.__getitem__', ['index'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['index'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 315):
        
        # Assigning a Name to a Attribute (line 315):
        # Getting the type of 'True' (line 315)
        True_161065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 24), 'True')
        # Getting the type of 'self' (line 315)
        self_161066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'self')
        # Setting the type of the member '_getitem' of a type (line 315)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), self_161066, '_getitem', True_161065)
        
        # Try-finally block (line 317)
        
        # Assigning a Call to a Name (line 318):
        
        # Assigning a Call to a Name (line 318):
        
        # Call to __getitem__(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'self' (line 318)
        self_161070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 40), 'self', False)
        # Getting the type of 'index' (line 318)
        index_161071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 46), 'index', False)
        # Processing the call keyword arguments (line 318)
        kwargs_161072 = {}
        # Getting the type of 'N' (line 318)
        N_161067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 18), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 318)
        ndarray_161068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 18), N_161067, 'ndarray')
        # Obtaining the member '__getitem__' of a type (line 318)
        getitem___161069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 18), ndarray_161068, '__getitem__')
        # Calling __getitem__(args, kwargs) (line 318)
        getitem___call_result_161073 = invoke(stypy.reporting.localization.Localization(__file__, 318, 18), getitem___161069, *[self_161070, index_161071], **kwargs_161072)
        
        # Assigning a type to the variable 'out' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'out', getitem___call_result_161073)
        
        # finally branch of the try-finally block (line 317)
        
        # Assigning a Name to a Attribute (line 320):
        
        # Assigning a Name to a Attribute (line 320):
        # Getting the type of 'False' (line 320)
        False_161074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 28), 'False')
        # Getting the type of 'self' (line 320)
        self_161075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'self')
        # Setting the type of the member '_getitem' of a type (line 320)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), self_161075, '_getitem', False_161074)
        
        
        
        
        # Call to isinstance(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'out' (line 322)
        out_161077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 26), 'out', False)
        # Getting the type of 'N' (line 322)
        N_161078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 31), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 322)
        ndarray_161079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 31), N_161078, 'ndarray')
        # Processing the call keyword arguments (line 322)
        kwargs_161080 = {}
        # Getting the type of 'isinstance' (line 322)
        isinstance_161076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 322)
        isinstance_call_result_161081 = invoke(stypy.reporting.localization.Localization(__file__, 322, 15), isinstance_161076, *[out_161077, ndarray_161079], **kwargs_161080)
        
        # Applying the 'not' unary operator (line 322)
        result_not__161082 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 11), 'not', isinstance_call_result_161081)
        
        # Testing the type of an if condition (line 322)
        if_condition_161083 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 8), result_not__161082)
        # Assigning a type to the variable 'if_condition_161083' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'if_condition_161083', if_condition_161083)
        # SSA begins for if statement (line 322)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'out' (line 323)
        out_161084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'stypy_return_type', out_161084)
        # SSA join for if statement (line 322)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'out' (line 325)
        out_161085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 11), 'out')
        # Obtaining the member 'ndim' of a type (line 325)
        ndim_161086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 11), out_161085, 'ndim')
        int_161087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 23), 'int')
        # Applying the binary operator '==' (line 325)
        result_eq_161088 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 11), '==', ndim_161086, int_161087)
        
        # Testing the type of an if condition (line 325)
        if_condition_161089 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 8), result_eq_161088)
        # Assigning a type to the variable 'if_condition_161089' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'if_condition_161089', if_condition_161089)
        # SSA begins for if statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 326)
        tuple_161090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 326)
        
        # Getting the type of 'out' (line 326)
        out_161091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'out')
        # Obtaining the member '__getitem__' of a type (line 326)
        getitem___161092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 19), out_161091, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 326)
        subscript_call_result_161093 = invoke(stypy.reporting.localization.Localization(__file__, 326, 19), getitem___161092, tuple_161090)
        
        # Assigning a type to the variable 'stypy_return_type' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'stypy_return_type', subscript_call_result_161093)
        # SSA join for if statement (line 325)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'out' (line 327)
        out_161094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 11), 'out')
        # Obtaining the member 'ndim' of a type (line 327)
        ndim_161095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 11), out_161094, 'ndim')
        int_161096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 23), 'int')
        # Applying the binary operator '==' (line 327)
        result_eq_161097 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 11), '==', ndim_161095, int_161096)
        
        # Testing the type of an if condition (line 327)
        if_condition_161098 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 8), result_eq_161097)
        # Assigning a type to the variable 'if_condition_161098' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'if_condition_161098', if_condition_161098)
        # SSA begins for if statement (line 327)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 328):
        
        # Assigning a Subscript to a Name (line 328):
        
        # Obtaining the type of the subscript
        int_161099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 27), 'int')
        # Getting the type of 'out' (line 328)
        out_161100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 17), 'out')
        # Obtaining the member 'shape' of a type (line 328)
        shape_161101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 17), out_161100, 'shape')
        # Obtaining the member '__getitem__' of a type (line 328)
        getitem___161102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 17), shape_161101, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 328)
        subscript_call_result_161103 = invoke(stypy.reporting.localization.Localization(__file__, 328, 17), getitem___161102, int_161099)
        
        # Assigning a type to the variable 'sh' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'sh', subscript_call_result_161103)
        
        
        # SSA begins for try-except statement (line 330)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 331):
        
        # Assigning a Call to a Name (line 331):
        
        # Call to len(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'index' (line 331)
        index_161105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 24), 'index', False)
        # Processing the call keyword arguments (line 331)
        kwargs_161106 = {}
        # Getting the type of 'len' (line 331)
        len_161104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'len', False)
        # Calling len(args, kwargs) (line 331)
        len_call_result_161107 = invoke(stypy.reporting.localization.Localization(__file__, 331, 20), len_161104, *[index_161105], **kwargs_161106)
        
        # Assigning a type to the variable 'n' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'n', len_call_result_161107)
        # SSA branch for the except part of a try statement (line 330)
        # SSA branch for the except '<any exception>' branch of a try statement (line 330)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Name (line 333):
        
        # Assigning a Num to a Name (line 333):
        int_161108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 20), 'int')
        # Assigning a type to the variable 'n' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'n', int_161108)
        # SSA join for try-except statement (line 330)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'n' (line 334)
        n_161109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 15), 'n')
        int_161110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 19), 'int')
        # Applying the binary operator '>' (line 334)
        result_gt_161111 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 15), '>', n_161109, int_161110)
        
        
        # Call to isscalar(...): (line 334)
        # Processing the call arguments (line 334)
        
        # Obtaining the type of the subscript
        int_161113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 40), 'int')
        # Getting the type of 'index' (line 334)
        index_161114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 34), 'index', False)
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___161115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 34), index_161114, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 334)
        subscript_call_result_161116 = invoke(stypy.reporting.localization.Localization(__file__, 334, 34), getitem___161115, int_161113)
        
        # Processing the call keyword arguments (line 334)
        kwargs_161117 = {}
        # Getting the type of 'isscalar' (line 334)
        isscalar_161112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 25), 'isscalar', False)
        # Calling isscalar(args, kwargs) (line 334)
        isscalar_call_result_161118 = invoke(stypy.reporting.localization.Localization(__file__, 334, 25), isscalar_161112, *[subscript_call_result_161116], **kwargs_161117)
        
        # Applying the binary operator 'and' (line 334)
        result_and_keyword_161119 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 15), 'and', result_gt_161111, isscalar_call_result_161118)
        
        # Testing the type of an if condition (line 334)
        if_condition_161120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 12), result_and_keyword_161119)
        # Assigning a type to the variable 'if_condition_161120' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'if_condition_161120', if_condition_161120)
        # SSA begins for if statement (line 334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Attribute (line 335):
        
        # Assigning a Tuple to a Attribute (line 335):
        
        # Obtaining an instance of the builtin type 'tuple' (line 335)
        tuple_161121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 335)
        # Adding element type (line 335)
        # Getting the type of 'sh' (line 335)
        sh_161122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 29), 'sh')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 29), tuple_161121, sh_161122)
        # Adding element type (line 335)
        int_161123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 29), tuple_161121, int_161123)
        
        # Getting the type of 'out' (line 335)
        out_161124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'out')
        # Setting the type of the member 'shape' of a type (line 335)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 16), out_161124, 'shape', tuple_161121)
        # SSA branch for the else part of an if statement (line 334)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Tuple to a Attribute (line 337):
        
        # Assigning a Tuple to a Attribute (line 337):
        
        # Obtaining an instance of the builtin type 'tuple' (line 337)
        tuple_161125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 337)
        # Adding element type (line 337)
        int_161126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 29), tuple_161125, int_161126)
        # Adding element type (line 337)
        # Getting the type of 'sh' (line 337)
        sh_161127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 32), 'sh')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 29), tuple_161125, sh_161127)
        
        # Getting the type of 'out' (line 337)
        out_161128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 16), 'out')
        # Setting the type of the member 'shape' of a type (line 337)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 16), out_161128, 'shape', tuple_161125)
        # SSA join for if statement (line 334)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 327)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'out' (line 338)
        out_161129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 15), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'stypy_return_type', out_161129)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 314)
        stypy_return_type_161130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161130)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_161130


    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 340, 4, False)
        # Assigning a type to the variable 'self' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.__mul__.__dict__.__setitem__('stypy_localization', localization)
        matrix.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.__mul__.__dict__.__setitem__('stypy_function_name', 'matrix.__mul__')
        matrix.__mul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        matrix.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.__mul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mul__(...)' code ##################

        
        
        # Call to isinstance(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'other' (line 341)
        other_161132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 22), 'other', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 341)
        tuple_161133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 341)
        # Adding element type (line 341)
        # Getting the type of 'N' (line 341)
        N_161134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 30), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 341)
        ndarray_161135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 30), N_161134, 'ndarray')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 30), tuple_161133, ndarray_161135)
        # Adding element type (line 341)
        # Getting the type of 'list' (line 341)
        list_161136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 41), 'list', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 30), tuple_161133, list_161136)
        # Adding element type (line 341)
        # Getting the type of 'tuple' (line 341)
        tuple_161137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 47), 'tuple', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 30), tuple_161133, tuple_161137)
        
        # Processing the call keyword arguments (line 341)
        kwargs_161138 = {}
        # Getting the type of 'isinstance' (line 341)
        isinstance_161131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 341)
        isinstance_call_result_161139 = invoke(stypy.reporting.localization.Localization(__file__, 341, 11), isinstance_161131, *[other_161132, tuple_161133], **kwargs_161138)
        
        # Testing the type of an if condition (line 341)
        if_condition_161140 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 8), isinstance_call_result_161139)
        # Assigning a type to the variable 'if_condition_161140' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'if_condition_161140', if_condition_161140)
        # SSA begins for if statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to dot(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'self' (line 343)
        self_161143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 25), 'self', False)
        
        # Call to asmatrix(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'other' (line 343)
        other_161145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 40), 'other', False)
        # Processing the call keyword arguments (line 343)
        kwargs_161146 = {}
        # Getting the type of 'asmatrix' (line 343)
        asmatrix_161144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 31), 'asmatrix', False)
        # Calling asmatrix(args, kwargs) (line 343)
        asmatrix_call_result_161147 = invoke(stypy.reporting.localization.Localization(__file__, 343, 31), asmatrix_161144, *[other_161145], **kwargs_161146)
        
        # Processing the call keyword arguments (line 343)
        kwargs_161148 = {}
        # Getting the type of 'N' (line 343)
        N_161141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 19), 'N', False)
        # Obtaining the member 'dot' of a type (line 343)
        dot_161142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 19), N_161141, 'dot')
        # Calling dot(args, kwargs) (line 343)
        dot_call_result_161149 = invoke(stypy.reporting.localization.Localization(__file__, 343, 19), dot_161142, *[self_161143, asmatrix_call_result_161147], **kwargs_161148)
        
        # Assigning a type to the variable 'stypy_return_type' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'stypy_return_type', dot_call_result_161149)
        # SSA join for if statement (line 341)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Call to isscalar(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'other' (line 344)
        other_161151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 20), 'other', False)
        # Processing the call keyword arguments (line 344)
        kwargs_161152 = {}
        # Getting the type of 'isscalar' (line 344)
        isscalar_161150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 11), 'isscalar', False)
        # Calling isscalar(args, kwargs) (line 344)
        isscalar_call_result_161153 = invoke(stypy.reporting.localization.Localization(__file__, 344, 11), isscalar_161150, *[other_161151], **kwargs_161152)
        
        
        
        # Call to hasattr(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'other' (line 344)
        other_161155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 42), 'other', False)
        str_161156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 49), 'str', '__rmul__')
        # Processing the call keyword arguments (line 344)
        kwargs_161157 = {}
        # Getting the type of 'hasattr' (line 344)
        hasattr_161154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 34), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 344)
        hasattr_call_result_161158 = invoke(stypy.reporting.localization.Localization(__file__, 344, 34), hasattr_161154, *[other_161155, str_161156], **kwargs_161157)
        
        # Applying the 'not' unary operator (line 344)
        result_not__161159 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 30), 'not', hasattr_call_result_161158)
        
        # Applying the binary operator 'or' (line 344)
        result_or_keyword_161160 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 11), 'or', isscalar_call_result_161153, result_not__161159)
        
        # Testing the type of an if condition (line 344)
        if_condition_161161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 344, 8), result_or_keyword_161160)
        # Assigning a type to the variable 'if_condition_161161' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'if_condition_161161', if_condition_161161)
        # SSA begins for if statement (line 344)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to dot(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'self' (line 345)
        self_161164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 25), 'self', False)
        # Getting the type of 'other' (line 345)
        other_161165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 31), 'other', False)
        # Processing the call keyword arguments (line 345)
        kwargs_161166 = {}
        # Getting the type of 'N' (line 345)
        N_161162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'N', False)
        # Obtaining the member 'dot' of a type (line 345)
        dot_161163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 19), N_161162, 'dot')
        # Calling dot(args, kwargs) (line 345)
        dot_call_result_161167 = invoke(stypy.reporting.localization.Localization(__file__, 345, 19), dot_161163, *[self_161164, other_161165], **kwargs_161166)
        
        # Assigning a type to the variable 'stypy_return_type' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'stypy_return_type', dot_call_result_161167)
        # SSA join for if statement (line 344)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'NotImplemented' (line 346)
        NotImplemented_161168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 15), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'stypy_return_type', NotImplemented_161168)
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 340)
        stypy_return_type_161169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161169)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_161169


    @norecursion
    def __rmul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rmul__'
        module_type_store = module_type_store.open_function_context('__rmul__', 348, 4, False)
        # Assigning a type to the variable 'self' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.__rmul__.__dict__.__setitem__('stypy_localization', localization)
        matrix.__rmul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.__rmul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.__rmul__.__dict__.__setitem__('stypy_function_name', 'matrix.__rmul__')
        matrix.__rmul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        matrix.__rmul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.__rmul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.__rmul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.__rmul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.__rmul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.__rmul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.__rmul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rmul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rmul__(...)' code ##################

        
        # Call to dot(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'other' (line 349)
        other_161172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 21), 'other', False)
        # Getting the type of 'self' (line 349)
        self_161173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 28), 'self', False)
        # Processing the call keyword arguments (line 349)
        kwargs_161174 = {}
        # Getting the type of 'N' (line 349)
        N_161170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'N', False)
        # Obtaining the member 'dot' of a type (line 349)
        dot_161171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 15), N_161170, 'dot')
        # Calling dot(args, kwargs) (line 349)
        dot_call_result_161175 = invoke(stypy.reporting.localization.Localization(__file__, 349, 15), dot_161171, *[other_161172, self_161173], **kwargs_161174)
        
        # Assigning a type to the variable 'stypy_return_type' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'stypy_return_type', dot_call_result_161175)
        
        # ################# End of '__rmul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rmul__' in the type store
        # Getting the type of 'stypy_return_type' (line 348)
        stypy_return_type_161176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161176)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rmul__'
        return stypy_return_type_161176


    @norecursion
    def __imul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__imul__'
        module_type_store = module_type_store.open_function_context('__imul__', 351, 4, False)
        # Assigning a type to the variable 'self' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.__imul__.__dict__.__setitem__('stypy_localization', localization)
        matrix.__imul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.__imul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.__imul__.__dict__.__setitem__('stypy_function_name', 'matrix.__imul__')
        matrix.__imul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        matrix.__imul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.__imul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.__imul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.__imul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.__imul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.__imul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.__imul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__imul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__imul__(...)' code ##################

        
        # Assigning a BinOp to a Subscript (line 352):
        
        # Assigning a BinOp to a Subscript (line 352):
        # Getting the type of 'self' (line 352)
        self_161177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 18), 'self')
        # Getting the type of 'other' (line 352)
        other_161178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 25), 'other')
        # Applying the binary operator '*' (line 352)
        result_mul_161179 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 18), '*', self_161177, other_161178)
        
        # Getting the type of 'self' (line 352)
        self_161180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'self')
        slice_161181 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 352, 8), None, None, None)
        # Storing an element on a container (line 352)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 8), self_161180, (slice_161181, result_mul_161179))
        # Getting the type of 'self' (line 353)
        self_161182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'stypy_return_type', self_161182)
        
        # ################# End of '__imul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__imul__' in the type store
        # Getting the type of 'stypy_return_type' (line 351)
        stypy_return_type_161183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161183)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__imul__'
        return stypy_return_type_161183


    @norecursion
    def __pow__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pow__'
        module_type_store = module_type_store.open_function_context('__pow__', 355, 4, False)
        # Assigning a type to the variable 'self' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.__pow__.__dict__.__setitem__('stypy_localization', localization)
        matrix.__pow__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.__pow__.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.__pow__.__dict__.__setitem__('stypy_function_name', 'matrix.__pow__')
        matrix.__pow__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        matrix.__pow__.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.__pow__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.__pow__.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.__pow__.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.__pow__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.__pow__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.__pow__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__pow__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__pow__(...)' code ##################

        
        # Call to matrix_power(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'self' (line 356)
        self_161185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 28), 'self', False)
        # Getting the type of 'other' (line 356)
        other_161186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 34), 'other', False)
        # Processing the call keyword arguments (line 356)
        kwargs_161187 = {}
        # Getting the type of 'matrix_power' (line 356)
        matrix_power_161184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 15), 'matrix_power', False)
        # Calling matrix_power(args, kwargs) (line 356)
        matrix_power_call_result_161188 = invoke(stypy.reporting.localization.Localization(__file__, 356, 15), matrix_power_161184, *[self_161185, other_161186], **kwargs_161187)
        
        # Assigning a type to the variable 'stypy_return_type' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'stypy_return_type', matrix_power_call_result_161188)
        
        # ################# End of '__pow__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pow__' in the type store
        # Getting the type of 'stypy_return_type' (line 355)
        stypy_return_type_161189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161189)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pow__'
        return stypy_return_type_161189


    @norecursion
    def __ipow__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ipow__'
        module_type_store = module_type_store.open_function_context('__ipow__', 358, 4, False)
        # Assigning a type to the variable 'self' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.__ipow__.__dict__.__setitem__('stypy_localization', localization)
        matrix.__ipow__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.__ipow__.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.__ipow__.__dict__.__setitem__('stypy_function_name', 'matrix.__ipow__')
        matrix.__ipow__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        matrix.__ipow__.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.__ipow__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.__ipow__.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.__ipow__.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.__ipow__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.__ipow__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.__ipow__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ipow__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ipow__(...)' code ##################

        
        # Assigning a BinOp to a Subscript (line 359):
        
        # Assigning a BinOp to a Subscript (line 359):
        # Getting the type of 'self' (line 359)
        self_161190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 18), 'self')
        # Getting the type of 'other' (line 359)
        other_161191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 26), 'other')
        # Applying the binary operator '**' (line 359)
        result_pow_161192 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 18), '**', self_161190, other_161191)
        
        # Getting the type of 'self' (line 359)
        self_161193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'self')
        slice_161194 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 359, 8), None, None, None)
        # Storing an element on a container (line 359)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 8), self_161193, (slice_161194, result_pow_161192))
        # Getting the type of 'self' (line 360)
        self_161195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'stypy_return_type', self_161195)
        
        # ################# End of '__ipow__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ipow__' in the type store
        # Getting the type of 'stypy_return_type' (line 358)
        stypy_return_type_161196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ipow__'
        return stypy_return_type_161196


    @norecursion
    def __rpow__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rpow__'
        module_type_store = module_type_store.open_function_context('__rpow__', 362, 4, False)
        # Assigning a type to the variable 'self' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.__rpow__.__dict__.__setitem__('stypy_localization', localization)
        matrix.__rpow__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.__rpow__.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.__rpow__.__dict__.__setitem__('stypy_function_name', 'matrix.__rpow__')
        matrix.__rpow__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        matrix.__rpow__.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.__rpow__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.__rpow__.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.__rpow__.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.__rpow__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.__rpow__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.__rpow__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rpow__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rpow__(...)' code ##################

        # Getting the type of 'NotImplemented' (line 363)
        NotImplemented_161197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'stypy_return_type', NotImplemented_161197)
        
        # ################# End of '__rpow__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rpow__' in the type store
        # Getting the type of 'stypy_return_type' (line 362)
        stypy_return_type_161198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161198)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rpow__'
        return stypy_return_type_161198


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 365, 4, False)
        # Assigning a type to the variable 'self' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        matrix.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'matrix.__repr__')
        matrix.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        matrix.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        
        # Assigning a Call to a Name (line 366):
        
        # Assigning a Call to a Name (line 366):
        
        # Call to replace(...): (line 366)
        # Processing the call arguments (line 366)
        str_161207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 43), 'str', 'array')
        str_161208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 52), 'str', 'matrix')
        # Processing the call keyword arguments (line 366)
        kwargs_161209 = {}
        
        # Call to repr(...): (line 366)
        # Processing the call arguments (line 366)
        
        # Call to __array__(...): (line 366)
        # Processing the call keyword arguments (line 366)
        kwargs_161202 = {}
        # Getting the type of 'self' (line 366)
        self_161200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 17), 'self', False)
        # Obtaining the member '__array__' of a type (line 366)
        array___161201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 17), self_161200, '__array__')
        # Calling __array__(args, kwargs) (line 366)
        array___call_result_161203 = invoke(stypy.reporting.localization.Localization(__file__, 366, 17), array___161201, *[], **kwargs_161202)
        
        # Processing the call keyword arguments (line 366)
        kwargs_161204 = {}
        # Getting the type of 'repr' (line 366)
        repr_161199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'repr', False)
        # Calling repr(args, kwargs) (line 366)
        repr_call_result_161205 = invoke(stypy.reporting.localization.Localization(__file__, 366, 12), repr_161199, *[array___call_result_161203], **kwargs_161204)
        
        # Obtaining the member 'replace' of a type (line 366)
        replace_161206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 12), repr_call_result_161205, 'replace')
        # Calling replace(args, kwargs) (line 366)
        replace_call_result_161210 = invoke(stypy.reporting.localization.Localization(__file__, 366, 12), replace_161206, *[str_161207, str_161208], **kwargs_161209)
        
        # Assigning a type to the variable 's' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 's', replace_call_result_161210)
        
        # Assigning a Call to a Name (line 369):
        
        # Assigning a Call to a Name (line 369):
        
        # Call to splitlines(...): (line 369)
        # Processing the call keyword arguments (line 369)
        kwargs_161213 = {}
        # Getting the type of 's' (line 369)
        s_161211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 's', False)
        # Obtaining the member 'splitlines' of a type (line 369)
        splitlines_161212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 12), s_161211, 'splitlines')
        # Calling splitlines(args, kwargs) (line 369)
        splitlines_call_result_161214 = invoke(stypy.reporting.localization.Localization(__file__, 369, 12), splitlines_161212, *[], **kwargs_161213)
        
        # Assigning a type to the variable 'l' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'l', splitlines_call_result_161214)
        
        
        # Call to range(...): (line 370)
        # Processing the call arguments (line 370)
        int_161216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 23), 'int')
        
        # Call to len(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'l' (line 370)
        l_161218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 30), 'l', False)
        # Processing the call keyword arguments (line 370)
        kwargs_161219 = {}
        # Getting the type of 'len' (line 370)
        len_161217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 26), 'len', False)
        # Calling len(args, kwargs) (line 370)
        len_call_result_161220 = invoke(stypy.reporting.localization.Localization(__file__, 370, 26), len_161217, *[l_161218], **kwargs_161219)
        
        # Processing the call keyword arguments (line 370)
        kwargs_161221 = {}
        # Getting the type of 'range' (line 370)
        range_161215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 17), 'range', False)
        # Calling range(args, kwargs) (line 370)
        range_call_result_161222 = invoke(stypy.reporting.localization.Localization(__file__, 370, 17), range_161215, *[int_161216, len_call_result_161220], **kwargs_161221)
        
        # Testing the type of a for loop iterable (line 370)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 370, 8), range_call_result_161222)
        # Getting the type of the for loop variable (line 370)
        for_loop_var_161223 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 370, 8), range_call_result_161222)
        # Assigning a type to the variable 'i' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'i', for_loop_var_161223)
        # SSA begins for a for statement (line 370)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 371)
        i_161224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 17), 'i')
        # Getting the type of 'l' (line 371)
        l_161225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 15), 'l')
        # Obtaining the member '__getitem__' of a type (line 371)
        getitem___161226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 15), l_161225, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 371)
        subscript_call_result_161227 = invoke(stypy.reporting.localization.Localization(__file__, 371, 15), getitem___161226, i_161224)
        
        # Testing the type of an if condition (line 371)
        if_condition_161228 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 12), subscript_call_result_161227)
        # Assigning a type to the variable 'if_condition_161228' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'if_condition_161228', if_condition_161228)
        # SSA begins for if statement (line 371)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Subscript (line 372):
        
        # Assigning a BinOp to a Subscript (line 372):
        str_161229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 23), 'str', ' ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 372)
        i_161230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 31), 'i')
        # Getting the type of 'l' (line 372)
        l_161231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 29), 'l')
        # Obtaining the member '__getitem__' of a type (line 372)
        getitem___161232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 29), l_161231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 372)
        subscript_call_result_161233 = invoke(stypy.reporting.localization.Localization(__file__, 372, 29), getitem___161232, i_161230)
        
        # Applying the binary operator '+' (line 372)
        result_add_161234 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 23), '+', str_161229, subscript_call_result_161233)
        
        # Getting the type of 'l' (line 372)
        l_161235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'l')
        # Getting the type of 'i' (line 372)
        i_161236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 18), 'i')
        # Storing an element on a container (line 372)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 16), l_161235, (i_161236, result_add_161234))
        # SSA join for if statement (line 371)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to join(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'l' (line 373)
        l_161239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 25), 'l', False)
        # Processing the call keyword arguments (line 373)
        kwargs_161240 = {}
        str_161237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 15), 'str', '\n')
        # Obtaining the member 'join' of a type (line 373)
        join_161238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 15), str_161237, 'join')
        # Calling join(args, kwargs) (line 373)
        join_call_result_161241 = invoke(stypy.reporting.localization.Localization(__file__, 373, 15), join_161238, *[l_161239], **kwargs_161240)
        
        # Assigning a type to the variable 'stypy_return_type' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'stypy_return_type', join_call_result_161241)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 365)
        stypy_return_type_161242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161242)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_161242


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 375, 4, False)
        # Assigning a type to the variable 'self' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        matrix.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.stypy__str__.__dict__.__setitem__('stypy_function_name', 'matrix.__str__')
        matrix.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        matrix.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        
        # Call to str(...): (line 376)
        # Processing the call arguments (line 376)
        
        # Call to __array__(...): (line 376)
        # Processing the call keyword arguments (line 376)
        kwargs_161246 = {}
        # Getting the type of 'self' (line 376)
        self_161244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 'self', False)
        # Obtaining the member '__array__' of a type (line 376)
        array___161245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 19), self_161244, '__array__')
        # Calling __array__(args, kwargs) (line 376)
        array___call_result_161247 = invoke(stypy.reporting.localization.Localization(__file__, 376, 19), array___161245, *[], **kwargs_161246)
        
        # Processing the call keyword arguments (line 376)
        kwargs_161248 = {}
        # Getting the type of 'str' (line 376)
        str_161243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 15), 'str', False)
        # Calling str(args, kwargs) (line 376)
        str_call_result_161249 = invoke(stypy.reporting.localization.Localization(__file__, 376, 15), str_161243, *[array___call_result_161247], **kwargs_161248)
        
        # Assigning a type to the variable 'stypy_return_type' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'stypy_return_type', str_call_result_161249)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 375)
        stypy_return_type_161250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161250)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_161250


    @norecursion
    def _align(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_align'
        module_type_store = module_type_store.open_function_context('_align', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix._align.__dict__.__setitem__('stypy_localization', localization)
        matrix._align.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix._align.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix._align.__dict__.__setitem__('stypy_function_name', 'matrix._align')
        matrix._align.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        matrix._align.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix._align.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix._align.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix._align.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix._align.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix._align.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix._align', ['axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_align', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_align(...)' code ##################

        str_161251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, (-1)), 'str', 'A convenience function for operations that need to preserve axis\n        orientation.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 382)
        # Getting the type of 'axis' (line 382)
        axis_161252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 11), 'axis')
        # Getting the type of 'None' (line 382)
        None_161253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 19), 'None')
        
        (may_be_161254, more_types_in_union_161255) = may_be_none(axis_161252, None_161253)

        if may_be_161254:

            if more_types_in_union_161255:
                # Runtime conditional SSA (line 382)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Obtaining the type of the subscript
            
            # Obtaining an instance of the builtin type 'tuple' (line 383)
            tuple_161256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 24), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 383)
            # Adding element type (line 383)
            int_161257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 24), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 24), tuple_161256, int_161257)
            # Adding element type (line 383)
            int_161258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 27), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 24), tuple_161256, int_161258)
            
            # Getting the type of 'self' (line 383)
            self_161259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), 'self')
            # Obtaining the member '__getitem__' of a type (line 383)
            getitem___161260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 19), self_161259, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 383)
            subscript_call_result_161261 = invoke(stypy.reporting.localization.Localization(__file__, 383, 19), getitem___161260, tuple_161256)
            
            # Assigning a type to the variable 'stypy_return_type' (line 383)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'stypy_return_type', subscript_call_result_161261)

            if more_types_in_union_161255:
                # Runtime conditional SSA for else branch (line 382)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_161254) or more_types_in_union_161255):
            
            
            # Getting the type of 'axis' (line 384)
            axis_161262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 13), 'axis')
            int_161263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 19), 'int')
            # Applying the binary operator '==' (line 384)
            result_eq_161264 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 13), '==', axis_161262, int_161263)
            
            # Testing the type of an if condition (line 384)
            if_condition_161265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 13), result_eq_161264)
            # Assigning a type to the variable 'if_condition_161265' (line 384)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 13), 'if_condition_161265', if_condition_161265)
            # SSA begins for if statement (line 384)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 385)
            self_161266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 19), 'self')
            # Assigning a type to the variable 'stypy_return_type' (line 385)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'stypy_return_type', self_161266)
            # SSA branch for the else part of an if statement (line 384)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'axis' (line 386)
            axis_161267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 13), 'axis')
            int_161268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 19), 'int')
            # Applying the binary operator '==' (line 386)
            result_eq_161269 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 13), '==', axis_161267, int_161268)
            
            # Testing the type of an if condition (line 386)
            if_condition_161270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 386, 13), result_eq_161269)
            # Assigning a type to the variable 'if_condition_161270' (line 386)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 13), 'if_condition_161270', if_condition_161270)
            # SSA begins for if statement (line 386)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to transpose(...): (line 387)
            # Processing the call keyword arguments (line 387)
            kwargs_161273 = {}
            # Getting the type of 'self' (line 387)
            self_161271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 19), 'self', False)
            # Obtaining the member 'transpose' of a type (line 387)
            transpose_161272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 19), self_161271, 'transpose')
            # Calling transpose(args, kwargs) (line 387)
            transpose_call_result_161274 = invoke(stypy.reporting.localization.Localization(__file__, 387, 19), transpose_161272, *[], **kwargs_161273)
            
            # Assigning a type to the variable 'stypy_return_type' (line 387)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'stypy_return_type', transpose_call_result_161274)
            # SSA branch for the else part of an if statement (line 386)
            module_type_store.open_ssa_branch('else')
            
            # Call to ValueError(...): (line 389)
            # Processing the call arguments (line 389)
            str_161276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 29), 'str', 'unsupported axis')
            # Processing the call keyword arguments (line 389)
            kwargs_161277 = {}
            # Getting the type of 'ValueError' (line 389)
            ValueError_161275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 389)
            ValueError_call_result_161278 = invoke(stypy.reporting.localization.Localization(__file__, 389, 18), ValueError_161275, *[str_161276], **kwargs_161277)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 389, 12), ValueError_call_result_161278, 'raise parameter', BaseException)
            # SSA join for if statement (line 386)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 384)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_161254 and more_types_in_union_161255):
                # SSA join for if statement (line 382)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_align(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_align' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_161279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161279)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_align'
        return stypy_return_type_161279


    @norecursion
    def _collapse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_collapse'
        module_type_store = module_type_store.open_function_context('_collapse', 391, 4, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix._collapse.__dict__.__setitem__('stypy_localization', localization)
        matrix._collapse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix._collapse.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix._collapse.__dict__.__setitem__('stypy_function_name', 'matrix._collapse')
        matrix._collapse.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        matrix._collapse.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix._collapse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix._collapse.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix._collapse.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix._collapse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix._collapse.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix._collapse', ['axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_collapse', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_collapse(...)' code ##################

        str_161280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, (-1)), 'str', 'A convenience function for operations that want to collapse\n        to a scalar like _align, but are using keepdims=True\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 395)
        # Getting the type of 'axis' (line 395)
        axis_161281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 11), 'axis')
        # Getting the type of 'None' (line 395)
        None_161282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 19), 'None')
        
        (may_be_161283, more_types_in_union_161284) = may_be_none(axis_161281, None_161282)

        if may_be_161283:

            if more_types_in_union_161284:
                # Runtime conditional SSA (line 395)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Obtaining the type of the subscript
            
            # Obtaining an instance of the builtin type 'tuple' (line 396)
            tuple_161285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 24), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 396)
            # Adding element type (line 396)
            int_161286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 24), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 24), tuple_161285, int_161286)
            # Adding element type (line 396)
            int_161287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 27), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 24), tuple_161285, int_161287)
            
            # Getting the type of 'self' (line 396)
            self_161288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 'self')
            # Obtaining the member '__getitem__' of a type (line 396)
            getitem___161289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 19), self_161288, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 396)
            subscript_call_result_161290 = invoke(stypy.reporting.localization.Localization(__file__, 396, 19), getitem___161289, tuple_161285)
            
            # Assigning a type to the variable 'stypy_return_type' (line 396)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'stypy_return_type', subscript_call_result_161290)

            if more_types_in_union_161284:
                # Runtime conditional SSA for else branch (line 395)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_161283) or more_types_in_union_161284):
            # Getting the type of 'self' (line 398)
            self_161291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), 'self')
            # Assigning a type to the variable 'stypy_return_type' (line 398)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'stypy_return_type', self_161291)

            if (may_be_161283 and more_types_in_union_161284):
                # SSA join for if statement (line 395)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_collapse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_collapse' in the type store
        # Getting the type of 'stypy_return_type' (line 391)
        stypy_return_type_161292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161292)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_collapse'
        return stypy_return_type_161292


    @norecursion
    def tolist(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tolist'
        module_type_store = module_type_store.open_function_context('tolist', 402, 4, False)
        # Assigning a type to the variable 'self' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.tolist.__dict__.__setitem__('stypy_localization', localization)
        matrix.tolist.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.tolist.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.tolist.__dict__.__setitem__('stypy_function_name', 'matrix.tolist')
        matrix.tolist.__dict__.__setitem__('stypy_param_names_list', [])
        matrix.tolist.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.tolist.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.tolist.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.tolist.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.tolist.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.tolist.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.tolist', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tolist', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tolist(...)' code ##################

        str_161293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, (-1)), 'str', '\n        Return the matrix as a (possibly nested) list.\n\n        See `ndarray.tolist` for full documentation.\n\n        See Also\n        --------\n        ndarray.tolist\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.tolist()\n        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]\n\n        ')
        
        # Call to tolist(...): (line 422)
        # Processing the call keyword arguments (line 422)
        kwargs_161299 = {}
        
        # Call to __array__(...): (line 422)
        # Processing the call keyword arguments (line 422)
        kwargs_161296 = {}
        # Getting the type of 'self' (line 422)
        self_161294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'self', False)
        # Obtaining the member '__array__' of a type (line 422)
        array___161295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 15), self_161294, '__array__')
        # Calling __array__(args, kwargs) (line 422)
        array___call_result_161297 = invoke(stypy.reporting.localization.Localization(__file__, 422, 15), array___161295, *[], **kwargs_161296)
        
        # Obtaining the member 'tolist' of a type (line 422)
        tolist_161298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 15), array___call_result_161297, 'tolist')
        # Calling tolist(args, kwargs) (line 422)
        tolist_call_result_161300 = invoke(stypy.reporting.localization.Localization(__file__, 422, 15), tolist_161298, *[], **kwargs_161299)
        
        # Assigning a type to the variable 'stypy_return_type' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'stypy_return_type', tolist_call_result_161300)
        
        # ################# End of 'tolist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tolist' in the type store
        # Getting the type of 'stypy_return_type' (line 402)
        stypy_return_type_161301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161301)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tolist'
        return stypy_return_type_161301


    @norecursion
    def sum(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 425)
        None_161302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 23), 'None')
        # Getting the type of 'None' (line 425)
        None_161303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 35), 'None')
        # Getting the type of 'None' (line 425)
        None_161304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 45), 'None')
        defaults = [None_161302, None_161303, None_161304]
        # Create a new context for function 'sum'
        module_type_store = module_type_store.open_function_context('sum', 425, 4, False)
        # Assigning a type to the variable 'self' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.sum.__dict__.__setitem__('stypy_localization', localization)
        matrix.sum.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.sum.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.sum.__dict__.__setitem__('stypy_function_name', 'matrix.sum')
        matrix.sum.__dict__.__setitem__('stypy_param_names_list', ['axis', 'dtype', 'out'])
        matrix.sum.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.sum.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.sum.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.sum.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.sum.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.sum.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.sum', ['axis', 'dtype', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sum', localization, ['axis', 'dtype', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sum(...)' code ##################

        str_161305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, (-1)), 'str', "\n        Returns the sum of the matrix elements, along the given axis.\n\n        Refer to `numpy.sum` for full documentation.\n\n        See Also\n        --------\n        numpy.sum\n\n        Notes\n        -----\n        This is the same as `ndarray.sum`, except that where an `ndarray` would\n        be returned, a `matrix` object is returned instead.\n\n        Examples\n        --------\n        >>> x = np.matrix([[1, 2], [4, 3]])\n        >>> x.sum()\n        10\n        >>> x.sum(axis=1)\n        matrix([[3],\n                [7]])\n        >>> x.sum(axis=1, dtype='float')\n        matrix([[ 3.],\n                [ 7.]])\n        >>> out = np.zeros((1, 2), dtype='float')\n        >>> x.sum(axis=1, dtype='float', out=out)\n        matrix([[ 3.],\n                [ 7.]])\n\n        ")
        
        # Call to _collapse(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'axis' (line 457)
        axis_161318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 78), 'axis', False)
        # Processing the call keyword arguments (line 457)
        kwargs_161319 = {}
        
        # Call to sum(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'self' (line 457)
        self_161309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 29), 'self', False)
        # Getting the type of 'axis' (line 457)
        axis_161310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 35), 'axis', False)
        # Getting the type of 'dtype' (line 457)
        dtype_161311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 41), 'dtype', False)
        # Getting the type of 'out' (line 457)
        out_161312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 48), 'out', False)
        # Processing the call keyword arguments (line 457)
        # Getting the type of 'True' (line 457)
        True_161313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 62), 'True', False)
        keyword_161314 = True_161313
        kwargs_161315 = {'keepdims': keyword_161314}
        # Getting the type of 'N' (line 457)
        N_161306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 457)
        ndarray_161307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 15), N_161306, 'ndarray')
        # Obtaining the member 'sum' of a type (line 457)
        sum_161308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 15), ndarray_161307, 'sum')
        # Calling sum(args, kwargs) (line 457)
        sum_call_result_161316 = invoke(stypy.reporting.localization.Localization(__file__, 457, 15), sum_161308, *[self_161309, axis_161310, dtype_161311, out_161312], **kwargs_161315)
        
        # Obtaining the member '_collapse' of a type (line 457)
        _collapse_161317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 15), sum_call_result_161316, '_collapse')
        # Calling _collapse(args, kwargs) (line 457)
        _collapse_call_result_161320 = invoke(stypy.reporting.localization.Localization(__file__, 457, 15), _collapse_161317, *[axis_161318], **kwargs_161319)
        
        # Assigning a type to the variable 'stypy_return_type' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'stypy_return_type', _collapse_call_result_161320)
        
        # ################# End of 'sum(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sum' in the type store
        # Getting the type of 'stypy_return_type' (line 425)
        stypy_return_type_161321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161321)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sum'
        return stypy_return_type_161321


    @norecursion
    def squeeze(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 461)
        None_161322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 27), 'None')
        defaults = [None_161322]
        # Create a new context for function 'squeeze'
        module_type_store = module_type_store.open_function_context('squeeze', 461, 4, False)
        # Assigning a type to the variable 'self' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.squeeze.__dict__.__setitem__('stypy_localization', localization)
        matrix.squeeze.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.squeeze.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.squeeze.__dict__.__setitem__('stypy_function_name', 'matrix.squeeze')
        matrix.squeeze.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        matrix.squeeze.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.squeeze.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.squeeze.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.squeeze.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.squeeze.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.squeeze.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.squeeze', ['axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'squeeze', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'squeeze(...)' code ##################

        str_161323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, (-1)), 'str', '\n        Return a possibly reshaped matrix.\n\n        Refer to `numpy.squeeze` for more documentation.\n\n        Parameters\n        ----------\n        axis : None or int or tuple of ints, optional\n            Selects a subset of the single-dimensional entries in the shape.\n            If an axis is selected with shape entry greater than one,\n            an error is raised.\n\n        Returns\n        -------\n        squeezed : matrix\n            The matrix, but as a (1, N) matrix if it had shape (N, 1).\n\n        See Also\n        --------\n        numpy.squeeze : related function\n\n        Notes\n        -----\n        If `m` has a single column then that column is returned\n        as the single row of a matrix.  Otherwise `m` is returned.\n        The returned matrix is always either `m` itself or a view into `m`.\n        Supplying an axis keyword argument will not affect the returned matrix\n        but it may cause an error to be raised.\n\n        Examples\n        --------\n        >>> c = np.matrix([[1], [2]])\n        >>> c\n        matrix([[1],\n                [2]])\n        >>> c.squeeze()\n        matrix([[1, 2]])\n        >>> r = c.T\n        >>> r\n        matrix([[1, 2]])\n        >>> r.squeeze()\n        matrix([[1, 2]])\n        >>> m = np.matrix([[1, 2], [3, 4]])\n        >>> m.squeeze()\n        matrix([[1, 2],\n                [3, 4]])\n\n        ')
        
        # Call to squeeze(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'self' (line 510)
        self_161327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 33), 'self', False)
        # Processing the call keyword arguments (line 510)
        # Getting the type of 'axis' (line 510)
        axis_161328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 44), 'axis', False)
        keyword_161329 = axis_161328
        kwargs_161330 = {'axis': keyword_161329}
        # Getting the type of 'N' (line 510)
        N_161324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 510)
        ndarray_161325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 15), N_161324, 'ndarray')
        # Obtaining the member 'squeeze' of a type (line 510)
        squeeze_161326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 15), ndarray_161325, 'squeeze')
        # Calling squeeze(args, kwargs) (line 510)
        squeeze_call_result_161331 = invoke(stypy.reporting.localization.Localization(__file__, 510, 15), squeeze_161326, *[self_161327], **kwargs_161330)
        
        # Assigning a type to the variable 'stypy_return_type' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'stypy_return_type', squeeze_call_result_161331)
        
        # ################# End of 'squeeze(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'squeeze' in the type store
        # Getting the type of 'stypy_return_type' (line 461)
        stypy_return_type_161332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161332)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'squeeze'
        return stypy_return_type_161332


    @norecursion
    def flatten(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_161333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 28), 'str', 'C')
        defaults = [str_161333]
        # Create a new context for function 'flatten'
        module_type_store = module_type_store.open_function_context('flatten', 514, 4, False)
        # Assigning a type to the variable 'self' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.flatten.__dict__.__setitem__('stypy_localization', localization)
        matrix.flatten.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.flatten.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.flatten.__dict__.__setitem__('stypy_function_name', 'matrix.flatten')
        matrix.flatten.__dict__.__setitem__('stypy_param_names_list', ['order'])
        matrix.flatten.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.flatten.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.flatten.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.flatten.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.flatten.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.flatten.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.flatten', ['order'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'flatten', localization, ['order'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'flatten(...)' code ##################

        str_161334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, (-1)), 'str', "\n        Return a flattened copy of the matrix.\n\n        All `N` elements of the matrix are placed into a single row.\n\n        Parameters\n        ----------\n        order : {'C', 'F', 'A', 'K'}, optional\n            'C' means to flatten in row-major (C-style) order. 'F' means to\n            flatten in column-major (Fortran-style) order. 'A' means to\n            flatten in column-major order if `m` is Fortran *contiguous* in\n            memory, row-major order otherwise. 'K' means to flatten `m` in\n            the order the elements occur in memory. The default is 'C'.\n\n        Returns\n        -------\n        y : matrix\n            A copy of the matrix, flattened to a `(1, N)` matrix where `N`\n            is the number of elements in the original matrix.\n\n        See Also\n        --------\n        ravel : Return a flattened array.\n        flat : A 1-D flat iterator over the matrix.\n\n        Examples\n        --------\n        >>> m = np.matrix([[1,2], [3,4]])\n        >>> m.flatten()\n        matrix([[1, 2, 3, 4]])\n        >>> m.flatten('F')\n        matrix([[1, 3, 2, 4]])\n\n        ")
        
        # Call to flatten(...): (line 549)
        # Processing the call arguments (line 549)
        # Getting the type of 'self' (line 549)
        self_161338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 33), 'self', False)
        # Processing the call keyword arguments (line 549)
        # Getting the type of 'order' (line 549)
        order_161339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 45), 'order', False)
        keyword_161340 = order_161339
        kwargs_161341 = {'order': keyword_161340}
        # Getting the type of 'N' (line 549)
        N_161335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 549)
        ndarray_161336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 15), N_161335, 'ndarray')
        # Obtaining the member 'flatten' of a type (line 549)
        flatten_161337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 15), ndarray_161336, 'flatten')
        # Calling flatten(args, kwargs) (line 549)
        flatten_call_result_161342 = invoke(stypy.reporting.localization.Localization(__file__, 549, 15), flatten_161337, *[self_161338], **kwargs_161341)
        
        # Assigning a type to the variable 'stypy_return_type' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'stypy_return_type', flatten_call_result_161342)
        
        # ################# End of 'flatten(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'flatten' in the type store
        # Getting the type of 'stypy_return_type' (line 514)
        stypy_return_type_161343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161343)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'flatten'
        return stypy_return_type_161343


    @norecursion
    def mean(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 551)
        None_161344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 24), 'None')
        # Getting the type of 'None' (line 551)
        None_161345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 36), 'None')
        # Getting the type of 'None' (line 551)
        None_161346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 46), 'None')
        defaults = [None_161344, None_161345, None_161346]
        # Create a new context for function 'mean'
        module_type_store = module_type_store.open_function_context('mean', 551, 4, False)
        # Assigning a type to the variable 'self' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.mean.__dict__.__setitem__('stypy_localization', localization)
        matrix.mean.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.mean.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.mean.__dict__.__setitem__('stypy_function_name', 'matrix.mean')
        matrix.mean.__dict__.__setitem__('stypy_param_names_list', ['axis', 'dtype', 'out'])
        matrix.mean.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.mean.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.mean.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.mean.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.mean.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.mean.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.mean', ['axis', 'dtype', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mean', localization, ['axis', 'dtype', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mean(...)' code ##################

        str_161347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, (-1)), 'str', '\n        Returns the average of the matrix elements along the given axis.\n\n        Refer to `numpy.mean` for full documentation.\n\n        See Also\n        --------\n        numpy.mean\n\n        Notes\n        -----\n        Same as `ndarray.mean` except that, where that returns an `ndarray`,\n        this returns a `matrix` object.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3, 4)))\n        >>> x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.mean()\n        5.5\n        >>> x.mean(0)\n        matrix([[ 4.,  5.,  6.,  7.]])\n        >>> x.mean(1)\n        matrix([[ 1.5],\n                [ 5.5],\n                [ 9.5]])\n\n        ')
        
        # Call to _collapse(...): (line 583)
        # Processing the call arguments (line 583)
        # Getting the type of 'axis' (line 583)
        axis_161360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 79), 'axis', False)
        # Processing the call keyword arguments (line 583)
        kwargs_161361 = {}
        
        # Call to mean(...): (line 583)
        # Processing the call arguments (line 583)
        # Getting the type of 'self' (line 583)
        self_161351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 30), 'self', False)
        # Getting the type of 'axis' (line 583)
        axis_161352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 36), 'axis', False)
        # Getting the type of 'dtype' (line 583)
        dtype_161353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 42), 'dtype', False)
        # Getting the type of 'out' (line 583)
        out_161354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 49), 'out', False)
        # Processing the call keyword arguments (line 583)
        # Getting the type of 'True' (line 583)
        True_161355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 63), 'True', False)
        keyword_161356 = True_161355
        kwargs_161357 = {'keepdims': keyword_161356}
        # Getting the type of 'N' (line 583)
        N_161348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 583)
        ndarray_161349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 15), N_161348, 'ndarray')
        # Obtaining the member 'mean' of a type (line 583)
        mean_161350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 15), ndarray_161349, 'mean')
        # Calling mean(args, kwargs) (line 583)
        mean_call_result_161358 = invoke(stypy.reporting.localization.Localization(__file__, 583, 15), mean_161350, *[self_161351, axis_161352, dtype_161353, out_161354], **kwargs_161357)
        
        # Obtaining the member '_collapse' of a type (line 583)
        _collapse_161359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 15), mean_call_result_161358, '_collapse')
        # Calling _collapse(args, kwargs) (line 583)
        _collapse_call_result_161362 = invoke(stypy.reporting.localization.Localization(__file__, 583, 15), _collapse_161359, *[axis_161360], **kwargs_161361)
        
        # Assigning a type to the variable 'stypy_return_type' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'stypy_return_type', _collapse_call_result_161362)
        
        # ################# End of 'mean(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mean' in the type store
        # Getting the type of 'stypy_return_type' (line 551)
        stypy_return_type_161363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161363)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mean'
        return stypy_return_type_161363


    @norecursion
    def std(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 585)
        None_161364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 23), 'None')
        # Getting the type of 'None' (line 585)
        None_161365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 35), 'None')
        # Getting the type of 'None' (line 585)
        None_161366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 45), 'None')
        int_161367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 56), 'int')
        defaults = [None_161364, None_161365, None_161366, int_161367]
        # Create a new context for function 'std'
        module_type_store = module_type_store.open_function_context('std', 585, 4, False)
        # Assigning a type to the variable 'self' (line 586)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.std.__dict__.__setitem__('stypy_localization', localization)
        matrix.std.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.std.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.std.__dict__.__setitem__('stypy_function_name', 'matrix.std')
        matrix.std.__dict__.__setitem__('stypy_param_names_list', ['axis', 'dtype', 'out', 'ddof'])
        matrix.std.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.std.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.std.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.std.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.std.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.std.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.std', ['axis', 'dtype', 'out', 'ddof'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'std', localization, ['axis', 'dtype', 'out', 'ddof'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'std(...)' code ##################

        str_161368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, (-1)), 'str', '\n        Return the standard deviation of the array elements along the given axis.\n\n        Refer to `numpy.std` for full documentation.\n\n        See Also\n        --------\n        numpy.std\n\n        Notes\n        -----\n        This is the same as `ndarray.std`, except that where an `ndarray` would\n        be returned, a `matrix` object is returned instead.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3, 4)))\n        >>> x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.std()\n        3.4520525295346629\n        >>> x.std(0)\n        matrix([[ 3.26598632,  3.26598632,  3.26598632,  3.26598632]])\n        >>> x.std(1)\n        matrix([[ 1.11803399],\n                [ 1.11803399],\n                [ 1.11803399]])\n\n        ')
        
        # Call to _collapse(...): (line 617)
        # Processing the call arguments (line 617)
        # Getting the type of 'axis' (line 617)
        axis_161382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 84), 'axis', False)
        # Processing the call keyword arguments (line 617)
        kwargs_161383 = {}
        
        # Call to std(...): (line 617)
        # Processing the call arguments (line 617)
        # Getting the type of 'self' (line 617)
        self_161372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 29), 'self', False)
        # Getting the type of 'axis' (line 617)
        axis_161373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 35), 'axis', False)
        # Getting the type of 'dtype' (line 617)
        dtype_161374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 41), 'dtype', False)
        # Getting the type of 'out' (line 617)
        out_161375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 48), 'out', False)
        # Getting the type of 'ddof' (line 617)
        ddof_161376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 53), 'ddof', False)
        # Processing the call keyword arguments (line 617)
        # Getting the type of 'True' (line 617)
        True_161377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 68), 'True', False)
        keyword_161378 = True_161377
        kwargs_161379 = {'keepdims': keyword_161378}
        # Getting the type of 'N' (line 617)
        N_161369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 617)
        ndarray_161370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 15), N_161369, 'ndarray')
        # Obtaining the member 'std' of a type (line 617)
        std_161371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 15), ndarray_161370, 'std')
        # Calling std(args, kwargs) (line 617)
        std_call_result_161380 = invoke(stypy.reporting.localization.Localization(__file__, 617, 15), std_161371, *[self_161372, axis_161373, dtype_161374, out_161375, ddof_161376], **kwargs_161379)
        
        # Obtaining the member '_collapse' of a type (line 617)
        _collapse_161381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 15), std_call_result_161380, '_collapse')
        # Calling _collapse(args, kwargs) (line 617)
        _collapse_call_result_161384 = invoke(stypy.reporting.localization.Localization(__file__, 617, 15), _collapse_161381, *[axis_161382], **kwargs_161383)
        
        # Assigning a type to the variable 'stypy_return_type' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'stypy_return_type', _collapse_call_result_161384)
        
        # ################# End of 'std(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'std' in the type store
        # Getting the type of 'stypy_return_type' (line 585)
        stypy_return_type_161385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161385)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'std'
        return stypy_return_type_161385


    @norecursion
    def var(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 619)
        None_161386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 23), 'None')
        # Getting the type of 'None' (line 619)
        None_161387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 35), 'None')
        # Getting the type of 'None' (line 619)
        None_161388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 45), 'None')
        int_161389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 56), 'int')
        defaults = [None_161386, None_161387, None_161388, int_161389]
        # Create a new context for function 'var'
        module_type_store = module_type_store.open_function_context('var', 619, 4, False)
        # Assigning a type to the variable 'self' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.var.__dict__.__setitem__('stypy_localization', localization)
        matrix.var.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.var.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.var.__dict__.__setitem__('stypy_function_name', 'matrix.var')
        matrix.var.__dict__.__setitem__('stypy_param_names_list', ['axis', 'dtype', 'out', 'ddof'])
        matrix.var.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.var.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.var.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.var.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.var.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.var.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.var', ['axis', 'dtype', 'out', 'ddof'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'var', localization, ['axis', 'dtype', 'out', 'ddof'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'var(...)' code ##################

        str_161390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, (-1)), 'str', '\n        Returns the variance of the matrix elements, along the given axis.\n\n        Refer to `numpy.var` for full documentation.\n\n        See Also\n        --------\n        numpy.var\n\n        Notes\n        -----\n        This is the same as `ndarray.var`, except that where an `ndarray` would\n        be returned, a `matrix` object is returned instead.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3, 4)))\n        >>> x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.var()\n        11.916666666666666\n        >>> x.var(0)\n        matrix([[ 10.66666667,  10.66666667,  10.66666667,  10.66666667]])\n        >>> x.var(1)\n        matrix([[ 1.25],\n                [ 1.25],\n                [ 1.25]])\n\n        ')
        
        # Call to _collapse(...): (line 651)
        # Processing the call arguments (line 651)
        # Getting the type of 'axis' (line 651)
        axis_161404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 84), 'axis', False)
        # Processing the call keyword arguments (line 651)
        kwargs_161405 = {}
        
        # Call to var(...): (line 651)
        # Processing the call arguments (line 651)
        # Getting the type of 'self' (line 651)
        self_161394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 29), 'self', False)
        # Getting the type of 'axis' (line 651)
        axis_161395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 35), 'axis', False)
        # Getting the type of 'dtype' (line 651)
        dtype_161396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 41), 'dtype', False)
        # Getting the type of 'out' (line 651)
        out_161397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 48), 'out', False)
        # Getting the type of 'ddof' (line 651)
        ddof_161398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 53), 'ddof', False)
        # Processing the call keyword arguments (line 651)
        # Getting the type of 'True' (line 651)
        True_161399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 68), 'True', False)
        keyword_161400 = True_161399
        kwargs_161401 = {'keepdims': keyword_161400}
        # Getting the type of 'N' (line 651)
        N_161391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 651)
        ndarray_161392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 15), N_161391, 'ndarray')
        # Obtaining the member 'var' of a type (line 651)
        var_161393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 15), ndarray_161392, 'var')
        # Calling var(args, kwargs) (line 651)
        var_call_result_161402 = invoke(stypy.reporting.localization.Localization(__file__, 651, 15), var_161393, *[self_161394, axis_161395, dtype_161396, out_161397, ddof_161398], **kwargs_161401)
        
        # Obtaining the member '_collapse' of a type (line 651)
        _collapse_161403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 15), var_call_result_161402, '_collapse')
        # Calling _collapse(args, kwargs) (line 651)
        _collapse_call_result_161406 = invoke(stypy.reporting.localization.Localization(__file__, 651, 15), _collapse_161403, *[axis_161404], **kwargs_161405)
        
        # Assigning a type to the variable 'stypy_return_type' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 8), 'stypy_return_type', _collapse_call_result_161406)
        
        # ################# End of 'var(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'var' in the type store
        # Getting the type of 'stypy_return_type' (line 619)
        stypy_return_type_161407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161407)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'var'
        return stypy_return_type_161407


    @norecursion
    def prod(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 653)
        None_161408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 24), 'None')
        # Getting the type of 'None' (line 653)
        None_161409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 36), 'None')
        # Getting the type of 'None' (line 653)
        None_161410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 46), 'None')
        defaults = [None_161408, None_161409, None_161410]
        # Create a new context for function 'prod'
        module_type_store = module_type_store.open_function_context('prod', 653, 4, False)
        # Assigning a type to the variable 'self' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.prod.__dict__.__setitem__('stypy_localization', localization)
        matrix.prod.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.prod.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.prod.__dict__.__setitem__('stypy_function_name', 'matrix.prod')
        matrix.prod.__dict__.__setitem__('stypy_param_names_list', ['axis', 'dtype', 'out'])
        matrix.prod.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.prod.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.prod.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.prod.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.prod.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.prod.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.prod', ['axis', 'dtype', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'prod', localization, ['axis', 'dtype', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'prod(...)' code ##################

        str_161411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, (-1)), 'str', '\n        Return the product of the array elements over the given axis.\n\n        Refer to `prod` for full documentation.\n\n        See Also\n        --------\n        prod, ndarray.prod\n\n        Notes\n        -----\n        Same as `ndarray.prod`, except, where that returns an `ndarray`, this\n        returns a `matrix` object instead.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.prod()\n        0\n        >>> x.prod(0)\n        matrix([[  0,  45, 120, 231]])\n        >>> x.prod(1)\n        matrix([[   0],\n                [ 840],\n                [7920]])\n\n        ')
        
        # Call to _collapse(...): (line 684)
        # Processing the call arguments (line 684)
        # Getting the type of 'axis' (line 684)
        axis_161424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 79), 'axis', False)
        # Processing the call keyword arguments (line 684)
        kwargs_161425 = {}
        
        # Call to prod(...): (line 684)
        # Processing the call arguments (line 684)
        # Getting the type of 'self' (line 684)
        self_161415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 30), 'self', False)
        # Getting the type of 'axis' (line 684)
        axis_161416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 36), 'axis', False)
        # Getting the type of 'dtype' (line 684)
        dtype_161417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 42), 'dtype', False)
        # Getting the type of 'out' (line 684)
        out_161418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 49), 'out', False)
        # Processing the call keyword arguments (line 684)
        # Getting the type of 'True' (line 684)
        True_161419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 63), 'True', False)
        keyword_161420 = True_161419
        kwargs_161421 = {'keepdims': keyword_161420}
        # Getting the type of 'N' (line 684)
        N_161412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 684)
        ndarray_161413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 15), N_161412, 'ndarray')
        # Obtaining the member 'prod' of a type (line 684)
        prod_161414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 15), ndarray_161413, 'prod')
        # Calling prod(args, kwargs) (line 684)
        prod_call_result_161422 = invoke(stypy.reporting.localization.Localization(__file__, 684, 15), prod_161414, *[self_161415, axis_161416, dtype_161417, out_161418], **kwargs_161421)
        
        # Obtaining the member '_collapse' of a type (line 684)
        _collapse_161423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 15), prod_call_result_161422, '_collapse')
        # Calling _collapse(args, kwargs) (line 684)
        _collapse_call_result_161426 = invoke(stypy.reporting.localization.Localization(__file__, 684, 15), _collapse_161423, *[axis_161424], **kwargs_161425)
        
        # Assigning a type to the variable 'stypy_return_type' (line 684)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'stypy_return_type', _collapse_call_result_161426)
        
        # ################# End of 'prod(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'prod' in the type store
        # Getting the type of 'stypy_return_type' (line 653)
        stypy_return_type_161427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161427)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'prod'
        return stypy_return_type_161427


    @norecursion
    def any(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 686)
        None_161428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 23), 'None')
        # Getting the type of 'None' (line 686)
        None_161429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 33), 'None')
        defaults = [None_161428, None_161429]
        # Create a new context for function 'any'
        module_type_store = module_type_store.open_function_context('any', 686, 4, False)
        # Assigning a type to the variable 'self' (line 687)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.any.__dict__.__setitem__('stypy_localization', localization)
        matrix.any.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.any.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.any.__dict__.__setitem__('stypy_function_name', 'matrix.any')
        matrix.any.__dict__.__setitem__('stypy_param_names_list', ['axis', 'out'])
        matrix.any.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.any.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.any.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.any.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.any.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.any.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.any', ['axis', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'any', localization, ['axis', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'any(...)' code ##################

        str_161430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, (-1)), 'str', '\n        Test whether any array element along a given axis evaluates to True.\n\n        Refer to `numpy.any` for full documentation.\n\n        Parameters\n        ----------\n        axis : int, optional\n            Axis along which logical OR is performed\n        out : ndarray, optional\n            Output to existing array instead of creating new one, must have\n            same shape as expected output\n\n        Returns\n        -------\n            any : bool, ndarray\n                Returns a single bool if `axis` is ``None``; otherwise,\n                returns `ndarray`\n\n        ')
        
        # Call to _collapse(...): (line 707)
        # Processing the call arguments (line 707)
        # Getting the type of 'axis' (line 707)
        axis_161442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 71), 'axis', False)
        # Processing the call keyword arguments (line 707)
        kwargs_161443 = {}
        
        # Call to any(...): (line 707)
        # Processing the call arguments (line 707)
        # Getting the type of 'self' (line 707)
        self_161434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 29), 'self', False)
        # Getting the type of 'axis' (line 707)
        axis_161435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 35), 'axis', False)
        # Getting the type of 'out' (line 707)
        out_161436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 41), 'out', False)
        # Processing the call keyword arguments (line 707)
        # Getting the type of 'True' (line 707)
        True_161437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 55), 'True', False)
        keyword_161438 = True_161437
        kwargs_161439 = {'keepdims': keyword_161438}
        # Getting the type of 'N' (line 707)
        N_161431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 707)
        ndarray_161432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 15), N_161431, 'ndarray')
        # Obtaining the member 'any' of a type (line 707)
        any_161433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 15), ndarray_161432, 'any')
        # Calling any(args, kwargs) (line 707)
        any_call_result_161440 = invoke(stypy.reporting.localization.Localization(__file__, 707, 15), any_161433, *[self_161434, axis_161435, out_161436], **kwargs_161439)
        
        # Obtaining the member '_collapse' of a type (line 707)
        _collapse_161441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 15), any_call_result_161440, '_collapse')
        # Calling _collapse(args, kwargs) (line 707)
        _collapse_call_result_161444 = invoke(stypy.reporting.localization.Localization(__file__, 707, 15), _collapse_161441, *[axis_161442], **kwargs_161443)
        
        # Assigning a type to the variable 'stypy_return_type' (line 707)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'stypy_return_type', _collapse_call_result_161444)
        
        # ################# End of 'any(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'any' in the type store
        # Getting the type of 'stypy_return_type' (line 686)
        stypy_return_type_161445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161445)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'any'
        return stypy_return_type_161445


    @norecursion
    def all(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 709)
        None_161446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 23), 'None')
        # Getting the type of 'None' (line 709)
        None_161447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 33), 'None')
        defaults = [None_161446, None_161447]
        # Create a new context for function 'all'
        module_type_store = module_type_store.open_function_context('all', 709, 4, False)
        # Assigning a type to the variable 'self' (line 710)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.all.__dict__.__setitem__('stypy_localization', localization)
        matrix.all.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.all.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.all.__dict__.__setitem__('stypy_function_name', 'matrix.all')
        matrix.all.__dict__.__setitem__('stypy_param_names_list', ['axis', 'out'])
        matrix.all.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.all.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.all.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.all.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.all.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.all.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.all', ['axis', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'all', localization, ['axis', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'all(...)' code ##################

        str_161448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, (-1)), 'str', '\n        Test whether all matrix elements along a given axis evaluate to True.\n\n        Parameters\n        ----------\n        See `numpy.all` for complete descriptions\n\n        See Also\n        --------\n        numpy.all\n\n        Notes\n        -----\n        This is the same as `ndarray.all`, but it returns a `matrix` object.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> y = x[0]; y\n        matrix([[0, 1, 2, 3]])\n        >>> (x == y)\n        matrix([[ True,  True,  True,  True],\n                [False, False, False, False],\n                [False, False, False, False]], dtype=bool)\n        >>> (x == y).all()\n        False\n        >>> (x == y).all(0)\n        matrix([[False, False, False, False]], dtype=bool)\n        >>> (x == y).all(1)\n        matrix([[ True],\n                [False],\n                [False]], dtype=bool)\n\n        ')
        
        # Call to _collapse(...): (line 747)
        # Processing the call arguments (line 747)
        # Getting the type of 'axis' (line 747)
        axis_161460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 71), 'axis', False)
        # Processing the call keyword arguments (line 747)
        kwargs_161461 = {}
        
        # Call to all(...): (line 747)
        # Processing the call arguments (line 747)
        # Getting the type of 'self' (line 747)
        self_161452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 29), 'self', False)
        # Getting the type of 'axis' (line 747)
        axis_161453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 35), 'axis', False)
        # Getting the type of 'out' (line 747)
        out_161454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 41), 'out', False)
        # Processing the call keyword arguments (line 747)
        # Getting the type of 'True' (line 747)
        True_161455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 55), 'True', False)
        keyword_161456 = True_161455
        kwargs_161457 = {'keepdims': keyword_161456}
        # Getting the type of 'N' (line 747)
        N_161449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 747)
        ndarray_161450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 15), N_161449, 'ndarray')
        # Obtaining the member 'all' of a type (line 747)
        all_161451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 15), ndarray_161450, 'all')
        # Calling all(args, kwargs) (line 747)
        all_call_result_161458 = invoke(stypy.reporting.localization.Localization(__file__, 747, 15), all_161451, *[self_161452, axis_161453, out_161454], **kwargs_161457)
        
        # Obtaining the member '_collapse' of a type (line 747)
        _collapse_161459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 15), all_call_result_161458, '_collapse')
        # Calling _collapse(args, kwargs) (line 747)
        _collapse_call_result_161462 = invoke(stypy.reporting.localization.Localization(__file__, 747, 15), _collapse_161459, *[axis_161460], **kwargs_161461)
        
        # Assigning a type to the variable 'stypy_return_type' (line 747)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 8), 'stypy_return_type', _collapse_call_result_161462)
        
        # ################# End of 'all(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'all' in the type store
        # Getting the type of 'stypy_return_type' (line 709)
        stypy_return_type_161463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161463)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'all'
        return stypy_return_type_161463


    @norecursion
    def max(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 749)
        None_161464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 23), 'None')
        # Getting the type of 'None' (line 749)
        None_161465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 33), 'None')
        defaults = [None_161464, None_161465]
        # Create a new context for function 'max'
        module_type_store = module_type_store.open_function_context('max', 749, 4, False)
        # Assigning a type to the variable 'self' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.max.__dict__.__setitem__('stypy_localization', localization)
        matrix.max.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.max.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.max.__dict__.__setitem__('stypy_function_name', 'matrix.max')
        matrix.max.__dict__.__setitem__('stypy_param_names_list', ['axis', 'out'])
        matrix.max.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.max.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.max.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.max.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.max.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.max.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.max', ['axis', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'max', localization, ['axis', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'max(...)' code ##################

        str_161466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, (-1)), 'str', '\n        Return the maximum value along an axis.\n\n        Parameters\n        ----------\n        See `amax` for complete descriptions\n\n        See Also\n        --------\n        amax, ndarray.max\n\n        Notes\n        -----\n        This is the same as `ndarray.max`, but returns a `matrix` object\n        where `ndarray.max` would return an ndarray.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.max()\n        11\n        >>> x.max(0)\n        matrix([[ 8,  9, 10, 11]])\n        >>> x.max(1)\n        matrix([[ 3],\n                [ 7],\n                [11]])\n\n        ')
        
        # Call to _collapse(...): (line 782)
        # Processing the call arguments (line 782)
        # Getting the type of 'axis' (line 782)
        axis_161478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 71), 'axis', False)
        # Processing the call keyword arguments (line 782)
        kwargs_161479 = {}
        
        # Call to max(...): (line 782)
        # Processing the call arguments (line 782)
        # Getting the type of 'self' (line 782)
        self_161470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 29), 'self', False)
        # Getting the type of 'axis' (line 782)
        axis_161471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 35), 'axis', False)
        # Getting the type of 'out' (line 782)
        out_161472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 41), 'out', False)
        # Processing the call keyword arguments (line 782)
        # Getting the type of 'True' (line 782)
        True_161473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 55), 'True', False)
        keyword_161474 = True_161473
        kwargs_161475 = {'keepdims': keyword_161474}
        # Getting the type of 'N' (line 782)
        N_161467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 782)
        ndarray_161468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 15), N_161467, 'ndarray')
        # Obtaining the member 'max' of a type (line 782)
        max_161469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 15), ndarray_161468, 'max')
        # Calling max(args, kwargs) (line 782)
        max_call_result_161476 = invoke(stypy.reporting.localization.Localization(__file__, 782, 15), max_161469, *[self_161470, axis_161471, out_161472], **kwargs_161475)
        
        # Obtaining the member '_collapse' of a type (line 782)
        _collapse_161477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 15), max_call_result_161476, '_collapse')
        # Calling _collapse(args, kwargs) (line 782)
        _collapse_call_result_161480 = invoke(stypy.reporting.localization.Localization(__file__, 782, 15), _collapse_161477, *[axis_161478], **kwargs_161479)
        
        # Assigning a type to the variable 'stypy_return_type' (line 782)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 8), 'stypy_return_type', _collapse_call_result_161480)
        
        # ################# End of 'max(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'max' in the type store
        # Getting the type of 'stypy_return_type' (line 749)
        stypy_return_type_161481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161481)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'max'
        return stypy_return_type_161481


    @norecursion
    def argmax(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 784)
        None_161482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 26), 'None')
        # Getting the type of 'None' (line 784)
        None_161483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 36), 'None')
        defaults = [None_161482, None_161483]
        # Create a new context for function 'argmax'
        module_type_store = module_type_store.open_function_context('argmax', 784, 4, False)
        # Assigning a type to the variable 'self' (line 785)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.argmax.__dict__.__setitem__('stypy_localization', localization)
        matrix.argmax.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.argmax.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.argmax.__dict__.__setitem__('stypy_function_name', 'matrix.argmax')
        matrix.argmax.__dict__.__setitem__('stypy_param_names_list', ['axis', 'out'])
        matrix.argmax.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.argmax.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.argmax.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.argmax.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.argmax.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.argmax.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.argmax', ['axis', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'argmax', localization, ['axis', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'argmax(...)' code ##################

        str_161484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, (-1)), 'str', '\n        Indexes of the maximum values along an axis.\n\n        Return the indexes of the first occurrences of the maximum values\n        along the specified axis.  If axis is None, the index is for the\n        flattened matrix.\n\n        Parameters\n        ----------\n        See `numpy.argmax` for complete descriptions\n\n        See Also\n        --------\n        numpy.argmax\n\n        Notes\n        -----\n        This is the same as `ndarray.argmax`, but returns a `matrix` object\n        where `ndarray.argmax` would return an `ndarray`.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.argmax()\n        11\n        >>> x.argmax(0)\n        matrix([[2, 2, 2, 2]])\n        >>> x.argmax(1)\n        matrix([[3],\n                [3],\n                [3]])\n\n        ')
        
        # Call to _align(...): (line 821)
        # Processing the call arguments (line 821)
        # Getting the type of 'axis' (line 821)
        axis_161494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 56), 'axis', False)
        # Processing the call keyword arguments (line 821)
        kwargs_161495 = {}
        
        # Call to argmax(...): (line 821)
        # Processing the call arguments (line 821)
        # Getting the type of 'self' (line 821)
        self_161488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 32), 'self', False)
        # Getting the type of 'axis' (line 821)
        axis_161489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 38), 'axis', False)
        # Getting the type of 'out' (line 821)
        out_161490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 44), 'out', False)
        # Processing the call keyword arguments (line 821)
        kwargs_161491 = {}
        # Getting the type of 'N' (line 821)
        N_161485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 821)
        ndarray_161486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 15), N_161485, 'ndarray')
        # Obtaining the member 'argmax' of a type (line 821)
        argmax_161487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 15), ndarray_161486, 'argmax')
        # Calling argmax(args, kwargs) (line 821)
        argmax_call_result_161492 = invoke(stypy.reporting.localization.Localization(__file__, 821, 15), argmax_161487, *[self_161488, axis_161489, out_161490], **kwargs_161491)
        
        # Obtaining the member '_align' of a type (line 821)
        _align_161493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 15), argmax_call_result_161492, '_align')
        # Calling _align(args, kwargs) (line 821)
        _align_call_result_161496 = invoke(stypy.reporting.localization.Localization(__file__, 821, 15), _align_161493, *[axis_161494], **kwargs_161495)
        
        # Assigning a type to the variable 'stypy_return_type' (line 821)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 8), 'stypy_return_type', _align_call_result_161496)
        
        # ################# End of 'argmax(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'argmax' in the type store
        # Getting the type of 'stypy_return_type' (line 784)
        stypy_return_type_161497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161497)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'argmax'
        return stypy_return_type_161497


    @norecursion
    def min(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 823)
        None_161498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 23), 'None')
        # Getting the type of 'None' (line 823)
        None_161499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 33), 'None')
        defaults = [None_161498, None_161499]
        # Create a new context for function 'min'
        module_type_store = module_type_store.open_function_context('min', 823, 4, False)
        # Assigning a type to the variable 'self' (line 824)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.min.__dict__.__setitem__('stypy_localization', localization)
        matrix.min.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.min.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.min.__dict__.__setitem__('stypy_function_name', 'matrix.min')
        matrix.min.__dict__.__setitem__('stypy_param_names_list', ['axis', 'out'])
        matrix.min.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.min.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.min.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.min.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.min.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.min.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.min', ['axis', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'min', localization, ['axis', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'min(...)' code ##################

        str_161500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, (-1)), 'str', '\n        Return the minimum value along an axis.\n\n        Parameters\n        ----------\n        See `amin` for complete descriptions.\n\n        See Also\n        --------\n        amin, ndarray.min\n\n        Notes\n        -----\n        This is the same as `ndarray.min`, but returns a `matrix` object\n        where `ndarray.min` would return an ndarray.\n\n        Examples\n        --------\n        >>> x = -np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[  0,  -1,  -2,  -3],\n                [ -4,  -5,  -6,  -7],\n                [ -8,  -9, -10, -11]])\n        >>> x.min()\n        -11\n        >>> x.min(0)\n        matrix([[ -8,  -9, -10, -11]])\n        >>> x.min(1)\n        matrix([[ -3],\n                [ -7],\n                [-11]])\n\n        ')
        
        # Call to _collapse(...): (line 856)
        # Processing the call arguments (line 856)
        # Getting the type of 'axis' (line 856)
        axis_161512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 71), 'axis', False)
        # Processing the call keyword arguments (line 856)
        kwargs_161513 = {}
        
        # Call to min(...): (line 856)
        # Processing the call arguments (line 856)
        # Getting the type of 'self' (line 856)
        self_161504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 29), 'self', False)
        # Getting the type of 'axis' (line 856)
        axis_161505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 35), 'axis', False)
        # Getting the type of 'out' (line 856)
        out_161506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 41), 'out', False)
        # Processing the call keyword arguments (line 856)
        # Getting the type of 'True' (line 856)
        True_161507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 55), 'True', False)
        keyword_161508 = True_161507
        kwargs_161509 = {'keepdims': keyword_161508}
        # Getting the type of 'N' (line 856)
        N_161501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 856)
        ndarray_161502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 15), N_161501, 'ndarray')
        # Obtaining the member 'min' of a type (line 856)
        min_161503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 15), ndarray_161502, 'min')
        # Calling min(args, kwargs) (line 856)
        min_call_result_161510 = invoke(stypy.reporting.localization.Localization(__file__, 856, 15), min_161503, *[self_161504, axis_161505, out_161506], **kwargs_161509)
        
        # Obtaining the member '_collapse' of a type (line 856)
        _collapse_161511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 15), min_call_result_161510, '_collapse')
        # Calling _collapse(args, kwargs) (line 856)
        _collapse_call_result_161514 = invoke(stypy.reporting.localization.Localization(__file__, 856, 15), _collapse_161511, *[axis_161512], **kwargs_161513)
        
        # Assigning a type to the variable 'stypy_return_type' (line 856)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 8), 'stypy_return_type', _collapse_call_result_161514)
        
        # ################# End of 'min(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'min' in the type store
        # Getting the type of 'stypy_return_type' (line 823)
        stypy_return_type_161515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161515)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'min'
        return stypy_return_type_161515


    @norecursion
    def argmin(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 858)
        None_161516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 26), 'None')
        # Getting the type of 'None' (line 858)
        None_161517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 36), 'None')
        defaults = [None_161516, None_161517]
        # Create a new context for function 'argmin'
        module_type_store = module_type_store.open_function_context('argmin', 858, 4, False)
        # Assigning a type to the variable 'self' (line 859)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.argmin.__dict__.__setitem__('stypy_localization', localization)
        matrix.argmin.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.argmin.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.argmin.__dict__.__setitem__('stypy_function_name', 'matrix.argmin')
        matrix.argmin.__dict__.__setitem__('stypy_param_names_list', ['axis', 'out'])
        matrix.argmin.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.argmin.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.argmin.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.argmin.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.argmin.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.argmin.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.argmin', ['axis', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'argmin', localization, ['axis', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'argmin(...)' code ##################

        str_161518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, (-1)), 'str', '\n        Indexes of the minimum values along an axis.\n\n        Return the indexes of the first occurrences of the minimum values\n        along the specified axis.  If axis is None, the index is for the\n        flattened matrix.\n\n        Parameters\n        ----------\n        See `numpy.argmin` for complete descriptions.\n\n        See Also\n        --------\n        numpy.argmin\n\n        Notes\n        -----\n        This is the same as `ndarray.argmin`, but returns a `matrix` object\n        where `ndarray.argmin` would return an `ndarray`.\n\n        Examples\n        --------\n        >>> x = -np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[  0,  -1,  -2,  -3],\n                [ -4,  -5,  -6,  -7],\n                [ -8,  -9, -10, -11]])\n        >>> x.argmin()\n        11\n        >>> x.argmin(0)\n        matrix([[2, 2, 2, 2]])\n        >>> x.argmin(1)\n        matrix([[3],\n                [3],\n                [3]])\n\n        ')
        
        # Call to _align(...): (line 895)
        # Processing the call arguments (line 895)
        # Getting the type of 'axis' (line 895)
        axis_161528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 56), 'axis', False)
        # Processing the call keyword arguments (line 895)
        kwargs_161529 = {}
        
        # Call to argmin(...): (line 895)
        # Processing the call arguments (line 895)
        # Getting the type of 'self' (line 895)
        self_161522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 32), 'self', False)
        # Getting the type of 'axis' (line 895)
        axis_161523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 38), 'axis', False)
        # Getting the type of 'out' (line 895)
        out_161524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 44), 'out', False)
        # Processing the call keyword arguments (line 895)
        kwargs_161525 = {}
        # Getting the type of 'N' (line 895)
        N_161519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 895)
        ndarray_161520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 15), N_161519, 'ndarray')
        # Obtaining the member 'argmin' of a type (line 895)
        argmin_161521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 15), ndarray_161520, 'argmin')
        # Calling argmin(args, kwargs) (line 895)
        argmin_call_result_161526 = invoke(stypy.reporting.localization.Localization(__file__, 895, 15), argmin_161521, *[self_161522, axis_161523, out_161524], **kwargs_161525)
        
        # Obtaining the member '_align' of a type (line 895)
        _align_161527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 15), argmin_call_result_161526, '_align')
        # Calling _align(args, kwargs) (line 895)
        _align_call_result_161530 = invoke(stypy.reporting.localization.Localization(__file__, 895, 15), _align_161527, *[axis_161528], **kwargs_161529)
        
        # Assigning a type to the variable 'stypy_return_type' (line 895)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 8), 'stypy_return_type', _align_call_result_161530)
        
        # ################# End of 'argmin(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'argmin' in the type store
        # Getting the type of 'stypy_return_type' (line 858)
        stypy_return_type_161531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161531)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'argmin'
        return stypy_return_type_161531


    @norecursion
    def ptp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 897)
        None_161532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 23), 'None')
        # Getting the type of 'None' (line 897)
        None_161533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 33), 'None')
        defaults = [None_161532, None_161533]
        # Create a new context for function 'ptp'
        module_type_store = module_type_store.open_function_context('ptp', 897, 4, False)
        # Assigning a type to the variable 'self' (line 898)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 898, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.ptp.__dict__.__setitem__('stypy_localization', localization)
        matrix.ptp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.ptp.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.ptp.__dict__.__setitem__('stypy_function_name', 'matrix.ptp')
        matrix.ptp.__dict__.__setitem__('stypy_param_names_list', ['axis', 'out'])
        matrix.ptp.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.ptp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.ptp.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.ptp.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.ptp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.ptp.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.ptp', ['axis', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ptp', localization, ['axis', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ptp(...)' code ##################

        str_161534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 927, (-1)), 'str', '\n        Peak-to-peak (maximum - minimum) value along the given axis.\n\n        Refer to `numpy.ptp` for full documentation.\n\n        See Also\n        --------\n        numpy.ptp\n\n        Notes\n        -----\n        Same as `ndarray.ptp`, except, where that would return an `ndarray` object,\n        this returns a `matrix` object.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.ptp()\n        11\n        >>> x.ptp(0)\n        matrix([[8, 8, 8, 8]])\n        >>> x.ptp(1)\n        matrix([[3],\n                [3],\n                [3]])\n\n        ')
        
        # Call to _align(...): (line 928)
        # Processing the call arguments (line 928)
        # Getting the type of 'axis' (line 928)
        axis_161544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 53), 'axis', False)
        # Processing the call keyword arguments (line 928)
        kwargs_161545 = {}
        
        # Call to ptp(...): (line 928)
        # Processing the call arguments (line 928)
        # Getting the type of 'self' (line 928)
        self_161538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 29), 'self', False)
        # Getting the type of 'axis' (line 928)
        axis_161539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 35), 'axis', False)
        # Getting the type of 'out' (line 928)
        out_161540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 41), 'out', False)
        # Processing the call keyword arguments (line 928)
        kwargs_161541 = {}
        # Getting the type of 'N' (line 928)
        N_161535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 928)
        ndarray_161536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 15), N_161535, 'ndarray')
        # Obtaining the member 'ptp' of a type (line 928)
        ptp_161537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 15), ndarray_161536, 'ptp')
        # Calling ptp(args, kwargs) (line 928)
        ptp_call_result_161542 = invoke(stypy.reporting.localization.Localization(__file__, 928, 15), ptp_161537, *[self_161538, axis_161539, out_161540], **kwargs_161541)
        
        # Obtaining the member '_align' of a type (line 928)
        _align_161543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 15), ptp_call_result_161542, '_align')
        # Calling _align(args, kwargs) (line 928)
        _align_call_result_161546 = invoke(stypy.reporting.localization.Localization(__file__, 928, 15), _align_161543, *[axis_161544], **kwargs_161545)
        
        # Assigning a type to the variable 'stypy_return_type' (line 928)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 928, 8), 'stypy_return_type', _align_call_result_161546)
        
        # ################# End of 'ptp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ptp' in the type store
        # Getting the type of 'stypy_return_type' (line 897)
        stypy_return_type_161547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161547)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ptp'
        return stypy_return_type_161547


    @norecursion
    def getI(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getI'
        module_type_store = module_type_store.open_function_context('getI', 930, 4, False)
        # Assigning a type to the variable 'self' (line 931)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.getI.__dict__.__setitem__('stypy_localization', localization)
        matrix.getI.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.getI.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.getI.__dict__.__setitem__('stypy_function_name', 'matrix.getI')
        matrix.getI.__dict__.__setitem__('stypy_param_names_list', [])
        matrix.getI.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.getI.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.getI.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.getI.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.getI.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.getI.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.getI', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getI', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getI(...)' code ##################

        str_161548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, (-1)), 'str', "\n        Returns the (multiplicative) inverse of invertible `self`.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        ret : matrix object\n            If `self` is non-singular, `ret` is such that ``ret * self`` ==\n            ``self * ret`` == ``np.matrix(np.eye(self[0,:].size)`` all return\n            ``True``.\n\n        Raises\n        ------\n        numpy.linalg.LinAlgError: Singular matrix\n            If `self` is singular.\n\n        See Also\n        --------\n        linalg.inv\n\n        Examples\n        --------\n        >>> m = np.matrix('[1, 2; 3, 4]'); m\n        matrix([[1, 2],\n                [3, 4]])\n        >>> m.getI()\n        matrix([[-2. ,  1. ],\n                [ 1.5, -0.5]])\n        >>> m.getI() * m\n        matrix([[ 1.,  0.],\n                [ 0.,  1.]])\n\n        ")
        
        # Assigning a Attribute to a Tuple (line 967):
        
        # Assigning a Subscript to a Name (line 967):
        
        # Obtaining the type of the subscript
        int_161549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 8), 'int')
        # Getting the type of 'self' (line 967)
        self_161550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 15), 'self')
        # Obtaining the member 'shape' of a type (line 967)
        shape_161551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 967, 15), self_161550, 'shape')
        # Obtaining the member '__getitem__' of a type (line 967)
        getitem___161552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 967, 8), shape_161551, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 967)
        subscript_call_result_161553 = invoke(stypy.reporting.localization.Localization(__file__, 967, 8), getitem___161552, int_161549)
        
        # Assigning a type to the variable 'tuple_var_assignment_160496' (line 967)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 8), 'tuple_var_assignment_160496', subscript_call_result_161553)
        
        # Assigning a Subscript to a Name (line 967):
        
        # Obtaining the type of the subscript
        int_161554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 8), 'int')
        # Getting the type of 'self' (line 967)
        self_161555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 15), 'self')
        # Obtaining the member 'shape' of a type (line 967)
        shape_161556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 967, 15), self_161555, 'shape')
        # Obtaining the member '__getitem__' of a type (line 967)
        getitem___161557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 967, 8), shape_161556, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 967)
        subscript_call_result_161558 = invoke(stypy.reporting.localization.Localization(__file__, 967, 8), getitem___161557, int_161554)
        
        # Assigning a type to the variable 'tuple_var_assignment_160497' (line 967)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 8), 'tuple_var_assignment_160497', subscript_call_result_161558)
        
        # Assigning a Name to a Name (line 967):
        # Getting the type of 'tuple_var_assignment_160496' (line 967)
        tuple_var_assignment_160496_161559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 8), 'tuple_var_assignment_160496')
        # Assigning a type to the variable 'M' (line 967)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 8), 'M', tuple_var_assignment_160496_161559)
        
        # Assigning a Name to a Name (line 967):
        # Getting the type of 'tuple_var_assignment_160497' (line 967)
        tuple_var_assignment_160497_161560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 8), 'tuple_var_assignment_160497')
        # Assigning a type to the variable 'N' (line 967)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 11), 'N', tuple_var_assignment_160497_161560)
        
        
        # Getting the type of 'M' (line 968)
        M_161561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 11), 'M')
        # Getting the type of 'N' (line 968)
        N_161562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 16), 'N')
        # Applying the binary operator '==' (line 968)
        result_eq_161563 = python_operator(stypy.reporting.localization.Localization(__file__, 968, 11), '==', M_161561, N_161562)
        
        # Testing the type of an if condition (line 968)
        if_condition_161564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 968, 8), result_eq_161563)
        # Assigning a type to the variable 'if_condition_161564' (line 968)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 968, 8), 'if_condition_161564', if_condition_161564)
        # SSA begins for if statement (line 968)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 969, 12))
        
        # 'from numpy.dual import func' statement (line 969)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/matrixlib/')
        import_161565 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 969, 12), 'numpy.dual')

        if (type(import_161565) is not StypyTypeError):

            if (import_161565 != 'pyd_module'):
                __import__(import_161565)
                sys_modules_161566 = sys.modules[import_161565]
                import_from_module(stypy.reporting.localization.Localization(__file__, 969, 12), 'numpy.dual', sys_modules_161566.module_type_store, module_type_store, ['inv'])
                nest_module(stypy.reporting.localization.Localization(__file__, 969, 12), __file__, sys_modules_161566, sys_modules_161566.module_type_store, module_type_store)
            else:
                from numpy.dual import inv as func

                import_from_module(stypy.reporting.localization.Localization(__file__, 969, 12), 'numpy.dual', None, module_type_store, ['inv'], [func])

        else:
            # Assigning a type to the variable 'numpy.dual' (line 969)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 12), 'numpy.dual', import_161565)

        # Adding an alias
        module_type_store.add_alias('func', 'inv')
        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/matrixlib/')
        
        # SSA branch for the else part of an if statement (line 968)
        module_type_store.open_ssa_branch('else')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 971, 12))
        
        # 'from numpy.dual import func' statement (line 971)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/matrixlib/')
        import_161567 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 971, 12), 'numpy.dual')

        if (type(import_161567) is not StypyTypeError):

            if (import_161567 != 'pyd_module'):
                __import__(import_161567)
                sys_modules_161568 = sys.modules[import_161567]
                import_from_module(stypy.reporting.localization.Localization(__file__, 971, 12), 'numpy.dual', sys_modules_161568.module_type_store, module_type_store, ['pinv'])
                nest_module(stypy.reporting.localization.Localization(__file__, 971, 12), __file__, sys_modules_161568, sys_modules_161568.module_type_store, module_type_store)
            else:
                from numpy.dual import pinv as func

                import_from_module(stypy.reporting.localization.Localization(__file__, 971, 12), 'numpy.dual', None, module_type_store, ['pinv'], [func])

        else:
            # Assigning a type to the variable 'numpy.dual' (line 971)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 971, 12), 'numpy.dual', import_161567)

        # Adding an alias
        module_type_store.add_alias('func', 'pinv')
        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/matrixlib/')
        
        # SSA join for if statement (line 968)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to asmatrix(...): (line 972)
        # Processing the call arguments (line 972)
        
        # Call to func(...): (line 972)
        # Processing the call arguments (line 972)
        # Getting the type of 'self' (line 972)
        self_161571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 29), 'self', False)
        # Processing the call keyword arguments (line 972)
        kwargs_161572 = {}
        # Getting the type of 'func' (line 972)
        func_161570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 24), 'func', False)
        # Calling func(args, kwargs) (line 972)
        func_call_result_161573 = invoke(stypy.reporting.localization.Localization(__file__, 972, 24), func_161570, *[self_161571], **kwargs_161572)
        
        # Processing the call keyword arguments (line 972)
        kwargs_161574 = {}
        # Getting the type of 'asmatrix' (line 972)
        asmatrix_161569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 15), 'asmatrix', False)
        # Calling asmatrix(args, kwargs) (line 972)
        asmatrix_call_result_161575 = invoke(stypy.reporting.localization.Localization(__file__, 972, 15), asmatrix_161569, *[func_call_result_161573], **kwargs_161574)
        
        # Assigning a type to the variable 'stypy_return_type' (line 972)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 8), 'stypy_return_type', asmatrix_call_result_161575)
        
        # ################# End of 'getI(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getI' in the type store
        # Getting the type of 'stypy_return_type' (line 930)
        stypy_return_type_161576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161576)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getI'
        return stypy_return_type_161576


    @norecursion
    def getA(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getA'
        module_type_store = module_type_store.open_function_context('getA', 974, 4, False)
        # Assigning a type to the variable 'self' (line 975)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.getA.__dict__.__setitem__('stypy_localization', localization)
        matrix.getA.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.getA.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.getA.__dict__.__setitem__('stypy_function_name', 'matrix.getA')
        matrix.getA.__dict__.__setitem__('stypy_param_names_list', [])
        matrix.getA.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.getA.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.getA.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.getA.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.getA.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.getA.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.getA', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getA', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getA(...)' code ##################

        str_161577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1000, (-1)), 'str', '\n        Return `self` as an `ndarray` object.\n\n        Equivalent to ``np.asarray(self)``.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        ret : ndarray\n            `self` as an `ndarray`\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.getA()\n        array([[ 0,  1,  2,  3],\n               [ 4,  5,  6,  7],\n               [ 8,  9, 10, 11]])\n\n        ')
        
        # Call to __array__(...): (line 1001)
        # Processing the call keyword arguments (line 1001)
        kwargs_161580 = {}
        # Getting the type of 'self' (line 1001)
        self_161578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 15), 'self', False)
        # Obtaining the member '__array__' of a type (line 1001)
        array___161579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 15), self_161578, '__array__')
        # Calling __array__(args, kwargs) (line 1001)
        array___call_result_161581 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 15), array___161579, *[], **kwargs_161580)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1001)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1001, 8), 'stypy_return_type', array___call_result_161581)
        
        # ################# End of 'getA(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getA' in the type store
        # Getting the type of 'stypy_return_type' (line 974)
        stypy_return_type_161582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161582)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getA'
        return stypy_return_type_161582


    @norecursion
    def getA1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getA1'
        module_type_store = module_type_store.open_function_context('getA1', 1003, 4, False)
        # Assigning a type to the variable 'self' (line 1004)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1004, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.getA1.__dict__.__setitem__('stypy_localization', localization)
        matrix.getA1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.getA1.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.getA1.__dict__.__setitem__('stypy_function_name', 'matrix.getA1')
        matrix.getA1.__dict__.__setitem__('stypy_param_names_list', [])
        matrix.getA1.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.getA1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.getA1.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.getA1.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.getA1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.getA1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.getA1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getA1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getA1(...)' code ##################

        str_161583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1027, (-1)), 'str', '\n        Return `self` as a flattened `ndarray`.\n\n        Equivalent to ``np.asarray(x).ravel()``\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        ret : ndarray\n            `self`, 1-D, as an `ndarray`\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.getA1()\n        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])\n\n        ')
        
        # Call to ravel(...): (line 1028)
        # Processing the call keyword arguments (line 1028)
        kwargs_161589 = {}
        
        # Call to __array__(...): (line 1028)
        # Processing the call keyword arguments (line 1028)
        kwargs_161586 = {}
        # Getting the type of 'self' (line 1028)
        self_161584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 15), 'self', False)
        # Obtaining the member '__array__' of a type (line 1028)
        array___161585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1028, 15), self_161584, '__array__')
        # Calling __array__(args, kwargs) (line 1028)
        array___call_result_161587 = invoke(stypy.reporting.localization.Localization(__file__, 1028, 15), array___161585, *[], **kwargs_161586)
        
        # Obtaining the member 'ravel' of a type (line 1028)
        ravel_161588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1028, 15), array___call_result_161587, 'ravel')
        # Calling ravel(args, kwargs) (line 1028)
        ravel_call_result_161590 = invoke(stypy.reporting.localization.Localization(__file__, 1028, 15), ravel_161588, *[], **kwargs_161589)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1028)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1028, 8), 'stypy_return_type', ravel_call_result_161590)
        
        # ################# End of 'getA1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getA1' in the type store
        # Getting the type of 'stypy_return_type' (line 1003)
        stypy_return_type_161591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161591)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getA1'
        return stypy_return_type_161591


    @norecursion
    def ravel(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_161592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 26), 'str', 'C')
        defaults = [str_161592]
        # Create a new context for function 'ravel'
        module_type_store = module_type_store.open_function_context('ravel', 1031, 4, False)
        # Assigning a type to the variable 'self' (line 1032)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1032, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.ravel.__dict__.__setitem__('stypy_localization', localization)
        matrix.ravel.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.ravel.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.ravel.__dict__.__setitem__('stypy_function_name', 'matrix.ravel')
        matrix.ravel.__dict__.__setitem__('stypy_param_names_list', ['order'])
        matrix.ravel.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.ravel.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.ravel.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.ravel.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.ravel.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.ravel.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.ravel', ['order'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ravel', localization, ['order'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ravel(...)' code ##################

        str_161593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1066, (-1)), 'str', "\n        Return a flattened matrix.\n\n        Refer to `numpy.ravel` for more documentation.\n\n        Parameters\n        ----------\n        order : {'C', 'F', 'A', 'K'}, optional\n            The elements of `m` are read using this index order. 'C' means to\n            index the elements in C-like order, with the last axis index\n            changing fastest, back to the first axis index changing slowest.\n            'F' means to index the elements in Fortran-like index order, with\n            the first index changing fastest, and the last index changing\n            slowest. Note that the 'C' and 'F' options take no account of the\n            memory layout of the underlying array, and only refer to the order\n            of axis indexing.  'A' means to read the elements in Fortran-like\n            index order if `m` is Fortran *contiguous* in memory, C-like order\n            otherwise.  'K' means to read the elements in the order they occur\n            in memory, except for reversing the data when strides are negative.\n            By default, 'C' index order is used.\n\n        Returns\n        -------\n        ret : matrix\n            Return the matrix flattened to shape `(1, N)` where `N`\n            is the number of elements in the original matrix.\n            A copy is made only if necessary.\n\n        See Also\n        --------\n        matrix.flatten : returns a similar output matrix but always a copy\n        matrix.flat : a flat iterator on the array.\n        numpy.ravel : related function which returns an ndarray\n\n        ")
        
        # Call to ravel(...): (line 1067)
        # Processing the call arguments (line 1067)
        # Getting the type of 'self' (line 1067)
        self_161597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 31), 'self', False)
        # Processing the call keyword arguments (line 1067)
        # Getting the type of 'order' (line 1067)
        order_161598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 43), 'order', False)
        keyword_161599 = order_161598
        kwargs_161600 = {'order': keyword_161599}
        # Getting the type of 'N' (line 1067)
        N_161594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 15), 'N', False)
        # Obtaining the member 'ndarray' of a type (line 1067)
        ndarray_161595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 15), N_161594, 'ndarray')
        # Obtaining the member 'ravel' of a type (line 1067)
        ravel_161596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 15), ndarray_161595, 'ravel')
        # Calling ravel(args, kwargs) (line 1067)
        ravel_call_result_161601 = invoke(stypy.reporting.localization.Localization(__file__, 1067, 15), ravel_161596, *[self_161597], **kwargs_161600)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1067)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1067, 8), 'stypy_return_type', ravel_call_result_161601)
        
        # ################# End of 'ravel(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ravel' in the type store
        # Getting the type of 'stypy_return_type' (line 1031)
        stypy_return_type_161602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161602)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ravel'
        return stypy_return_type_161602


    @norecursion
    def getT(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getT'
        module_type_store = module_type_store.open_function_context('getT', 1070, 4, False)
        # Assigning a type to the variable 'self' (line 1071)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1071, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.getT.__dict__.__setitem__('stypy_localization', localization)
        matrix.getT.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.getT.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.getT.__dict__.__setitem__('stypy_function_name', 'matrix.getT')
        matrix.getT.__dict__.__setitem__('stypy_param_names_list', [])
        matrix.getT.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.getT.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.getT.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.getT.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.getT.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.getT.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.getT', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getT', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getT(...)' code ##################

        str_161603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, (-1)), 'str', "\n        Returns the transpose of the matrix.\n\n        Does *not* conjugate!  For the complex conjugate transpose, use ``.H``.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        ret : matrix object\n            The (non-conjugated) transpose of the matrix.\n\n        See Also\n        --------\n        transpose, getH\n\n        Examples\n        --------\n        >>> m = np.matrix('[1, 2; 3, 4]')\n        >>> m\n        matrix([[1, 2],\n                [3, 4]])\n        >>> m.getT()\n        matrix([[1, 3],\n                [2, 4]])\n\n        ")
        
        # Call to transpose(...): (line 1100)
        # Processing the call keyword arguments (line 1100)
        kwargs_161606 = {}
        # Getting the type of 'self' (line 1100)
        self_161604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 15), 'self', False)
        # Obtaining the member 'transpose' of a type (line 1100)
        transpose_161605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1100, 15), self_161604, 'transpose')
        # Calling transpose(args, kwargs) (line 1100)
        transpose_call_result_161607 = invoke(stypy.reporting.localization.Localization(__file__, 1100, 15), transpose_161605, *[], **kwargs_161606)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1100, 8), 'stypy_return_type', transpose_call_result_161607)
        
        # ################# End of 'getT(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getT' in the type store
        # Getting the type of 'stypy_return_type' (line 1070)
        stypy_return_type_161608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161608)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getT'
        return stypy_return_type_161608


    @norecursion
    def getH(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getH'
        module_type_store = module_type_store.open_function_context('getH', 1102, 4, False)
        # Assigning a type to the variable 'self' (line 1103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1103, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        matrix.getH.__dict__.__setitem__('stypy_localization', localization)
        matrix.getH.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        matrix.getH.__dict__.__setitem__('stypy_type_store', module_type_store)
        matrix.getH.__dict__.__setitem__('stypy_function_name', 'matrix.getH')
        matrix.getH.__dict__.__setitem__('stypy_param_names_list', [])
        matrix.getH.__dict__.__setitem__('stypy_varargs_param_name', None)
        matrix.getH.__dict__.__setitem__('stypy_kwargs_param_name', None)
        matrix.getH.__dict__.__setitem__('stypy_call_defaults', defaults)
        matrix.getH.__dict__.__setitem__('stypy_call_varargs', varargs)
        matrix.getH.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        matrix.getH.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.getH', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getH', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getH(...)' code ##################

        str_161609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1130, (-1)), 'str', '\n        Returns the (complex) conjugate transpose of `self`.\n\n        Equivalent to ``np.transpose(self)`` if `self` is real-valued.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        ret : matrix object\n            complex conjugate transpose of `self`\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4)))\n        >>> z = x - 1j*x; z\n        matrix([[  0. +0.j,   1. -1.j,   2. -2.j,   3. -3.j],\n                [  4. -4.j,   5. -5.j,   6. -6.j,   7. -7.j],\n                [  8. -8.j,   9. -9.j,  10.-10.j,  11.-11.j]])\n        >>> z.getH()\n        matrix([[  0. +0.j,   4. +4.j,   8. +8.j],\n                [  1. +1.j,   5. +5.j,   9. +9.j],\n                [  2. +2.j,   6. +6.j,  10.+10.j],\n                [  3. +3.j,   7. +7.j,  11.+11.j]])\n\n        ')
        
        
        # Call to issubclass(...): (line 1131)
        # Processing the call arguments (line 1131)
        # Getting the type of 'self' (line 1131)
        self_161611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 22), 'self', False)
        # Obtaining the member 'dtype' of a type (line 1131)
        dtype_161612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1131, 22), self_161611, 'dtype')
        # Obtaining the member 'type' of a type (line 1131)
        type_161613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1131, 22), dtype_161612, 'type')
        # Getting the type of 'N' (line 1131)
        N_161614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 39), 'N', False)
        # Obtaining the member 'complexfloating' of a type (line 1131)
        complexfloating_161615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1131, 39), N_161614, 'complexfloating')
        # Processing the call keyword arguments (line 1131)
        kwargs_161616 = {}
        # Getting the type of 'issubclass' (line 1131)
        issubclass_161610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 11), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 1131)
        issubclass_call_result_161617 = invoke(stypy.reporting.localization.Localization(__file__, 1131, 11), issubclass_161610, *[type_161613, complexfloating_161615], **kwargs_161616)
        
        # Testing the type of an if condition (line 1131)
        if_condition_161618 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1131, 8), issubclass_call_result_161617)
        # Assigning a type to the variable 'if_condition_161618' (line 1131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1131, 8), 'if_condition_161618', if_condition_161618)
        # SSA begins for if statement (line 1131)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to conjugate(...): (line 1132)
        # Processing the call keyword arguments (line 1132)
        kwargs_161624 = {}
        
        # Call to transpose(...): (line 1132)
        # Processing the call keyword arguments (line 1132)
        kwargs_161621 = {}
        # Getting the type of 'self' (line 1132)
        self_161619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 19), 'self', False)
        # Obtaining the member 'transpose' of a type (line 1132)
        transpose_161620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1132, 19), self_161619, 'transpose')
        # Calling transpose(args, kwargs) (line 1132)
        transpose_call_result_161622 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 19), transpose_161620, *[], **kwargs_161621)
        
        # Obtaining the member 'conjugate' of a type (line 1132)
        conjugate_161623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1132, 19), transpose_call_result_161622, 'conjugate')
        # Calling conjugate(args, kwargs) (line 1132)
        conjugate_call_result_161625 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 19), conjugate_161623, *[], **kwargs_161624)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 12), 'stypy_return_type', conjugate_call_result_161625)
        # SSA branch for the else part of an if statement (line 1131)
        module_type_store.open_ssa_branch('else')
        
        # Call to transpose(...): (line 1134)
        # Processing the call keyword arguments (line 1134)
        kwargs_161628 = {}
        # Getting the type of 'self' (line 1134)
        self_161626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 19), 'self', False)
        # Obtaining the member 'transpose' of a type (line 1134)
        transpose_161627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1134, 19), self_161626, 'transpose')
        # Calling transpose(args, kwargs) (line 1134)
        transpose_call_result_161629 = invoke(stypy.reporting.localization.Localization(__file__, 1134, 19), transpose_161627, *[], **kwargs_161628)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1134, 12), 'stypy_return_type', transpose_call_result_161629)
        # SSA join for if statement (line 1131)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'getH(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getH' in the type store
        # Getting the type of 'stypy_return_type' (line 1102)
        stypy_return_type_161630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161630)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getH'
        return stypy_return_type_161630

    
    # Assigning a Call to a Name (line 1136):
    
    # Assigning a Call to a Name (line 1137):
    
    # Assigning a Call to a Name (line 1138):
    
    # Assigning a Call to a Name (line 1139):
    
    # Assigning a Call to a Name (line 1140):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 208, 0, False)
        # Assigning a type to the variable 'self' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'matrix.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'matrix' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'matrix', matrix)

# Assigning a Num to a Name (line 245):
float_161631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 25), 'float')
# Getting the type of 'matrix'
matrix_161632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'matrix')
# Setting the type of the member '__array_priority__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), matrix_161632, '__array_priority__', float_161631)

# Assigning a Call to a Name (line 1136):

# Call to property(...): (line 1136)
# Processing the call arguments (line 1136)
# Getting the type of 'matrix'
matrix_161634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'matrix', False)
# Obtaining the member 'getT' of a type
getT_161635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), matrix_161634, 'getT')
# Getting the type of 'None' (line 1136)
None_161636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 23), 'None', False)
# Processing the call keyword arguments (line 1136)
kwargs_161637 = {}
# Getting the type of 'property' (line 1136)
property_161633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 8), 'property', False)
# Calling property(args, kwargs) (line 1136)
property_call_result_161638 = invoke(stypy.reporting.localization.Localization(__file__, 1136, 8), property_161633, *[getT_161635, None_161636], **kwargs_161637)

# Getting the type of 'matrix'
matrix_161639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'matrix')
# Setting the type of the member 'T' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), matrix_161639, 'T', property_call_result_161638)

# Assigning a Call to a Name (line 1137):

# Call to property(...): (line 1137)
# Processing the call arguments (line 1137)
# Getting the type of 'matrix'
matrix_161641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'matrix', False)
# Obtaining the member 'getA' of a type
getA_161642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), matrix_161641, 'getA')
# Getting the type of 'None' (line 1137)
None_161643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 23), 'None', False)
# Processing the call keyword arguments (line 1137)
kwargs_161644 = {}
# Getting the type of 'property' (line 1137)
property_161640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 8), 'property', False)
# Calling property(args, kwargs) (line 1137)
property_call_result_161645 = invoke(stypy.reporting.localization.Localization(__file__, 1137, 8), property_161640, *[getA_161642, None_161643], **kwargs_161644)

# Getting the type of 'matrix'
matrix_161646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'matrix')
# Setting the type of the member 'A' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), matrix_161646, 'A', property_call_result_161645)

# Assigning a Call to a Name (line 1138):

# Call to property(...): (line 1138)
# Processing the call arguments (line 1138)
# Getting the type of 'matrix'
matrix_161648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'matrix', False)
# Obtaining the member 'getA1' of a type
getA1_161649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), matrix_161648, 'getA1')
# Getting the type of 'None' (line 1138)
None_161650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 25), 'None', False)
# Processing the call keyword arguments (line 1138)
kwargs_161651 = {}
# Getting the type of 'property' (line 1138)
property_161647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 9), 'property', False)
# Calling property(args, kwargs) (line 1138)
property_call_result_161652 = invoke(stypy.reporting.localization.Localization(__file__, 1138, 9), property_161647, *[getA1_161649, None_161650], **kwargs_161651)

# Getting the type of 'matrix'
matrix_161653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'matrix')
# Setting the type of the member 'A1' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), matrix_161653, 'A1', property_call_result_161652)

# Assigning a Call to a Name (line 1139):

# Call to property(...): (line 1139)
# Processing the call arguments (line 1139)
# Getting the type of 'matrix'
matrix_161655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'matrix', False)
# Obtaining the member 'getH' of a type
getH_161656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), matrix_161655, 'getH')
# Getting the type of 'None' (line 1139)
None_161657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 23), 'None', False)
# Processing the call keyword arguments (line 1139)
kwargs_161658 = {}
# Getting the type of 'property' (line 1139)
property_161654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 8), 'property', False)
# Calling property(args, kwargs) (line 1139)
property_call_result_161659 = invoke(stypy.reporting.localization.Localization(__file__, 1139, 8), property_161654, *[getH_161656, None_161657], **kwargs_161658)

# Getting the type of 'matrix'
matrix_161660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'matrix')
# Setting the type of the member 'H' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), matrix_161660, 'H', property_call_result_161659)

# Assigning a Call to a Name (line 1140):

# Call to property(...): (line 1140)
# Processing the call arguments (line 1140)
# Getting the type of 'matrix'
matrix_161662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'matrix', False)
# Obtaining the member 'getI' of a type
getI_161663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), matrix_161662, 'getI')
# Getting the type of 'None' (line 1140)
None_161664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 23), 'None', False)
# Processing the call keyword arguments (line 1140)
kwargs_161665 = {}
# Getting the type of 'property' (line 1140)
property_161661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 8), 'property', False)
# Calling property(args, kwargs) (line 1140)
property_call_result_161666 = invoke(stypy.reporting.localization.Localization(__file__, 1140, 8), property_161661, *[getI_161663, None_161664], **kwargs_161665)

# Getting the type of 'matrix'
matrix_161667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'matrix')
# Setting the type of the member 'I' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), matrix_161667, 'I', property_call_result_161666)

@norecursion
def _from_string(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_from_string'
    module_type_store = module_type_store.open_function_context('_from_string', 1142, 0, False)
    
    # Passed parameters checking function
    _from_string.stypy_localization = localization
    _from_string.stypy_type_of_self = None
    _from_string.stypy_type_store = module_type_store
    _from_string.stypy_function_name = '_from_string'
    _from_string.stypy_param_names_list = ['str', 'gdict', 'ldict']
    _from_string.stypy_varargs_param_name = None
    _from_string.stypy_kwargs_param_name = None
    _from_string.stypy_call_defaults = defaults
    _from_string.stypy_call_varargs = varargs
    _from_string.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_from_string', ['str', 'gdict', 'ldict'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_from_string', localization, ['str', 'gdict', 'ldict'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_from_string(...)' code ##################

    
    # Assigning a Call to a Name (line 1143):
    
    # Assigning a Call to a Name (line 1143):
    
    # Call to split(...): (line 1143)
    # Processing the call arguments (line 1143)
    str_161670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1143, 21), 'str', ';')
    # Processing the call keyword arguments (line 1143)
    kwargs_161671 = {}
    # Getting the type of 'str' (line 1143)
    str_161668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 11), 'str', False)
    # Obtaining the member 'split' of a type (line 1143)
    split_161669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1143, 11), str_161668, 'split')
    # Calling split(args, kwargs) (line 1143)
    split_call_result_161672 = invoke(stypy.reporting.localization.Localization(__file__, 1143, 11), split_161669, *[str_161670], **kwargs_161671)
    
    # Assigning a type to the variable 'rows' (line 1143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1143, 4), 'rows', split_call_result_161672)
    
    # Assigning a List to a Name (line 1144):
    
    # Assigning a List to a Name (line 1144):
    
    # Obtaining an instance of the builtin type 'list' (line 1144)
    list_161673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1144, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1144)
    
    # Assigning a type to the variable 'rowtup' (line 1144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1144, 4), 'rowtup', list_161673)
    
    # Getting the type of 'rows' (line 1145)
    rows_161674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 15), 'rows')
    # Testing the type of a for loop iterable (line 1145)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1145, 4), rows_161674)
    # Getting the type of the for loop variable (line 1145)
    for_loop_var_161675 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1145, 4), rows_161674)
    # Assigning a type to the variable 'row' (line 1145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1145, 4), 'row', for_loop_var_161675)
    # SSA begins for a for statement (line 1145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1146):
    
    # Assigning a Call to a Name (line 1146):
    
    # Call to split(...): (line 1146)
    # Processing the call arguments (line 1146)
    str_161678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1146, 25), 'str', ',')
    # Processing the call keyword arguments (line 1146)
    kwargs_161679 = {}
    # Getting the type of 'row' (line 1146)
    row_161676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 15), 'row', False)
    # Obtaining the member 'split' of a type (line 1146)
    split_161677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1146, 15), row_161676, 'split')
    # Calling split(args, kwargs) (line 1146)
    split_call_result_161680 = invoke(stypy.reporting.localization.Localization(__file__, 1146, 15), split_161677, *[str_161678], **kwargs_161679)
    
    # Assigning a type to the variable 'trow' (line 1146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'trow', split_call_result_161680)
    
    # Assigning a List to a Name (line 1147):
    
    # Assigning a List to a Name (line 1147):
    
    # Obtaining an instance of the builtin type 'list' (line 1147)
    list_161681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1147, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1147)
    
    # Assigning a type to the variable 'newrow' (line 1147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1147, 8), 'newrow', list_161681)
    
    # Getting the type of 'trow' (line 1148)
    trow_161682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 17), 'trow')
    # Testing the type of a for loop iterable (line 1148)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1148, 8), trow_161682)
    # Getting the type of the for loop variable (line 1148)
    for_loop_var_161683 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1148, 8), trow_161682)
    # Assigning a type to the variable 'x' (line 1148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 8), 'x', for_loop_var_161683)
    # SSA begins for a for statement (line 1148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to extend(...): (line 1149)
    # Processing the call arguments (line 1149)
    
    # Call to split(...): (line 1149)
    # Processing the call keyword arguments (line 1149)
    kwargs_161688 = {}
    # Getting the type of 'x' (line 1149)
    x_161686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 26), 'x', False)
    # Obtaining the member 'split' of a type (line 1149)
    split_161687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1149, 26), x_161686, 'split')
    # Calling split(args, kwargs) (line 1149)
    split_call_result_161689 = invoke(stypy.reporting.localization.Localization(__file__, 1149, 26), split_161687, *[], **kwargs_161688)
    
    # Processing the call keyword arguments (line 1149)
    kwargs_161690 = {}
    # Getting the type of 'newrow' (line 1149)
    newrow_161684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 12), 'newrow', False)
    # Obtaining the member 'extend' of a type (line 1149)
    extend_161685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1149, 12), newrow_161684, 'extend')
    # Calling extend(args, kwargs) (line 1149)
    extend_call_result_161691 = invoke(stypy.reporting.localization.Localization(__file__, 1149, 12), extend_161685, *[split_call_result_161689], **kwargs_161690)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 1150):
    
    # Assigning a Name to a Name (line 1150):
    # Getting the type of 'newrow' (line 1150)
    newrow_161692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 15), 'newrow')
    # Assigning a type to the variable 'trow' (line 1150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1150, 8), 'trow', newrow_161692)
    
    # Assigning a List to a Name (line 1151):
    
    # Assigning a List to a Name (line 1151):
    
    # Obtaining an instance of the builtin type 'list' (line 1151)
    list_161693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1151, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1151)
    
    # Assigning a type to the variable 'coltup' (line 1151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1151, 8), 'coltup', list_161693)
    
    # Getting the type of 'trow' (line 1152)
    trow_161694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 19), 'trow')
    # Testing the type of a for loop iterable (line 1152)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1152, 8), trow_161694)
    # Getting the type of the for loop variable (line 1152)
    for_loop_var_161695 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1152, 8), trow_161694)
    # Assigning a type to the variable 'col' (line 1152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1152, 8), 'col', for_loop_var_161695)
    # SSA begins for a for statement (line 1152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1153):
    
    # Assigning a Call to a Name (line 1153):
    
    # Call to strip(...): (line 1153)
    # Processing the call keyword arguments (line 1153)
    kwargs_161698 = {}
    # Getting the type of 'col' (line 1153)
    col_161696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 18), 'col', False)
    # Obtaining the member 'strip' of a type (line 1153)
    strip_161697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 18), col_161696, 'strip')
    # Calling strip(args, kwargs) (line 1153)
    strip_call_result_161699 = invoke(stypy.reporting.localization.Localization(__file__, 1153, 18), strip_161697, *[], **kwargs_161698)
    
    # Assigning a type to the variable 'col' (line 1153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'col', strip_call_result_161699)
    
    
    # SSA begins for try-except statement (line 1154)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 1155):
    
    # Assigning a Subscript to a Name (line 1155):
    
    # Obtaining the type of the subscript
    # Getting the type of 'col' (line 1155)
    col_161700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 32), 'col')
    # Getting the type of 'ldict' (line 1155)
    ldict_161701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 26), 'ldict')
    # Obtaining the member '__getitem__' of a type (line 1155)
    getitem___161702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1155, 26), ldict_161701, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1155)
    subscript_call_result_161703 = invoke(stypy.reporting.localization.Localization(__file__, 1155, 26), getitem___161702, col_161700)
    
    # Assigning a type to the variable 'thismat' (line 1155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1155, 16), 'thismat', subscript_call_result_161703)
    # SSA branch for the except part of a try statement (line 1154)
    # SSA branch for the except 'KeyError' branch of a try statement (line 1154)
    module_type_store.open_ssa_branch('except')
    
    
    # SSA begins for try-except statement (line 1157)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 1158):
    
    # Assigning a Subscript to a Name (line 1158):
    
    # Obtaining the type of the subscript
    # Getting the type of 'col' (line 1158)
    col_161704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 36), 'col')
    # Getting the type of 'gdict' (line 1158)
    gdict_161705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 30), 'gdict')
    # Obtaining the member '__getitem__' of a type (line 1158)
    getitem___161706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1158, 30), gdict_161705, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1158)
    subscript_call_result_161707 = invoke(stypy.reporting.localization.Localization(__file__, 1158, 30), getitem___161706, col_161704)
    
    # Assigning a type to the variable 'thismat' (line 1158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 20), 'thismat', subscript_call_result_161707)
    # SSA branch for the except part of a try statement (line 1157)
    # SSA branch for the except 'KeyError' branch of a try statement (line 1157)
    module_type_store.open_ssa_branch('except')
    
    # Call to KeyError(...): (line 1160)
    # Processing the call arguments (line 1160)
    str_161709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1160, 35), 'str', '%s not found')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1160)
    tuple_161710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1160, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1160)
    # Adding element type (line 1160)
    # Getting the type of 'col' (line 1160)
    col_161711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 53), 'col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1160, 53), tuple_161710, col_161711)
    
    # Applying the binary operator '%' (line 1160)
    result_mod_161712 = python_operator(stypy.reporting.localization.Localization(__file__, 1160, 35), '%', str_161709, tuple_161710)
    
    # Processing the call keyword arguments (line 1160)
    kwargs_161713 = {}
    # Getting the type of 'KeyError' (line 1160)
    KeyError_161708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 26), 'KeyError', False)
    # Calling KeyError(args, kwargs) (line 1160)
    KeyError_call_result_161714 = invoke(stypy.reporting.localization.Localization(__file__, 1160, 26), KeyError_161708, *[result_mod_161712], **kwargs_161713)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1160, 20), KeyError_call_result_161714, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 1157)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 1154)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 1162)
    # Processing the call arguments (line 1162)
    # Getting the type of 'thismat' (line 1162)
    thismat_161717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 26), 'thismat', False)
    # Processing the call keyword arguments (line 1162)
    kwargs_161718 = {}
    # Getting the type of 'coltup' (line 1162)
    coltup_161715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 12), 'coltup', False)
    # Obtaining the member 'append' of a type (line 1162)
    append_161716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1162, 12), coltup_161715, 'append')
    # Calling append(args, kwargs) (line 1162)
    append_call_result_161719 = invoke(stypy.reporting.localization.Localization(__file__, 1162, 12), append_161716, *[thismat_161717], **kwargs_161718)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 1163)
    # Processing the call arguments (line 1163)
    
    # Call to concatenate(...): (line 1163)
    # Processing the call arguments (line 1163)
    # Getting the type of 'coltup' (line 1163)
    coltup_161723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 34), 'coltup', False)
    # Processing the call keyword arguments (line 1163)
    int_161724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1163, 47), 'int')
    keyword_161725 = int_161724
    kwargs_161726 = {'axis': keyword_161725}
    # Getting the type of 'concatenate' (line 1163)
    concatenate_161722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 22), 'concatenate', False)
    # Calling concatenate(args, kwargs) (line 1163)
    concatenate_call_result_161727 = invoke(stypy.reporting.localization.Localization(__file__, 1163, 22), concatenate_161722, *[coltup_161723], **kwargs_161726)
    
    # Processing the call keyword arguments (line 1163)
    kwargs_161728 = {}
    # Getting the type of 'rowtup' (line 1163)
    rowtup_161720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 8), 'rowtup', False)
    # Obtaining the member 'append' of a type (line 1163)
    append_161721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1163, 8), rowtup_161720, 'append')
    # Calling append(args, kwargs) (line 1163)
    append_call_result_161729 = invoke(stypy.reporting.localization.Localization(__file__, 1163, 8), append_161721, *[concatenate_call_result_161727], **kwargs_161728)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to concatenate(...): (line 1164)
    # Processing the call arguments (line 1164)
    # Getting the type of 'rowtup' (line 1164)
    rowtup_161731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 23), 'rowtup', False)
    # Processing the call keyword arguments (line 1164)
    int_161732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1164, 36), 'int')
    keyword_161733 = int_161732
    kwargs_161734 = {'axis': keyword_161733}
    # Getting the type of 'concatenate' (line 1164)
    concatenate_161730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 11), 'concatenate', False)
    # Calling concatenate(args, kwargs) (line 1164)
    concatenate_call_result_161735 = invoke(stypy.reporting.localization.Localization(__file__, 1164, 11), concatenate_161730, *[rowtup_161731], **kwargs_161734)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'stypy_return_type', concatenate_call_result_161735)
    
    # ################# End of '_from_string(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_from_string' in the type store
    # Getting the type of 'stypy_return_type' (line 1142)
    stypy_return_type_161736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_161736)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_from_string'
    return stypy_return_type_161736

# Assigning a type to the variable '_from_string' (line 1142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1142, 0), '_from_string', _from_string)

@norecursion
def bmat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1167)
    None_161737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 20), 'None')
    # Getting the type of 'None' (line 1167)
    None_161738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 32), 'None')
    defaults = [None_161737, None_161738]
    # Create a new context for function 'bmat'
    module_type_store = module_type_store.open_function_context('bmat', 1167, 0, False)
    
    # Passed parameters checking function
    bmat.stypy_localization = localization
    bmat.stypy_type_of_self = None
    bmat.stypy_type_store = module_type_store
    bmat.stypy_function_name = 'bmat'
    bmat.stypy_param_names_list = ['obj', 'ldict', 'gdict']
    bmat.stypy_varargs_param_name = None
    bmat.stypy_kwargs_param_name = None
    bmat.stypy_call_defaults = defaults
    bmat.stypy_call_varargs = varargs
    bmat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bmat', ['obj', 'ldict', 'gdict'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bmat', localization, ['obj', 'ldict', 'gdict'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bmat(...)' code ##################

    str_161739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1217, (-1)), 'str', "\n    Build a matrix object from a string, nested sequence, or array.\n\n    Parameters\n    ----------\n    obj : str or array_like\n        Input data.  Names of variables in the current scope may be\n        referenced, even if `obj` is a string.\n    ldict : dict, optional\n        A dictionary that replaces local operands in current frame.\n        Ignored if `obj` is not a string or `gdict` is `None`.\n    gdict : dict, optional\n        A dictionary that replaces global operands in current frame.\n        Ignored if `obj` is not a string.\n\n    Returns\n    -------\n    out : matrix\n        Returns a matrix object, which is a specialized 2-D array.\n\n    See Also\n    --------\n    matrix\n\n    Examples\n    --------\n    >>> A = np.mat('1 1; 1 1')\n    >>> B = np.mat('2 2; 2 2')\n    >>> C = np.mat('3 4; 5 6')\n    >>> D = np.mat('7 8; 9 0')\n\n    All the following expressions construct the same block matrix:\n\n    >>> np.bmat([[A, B], [C, D]])\n    matrix([[1, 1, 2, 2],\n            [1, 1, 2, 2],\n            [3, 4, 7, 8],\n            [5, 6, 9, 0]])\n    >>> np.bmat(np.r_[np.c_[A, B], np.c_[C, D]])\n    matrix([[1, 1, 2, 2],\n            [1, 1, 2, 2],\n            [3, 4, 7, 8],\n            [5, 6, 9, 0]])\n    >>> np.bmat('A,B; C,D')\n    matrix([[1, 1, 2, 2],\n            [1, 1, 2, 2],\n            [3, 4, 7, 8],\n            [5, 6, 9, 0]])\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 1218)
    # Getting the type of 'str' (line 1218)
    str_161740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 23), 'str')
    # Getting the type of 'obj' (line 1218)
    obj_161741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 18), 'obj')
    
    (may_be_161742, more_types_in_union_161743) = may_be_subtype(str_161740, obj_161741)

    if may_be_161742:

        if more_types_in_union_161743:
            # Runtime conditional SSA (line 1218)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'obj' (line 1218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1218, 4), 'obj', remove_not_subtype_from_union(obj_161741, str))
        
        # Type idiom detected: calculating its left and rigth part (line 1219)
        # Getting the type of 'gdict' (line 1219)
        gdict_161744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 11), 'gdict')
        # Getting the type of 'None' (line 1219)
        None_161745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 20), 'None')
        
        (may_be_161746, more_types_in_union_161747) = may_be_none(gdict_161744, None_161745)

        if may_be_161746:

            if more_types_in_union_161747:
                # Runtime conditional SSA (line 1219)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 1221):
            
            # Assigning a Attribute to a Name (line 1221):
            
            # Call to _getframe(...): (line 1221)
            # Processing the call keyword arguments (line 1221)
            kwargs_161750 = {}
            # Getting the type of 'sys' (line 1221)
            sys_161748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 20), 'sys', False)
            # Obtaining the member '_getframe' of a type (line 1221)
            _getframe_161749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1221, 20), sys_161748, '_getframe')
            # Calling _getframe(args, kwargs) (line 1221)
            _getframe_call_result_161751 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 20), _getframe_161749, *[], **kwargs_161750)
            
            # Obtaining the member 'f_back' of a type (line 1221)
            f_back_161752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1221, 20), _getframe_call_result_161751, 'f_back')
            # Assigning a type to the variable 'frame' (line 1221)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 12), 'frame', f_back_161752)
            
            # Assigning a Attribute to a Name (line 1222):
            
            # Assigning a Attribute to a Name (line 1222):
            # Getting the type of 'frame' (line 1222)
            frame_161753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 24), 'frame')
            # Obtaining the member 'f_globals' of a type (line 1222)
            f_globals_161754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1222, 24), frame_161753, 'f_globals')
            # Assigning a type to the variable 'glob_dict' (line 1222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1222, 12), 'glob_dict', f_globals_161754)
            
            # Assigning a Attribute to a Name (line 1223):
            
            # Assigning a Attribute to a Name (line 1223):
            # Getting the type of 'frame' (line 1223)
            frame_161755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 23), 'frame')
            # Obtaining the member 'f_locals' of a type (line 1223)
            f_locals_161756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1223, 23), frame_161755, 'f_locals')
            # Assigning a type to the variable 'loc_dict' (line 1223)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1223, 12), 'loc_dict', f_locals_161756)

            if more_types_in_union_161747:
                # Runtime conditional SSA for else branch (line 1219)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_161746) or more_types_in_union_161747):
            
            # Assigning a Name to a Name (line 1225):
            
            # Assigning a Name to a Name (line 1225):
            # Getting the type of 'gdict' (line 1225)
            gdict_161757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 24), 'gdict')
            # Assigning a type to the variable 'glob_dict' (line 1225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1225, 12), 'glob_dict', gdict_161757)
            
            # Assigning a Name to a Name (line 1226):
            
            # Assigning a Name to a Name (line 1226):
            # Getting the type of 'ldict' (line 1226)
            ldict_161758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 23), 'ldict')
            # Assigning a type to the variable 'loc_dict' (line 1226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1226, 12), 'loc_dict', ldict_161758)

            if (may_be_161746 and more_types_in_union_161747):
                # SSA join for if statement (line 1219)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to matrix(...): (line 1228)
        # Processing the call arguments (line 1228)
        
        # Call to _from_string(...): (line 1228)
        # Processing the call arguments (line 1228)
        # Getting the type of 'obj' (line 1228)
        obj_161761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 35), 'obj', False)
        # Getting the type of 'glob_dict' (line 1228)
        glob_dict_161762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 40), 'glob_dict', False)
        # Getting the type of 'loc_dict' (line 1228)
        loc_dict_161763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 51), 'loc_dict', False)
        # Processing the call keyword arguments (line 1228)
        kwargs_161764 = {}
        # Getting the type of '_from_string' (line 1228)
        _from_string_161760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 22), '_from_string', False)
        # Calling _from_string(args, kwargs) (line 1228)
        _from_string_call_result_161765 = invoke(stypy.reporting.localization.Localization(__file__, 1228, 22), _from_string_161760, *[obj_161761, glob_dict_161762, loc_dict_161763], **kwargs_161764)
        
        # Processing the call keyword arguments (line 1228)
        kwargs_161766 = {}
        # Getting the type of 'matrix' (line 1228)
        matrix_161759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 15), 'matrix', False)
        # Calling matrix(args, kwargs) (line 1228)
        matrix_call_result_161767 = invoke(stypy.reporting.localization.Localization(__file__, 1228, 15), matrix_161759, *[_from_string_call_result_161765], **kwargs_161766)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1228, 8), 'stypy_return_type', matrix_call_result_161767)

        if more_types_in_union_161743:
            # SSA join for if statement (line 1218)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to isinstance(...): (line 1230)
    # Processing the call arguments (line 1230)
    # Getting the type of 'obj' (line 1230)
    obj_161769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 18), 'obj', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1230)
    tuple_161770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1230, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1230)
    # Adding element type (line 1230)
    # Getting the type of 'tuple' (line 1230)
    tuple_161771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 24), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1230, 24), tuple_161770, tuple_161771)
    # Adding element type (line 1230)
    # Getting the type of 'list' (line 1230)
    list_161772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 31), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1230, 24), tuple_161770, list_161772)
    
    # Processing the call keyword arguments (line 1230)
    kwargs_161773 = {}
    # Getting the type of 'isinstance' (line 1230)
    isinstance_161768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1230)
    isinstance_call_result_161774 = invoke(stypy.reporting.localization.Localization(__file__, 1230, 7), isinstance_161768, *[obj_161769, tuple_161770], **kwargs_161773)
    
    # Testing the type of an if condition (line 1230)
    if_condition_161775 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1230, 4), isinstance_call_result_161774)
    # Assigning a type to the variable 'if_condition_161775' (line 1230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1230, 4), 'if_condition_161775', if_condition_161775)
    # SSA begins for if statement (line 1230)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 1232):
    
    # Assigning a List to a Name (line 1232):
    
    # Obtaining an instance of the builtin type 'list' (line 1232)
    list_161776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1232)
    
    # Assigning a type to the variable 'arr_rows' (line 1232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 8), 'arr_rows', list_161776)
    
    # Getting the type of 'obj' (line 1233)
    obj_161777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 19), 'obj')
    # Testing the type of a for loop iterable (line 1233)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1233, 8), obj_161777)
    # Getting the type of the for loop variable (line 1233)
    for_loop_var_161778 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1233, 8), obj_161777)
    # Assigning a type to the variable 'row' (line 1233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1233, 8), 'row', for_loop_var_161778)
    # SSA begins for a for statement (line 1233)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to isinstance(...): (line 1234)
    # Processing the call arguments (line 1234)
    # Getting the type of 'row' (line 1234)
    row_161780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 26), 'row', False)
    # Getting the type of 'N' (line 1234)
    N_161781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 31), 'N', False)
    # Obtaining the member 'ndarray' of a type (line 1234)
    ndarray_161782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1234, 31), N_161781, 'ndarray')
    # Processing the call keyword arguments (line 1234)
    kwargs_161783 = {}
    # Getting the type of 'isinstance' (line 1234)
    isinstance_161779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1234)
    isinstance_call_result_161784 = invoke(stypy.reporting.localization.Localization(__file__, 1234, 15), isinstance_161779, *[row_161780, ndarray_161782], **kwargs_161783)
    
    # Testing the type of an if condition (line 1234)
    if_condition_161785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1234, 12), isinstance_call_result_161784)
    # Assigning a type to the variable 'if_condition_161785' (line 1234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1234, 12), 'if_condition_161785', if_condition_161785)
    # SSA begins for if statement (line 1234)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to matrix(...): (line 1235)
    # Processing the call arguments (line 1235)
    
    # Call to concatenate(...): (line 1235)
    # Processing the call arguments (line 1235)
    # Getting the type of 'obj' (line 1235)
    obj_161788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 42), 'obj', False)
    # Processing the call keyword arguments (line 1235)
    int_161789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 52), 'int')
    keyword_161790 = int_161789
    kwargs_161791 = {'axis': keyword_161790}
    # Getting the type of 'concatenate' (line 1235)
    concatenate_161787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 30), 'concatenate', False)
    # Calling concatenate(args, kwargs) (line 1235)
    concatenate_call_result_161792 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 30), concatenate_161787, *[obj_161788], **kwargs_161791)
    
    # Processing the call keyword arguments (line 1235)
    kwargs_161793 = {}
    # Getting the type of 'matrix' (line 1235)
    matrix_161786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 23), 'matrix', False)
    # Calling matrix(args, kwargs) (line 1235)
    matrix_call_result_161794 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 23), matrix_161786, *[concatenate_call_result_161792], **kwargs_161793)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 16), 'stypy_return_type', matrix_call_result_161794)
    # SSA branch for the else part of an if statement (line 1234)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 1237)
    # Processing the call arguments (line 1237)
    
    # Call to concatenate(...): (line 1237)
    # Processing the call arguments (line 1237)
    # Getting the type of 'row' (line 1237)
    row_161798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 44), 'row', False)
    # Processing the call keyword arguments (line 1237)
    int_161799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1237, 54), 'int')
    keyword_161800 = int_161799
    kwargs_161801 = {'axis': keyword_161800}
    # Getting the type of 'concatenate' (line 1237)
    concatenate_161797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 32), 'concatenate', False)
    # Calling concatenate(args, kwargs) (line 1237)
    concatenate_call_result_161802 = invoke(stypy.reporting.localization.Localization(__file__, 1237, 32), concatenate_161797, *[row_161798], **kwargs_161801)
    
    # Processing the call keyword arguments (line 1237)
    kwargs_161803 = {}
    # Getting the type of 'arr_rows' (line 1237)
    arr_rows_161795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 16), 'arr_rows', False)
    # Obtaining the member 'append' of a type (line 1237)
    append_161796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1237, 16), arr_rows_161795, 'append')
    # Calling append(args, kwargs) (line 1237)
    append_call_result_161804 = invoke(stypy.reporting.localization.Localization(__file__, 1237, 16), append_161796, *[concatenate_call_result_161802], **kwargs_161803)
    
    # SSA join for if statement (line 1234)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to matrix(...): (line 1238)
    # Processing the call arguments (line 1238)
    
    # Call to concatenate(...): (line 1238)
    # Processing the call arguments (line 1238)
    # Getting the type of 'arr_rows' (line 1238)
    arr_rows_161807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 34), 'arr_rows', False)
    # Processing the call keyword arguments (line 1238)
    int_161808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1238, 49), 'int')
    keyword_161809 = int_161808
    kwargs_161810 = {'axis': keyword_161809}
    # Getting the type of 'concatenate' (line 1238)
    concatenate_161806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 22), 'concatenate', False)
    # Calling concatenate(args, kwargs) (line 1238)
    concatenate_call_result_161811 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 22), concatenate_161806, *[arr_rows_161807], **kwargs_161810)
    
    # Processing the call keyword arguments (line 1238)
    kwargs_161812 = {}
    # Getting the type of 'matrix' (line 1238)
    matrix_161805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 15), 'matrix', False)
    # Calling matrix(args, kwargs) (line 1238)
    matrix_call_result_161813 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 15), matrix_161805, *[concatenate_call_result_161811], **kwargs_161812)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1238, 8), 'stypy_return_type', matrix_call_result_161813)
    # SSA join for if statement (line 1230)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 1239)
    # Processing the call arguments (line 1239)
    # Getting the type of 'obj' (line 1239)
    obj_161815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 18), 'obj', False)
    # Getting the type of 'N' (line 1239)
    N_161816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 23), 'N', False)
    # Obtaining the member 'ndarray' of a type (line 1239)
    ndarray_161817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1239, 23), N_161816, 'ndarray')
    # Processing the call keyword arguments (line 1239)
    kwargs_161818 = {}
    # Getting the type of 'isinstance' (line 1239)
    isinstance_161814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1239)
    isinstance_call_result_161819 = invoke(stypy.reporting.localization.Localization(__file__, 1239, 7), isinstance_161814, *[obj_161815, ndarray_161817], **kwargs_161818)
    
    # Testing the type of an if condition (line 1239)
    if_condition_161820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1239, 4), isinstance_call_result_161819)
    # Assigning a type to the variable 'if_condition_161820' (line 1239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1239, 4), 'if_condition_161820', if_condition_161820)
    # SSA begins for if statement (line 1239)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to matrix(...): (line 1240)
    # Processing the call arguments (line 1240)
    # Getting the type of 'obj' (line 1240)
    obj_161822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 22), 'obj', False)
    # Processing the call keyword arguments (line 1240)
    kwargs_161823 = {}
    # Getting the type of 'matrix' (line 1240)
    matrix_161821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 15), 'matrix', False)
    # Calling matrix(args, kwargs) (line 1240)
    matrix_call_result_161824 = invoke(stypy.reporting.localization.Localization(__file__, 1240, 15), matrix_161821, *[obj_161822], **kwargs_161823)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1240, 8), 'stypy_return_type', matrix_call_result_161824)
    # SSA join for if statement (line 1239)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'bmat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bmat' in the type store
    # Getting the type of 'stypy_return_type' (line 1167)
    stypy_return_type_161825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_161825)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bmat'
    return stypy_return_type_161825

# Assigning a type to the variable 'bmat' (line 1167)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1167, 0), 'bmat', bmat)

# Assigning a Name to a Name (line 1242):

# Assigning a Name to a Name (line 1242):
# Getting the type of 'asmatrix' (line 1242)
asmatrix_161826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 6), 'asmatrix')
# Assigning a type to the variable 'mat' (line 1242)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1242, 0), 'mat', asmatrix_161826)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
