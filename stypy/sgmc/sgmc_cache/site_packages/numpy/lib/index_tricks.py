
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import sys
4: import math
5: 
6: import numpy.core.numeric as _nx
7: from numpy.core.numeric import (
8:     asarray, ScalarType, array, alltrue, cumprod, arange
9:     )
10: from numpy.core.numerictypes import find_common_type, issubdtype
11: 
12: from . import function_base
13: import numpy.matrixlib as matrix
14: from .function_base import diff
15: from numpy.core.multiarray import ravel_multi_index, unravel_index
16: from numpy.lib.stride_tricks import as_strided
17: 
18: makemat = matrix.matrix
19: 
20: 
21: __all__ = [
22:     'ravel_multi_index', 'unravel_index', 'mgrid', 'ogrid', 'r_', 'c_',
23:     's_', 'index_exp', 'ix_', 'ndenumerate', 'ndindex', 'fill_diagonal',
24:     'diag_indices', 'diag_indices_from'
25:     ]
26: 
27: 
28: def ix_(*args):
29:     '''
30:     Construct an open mesh from multiple sequences.
31: 
32:     This function takes N 1-D sequences and returns N outputs with N
33:     dimensions each, such that the shape is 1 in all but one dimension
34:     and the dimension with the non-unit shape value cycles through all
35:     N dimensions.
36: 
37:     Using `ix_` one can quickly construct index arrays that will index
38:     the cross product. ``a[np.ix_([1,3],[2,5])]`` returns the array
39:     ``[[a[1,2] a[1,5]], [a[3,2] a[3,5]]]``.
40: 
41:     Parameters
42:     ----------
43:     args : 1-D sequences
44: 
45:     Returns
46:     -------
47:     out : tuple of ndarrays
48:         N arrays with N dimensions each, with N the number of input
49:         sequences. Together these arrays form an open mesh.
50: 
51:     See Also
52:     --------
53:     ogrid, mgrid, meshgrid
54: 
55:     Examples
56:     --------
57:     >>> a = np.arange(10).reshape(2, 5)
58:     >>> a
59:     array([[0, 1, 2, 3, 4],
60:            [5, 6, 7, 8, 9]])
61:     >>> ixgrid = np.ix_([0,1], [2,4])
62:     >>> ixgrid
63:     (array([[0],
64:            [1]]), array([[2, 4]]))
65:     >>> ixgrid[0].shape, ixgrid[1].shape
66:     ((2, 1), (1, 2))
67:     >>> a[ixgrid]
68:     array([[2, 4],
69:            [7, 9]])
70: 
71:     '''
72:     out = []
73:     nd = len(args)
74:     for k, new in enumerate(args):
75:         new = asarray(new)
76:         if new.ndim != 1:
77:             raise ValueError("Cross index must be 1 dimensional")
78:         if new.size == 0:
79:             # Explicitly type empty arrays to avoid float default
80:             new = new.astype(_nx.intp)
81:         if issubdtype(new.dtype, _nx.bool_):
82:             new, = new.nonzero()
83:         new = new.reshape((1,)*k + (new.size,) + (1,)*(nd-k-1))
84:         out.append(new)
85:     return tuple(out)
86: 
87: class nd_grid(object):
88:     '''
89:     Construct a multi-dimensional "meshgrid".
90: 
91:     ``grid = nd_grid()`` creates an instance which will return a mesh-grid
92:     when indexed.  The dimension and number of the output arrays are equal
93:     to the number of indexing dimensions.  If the step length is not a
94:     complex number, then the stop is not inclusive.
95: 
96:     However, if the step length is a **complex number** (e.g. 5j), then the
97:     integer part of its magnitude is interpreted as specifying the
98:     number of points to create between the start and stop values, where
99:     the stop value **is inclusive**.
100: 
101:     If instantiated with an argument of ``sparse=True``, the mesh-grid is
102:     open (or not fleshed out) so that only one-dimension of each returned
103:     argument is greater than 1.
104: 
105:     Parameters
106:     ----------
107:     sparse : bool, optional
108:         Whether the grid is sparse or not. Default is False.
109: 
110:     Notes
111:     -----
112:     Two instances of `nd_grid` are made available in the NumPy namespace,
113:     `mgrid` and `ogrid`::
114: 
115:         mgrid = nd_grid(sparse=False)
116:         ogrid = nd_grid(sparse=True)
117: 
118:     Users should use these pre-defined instances instead of using `nd_grid`
119:     directly.
120: 
121:     Examples
122:     --------
123:     >>> mgrid = np.lib.index_tricks.nd_grid()
124:     >>> mgrid[0:5,0:5]
125:     array([[[0, 0, 0, 0, 0],
126:             [1, 1, 1, 1, 1],
127:             [2, 2, 2, 2, 2],
128:             [3, 3, 3, 3, 3],
129:             [4, 4, 4, 4, 4]],
130:            [[0, 1, 2, 3, 4],
131:             [0, 1, 2, 3, 4],
132:             [0, 1, 2, 3, 4],
133:             [0, 1, 2, 3, 4],
134:             [0, 1, 2, 3, 4]]])
135:     >>> mgrid[-1:1:5j]
136:     array([-1. , -0.5,  0. ,  0.5,  1. ])
137: 
138:     >>> ogrid = np.lib.index_tricks.nd_grid(sparse=True)
139:     >>> ogrid[0:5,0:5]
140:     [array([[0],
141:             [1],
142:             [2],
143:             [3],
144:             [4]]), array([[0, 1, 2, 3, 4]])]
145: 
146:     '''
147: 
148:     def __init__(self, sparse=False):
149:         self.sparse = sparse
150: 
151:     def __getitem__(self, key):
152:         try:
153:             size = []
154:             typ = int
155:             for k in range(len(key)):
156:                 step = key[k].step
157:                 start = key[k].start
158:                 if start is None:
159:                     start = 0
160:                 if step is None:
161:                     step = 1
162:                 if isinstance(step, complex):
163:                     size.append(int(abs(step)))
164:                     typ = float
165:                 else:
166:                     size.append(
167:                         int(math.ceil((key[k].stop - start)/(step*1.0))))
168:                 if (isinstance(step, float) or
169:                         isinstance(start, float) or
170:                         isinstance(key[k].stop, float)):
171:                     typ = float
172:             if self.sparse:
173:                 nn = [_nx.arange(_x, dtype=_t)
174:                         for _x, _t in zip(size, (typ,)*len(size))]
175:             else:
176:                 nn = _nx.indices(size, typ)
177:             for k in range(len(size)):
178:                 step = key[k].step
179:                 start = key[k].start
180:                 if start is None:
181:                     start = 0
182:                 if step is None:
183:                     step = 1
184:                 if isinstance(step, complex):
185:                     step = int(abs(step))
186:                     if step != 1:
187:                         step = (key[k].stop - start)/float(step-1)
188:                 nn[k] = (nn[k]*step+start)
189:             if self.sparse:
190:                 slobj = [_nx.newaxis]*len(size)
191:                 for k in range(len(size)):
192:                     slobj[k] = slice(None, None)
193:                     nn[k] = nn[k][slobj]
194:                     slobj[k] = _nx.newaxis
195:             return nn
196:         except (IndexError, TypeError):
197:             step = key.step
198:             stop = key.stop
199:             start = key.start
200:             if start is None:
201:                 start = 0
202:             if isinstance(step, complex):
203:                 step = abs(step)
204:                 length = int(step)
205:                 if step != 1:
206:                     step = (key.stop-start)/float(step-1)
207:                 stop = key.stop + step
208:                 return _nx.arange(0, length, 1, float)*step + start
209:             else:
210:                 return _nx.arange(start, stop, step)
211: 
212:     def __getslice__(self, i, j):
213:         return _nx.arange(i, j)
214: 
215:     def __len__(self):
216:         return 0
217: 
218: mgrid = nd_grid(sparse=False)
219: ogrid = nd_grid(sparse=True)
220: mgrid.__doc__ = None  # set in numpy.add_newdocs
221: ogrid.__doc__ = None  # set in numpy.add_newdocs
222: 
223: class AxisConcatenator(object):
224:     '''
225:     Translates slice objects to concatenation along an axis.
226: 
227:     For detailed documentation on usage, see `r_`.
228: 
229:     '''
230: 
231:     def _retval(self, res):
232:         if self.matrix:
233:             oldndim = res.ndim
234:             res = makemat(res)
235:             if oldndim == 1 and self.col:
236:                 res = res.T
237:         self.axis = self._axis
238:         self.matrix = self._matrix
239:         self.col = 0
240:         return res
241: 
242:     def __init__(self, axis=0, matrix=False, ndmin=1, trans1d=-1):
243:         self._axis = axis
244:         self._matrix = matrix
245:         self.axis = axis
246:         self.matrix = matrix
247:         self.col = 0
248:         self.trans1d = trans1d
249:         self.ndmin = ndmin
250: 
251:     def __getitem__(self, key):
252:         trans1d = self.trans1d
253:         ndmin = self.ndmin
254:         if isinstance(key, str):
255:             frame = sys._getframe().f_back
256:             mymat = matrix.bmat(key, frame.f_globals, frame.f_locals)
257:             return mymat
258:         if not isinstance(key, tuple):
259:             key = (key,)
260:         objs = []
261:         scalars = []
262:         arraytypes = []
263:         scalartypes = []
264:         for k in range(len(key)):
265:             scalar = False
266:             if isinstance(key[k], slice):
267:                 step = key[k].step
268:                 start = key[k].start
269:                 stop = key[k].stop
270:                 if start is None:
271:                     start = 0
272:                 if step is None:
273:                     step = 1
274:                 if isinstance(step, complex):
275:                     size = int(abs(step))
276:                     newobj = function_base.linspace(start, stop, num=size)
277:                 else:
278:                     newobj = _nx.arange(start, stop, step)
279:                 if ndmin > 1:
280:                     newobj = array(newobj, copy=False, ndmin=ndmin)
281:                     if trans1d != -1:
282:                         newobj = newobj.swapaxes(-1, trans1d)
283:             elif isinstance(key[k], str):
284:                 if k != 0:
285:                     raise ValueError("special directives must be the "
286:                             "first entry.")
287:                 key0 = key[0]
288:                 if key0 in 'rc':
289:                     self.matrix = True
290:                     self.col = (key0 == 'c')
291:                     continue
292:                 if ',' in key0:
293:                     vec = key0.split(',')
294:                     try:
295:                         self.axis, ndmin = \
296:                                    [int(x) for x in vec[:2]]
297:                         if len(vec) == 3:
298:                             trans1d = int(vec[2])
299:                         continue
300:                     except:
301:                         raise ValueError("unknown special directive")
302:                 try:
303:                     self.axis = int(key[k])
304:                     continue
305:                 except (ValueError, TypeError):
306:                     raise ValueError("unknown special directive")
307:             elif type(key[k]) in ScalarType:
308:                 newobj = array(key[k], ndmin=ndmin)
309:                 scalars.append(k)
310:                 scalar = True
311:                 scalartypes.append(newobj.dtype)
312:             else:
313:                 newobj = key[k]
314:                 if ndmin > 1:
315:                     tempobj = array(newobj, copy=False, subok=True)
316:                     newobj = array(newobj, copy=False, subok=True,
317:                                    ndmin=ndmin)
318:                     if trans1d != -1 and tempobj.ndim < ndmin:
319:                         k2 = ndmin-tempobj.ndim
320:                         if (trans1d < 0):
321:                             trans1d += k2 + 1
322:                         defaxes = list(range(ndmin))
323:                         k1 = trans1d
324:                         axes = defaxes[:k1] + defaxes[k2:] + \
325:                                defaxes[k1:k2]
326:                         newobj = newobj.transpose(axes)
327:                     del tempobj
328:             objs.append(newobj)
329:             if not scalar and isinstance(newobj, _nx.ndarray):
330:                 arraytypes.append(newobj.dtype)
331: 
332:         #  Esure that scalars won't up-cast unless warranted
333:         final_dtype = find_common_type(arraytypes, scalartypes)
334:         if final_dtype is not None:
335:             for k in scalars:
336:                 objs[k] = objs[k].astype(final_dtype)
337: 
338:         res = _nx.concatenate(tuple(objs), axis=self.axis)
339:         return self._retval(res)
340: 
341:     def __getslice__(self, i, j):
342:         res = _nx.arange(i, j)
343:         return self._retval(res)
344: 
345:     def __len__(self):
346:         return 0
347: 
348: # separate classes are used here instead of just making r_ = concatentor(0),
349: # etc. because otherwise we couldn't get the doc string to come out right
350: # in help(r_)
351: 
352: class RClass(AxisConcatenator):
353:     '''
354:     Translates slice objects to concatenation along the first axis.
355: 
356:     This is a simple way to build up arrays quickly. There are two use cases.
357: 
358:     1. If the index expression contains comma separated arrays, then stack
359:        them along their first axis.
360:     2. If the index expression contains slice notation or scalars then create
361:        a 1-D array with a range indicated by the slice notation.
362: 
363:     If slice notation is used, the syntax ``start:stop:step`` is equivalent
364:     to ``np.arange(start, stop, step)`` inside of the brackets. However, if
365:     ``step`` is an imaginary number (i.e. 100j) then its integer portion is
366:     interpreted as a number-of-points desired and the start and stop are
367:     inclusive. In other words ``start:stop:stepj`` is interpreted as
368:     ``np.linspace(start, stop, step, endpoint=1)`` inside of the brackets.
369:     After expansion of slice notation, all comma separated sequences are
370:     concatenated together.
371: 
372:     Optional character strings placed as the first element of the index
373:     expression can be used to change the output. The strings 'r' or 'c' result
374:     in matrix output. If the result is 1-D and 'r' is specified a 1 x N (row)
375:     matrix is produced. If the result is 1-D and 'c' is specified, then a N x 1
376:     (column) matrix is produced. If the result is 2-D then both provide the
377:     same matrix result.
378: 
379:     A string integer specifies which axis to stack multiple comma separated
380:     arrays along. A string of two comma-separated integers allows indication
381:     of the minimum number of dimensions to force each entry into as the
382:     second integer (the axis to concatenate along is still the first integer).
383: 
384:     A string with three comma-separated integers allows specification of the
385:     axis to concatenate along, the minimum number of dimensions to force the
386:     entries to, and which axis should contain the start of the arrays which
387:     are less than the specified number of dimensions. In other words the third
388:     integer allows you to specify where the 1's should be placed in the shape
389:     of the arrays that have their shapes upgraded. By default, they are placed
390:     in the front of the shape tuple. The third argument allows you to specify
391:     where the start of the array should be instead. Thus, a third argument of
392:     '0' would place the 1's at the end of the array shape. Negative integers
393:     specify where in the new shape tuple the last dimension of upgraded arrays
394:     should be placed, so the default is '-1'.
395: 
396:     Parameters
397:     ----------
398:     Not a function, so takes no parameters
399: 
400: 
401:     Returns
402:     -------
403:     A concatenated ndarray or matrix.
404: 
405:     See Also
406:     --------
407:     concatenate : Join a sequence of arrays along an existing axis.
408:     c_ : Translates slice objects to concatenation along the second axis.
409: 
410:     Examples
411:     --------
412:     >>> np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])]
413:     array([1, 2, 3, 0, 0, 4, 5, 6])
414:     >>> np.r_[-1:1:6j, [0]*3, 5, 6]
415:     array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ,  0. ,  0. ,  0. ,  5. ,  6. ])
416: 
417:     String integers specify the axis to concatenate along or the minimum
418:     number of dimensions to force entries into.
419: 
420:     >>> a = np.array([[0, 1, 2], [3, 4, 5]])
421:     >>> np.r_['-1', a, a] # concatenate along last axis
422:     array([[0, 1, 2, 0, 1, 2],
423:            [3, 4, 5, 3, 4, 5]])
424:     >>> np.r_['0,2', [1,2,3], [4,5,6]] # concatenate along first axis, dim>=2
425:     array([[1, 2, 3],
426:            [4, 5, 6]])
427: 
428:     >>> np.r_['0,2,0', [1,2,3], [4,5,6]]
429:     array([[1],
430:            [2],
431:            [3],
432:            [4],
433:            [5],
434:            [6]])
435:     >>> np.r_['1,2,0', [1,2,3], [4,5,6]]
436:     array([[1, 4],
437:            [2, 5],
438:            [3, 6]])
439: 
440:     Using 'r' or 'c' as a first string argument creates a matrix.
441: 
442:     >>> np.r_['r',[1,2,3], [4,5,6]]
443:     matrix([[1, 2, 3, 4, 5, 6]])
444: 
445:     '''
446: 
447:     def __init__(self):
448:         AxisConcatenator.__init__(self, 0)
449: 
450: r_ = RClass()
451: 
452: class CClass(AxisConcatenator):
453:     '''
454:     Translates slice objects to concatenation along the second axis.
455: 
456:     This is short-hand for ``np.r_['-1,2,0', index expression]``, which is
457:     useful because of its common occurrence. In particular, arrays will be
458:     stacked along their last axis after being upgraded to at least 2-D with
459:     1's post-pended to the shape (column vectors made out of 1-D arrays).
460: 
461:     For detailed documentation, see `r_`.
462: 
463:     Examples
464:     --------
465:     >>> np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
466:     array([[1, 2, 3, 0, 0, 4, 5, 6]])
467: 
468:     '''
469: 
470:     def __init__(self):
471:         AxisConcatenator.__init__(self, -1, ndmin=2, trans1d=0)
472: 
473: c_ = CClass()
474: 
475: class ndenumerate(object):
476:     '''
477:     Multidimensional index iterator.
478: 
479:     Return an iterator yielding pairs of array coordinates and values.
480: 
481:     Parameters
482:     ----------
483:     arr : ndarray
484:       Input array.
485: 
486:     See Also
487:     --------
488:     ndindex, flatiter
489: 
490:     Examples
491:     --------
492:     >>> a = np.array([[1, 2], [3, 4]])
493:     >>> for index, x in np.ndenumerate(a):
494:     ...     print(index, x)
495:     (0, 0) 1
496:     (0, 1) 2
497:     (1, 0) 3
498:     (1, 1) 4
499: 
500:     '''
501: 
502:     def __init__(self, arr):
503:         self.iter = asarray(arr).flat
504: 
505:     def __next__(self):
506:         '''
507:         Standard iterator method, returns the index tuple and array value.
508: 
509:         Returns
510:         -------
511:         coords : tuple of ints
512:             The indices of the current iteration.
513:         val : scalar
514:             The array element of the current iteration.
515: 
516:         '''
517:         return self.iter.coords, next(self.iter)
518: 
519:     def __iter__(self):
520:         return self
521: 
522:     next = __next__
523: 
524: 
525: class ndindex(object):
526:     '''
527:     An N-dimensional iterator object to index arrays.
528: 
529:     Given the shape of an array, an `ndindex` instance iterates over
530:     the N-dimensional index of the array. At each iteration a tuple
531:     of indices is returned, the last dimension is iterated over first.
532: 
533:     Parameters
534:     ----------
535:     `*args` : ints
536:       The size of each dimension of the array.
537: 
538:     See Also
539:     --------
540:     ndenumerate, flatiter
541: 
542:     Examples
543:     --------
544:     >>> for index in np.ndindex(3, 2, 1):
545:     ...     print(index)
546:     (0, 0, 0)
547:     (0, 1, 0)
548:     (1, 0, 0)
549:     (1, 1, 0)
550:     (2, 0, 0)
551:     (2, 1, 0)
552: 
553:     '''
554: 
555:     def __init__(self, *shape):
556:         if len(shape) == 1 and isinstance(shape[0], tuple):
557:             shape = shape[0]
558:         x = as_strided(_nx.zeros(1), shape=shape,
559:                        strides=_nx.zeros_like(shape))
560:         self._it = _nx.nditer(x, flags=['multi_index', 'zerosize_ok'],
561:                               order='C')
562: 
563:     def __iter__(self):
564:         return self
565: 
566:     def ndincr(self):
567:         '''
568:         Increment the multi-dimensional index by one.
569: 
570:         This method is for backward compatibility only: do not use.
571:         '''
572:         next(self)
573: 
574:     def __next__(self):
575:         '''
576:         Standard iterator method, updates the index and returns the index
577:         tuple.
578: 
579:         Returns
580:         -------
581:         val : tuple of ints
582:             Returns a tuple containing the indices of the current
583:             iteration.
584: 
585:         '''
586:         next(self._it)
587:         return self._it.multi_index
588: 
589:     next = __next__
590: 
591: 
592: # You can do all this with slice() plus a few special objects,
593: # but there's a lot to remember. This version is simpler because
594: # it uses the standard array indexing syntax.
595: #
596: # Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
597: # last revision: 1999-7-23
598: #
599: # Cosmetic changes by T. Oliphant 2001
600: #
601: #
602: 
603: class IndexExpression(object):
604:     '''
605:     A nicer way to build up index tuples for arrays.
606: 
607:     .. note::
608:        Use one of the two predefined instances `index_exp` or `s_`
609:        rather than directly using `IndexExpression`.
610: 
611:     For any index combination, including slicing and axis insertion,
612:     ``a[indices]`` is the same as ``a[np.index_exp[indices]]`` for any
613:     array `a`. However, ``np.index_exp[indices]`` can be used anywhere
614:     in Python code and returns a tuple of slice objects that can be
615:     used in the construction of complex index expressions.
616: 
617:     Parameters
618:     ----------
619:     maketuple : bool
620:         If True, always returns a tuple.
621: 
622:     See Also
623:     --------
624:     index_exp : Predefined instance that always returns a tuple:
625:        `index_exp = IndexExpression(maketuple=True)`.
626:     s_ : Predefined instance without tuple conversion:
627:        `s_ = IndexExpression(maketuple=False)`.
628: 
629:     Notes
630:     -----
631:     You can do all this with `slice()` plus a few special objects,
632:     but there's a lot to remember and this version is simpler because
633:     it uses the standard array indexing syntax.
634: 
635:     Examples
636:     --------
637:     >>> np.s_[2::2]
638:     slice(2, None, 2)
639:     >>> np.index_exp[2::2]
640:     (slice(2, None, 2),)
641: 
642:     >>> np.array([0, 1, 2, 3, 4])[np.s_[2::2]]
643:     array([2, 4])
644: 
645:     '''
646: 
647:     def __init__(self, maketuple):
648:         self.maketuple = maketuple
649: 
650:     def __getitem__(self, item):
651:         if self.maketuple and not isinstance(item, tuple):
652:             return (item,)
653:         else:
654:             return item
655: 
656: index_exp = IndexExpression(maketuple=True)
657: s_ = IndexExpression(maketuple=False)
658: 
659: # End contribution from Konrad.
660: 
661: 
662: # The following functions complement those in twodim_base, but are
663: # applicable to N-dimensions.
664: 
665: def fill_diagonal(a, val, wrap=False):
666:     '''Fill the main diagonal of the given array of any dimensionality.
667: 
668:     For an array `a` with ``a.ndim > 2``, the diagonal is the list of
669:     locations with indices ``a[i, i, ..., i]`` all identical. This function
670:     modifies the input array in-place, it does not return a value.
671: 
672:     Parameters
673:     ----------
674:     a : array, at least 2-D.
675:       Array whose diagonal is to be filled, it gets modified in-place.
676: 
677:     val : scalar
678:       Value to be written on the diagonal, its type must be compatible with
679:       that of the array a.
680: 
681:     wrap : bool
682:       For tall matrices in NumPy version up to 1.6.2, the
683:       diagonal "wrapped" after N columns. You can have this behavior
684:       with this option. This affects only tall matrices.
685: 
686:     See also
687:     --------
688:     diag_indices, diag_indices_from
689: 
690:     Notes
691:     -----
692:     .. versionadded:: 1.4.0
693: 
694:     This functionality can be obtained via `diag_indices`, but internally
695:     this version uses a much faster implementation that never constructs the
696:     indices and uses simple slicing.
697: 
698:     Examples
699:     --------
700:     >>> a = np.zeros((3, 3), int)
701:     >>> np.fill_diagonal(a, 5)
702:     >>> a
703:     array([[5, 0, 0],
704:            [0, 5, 0],
705:            [0, 0, 5]])
706: 
707:     The same function can operate on a 4-D array:
708: 
709:     >>> a = np.zeros((3, 3, 3, 3), int)
710:     >>> np.fill_diagonal(a, 4)
711: 
712:     We only show a few blocks for clarity:
713: 
714:     >>> a[0, 0]
715:     array([[4, 0, 0],
716:            [0, 0, 0],
717:            [0, 0, 0]])
718:     >>> a[1, 1]
719:     array([[0, 0, 0],
720:            [0, 4, 0],
721:            [0, 0, 0]])
722:     >>> a[2, 2]
723:     array([[0, 0, 0],
724:            [0, 0, 0],
725:            [0, 0, 4]])
726: 
727:     The wrap option affects only tall matrices:
728: 
729:     >>> # tall matrices no wrap
730:     >>> a = np.zeros((5, 3),int)
731:     >>> fill_diagonal(a, 4)
732:     >>> a
733:     array([[4, 0, 0],
734:            [0, 4, 0],
735:            [0, 0, 4],
736:            [0, 0, 0],
737:            [0, 0, 0]])
738: 
739:     >>> # tall matrices wrap
740:     >>> a = np.zeros((5, 3),int)
741:     >>> fill_diagonal(a, 4, wrap=True)
742:     >>> a
743:     array([[4, 0, 0],
744:            [0, 4, 0],
745:            [0, 0, 4],
746:            [0, 0, 0],
747:            [4, 0, 0]])
748: 
749:     >>> # wide matrices
750:     >>> a = np.zeros((3, 5),int)
751:     >>> fill_diagonal(a, 4, wrap=True)
752:     >>> a
753:     array([[4, 0, 0, 0, 0],
754:            [0, 4, 0, 0, 0],
755:            [0, 0, 4, 0, 0]])
756: 
757:     '''
758:     if a.ndim < 2:
759:         raise ValueError("array must be at least 2-d")
760:     end = None
761:     if a.ndim == 2:
762:         # Explicit, fast formula for the common case.  For 2-d arrays, we
763:         # accept rectangular ones.
764:         step = a.shape[1] + 1
765:         #This is needed to don't have tall matrix have the diagonal wrap.
766:         if not wrap:
767:             end = a.shape[1] * a.shape[1]
768:     else:
769:         # For more than d=2, the strided formula is only valid for arrays with
770:         # all dimensions equal, so we check first.
771:         if not alltrue(diff(a.shape) == 0):
772:             raise ValueError("All dimensions of input must be of equal length")
773:         step = 1 + (cumprod(a.shape[:-1])).sum()
774: 
775:     # Write the value out into the diagonal.
776:     a.flat[:end:step] = val
777: 
778: 
779: def diag_indices(n, ndim=2):
780:     '''
781:     Return the indices to access the main diagonal of an array.
782: 
783:     This returns a tuple of indices that can be used to access the main
784:     diagonal of an array `a` with ``a.ndim >= 2`` dimensions and shape
785:     (n, n, ..., n). For ``a.ndim = 2`` this is the usual diagonal, for
786:     ``a.ndim > 2`` this is the set of indices to access ``a[i, i, ..., i]``
787:     for ``i = [0..n-1]``.
788: 
789:     Parameters
790:     ----------
791:     n : int
792:       The size, along each dimension, of the arrays for which the returned
793:       indices can be used.
794: 
795:     ndim : int, optional
796:       The number of dimensions.
797: 
798:     See also
799:     --------
800:     diag_indices_from
801: 
802:     Notes
803:     -----
804:     .. versionadded:: 1.4.0
805: 
806:     Examples
807:     --------
808:     Create a set of indices to access the diagonal of a (4, 4) array:
809: 
810:     >>> di = np.diag_indices(4)
811:     >>> di
812:     (array([0, 1, 2, 3]), array([0, 1, 2, 3]))
813:     >>> a = np.arange(16).reshape(4, 4)
814:     >>> a
815:     array([[ 0,  1,  2,  3],
816:            [ 4,  5,  6,  7],
817:            [ 8,  9, 10, 11],
818:            [12, 13, 14, 15]])
819:     >>> a[di] = 100
820:     >>> a
821:     array([[100,   1,   2,   3],
822:            [  4, 100,   6,   7],
823:            [  8,   9, 100,  11],
824:            [ 12,  13,  14, 100]])
825: 
826:     Now, we create indices to manipulate a 3-D array:
827: 
828:     >>> d3 = np.diag_indices(2, 3)
829:     >>> d3
830:     (array([0, 1]), array([0, 1]), array([0, 1]))
831: 
832:     And use it to set the diagonal of an array of zeros to 1:
833: 
834:     >>> a = np.zeros((2, 2, 2), dtype=np.int)
835:     >>> a[d3] = 1
836:     >>> a
837:     array([[[1, 0],
838:             [0, 0]],
839:            [[0, 0],
840:             [0, 1]]])
841: 
842:     '''
843:     idx = arange(n)
844:     return (idx,) * ndim
845: 
846: 
847: def diag_indices_from(arr):
848:     '''
849:     Return the indices to access the main diagonal of an n-dimensional array.
850: 
851:     See `diag_indices` for full details.
852: 
853:     Parameters
854:     ----------
855:     arr : array, at least 2-D
856: 
857:     See Also
858:     --------
859:     diag_indices
860: 
861:     Notes
862:     -----
863:     .. versionadded:: 1.4.0
864: 
865:     '''
866: 
867:     if not arr.ndim >= 2:
868:         raise ValueError("input array must be at least 2-d")
869:     # For more than d=2, the strided formula is only valid for arrays with
870:     # all dimensions equal, so we check first.
871:     if not alltrue(diff(arr.shape) == 0):
872:         raise ValueError("All dimensions of input must be of equal length")
873: 
874:     return diag_indices(arr.shape[0], arr.ndim)
875: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import math' statement (line 4)
import math

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy.core.numeric' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_114437 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric')

if (type(import_114437) is not StypyTypeError):

    if (import_114437 != 'pyd_module'):
        __import__(import_114437)
        sys_modules_114438 = sys.modules[import_114437]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), '_nx', sys_modules_114438.module_type_store, module_type_store)
    else:
        import numpy.core.numeric as _nx

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), '_nx', numpy.core.numeric, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric', import_114437)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.core.numeric import asarray, ScalarType, array, alltrue, cumprod, arange' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_114439 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.numeric')

if (type(import_114439) is not StypyTypeError):

    if (import_114439 != 'pyd_module'):
        __import__(import_114439)
        sys_modules_114440 = sys.modules[import_114439]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.numeric', sys_modules_114440.module_type_store, module_type_store, ['asarray', 'ScalarType', 'array', 'alltrue', 'cumprod', 'arange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_114440, sys_modules_114440.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import asarray, ScalarType, array, alltrue, cumprod, arange

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.numeric', None, module_type_store, ['asarray', 'ScalarType', 'array', 'alltrue', 'cumprod', 'arange'], [asarray, ScalarType, array, alltrue, cumprod, arange])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.numeric', import_114439)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.core.numerictypes import find_common_type, issubdtype' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_114441 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core.numerictypes')

if (type(import_114441) is not StypyTypeError):

    if (import_114441 != 'pyd_module'):
        __import__(import_114441)
        sys_modules_114442 = sys.modules[import_114441]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core.numerictypes', sys_modules_114442.module_type_store, module_type_store, ['find_common_type', 'issubdtype'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_114442, sys_modules_114442.module_type_store, module_type_store)
    else:
        from numpy.core.numerictypes import find_common_type, issubdtype

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core.numerictypes', None, module_type_store, ['find_common_type', 'issubdtype'], [find_common_type, issubdtype])

else:
    # Assigning a type to the variable 'numpy.core.numerictypes' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core.numerictypes', import_114441)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.lib import function_base' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_114443 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.lib')

if (type(import_114443) is not StypyTypeError):

    if (import_114443 != 'pyd_module'):
        __import__(import_114443)
        sys_modules_114444 = sys.modules[import_114443]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.lib', sys_modules_114444.module_type_store, module_type_store, ['function_base'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_114444, sys_modules_114444.module_type_store, module_type_store)
    else:
        from numpy.lib import function_base

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.lib', None, module_type_store, ['function_base'], [function_base])

else:
    # Assigning a type to the variable 'numpy.lib' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.lib', import_114443)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy.matrixlib' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_114445 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.matrixlib')

if (type(import_114445) is not StypyTypeError):

    if (import_114445 != 'pyd_module'):
        __import__(import_114445)
        sys_modules_114446 = sys.modules[import_114445]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matrix', sys_modules_114446.module_type_store, module_type_store)
    else:
        import numpy.matrixlib as matrix

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matrix', numpy.matrixlib, module_type_store)

else:
    # Assigning a type to the variable 'numpy.matrixlib' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.matrixlib', import_114445)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.lib.function_base import diff' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_114447 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.lib.function_base')

if (type(import_114447) is not StypyTypeError):

    if (import_114447 != 'pyd_module'):
        __import__(import_114447)
        sys_modules_114448 = sys.modules[import_114447]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.lib.function_base', sys_modules_114448.module_type_store, module_type_store, ['diff'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_114448, sys_modules_114448.module_type_store, module_type_store)
    else:
        from numpy.lib.function_base import diff

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.lib.function_base', None, module_type_store, ['diff'], [diff])

else:
    # Assigning a type to the variable 'numpy.lib.function_base' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.lib.function_base', import_114447)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.core.multiarray import ravel_multi_index, unravel_index' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_114449 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core.multiarray')

if (type(import_114449) is not StypyTypeError):

    if (import_114449 != 'pyd_module'):
        __import__(import_114449)
        sys_modules_114450 = sys.modules[import_114449]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core.multiarray', sys_modules_114450.module_type_store, module_type_store, ['ravel_multi_index', 'unravel_index'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_114450, sys_modules_114450.module_type_store, module_type_store)
    else:
        from numpy.core.multiarray import ravel_multi_index, unravel_index

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core.multiarray', None, module_type_store, ['ravel_multi_index', 'unravel_index'], [ravel_multi_index, unravel_index])

else:
    # Assigning a type to the variable 'numpy.core.multiarray' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core.multiarray', import_114449)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from numpy.lib.stride_tricks import as_strided' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_114451 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.lib.stride_tricks')

if (type(import_114451) is not StypyTypeError):

    if (import_114451 != 'pyd_module'):
        __import__(import_114451)
        sys_modules_114452 = sys.modules[import_114451]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.lib.stride_tricks', sys_modules_114452.module_type_store, module_type_store, ['as_strided'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_114452, sys_modules_114452.module_type_store, module_type_store)
    else:
        from numpy.lib.stride_tricks import as_strided

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.lib.stride_tricks', None, module_type_store, ['as_strided'], [as_strided])

else:
    # Assigning a type to the variable 'numpy.lib.stride_tricks' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.lib.stride_tricks', import_114451)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a Attribute to a Name (line 18):

# Assigning a Attribute to a Name (line 18):
# Getting the type of 'matrix' (line 18)
matrix_114453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'matrix')
# Obtaining the member 'matrix' of a type (line 18)
matrix_114454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 10), matrix_114453, 'matrix')
# Assigning a type to the variable 'makemat' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'makemat', matrix_114454)

# Assigning a List to a Name (line 21):

# Assigning a List to a Name (line 21):
__all__ = ['ravel_multi_index', 'unravel_index', 'mgrid', 'ogrid', 'r_', 'c_', 's_', 'index_exp', 'ix_', 'ndenumerate', 'ndindex', 'fill_diagonal', 'diag_indices', 'diag_indices_from']
module_type_store.set_exportable_members(['ravel_multi_index', 'unravel_index', 'mgrid', 'ogrid', 'r_', 'c_', 's_', 'index_exp', 'ix_', 'ndenumerate', 'ndindex', 'fill_diagonal', 'diag_indices', 'diag_indices_from'])

# Obtaining an instance of the builtin type 'list' (line 21)
list_114455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_114456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 4), 'str', 'ravel_multi_index')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114456)
# Adding element type (line 21)
str_114457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'str', 'unravel_index')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114457)
# Adding element type (line 21)
str_114458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 42), 'str', 'mgrid')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114458)
# Adding element type (line 21)
str_114459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 51), 'str', 'ogrid')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114459)
# Adding element type (line 21)
str_114460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 60), 'str', 'r_')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114460)
# Adding element type (line 21)
str_114461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 66), 'str', 'c_')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114461)
# Adding element type (line 21)
str_114462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 4), 'str', 's_')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114462)
# Adding element type (line 21)
str_114463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 10), 'str', 'index_exp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114463)
# Adding element type (line 21)
str_114464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'str', 'ix_')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114464)
# Adding element type (line 21)
str_114465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'str', 'ndenumerate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114465)
# Adding element type (line 21)
str_114466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 45), 'str', 'ndindex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114466)
# Adding element type (line 21)
str_114467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 56), 'str', 'fill_diagonal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114467)
# Adding element type (line 21)
str_114468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'str', 'diag_indices')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114468)
# Adding element type (line 21)
str_114469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'str', 'diag_indices_from')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_114455, str_114469)

# Assigning a type to the variable '__all__' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), '__all__', list_114455)

@norecursion
def ix_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ix_'
    module_type_store = module_type_store.open_function_context('ix_', 28, 0, False)
    
    # Passed parameters checking function
    ix_.stypy_localization = localization
    ix_.stypy_type_of_self = None
    ix_.stypy_type_store = module_type_store
    ix_.stypy_function_name = 'ix_'
    ix_.stypy_param_names_list = []
    ix_.stypy_varargs_param_name = 'args'
    ix_.stypy_kwargs_param_name = None
    ix_.stypy_call_defaults = defaults
    ix_.stypy_call_varargs = varargs
    ix_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ix_', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ix_', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ix_(...)' code ##################

    str_114470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, (-1)), 'str', '\n    Construct an open mesh from multiple sequences.\n\n    This function takes N 1-D sequences and returns N outputs with N\n    dimensions each, such that the shape is 1 in all but one dimension\n    and the dimension with the non-unit shape value cycles through all\n    N dimensions.\n\n    Using `ix_` one can quickly construct index arrays that will index\n    the cross product. ``a[np.ix_([1,3],[2,5])]`` returns the array\n    ``[[a[1,2] a[1,5]], [a[3,2] a[3,5]]]``.\n\n    Parameters\n    ----------\n    args : 1-D sequences\n\n    Returns\n    -------\n    out : tuple of ndarrays\n        N arrays with N dimensions each, with N the number of input\n        sequences. Together these arrays form an open mesh.\n\n    See Also\n    --------\n    ogrid, mgrid, meshgrid\n\n    Examples\n    --------\n    >>> a = np.arange(10).reshape(2, 5)\n    >>> a\n    array([[0, 1, 2, 3, 4],\n           [5, 6, 7, 8, 9]])\n    >>> ixgrid = np.ix_([0,1], [2,4])\n    >>> ixgrid\n    (array([[0],\n           [1]]), array([[2, 4]]))\n    >>> ixgrid[0].shape, ixgrid[1].shape\n    ((2, 1), (1, 2))\n    >>> a[ixgrid]\n    array([[2, 4],\n           [7, 9]])\n\n    ')
    
    # Assigning a List to a Name (line 72):
    
    # Assigning a List to a Name (line 72):
    
    # Obtaining an instance of the builtin type 'list' (line 72)
    list_114471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 72)
    
    # Assigning a type to the variable 'out' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'out', list_114471)
    
    # Assigning a Call to a Name (line 73):
    
    # Assigning a Call to a Name (line 73):
    
    # Call to len(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'args' (line 73)
    args_114473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 13), 'args', False)
    # Processing the call keyword arguments (line 73)
    kwargs_114474 = {}
    # Getting the type of 'len' (line 73)
    len_114472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 9), 'len', False)
    # Calling len(args, kwargs) (line 73)
    len_call_result_114475 = invoke(stypy.reporting.localization.Localization(__file__, 73, 9), len_114472, *[args_114473], **kwargs_114474)
    
    # Assigning a type to the variable 'nd' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'nd', len_call_result_114475)
    
    
    # Call to enumerate(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'args' (line 74)
    args_114477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'args', False)
    # Processing the call keyword arguments (line 74)
    kwargs_114478 = {}
    # Getting the type of 'enumerate' (line 74)
    enumerate_114476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 74)
    enumerate_call_result_114479 = invoke(stypy.reporting.localization.Localization(__file__, 74, 18), enumerate_114476, *[args_114477], **kwargs_114478)
    
    # Testing the type of a for loop iterable (line 74)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 4), enumerate_call_result_114479)
    # Getting the type of the for loop variable (line 74)
    for_loop_var_114480 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 4), enumerate_call_result_114479)
    # Assigning a type to the variable 'k' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 4), for_loop_var_114480))
    # Assigning a type to the variable 'new' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'new', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 4), for_loop_var_114480))
    # SSA begins for a for statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 75):
    
    # Assigning a Call to a Name (line 75):
    
    # Call to asarray(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'new' (line 75)
    new_114482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'new', False)
    # Processing the call keyword arguments (line 75)
    kwargs_114483 = {}
    # Getting the type of 'asarray' (line 75)
    asarray_114481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'asarray', False)
    # Calling asarray(args, kwargs) (line 75)
    asarray_call_result_114484 = invoke(stypy.reporting.localization.Localization(__file__, 75, 14), asarray_114481, *[new_114482], **kwargs_114483)
    
    # Assigning a type to the variable 'new' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'new', asarray_call_result_114484)
    
    
    # Getting the type of 'new' (line 76)
    new_114485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'new')
    # Obtaining the member 'ndim' of a type (line 76)
    ndim_114486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 11), new_114485, 'ndim')
    int_114487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 23), 'int')
    # Applying the binary operator '!=' (line 76)
    result_ne_114488 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 11), '!=', ndim_114486, int_114487)
    
    # Testing the type of an if condition (line 76)
    if_condition_114489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), result_ne_114488)
    # Assigning a type to the variable 'if_condition_114489' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_114489', if_condition_114489)
    # SSA begins for if statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 77)
    # Processing the call arguments (line 77)
    str_114491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 29), 'str', 'Cross index must be 1 dimensional')
    # Processing the call keyword arguments (line 77)
    kwargs_114492 = {}
    # Getting the type of 'ValueError' (line 77)
    ValueError_114490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 77)
    ValueError_call_result_114493 = invoke(stypy.reporting.localization.Localization(__file__, 77, 18), ValueError_114490, *[str_114491], **kwargs_114492)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 77, 12), ValueError_call_result_114493, 'raise parameter', BaseException)
    # SSA join for if statement (line 76)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'new' (line 78)
    new_114494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'new')
    # Obtaining the member 'size' of a type (line 78)
    size_114495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 11), new_114494, 'size')
    int_114496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 23), 'int')
    # Applying the binary operator '==' (line 78)
    result_eq_114497 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), '==', size_114495, int_114496)
    
    # Testing the type of an if condition (line 78)
    if_condition_114498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_eq_114497)
    # Assigning a type to the variable 'if_condition_114498' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_114498', if_condition_114498)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to astype(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of '_nx' (line 80)
    _nx_114501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), '_nx', False)
    # Obtaining the member 'intp' of a type (line 80)
    intp_114502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 29), _nx_114501, 'intp')
    # Processing the call keyword arguments (line 80)
    kwargs_114503 = {}
    # Getting the type of 'new' (line 80)
    new_114499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'new', False)
    # Obtaining the member 'astype' of a type (line 80)
    astype_114500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 18), new_114499, 'astype')
    # Calling astype(args, kwargs) (line 80)
    astype_call_result_114504 = invoke(stypy.reporting.localization.Localization(__file__, 80, 18), astype_114500, *[intp_114502], **kwargs_114503)
    
    # Assigning a type to the variable 'new' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'new', astype_call_result_114504)
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to issubdtype(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'new' (line 81)
    new_114506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'new', False)
    # Obtaining the member 'dtype' of a type (line 81)
    dtype_114507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 22), new_114506, 'dtype')
    # Getting the type of '_nx' (line 81)
    _nx_114508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 33), '_nx', False)
    # Obtaining the member 'bool_' of a type (line 81)
    bool__114509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 33), _nx_114508, 'bool_')
    # Processing the call keyword arguments (line 81)
    kwargs_114510 = {}
    # Getting the type of 'issubdtype' (line 81)
    issubdtype_114505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'issubdtype', False)
    # Calling issubdtype(args, kwargs) (line 81)
    issubdtype_call_result_114511 = invoke(stypy.reporting.localization.Localization(__file__, 81, 11), issubdtype_114505, *[dtype_114507, bool__114509], **kwargs_114510)
    
    # Testing the type of an if condition (line 81)
    if_condition_114512 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), issubdtype_call_result_114511)
    # Assigning a type to the variable 'if_condition_114512' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_114512', if_condition_114512)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 82):
    
    # Assigning a Call to a Name:
    
    # Call to nonzero(...): (line 82)
    # Processing the call keyword arguments (line 82)
    kwargs_114515 = {}
    # Getting the type of 'new' (line 82)
    new_114513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'new', False)
    # Obtaining the member 'nonzero' of a type (line 82)
    nonzero_114514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 19), new_114513, 'nonzero')
    # Calling nonzero(args, kwargs) (line 82)
    nonzero_call_result_114516 = invoke(stypy.reporting.localization.Localization(__file__, 82, 19), nonzero_114514, *[], **kwargs_114515)
    
    # Assigning a type to the variable 'call_assignment_114433' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'call_assignment_114433', nonzero_call_result_114516)
    
    # Assigning a Call to a Name (line 82):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_114519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 12), 'int')
    # Processing the call keyword arguments
    kwargs_114520 = {}
    # Getting the type of 'call_assignment_114433' (line 82)
    call_assignment_114433_114517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'call_assignment_114433', False)
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___114518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), call_assignment_114433_114517, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_114521 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___114518, *[int_114519], **kwargs_114520)
    
    # Assigning a type to the variable 'call_assignment_114434' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'call_assignment_114434', getitem___call_result_114521)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'call_assignment_114434' (line 82)
    call_assignment_114434_114522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'call_assignment_114434')
    # Assigning a type to the variable 'new' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'new', call_assignment_114434_114522)
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to reshape(...): (line 83)
    # Processing the call arguments (line 83)
    
    # Obtaining an instance of the builtin type 'tuple' (line 83)
    tuple_114525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 83)
    # Adding element type (line 83)
    int_114526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 27), tuple_114525, int_114526)
    
    # Getting the type of 'k' (line 83)
    k_114527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 31), 'k', False)
    # Applying the binary operator '*' (line 83)
    result_mul_114528 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 26), '*', tuple_114525, k_114527)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 83)
    tuple_114529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 83)
    # Adding element type (line 83)
    # Getting the type of 'new' (line 83)
    new_114530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 36), 'new', False)
    # Obtaining the member 'size' of a type (line 83)
    size_114531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 36), new_114530, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 36), tuple_114529, size_114531)
    
    # Applying the binary operator '+' (line 83)
    result_add_114532 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 26), '+', result_mul_114528, tuple_114529)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 83)
    tuple_114533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 83)
    # Adding element type (line 83)
    int_114534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 50), tuple_114533, int_114534)
    
    # Getting the type of 'nd' (line 83)
    nd_114535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 55), 'nd', False)
    # Getting the type of 'k' (line 83)
    k_114536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 58), 'k', False)
    # Applying the binary operator '-' (line 83)
    result_sub_114537 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 55), '-', nd_114535, k_114536)
    
    int_114538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 60), 'int')
    # Applying the binary operator '-' (line 83)
    result_sub_114539 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 59), '-', result_sub_114537, int_114538)
    
    # Applying the binary operator '*' (line 83)
    result_mul_114540 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 49), '*', tuple_114533, result_sub_114539)
    
    # Applying the binary operator '+' (line 83)
    result_add_114541 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 47), '+', result_add_114532, result_mul_114540)
    
    # Processing the call keyword arguments (line 83)
    kwargs_114542 = {}
    # Getting the type of 'new' (line 83)
    new_114523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 14), 'new', False)
    # Obtaining the member 'reshape' of a type (line 83)
    reshape_114524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 14), new_114523, 'reshape')
    # Calling reshape(args, kwargs) (line 83)
    reshape_call_result_114543 = invoke(stypy.reporting.localization.Localization(__file__, 83, 14), reshape_114524, *[result_add_114541], **kwargs_114542)
    
    # Assigning a type to the variable 'new' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'new', reshape_call_result_114543)
    
    # Call to append(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'new' (line 84)
    new_114546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'new', False)
    # Processing the call keyword arguments (line 84)
    kwargs_114547 = {}
    # Getting the type of 'out' (line 84)
    out_114544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'out', False)
    # Obtaining the member 'append' of a type (line 84)
    append_114545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), out_114544, 'append')
    # Calling append(args, kwargs) (line 84)
    append_call_result_114548 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), append_114545, *[new_114546], **kwargs_114547)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to tuple(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'out' (line 85)
    out_114550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'out', False)
    # Processing the call keyword arguments (line 85)
    kwargs_114551 = {}
    # Getting the type of 'tuple' (line 85)
    tuple_114549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'tuple', False)
    # Calling tuple(args, kwargs) (line 85)
    tuple_call_result_114552 = invoke(stypy.reporting.localization.Localization(__file__, 85, 11), tuple_114549, *[out_114550], **kwargs_114551)
    
    # Assigning a type to the variable 'stypy_return_type' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type', tuple_call_result_114552)
    
    # ################# End of 'ix_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ix_' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_114553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114553)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ix_'
    return stypy_return_type_114553

# Assigning a type to the variable 'ix_' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'ix_', ix_)
# Declaration of the 'nd_grid' class

class nd_grid(object, ):
    str_114554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, (-1)), 'str', '\n    Construct a multi-dimensional "meshgrid".\n\n    ``grid = nd_grid()`` creates an instance which will return a mesh-grid\n    when indexed.  The dimension and number of the output arrays are equal\n    to the number of indexing dimensions.  If the step length is not a\n    complex number, then the stop is not inclusive.\n\n    However, if the step length is a **complex number** (e.g. 5j), then the\n    integer part of its magnitude is interpreted as specifying the\n    number of points to create between the start and stop values, where\n    the stop value **is inclusive**.\n\n    If instantiated with an argument of ``sparse=True``, the mesh-grid is\n    open (or not fleshed out) so that only one-dimension of each returned\n    argument is greater than 1.\n\n    Parameters\n    ----------\n    sparse : bool, optional\n        Whether the grid is sparse or not. Default is False.\n\n    Notes\n    -----\n    Two instances of `nd_grid` are made available in the NumPy namespace,\n    `mgrid` and `ogrid`::\n\n        mgrid = nd_grid(sparse=False)\n        ogrid = nd_grid(sparse=True)\n\n    Users should use these pre-defined instances instead of using `nd_grid`\n    directly.\n\n    Examples\n    --------\n    >>> mgrid = np.lib.index_tricks.nd_grid()\n    >>> mgrid[0:5,0:5]\n    array([[[0, 0, 0, 0, 0],\n            [1, 1, 1, 1, 1],\n            [2, 2, 2, 2, 2],\n            [3, 3, 3, 3, 3],\n            [4, 4, 4, 4, 4]],\n           [[0, 1, 2, 3, 4],\n            [0, 1, 2, 3, 4],\n            [0, 1, 2, 3, 4],\n            [0, 1, 2, 3, 4],\n            [0, 1, 2, 3, 4]]])\n    >>> mgrid[-1:1:5j]\n    array([-1. , -0.5,  0. ,  0.5,  1. ])\n\n    >>> ogrid = np.lib.index_tricks.nd_grid(sparse=True)\n    >>> ogrid[0:5,0:5]\n    [array([[0],\n            [1],\n            [2],\n            [3],\n            [4]]), array([[0, 1, 2, 3, 4]])]\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 148)
        False_114555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), 'False')
        defaults = [False_114555]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'nd_grid.__init__', ['sparse'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['sparse'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 149):
        
        # Assigning a Name to a Attribute (line 149):
        # Getting the type of 'sparse' (line 149)
        sparse_114556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 22), 'sparse')
        # Getting the type of 'self' (line 149)
        self_114557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self')
        # Setting the type of the member 'sparse' of a type (line 149)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_114557, 'sparse', sparse_114556)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        nd_grid.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        nd_grid.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        nd_grid.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        nd_grid.__getitem__.__dict__.__setitem__('stypy_function_name', 'nd_grid.__getitem__')
        nd_grid.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['key'])
        nd_grid.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        nd_grid.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        nd_grid.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        nd_grid.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        nd_grid.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        nd_grid.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'nd_grid.__getitem__', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a List to a Name (line 153):
        
        # Assigning a List to a Name (line 153):
        
        # Obtaining an instance of the builtin type 'list' (line 153)
        list_114558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 153)
        
        # Assigning a type to the variable 'size' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'size', list_114558)
        
        # Assigning a Name to a Name (line 154):
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'int' (line 154)
        int_114559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 18), 'int')
        # Assigning a type to the variable 'typ' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'typ', int_114559)
        
        
        # Call to range(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Call to len(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'key' (line 155)
        key_114562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 31), 'key', False)
        # Processing the call keyword arguments (line 155)
        kwargs_114563 = {}
        # Getting the type of 'len' (line 155)
        len_114561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 27), 'len', False)
        # Calling len(args, kwargs) (line 155)
        len_call_result_114564 = invoke(stypy.reporting.localization.Localization(__file__, 155, 27), len_114561, *[key_114562], **kwargs_114563)
        
        # Processing the call keyword arguments (line 155)
        kwargs_114565 = {}
        # Getting the type of 'range' (line 155)
        range_114560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'range', False)
        # Calling range(args, kwargs) (line 155)
        range_call_result_114566 = invoke(stypy.reporting.localization.Localization(__file__, 155, 21), range_114560, *[len_call_result_114564], **kwargs_114565)
        
        # Testing the type of a for loop iterable (line 155)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 155, 12), range_call_result_114566)
        # Getting the type of the for loop variable (line 155)
        for_loop_var_114567 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 155, 12), range_call_result_114566)
        # Assigning a type to the variable 'k' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'k', for_loop_var_114567)
        # SSA begins for a for statement (line 155)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Attribute to a Name (line 156):
        
        # Assigning a Attribute to a Name (line 156):
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 156)
        k_114568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 27), 'k')
        # Getting the type of 'key' (line 156)
        key_114569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 23), 'key')
        # Obtaining the member '__getitem__' of a type (line 156)
        getitem___114570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 23), key_114569, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 156)
        subscript_call_result_114571 = invoke(stypy.reporting.localization.Localization(__file__, 156, 23), getitem___114570, k_114568)
        
        # Obtaining the member 'step' of a type (line 156)
        step_114572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 23), subscript_call_result_114571, 'step')
        # Assigning a type to the variable 'step' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'step', step_114572)
        
        # Assigning a Attribute to a Name (line 157):
        
        # Assigning a Attribute to a Name (line 157):
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 157)
        k_114573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 28), 'k')
        # Getting the type of 'key' (line 157)
        key_114574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'key')
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___114575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 24), key_114574, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_114576 = invoke(stypy.reporting.localization.Localization(__file__, 157, 24), getitem___114575, k_114573)
        
        # Obtaining the member 'start' of a type (line 157)
        start_114577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 24), subscript_call_result_114576, 'start')
        # Assigning a type to the variable 'start' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'start', start_114577)
        
        # Type idiom detected: calculating its left and rigth part (line 158)
        # Getting the type of 'start' (line 158)
        start_114578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'start')
        # Getting the type of 'None' (line 158)
        None_114579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'None')
        
        (may_be_114580, more_types_in_union_114581) = may_be_none(start_114578, None_114579)

        if may_be_114580:

            if more_types_in_union_114581:
                # Runtime conditional SSA (line 158)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 159):
            
            # Assigning a Num to a Name (line 159):
            int_114582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 28), 'int')
            # Assigning a type to the variable 'start' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'start', int_114582)

            if more_types_in_union_114581:
                # SSA join for if statement (line 158)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 160)
        # Getting the type of 'step' (line 160)
        step_114583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 19), 'step')
        # Getting the type of 'None' (line 160)
        None_114584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 27), 'None')
        
        (may_be_114585, more_types_in_union_114586) = may_be_none(step_114583, None_114584)

        if may_be_114585:

            if more_types_in_union_114586:
                # Runtime conditional SSA (line 160)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 161):
            
            # Assigning a Num to a Name (line 161):
            int_114587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 27), 'int')
            # Assigning a type to the variable 'step' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'step', int_114587)

            if more_types_in_union_114586:
                # SSA join for if statement (line 160)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 162)
        # Getting the type of 'complex' (line 162)
        complex_114588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 36), 'complex')
        # Getting the type of 'step' (line 162)
        step_114589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 30), 'step')
        
        (may_be_114590, more_types_in_union_114591) = may_be_subtype(complex_114588, step_114589)

        if may_be_114590:

            if more_types_in_union_114591:
                # Runtime conditional SSA (line 162)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'step' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'step', remove_not_subtype_from_union(step_114589, complex))
            
            # Call to append(...): (line 163)
            # Processing the call arguments (line 163)
            
            # Call to int(...): (line 163)
            # Processing the call arguments (line 163)
            
            # Call to abs(...): (line 163)
            # Processing the call arguments (line 163)
            # Getting the type of 'step' (line 163)
            step_114596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 40), 'step', False)
            # Processing the call keyword arguments (line 163)
            kwargs_114597 = {}
            # Getting the type of 'abs' (line 163)
            abs_114595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 36), 'abs', False)
            # Calling abs(args, kwargs) (line 163)
            abs_call_result_114598 = invoke(stypy.reporting.localization.Localization(__file__, 163, 36), abs_114595, *[step_114596], **kwargs_114597)
            
            # Processing the call keyword arguments (line 163)
            kwargs_114599 = {}
            # Getting the type of 'int' (line 163)
            int_114594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 32), 'int', False)
            # Calling int(args, kwargs) (line 163)
            int_call_result_114600 = invoke(stypy.reporting.localization.Localization(__file__, 163, 32), int_114594, *[abs_call_result_114598], **kwargs_114599)
            
            # Processing the call keyword arguments (line 163)
            kwargs_114601 = {}
            # Getting the type of 'size' (line 163)
            size_114592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'size', False)
            # Obtaining the member 'append' of a type (line 163)
            append_114593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 20), size_114592, 'append')
            # Calling append(args, kwargs) (line 163)
            append_call_result_114602 = invoke(stypy.reporting.localization.Localization(__file__, 163, 20), append_114593, *[int_call_result_114600], **kwargs_114601)
            
            
            # Assigning a Name to a Name (line 164):
            
            # Assigning a Name to a Name (line 164):
            # Getting the type of 'float' (line 164)
            float_114603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 26), 'float')
            # Assigning a type to the variable 'typ' (line 164)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'typ', float_114603)

            if more_types_in_union_114591:
                # Runtime conditional SSA for else branch (line 162)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_114590) or more_types_in_union_114591):
            # Assigning a type to the variable 'step' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'step', remove_subtype_from_union(step_114589, complex))
            
            # Call to append(...): (line 166)
            # Processing the call arguments (line 166)
            
            # Call to int(...): (line 167)
            # Processing the call arguments (line 167)
            
            # Call to ceil(...): (line 167)
            # Processing the call arguments (line 167)
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 167)
            k_114609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 43), 'k', False)
            # Getting the type of 'key' (line 167)
            key_114610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'key', False)
            # Obtaining the member '__getitem__' of a type (line 167)
            getitem___114611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 39), key_114610, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 167)
            subscript_call_result_114612 = invoke(stypy.reporting.localization.Localization(__file__, 167, 39), getitem___114611, k_114609)
            
            # Obtaining the member 'stop' of a type (line 167)
            stop_114613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 39), subscript_call_result_114612, 'stop')
            # Getting the type of 'start' (line 167)
            start_114614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 53), 'start', False)
            # Applying the binary operator '-' (line 167)
            result_sub_114615 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 39), '-', stop_114613, start_114614)
            
            # Getting the type of 'step' (line 167)
            step_114616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 61), 'step', False)
            float_114617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 66), 'float')
            # Applying the binary operator '*' (line 167)
            result_mul_114618 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 61), '*', step_114616, float_114617)
            
            # Applying the binary operator 'div' (line 167)
            result_div_114619 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 38), 'div', result_sub_114615, result_mul_114618)
            
            # Processing the call keyword arguments (line 167)
            kwargs_114620 = {}
            # Getting the type of 'math' (line 167)
            math_114607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'math', False)
            # Obtaining the member 'ceil' of a type (line 167)
            ceil_114608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 28), math_114607, 'ceil')
            # Calling ceil(args, kwargs) (line 167)
            ceil_call_result_114621 = invoke(stypy.reporting.localization.Localization(__file__, 167, 28), ceil_114608, *[result_div_114619], **kwargs_114620)
            
            # Processing the call keyword arguments (line 167)
            kwargs_114622 = {}
            # Getting the type of 'int' (line 167)
            int_114606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'int', False)
            # Calling int(args, kwargs) (line 167)
            int_call_result_114623 = invoke(stypy.reporting.localization.Localization(__file__, 167, 24), int_114606, *[ceil_call_result_114621], **kwargs_114622)
            
            # Processing the call keyword arguments (line 166)
            kwargs_114624 = {}
            # Getting the type of 'size' (line 166)
            size_114604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'size', False)
            # Obtaining the member 'append' of a type (line 166)
            append_114605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 20), size_114604, 'append')
            # Calling append(args, kwargs) (line 166)
            append_call_result_114625 = invoke(stypy.reporting.localization.Localization(__file__, 166, 20), append_114605, *[int_call_result_114623], **kwargs_114624)
            

            if (may_be_114590 and more_types_in_union_114591):
                # SSA join for if statement (line 162)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'step' (line 168)
        step_114627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 31), 'step', False)
        # Getting the type of 'float' (line 168)
        float_114628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), 'float', False)
        # Processing the call keyword arguments (line 168)
        kwargs_114629 = {}
        # Getting the type of 'isinstance' (line 168)
        isinstance_114626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 168)
        isinstance_call_result_114630 = invoke(stypy.reporting.localization.Localization(__file__, 168, 20), isinstance_114626, *[step_114627, float_114628], **kwargs_114629)
        
        
        # Call to isinstance(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'start' (line 169)
        start_114632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'start', False)
        # Getting the type of 'float' (line 169)
        float_114633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 42), 'float', False)
        # Processing the call keyword arguments (line 169)
        kwargs_114634 = {}
        # Getting the type of 'isinstance' (line 169)
        isinstance_114631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 169)
        isinstance_call_result_114635 = invoke(stypy.reporting.localization.Localization(__file__, 169, 24), isinstance_114631, *[start_114632, float_114633], **kwargs_114634)
        
        # Applying the binary operator 'or' (line 168)
        result_or_keyword_114636 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 20), 'or', isinstance_call_result_114630, isinstance_call_result_114635)
        
        # Call to isinstance(...): (line 170)
        # Processing the call arguments (line 170)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 170)
        k_114638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 39), 'k', False)
        # Getting the type of 'key' (line 170)
        key_114639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 35), 'key', False)
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___114640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 35), key_114639, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_114641 = invoke(stypy.reporting.localization.Localization(__file__, 170, 35), getitem___114640, k_114638)
        
        # Obtaining the member 'stop' of a type (line 170)
        stop_114642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 35), subscript_call_result_114641, 'stop')
        # Getting the type of 'float' (line 170)
        float_114643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 48), 'float', False)
        # Processing the call keyword arguments (line 170)
        kwargs_114644 = {}
        # Getting the type of 'isinstance' (line 170)
        isinstance_114637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 170)
        isinstance_call_result_114645 = invoke(stypy.reporting.localization.Localization(__file__, 170, 24), isinstance_114637, *[stop_114642, float_114643], **kwargs_114644)
        
        # Applying the binary operator 'or' (line 168)
        result_or_keyword_114646 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 20), 'or', result_or_keyword_114636, isinstance_call_result_114645)
        
        # Testing the type of an if condition (line 168)
        if_condition_114647 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 16), result_or_keyword_114646)
        # Assigning a type to the variable 'if_condition_114647' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'if_condition_114647', if_condition_114647)
        # SSA begins for if statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 171):
        
        # Assigning a Name to a Name (line 171):
        # Getting the type of 'float' (line 171)
        float_114648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'float')
        # Assigning a type to the variable 'typ' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'typ', float_114648)
        # SSA join for if statement (line 168)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 172)
        self_114649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'self')
        # Obtaining the member 'sparse' of a type (line 172)
        sparse_114650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 15), self_114649, 'sparse')
        # Testing the type of an if condition (line 172)
        if_condition_114651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 12), sparse_114650)
        # Assigning a type to the variable 'if_condition_114651' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'if_condition_114651', if_condition_114651)
        # SSA begins for if statement (line 172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a ListComp to a Name (line 173):
        
        # Assigning a ListComp to a Name (line 173):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to zip(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'size' (line 174)
        size_114660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 42), 'size', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 174)
        tuple_114661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 174)
        # Adding element type (line 174)
        # Getting the type of 'typ' (line 174)
        typ_114662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 49), 'typ', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 49), tuple_114661, typ_114662)
        
        
        # Call to len(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'size' (line 174)
        size_114664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 59), 'size', False)
        # Processing the call keyword arguments (line 174)
        kwargs_114665 = {}
        # Getting the type of 'len' (line 174)
        len_114663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 55), 'len', False)
        # Calling len(args, kwargs) (line 174)
        len_call_result_114666 = invoke(stypy.reporting.localization.Localization(__file__, 174, 55), len_114663, *[size_114664], **kwargs_114665)
        
        # Applying the binary operator '*' (line 174)
        result_mul_114667 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 48), '*', tuple_114661, len_call_result_114666)
        
        # Processing the call keyword arguments (line 174)
        kwargs_114668 = {}
        # Getting the type of 'zip' (line 174)
        zip_114659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'zip', False)
        # Calling zip(args, kwargs) (line 174)
        zip_call_result_114669 = invoke(stypy.reporting.localization.Localization(__file__, 174, 38), zip_114659, *[size_114660, result_mul_114667], **kwargs_114668)
        
        comprehension_114670 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 22), zip_call_result_114669)
        # Assigning a type to the variable '_x' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), '_x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 22), comprehension_114670))
        # Assigning a type to the variable '_t' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), '_t', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 22), comprehension_114670))
        
        # Call to arange(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of '_x' (line 173)
        _x_114654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 33), '_x', False)
        # Processing the call keyword arguments (line 173)
        # Getting the type of '_t' (line 173)
        _t_114655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 43), '_t', False)
        keyword_114656 = _t_114655
        kwargs_114657 = {'dtype': keyword_114656}
        # Getting the type of '_nx' (line 173)
        _nx_114652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), '_nx', False)
        # Obtaining the member 'arange' of a type (line 173)
        arange_114653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 22), _nx_114652, 'arange')
        # Calling arange(args, kwargs) (line 173)
        arange_call_result_114658 = invoke(stypy.reporting.localization.Localization(__file__, 173, 22), arange_114653, *[_x_114654], **kwargs_114657)
        
        list_114671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 22), list_114671, arange_call_result_114658)
        # Assigning a type to the variable 'nn' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'nn', list_114671)
        # SSA branch for the else part of an if statement (line 172)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to indices(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'size' (line 176)
        size_114674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), 'size', False)
        # Getting the type of 'typ' (line 176)
        typ_114675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 39), 'typ', False)
        # Processing the call keyword arguments (line 176)
        kwargs_114676 = {}
        # Getting the type of '_nx' (line 176)
        _nx_114672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 21), '_nx', False)
        # Obtaining the member 'indices' of a type (line 176)
        indices_114673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 21), _nx_114672, 'indices')
        # Calling indices(args, kwargs) (line 176)
        indices_call_result_114677 = invoke(stypy.reporting.localization.Localization(__file__, 176, 21), indices_114673, *[size_114674, typ_114675], **kwargs_114676)
        
        # Assigning a type to the variable 'nn' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'nn', indices_call_result_114677)
        # SSA join for if statement (line 172)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to range(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Call to len(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'size' (line 177)
        size_114680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 31), 'size', False)
        # Processing the call keyword arguments (line 177)
        kwargs_114681 = {}
        # Getting the type of 'len' (line 177)
        len_114679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 27), 'len', False)
        # Calling len(args, kwargs) (line 177)
        len_call_result_114682 = invoke(stypy.reporting.localization.Localization(__file__, 177, 27), len_114679, *[size_114680], **kwargs_114681)
        
        # Processing the call keyword arguments (line 177)
        kwargs_114683 = {}
        # Getting the type of 'range' (line 177)
        range_114678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 21), 'range', False)
        # Calling range(args, kwargs) (line 177)
        range_call_result_114684 = invoke(stypy.reporting.localization.Localization(__file__, 177, 21), range_114678, *[len_call_result_114682], **kwargs_114683)
        
        # Testing the type of a for loop iterable (line 177)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 177, 12), range_call_result_114684)
        # Getting the type of the for loop variable (line 177)
        for_loop_var_114685 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 177, 12), range_call_result_114684)
        # Assigning a type to the variable 'k' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'k', for_loop_var_114685)
        # SSA begins for a for statement (line 177)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Attribute to a Name (line 178):
        
        # Assigning a Attribute to a Name (line 178):
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 178)
        k_114686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'k')
        # Getting the type of 'key' (line 178)
        key_114687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'key')
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___114688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 23), key_114687, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_114689 = invoke(stypy.reporting.localization.Localization(__file__, 178, 23), getitem___114688, k_114686)
        
        # Obtaining the member 'step' of a type (line 178)
        step_114690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 23), subscript_call_result_114689, 'step')
        # Assigning a type to the variable 'step' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'step', step_114690)
        
        # Assigning a Attribute to a Name (line 179):
        
        # Assigning a Attribute to a Name (line 179):
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 179)
        k_114691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 28), 'k')
        # Getting the type of 'key' (line 179)
        key_114692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 24), 'key')
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___114693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 24), key_114692, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_114694 = invoke(stypy.reporting.localization.Localization(__file__, 179, 24), getitem___114693, k_114691)
        
        # Obtaining the member 'start' of a type (line 179)
        start_114695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 24), subscript_call_result_114694, 'start')
        # Assigning a type to the variable 'start' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'start', start_114695)
        
        # Type idiom detected: calculating its left and rigth part (line 180)
        # Getting the type of 'start' (line 180)
        start_114696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'start')
        # Getting the type of 'None' (line 180)
        None_114697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'None')
        
        (may_be_114698, more_types_in_union_114699) = may_be_none(start_114696, None_114697)

        if may_be_114698:

            if more_types_in_union_114699:
                # Runtime conditional SSA (line 180)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 181):
            
            # Assigning a Num to a Name (line 181):
            int_114700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 28), 'int')
            # Assigning a type to the variable 'start' (line 181)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 20), 'start', int_114700)

            if more_types_in_union_114699:
                # SSA join for if statement (line 180)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 182)
        # Getting the type of 'step' (line 182)
        step_114701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 19), 'step')
        # Getting the type of 'None' (line 182)
        None_114702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 27), 'None')
        
        (may_be_114703, more_types_in_union_114704) = may_be_none(step_114701, None_114702)

        if may_be_114703:

            if more_types_in_union_114704:
                # Runtime conditional SSA (line 182)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 183):
            
            # Assigning a Num to a Name (line 183):
            int_114705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 27), 'int')
            # Assigning a type to the variable 'step' (line 183)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'step', int_114705)

            if more_types_in_union_114704:
                # SSA join for if statement (line 182)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 184)
        # Getting the type of 'complex' (line 184)
        complex_114706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 36), 'complex')
        # Getting the type of 'step' (line 184)
        step_114707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 30), 'step')
        
        (may_be_114708, more_types_in_union_114709) = may_be_subtype(complex_114706, step_114707)

        if may_be_114708:

            if more_types_in_union_114709:
                # Runtime conditional SSA (line 184)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'step' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'step', remove_not_subtype_from_union(step_114707, complex))
            
            # Assigning a Call to a Name (line 185):
            
            # Assigning a Call to a Name (line 185):
            
            # Call to int(...): (line 185)
            # Processing the call arguments (line 185)
            
            # Call to abs(...): (line 185)
            # Processing the call arguments (line 185)
            # Getting the type of 'step' (line 185)
            step_114712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 35), 'step', False)
            # Processing the call keyword arguments (line 185)
            kwargs_114713 = {}
            # Getting the type of 'abs' (line 185)
            abs_114711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 31), 'abs', False)
            # Calling abs(args, kwargs) (line 185)
            abs_call_result_114714 = invoke(stypy.reporting.localization.Localization(__file__, 185, 31), abs_114711, *[step_114712], **kwargs_114713)
            
            # Processing the call keyword arguments (line 185)
            kwargs_114715 = {}
            # Getting the type of 'int' (line 185)
            int_114710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 27), 'int', False)
            # Calling int(args, kwargs) (line 185)
            int_call_result_114716 = invoke(stypy.reporting.localization.Localization(__file__, 185, 27), int_114710, *[abs_call_result_114714], **kwargs_114715)
            
            # Assigning a type to the variable 'step' (line 185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'step', int_call_result_114716)
            
            
            # Getting the type of 'step' (line 186)
            step_114717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 23), 'step')
            int_114718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 31), 'int')
            # Applying the binary operator '!=' (line 186)
            result_ne_114719 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 23), '!=', step_114717, int_114718)
            
            # Testing the type of an if condition (line 186)
            if_condition_114720 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 20), result_ne_114719)
            # Assigning a type to the variable 'if_condition_114720' (line 186)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), 'if_condition_114720', if_condition_114720)
            # SSA begins for if statement (line 186)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 187):
            
            # Assigning a BinOp to a Name (line 187):
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 187)
            k_114721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 36), 'k')
            # Getting the type of 'key' (line 187)
            key_114722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 32), 'key')
            # Obtaining the member '__getitem__' of a type (line 187)
            getitem___114723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 32), key_114722, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 187)
            subscript_call_result_114724 = invoke(stypy.reporting.localization.Localization(__file__, 187, 32), getitem___114723, k_114721)
            
            # Obtaining the member 'stop' of a type (line 187)
            stop_114725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 32), subscript_call_result_114724, 'stop')
            # Getting the type of 'start' (line 187)
            start_114726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 46), 'start')
            # Applying the binary operator '-' (line 187)
            result_sub_114727 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 32), '-', stop_114725, start_114726)
            
            
            # Call to float(...): (line 187)
            # Processing the call arguments (line 187)
            # Getting the type of 'step' (line 187)
            step_114729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 59), 'step', False)
            int_114730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 64), 'int')
            # Applying the binary operator '-' (line 187)
            result_sub_114731 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 59), '-', step_114729, int_114730)
            
            # Processing the call keyword arguments (line 187)
            kwargs_114732 = {}
            # Getting the type of 'float' (line 187)
            float_114728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 53), 'float', False)
            # Calling float(args, kwargs) (line 187)
            float_call_result_114733 = invoke(stypy.reporting.localization.Localization(__file__, 187, 53), float_114728, *[result_sub_114731], **kwargs_114732)
            
            # Applying the binary operator 'div' (line 187)
            result_div_114734 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 31), 'div', result_sub_114727, float_call_result_114733)
            
            # Assigning a type to the variable 'step' (line 187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'step', result_div_114734)
            # SSA join for if statement (line 186)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_114709:
                # SSA join for if statement (line 184)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Subscript (line 188):
        
        # Assigning a BinOp to a Subscript (line 188):
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 188)
        k_114735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'k')
        # Getting the type of 'nn' (line 188)
        nn_114736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 25), 'nn')
        # Obtaining the member '__getitem__' of a type (line 188)
        getitem___114737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 25), nn_114736, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 188)
        subscript_call_result_114738 = invoke(stypy.reporting.localization.Localization(__file__, 188, 25), getitem___114737, k_114735)
        
        # Getting the type of 'step' (line 188)
        step_114739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 31), 'step')
        # Applying the binary operator '*' (line 188)
        result_mul_114740 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 25), '*', subscript_call_result_114738, step_114739)
        
        # Getting the type of 'start' (line 188)
        start_114741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 36), 'start')
        # Applying the binary operator '+' (line 188)
        result_add_114742 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 25), '+', result_mul_114740, start_114741)
        
        # Getting the type of 'nn' (line 188)
        nn_114743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'nn')
        # Getting the type of 'k' (line 188)
        k_114744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 19), 'k')
        # Storing an element on a container (line 188)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 16), nn_114743, (k_114744, result_add_114742))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 189)
        self_114745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'self')
        # Obtaining the member 'sparse' of a type (line 189)
        sparse_114746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 15), self_114745, 'sparse')
        # Testing the type of an if condition (line 189)
        if_condition_114747 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 12), sparse_114746)
        # Assigning a type to the variable 'if_condition_114747' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'if_condition_114747', if_condition_114747)
        # SSA begins for if statement (line 189)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 190):
        
        # Assigning a BinOp to a Name (line 190):
        
        # Obtaining an instance of the builtin type 'list' (line 190)
        list_114748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 190)
        # Adding element type (line 190)
        # Getting the type of '_nx' (line 190)
        _nx_114749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), '_nx')
        # Obtaining the member 'newaxis' of a type (line 190)
        newaxis_114750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 25), _nx_114749, 'newaxis')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 24), list_114748, newaxis_114750)
        
        
        # Call to len(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'size' (line 190)
        size_114752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 42), 'size', False)
        # Processing the call keyword arguments (line 190)
        kwargs_114753 = {}
        # Getting the type of 'len' (line 190)
        len_114751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 38), 'len', False)
        # Calling len(args, kwargs) (line 190)
        len_call_result_114754 = invoke(stypy.reporting.localization.Localization(__file__, 190, 38), len_114751, *[size_114752], **kwargs_114753)
        
        # Applying the binary operator '*' (line 190)
        result_mul_114755 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 24), '*', list_114748, len_call_result_114754)
        
        # Assigning a type to the variable 'slobj' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'slobj', result_mul_114755)
        
        
        # Call to range(...): (line 191)
        # Processing the call arguments (line 191)
        
        # Call to len(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'size' (line 191)
        size_114758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 35), 'size', False)
        # Processing the call keyword arguments (line 191)
        kwargs_114759 = {}
        # Getting the type of 'len' (line 191)
        len_114757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 31), 'len', False)
        # Calling len(args, kwargs) (line 191)
        len_call_result_114760 = invoke(stypy.reporting.localization.Localization(__file__, 191, 31), len_114757, *[size_114758], **kwargs_114759)
        
        # Processing the call keyword arguments (line 191)
        kwargs_114761 = {}
        # Getting the type of 'range' (line 191)
        range_114756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'range', False)
        # Calling range(args, kwargs) (line 191)
        range_call_result_114762 = invoke(stypy.reporting.localization.Localization(__file__, 191, 25), range_114756, *[len_call_result_114760], **kwargs_114761)
        
        # Testing the type of a for loop iterable (line 191)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 191, 16), range_call_result_114762)
        # Getting the type of the for loop variable (line 191)
        for_loop_var_114763 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 191, 16), range_call_result_114762)
        # Assigning a type to the variable 'k' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'k', for_loop_var_114763)
        # SSA begins for a for statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Subscript (line 192):
        
        # Assigning a Call to a Subscript (line 192):
        
        # Call to slice(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'None' (line 192)
        None_114765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 37), 'None', False)
        # Getting the type of 'None' (line 192)
        None_114766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 43), 'None', False)
        # Processing the call keyword arguments (line 192)
        kwargs_114767 = {}
        # Getting the type of 'slice' (line 192)
        slice_114764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 31), 'slice', False)
        # Calling slice(args, kwargs) (line 192)
        slice_call_result_114768 = invoke(stypy.reporting.localization.Localization(__file__, 192, 31), slice_114764, *[None_114765, None_114766], **kwargs_114767)
        
        # Getting the type of 'slobj' (line 192)
        slobj_114769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'slobj')
        # Getting the type of 'k' (line 192)
        k_114770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 26), 'k')
        # Storing an element on a container (line 192)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 20), slobj_114769, (k_114770, slice_call_result_114768))
        
        # Assigning a Subscript to a Subscript (line 193):
        
        # Assigning a Subscript to a Subscript (line 193):
        
        # Obtaining the type of the subscript
        # Getting the type of 'slobj' (line 193)
        slobj_114771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 34), 'slobj')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 193)
        k_114772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 31), 'k')
        # Getting the type of 'nn' (line 193)
        nn_114773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 28), 'nn')
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___114774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 28), nn_114773, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_114775 = invoke(stypy.reporting.localization.Localization(__file__, 193, 28), getitem___114774, k_114772)
        
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___114776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 28), subscript_call_result_114775, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_114777 = invoke(stypy.reporting.localization.Localization(__file__, 193, 28), getitem___114776, slobj_114771)
        
        # Getting the type of 'nn' (line 193)
        nn_114778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 20), 'nn')
        # Getting the type of 'k' (line 193)
        k_114779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'k')
        # Storing an element on a container (line 193)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 20), nn_114778, (k_114779, subscript_call_result_114777))
        
        # Assigning a Attribute to a Subscript (line 194):
        
        # Assigning a Attribute to a Subscript (line 194):
        # Getting the type of '_nx' (line 194)
        _nx_114780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 31), '_nx')
        # Obtaining the member 'newaxis' of a type (line 194)
        newaxis_114781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 31), _nx_114780, 'newaxis')
        # Getting the type of 'slobj' (line 194)
        slobj_114782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 20), 'slobj')
        # Getting the type of 'k' (line 194)
        k_114783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 26), 'k')
        # Storing an element on a container (line 194)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 20), slobj_114782, (k_114783, newaxis_114781))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 189)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'nn' (line 195)
        nn_114784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 19), 'nn')
        # Assigning a type to the variable 'stypy_return_type' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'stypy_return_type', nn_114784)
        # SSA branch for the except part of a try statement (line 152)
        # SSA branch for the except 'Tuple' branch of a try statement (line 152)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Attribute to a Name (line 197):
        
        # Assigning a Attribute to a Name (line 197):
        # Getting the type of 'key' (line 197)
        key_114785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'key')
        # Obtaining the member 'step' of a type (line 197)
        step_114786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 19), key_114785, 'step')
        # Assigning a type to the variable 'step' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'step', step_114786)
        
        # Assigning a Attribute to a Name (line 198):
        
        # Assigning a Attribute to a Name (line 198):
        # Getting the type of 'key' (line 198)
        key_114787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'key')
        # Obtaining the member 'stop' of a type (line 198)
        stop_114788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 19), key_114787, 'stop')
        # Assigning a type to the variable 'stop' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'stop', stop_114788)
        
        # Assigning a Attribute to a Name (line 199):
        
        # Assigning a Attribute to a Name (line 199):
        # Getting the type of 'key' (line 199)
        key_114789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'key')
        # Obtaining the member 'start' of a type (line 199)
        start_114790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 20), key_114789, 'start')
        # Assigning a type to the variable 'start' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'start', start_114790)
        
        # Type idiom detected: calculating its left and rigth part (line 200)
        # Getting the type of 'start' (line 200)
        start_114791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'start')
        # Getting the type of 'None' (line 200)
        None_114792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 24), 'None')
        
        (may_be_114793, more_types_in_union_114794) = may_be_none(start_114791, None_114792)

        if may_be_114793:

            if more_types_in_union_114794:
                # Runtime conditional SSA (line 200)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 201):
            
            # Assigning a Num to a Name (line 201):
            int_114795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 24), 'int')
            # Assigning a type to the variable 'start' (line 201)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'start', int_114795)

            if more_types_in_union_114794:
                # SSA join for if statement (line 200)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 202)
        # Getting the type of 'complex' (line 202)
        complex_114796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 32), 'complex')
        # Getting the type of 'step' (line 202)
        step_114797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 26), 'step')
        
        (may_be_114798, more_types_in_union_114799) = may_be_subtype(complex_114796, step_114797)

        if may_be_114798:

            if more_types_in_union_114799:
                # Runtime conditional SSA (line 202)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'step' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'step', remove_not_subtype_from_union(step_114797, complex))
            
            # Assigning a Call to a Name (line 203):
            
            # Assigning a Call to a Name (line 203):
            
            # Call to abs(...): (line 203)
            # Processing the call arguments (line 203)
            # Getting the type of 'step' (line 203)
            step_114801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 27), 'step', False)
            # Processing the call keyword arguments (line 203)
            kwargs_114802 = {}
            # Getting the type of 'abs' (line 203)
            abs_114800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'abs', False)
            # Calling abs(args, kwargs) (line 203)
            abs_call_result_114803 = invoke(stypy.reporting.localization.Localization(__file__, 203, 23), abs_114800, *[step_114801], **kwargs_114802)
            
            # Assigning a type to the variable 'step' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'step', abs_call_result_114803)
            
            # Assigning a Call to a Name (line 204):
            
            # Assigning a Call to a Name (line 204):
            
            # Call to int(...): (line 204)
            # Processing the call arguments (line 204)
            # Getting the type of 'step' (line 204)
            step_114805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 29), 'step', False)
            # Processing the call keyword arguments (line 204)
            kwargs_114806 = {}
            # Getting the type of 'int' (line 204)
            int_114804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 25), 'int', False)
            # Calling int(args, kwargs) (line 204)
            int_call_result_114807 = invoke(stypy.reporting.localization.Localization(__file__, 204, 25), int_114804, *[step_114805], **kwargs_114806)
            
            # Assigning a type to the variable 'length' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'length', int_call_result_114807)
            
            
            # Getting the type of 'step' (line 205)
            step_114808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'step')
            int_114809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 27), 'int')
            # Applying the binary operator '!=' (line 205)
            result_ne_114810 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 19), '!=', step_114808, int_114809)
            
            # Testing the type of an if condition (line 205)
            if_condition_114811 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 16), result_ne_114810)
            # Assigning a type to the variable 'if_condition_114811' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'if_condition_114811', if_condition_114811)
            # SSA begins for if statement (line 205)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 206):
            
            # Assigning a BinOp to a Name (line 206):
            # Getting the type of 'key' (line 206)
            key_114812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 28), 'key')
            # Obtaining the member 'stop' of a type (line 206)
            stop_114813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 28), key_114812, 'stop')
            # Getting the type of 'start' (line 206)
            start_114814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 37), 'start')
            # Applying the binary operator '-' (line 206)
            result_sub_114815 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 28), '-', stop_114813, start_114814)
            
            
            # Call to float(...): (line 206)
            # Processing the call arguments (line 206)
            # Getting the type of 'step' (line 206)
            step_114817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 50), 'step', False)
            int_114818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 55), 'int')
            # Applying the binary operator '-' (line 206)
            result_sub_114819 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 50), '-', step_114817, int_114818)
            
            # Processing the call keyword arguments (line 206)
            kwargs_114820 = {}
            # Getting the type of 'float' (line 206)
            float_114816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 44), 'float', False)
            # Calling float(args, kwargs) (line 206)
            float_call_result_114821 = invoke(stypy.reporting.localization.Localization(__file__, 206, 44), float_114816, *[result_sub_114819], **kwargs_114820)
            
            # Applying the binary operator 'div' (line 206)
            result_div_114822 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 27), 'div', result_sub_114815, float_call_result_114821)
            
            # Assigning a type to the variable 'step' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'step', result_div_114822)
            # SSA join for if statement (line 205)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a BinOp to a Name (line 207):
            
            # Assigning a BinOp to a Name (line 207):
            # Getting the type of 'key' (line 207)
            key_114823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 23), 'key')
            # Obtaining the member 'stop' of a type (line 207)
            stop_114824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 23), key_114823, 'stop')
            # Getting the type of 'step' (line 207)
            step_114825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 34), 'step')
            # Applying the binary operator '+' (line 207)
            result_add_114826 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 23), '+', stop_114824, step_114825)
            
            # Assigning a type to the variable 'stop' (line 207)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'stop', result_add_114826)
            
            # Call to arange(...): (line 208)
            # Processing the call arguments (line 208)
            int_114829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 34), 'int')
            # Getting the type of 'length' (line 208)
            length_114830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 37), 'length', False)
            int_114831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 45), 'int')
            # Getting the type of 'float' (line 208)
            float_114832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 48), 'float', False)
            # Processing the call keyword arguments (line 208)
            kwargs_114833 = {}
            # Getting the type of '_nx' (line 208)
            _nx_114827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), '_nx', False)
            # Obtaining the member 'arange' of a type (line 208)
            arange_114828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 23), _nx_114827, 'arange')
            # Calling arange(args, kwargs) (line 208)
            arange_call_result_114834 = invoke(stypy.reporting.localization.Localization(__file__, 208, 23), arange_114828, *[int_114829, length_114830, int_114831, float_114832], **kwargs_114833)
            
            # Getting the type of 'step' (line 208)
            step_114835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 55), 'step')
            # Applying the binary operator '*' (line 208)
            result_mul_114836 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 23), '*', arange_call_result_114834, step_114835)
            
            # Getting the type of 'start' (line 208)
            start_114837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 62), 'start')
            # Applying the binary operator '+' (line 208)
            result_add_114838 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 23), '+', result_mul_114836, start_114837)
            
            # Assigning a type to the variable 'stypy_return_type' (line 208)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'stypy_return_type', result_add_114838)

            if more_types_in_union_114799:
                # Runtime conditional SSA for else branch (line 202)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_114798) or more_types_in_union_114799):
            # Assigning a type to the variable 'step' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'step', remove_subtype_from_union(step_114797, complex))
            
            # Call to arange(...): (line 210)
            # Processing the call arguments (line 210)
            # Getting the type of 'start' (line 210)
            start_114841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 34), 'start', False)
            # Getting the type of 'stop' (line 210)
            stop_114842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 41), 'stop', False)
            # Getting the type of 'step' (line 210)
            step_114843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 47), 'step', False)
            # Processing the call keyword arguments (line 210)
            kwargs_114844 = {}
            # Getting the type of '_nx' (line 210)
            _nx_114839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 23), '_nx', False)
            # Obtaining the member 'arange' of a type (line 210)
            arange_114840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 23), _nx_114839, 'arange')
            # Calling arange(args, kwargs) (line 210)
            arange_call_result_114845 = invoke(stypy.reporting.localization.Localization(__file__, 210, 23), arange_114840, *[start_114841, stop_114842, step_114843], **kwargs_114844)
            
            # Assigning a type to the variable 'stypy_return_type' (line 210)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'stypy_return_type', arange_call_result_114845)

            if (may_be_114798 and more_types_in_union_114799):
                # SSA join for if statement (line 202)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for try-except statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_114846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114846)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_114846


    @norecursion
    def __getslice__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getslice__'
        module_type_store = module_type_store.open_function_context('__getslice__', 212, 4, False)
        # Assigning a type to the variable 'self' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        nd_grid.__getslice__.__dict__.__setitem__('stypy_localization', localization)
        nd_grid.__getslice__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        nd_grid.__getslice__.__dict__.__setitem__('stypy_type_store', module_type_store)
        nd_grid.__getslice__.__dict__.__setitem__('stypy_function_name', 'nd_grid.__getslice__')
        nd_grid.__getslice__.__dict__.__setitem__('stypy_param_names_list', ['i', 'j'])
        nd_grid.__getslice__.__dict__.__setitem__('stypy_varargs_param_name', None)
        nd_grid.__getslice__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        nd_grid.__getslice__.__dict__.__setitem__('stypy_call_defaults', defaults)
        nd_grid.__getslice__.__dict__.__setitem__('stypy_call_varargs', varargs)
        nd_grid.__getslice__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        nd_grid.__getslice__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'nd_grid.__getslice__', ['i', 'j'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getslice__', localization, ['i', 'j'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getslice__(...)' code ##################

        
        # Call to arange(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'i' (line 213)
        i_114849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 26), 'i', False)
        # Getting the type of 'j' (line 213)
        j_114850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 29), 'j', False)
        # Processing the call keyword arguments (line 213)
        kwargs_114851 = {}
        # Getting the type of '_nx' (line 213)
        _nx_114847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), '_nx', False)
        # Obtaining the member 'arange' of a type (line 213)
        arange_114848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 15), _nx_114847, 'arange')
        # Calling arange(args, kwargs) (line 213)
        arange_call_result_114852 = invoke(stypy.reporting.localization.Localization(__file__, 213, 15), arange_114848, *[i_114849, j_114850], **kwargs_114851)
        
        # Assigning a type to the variable 'stypy_return_type' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'stypy_return_type', arange_call_result_114852)
        
        # ################# End of '__getslice__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getslice__' in the type store
        # Getting the type of 'stypy_return_type' (line 212)
        stypy_return_type_114853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114853)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getslice__'
        return stypy_return_type_114853


    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 215, 4, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        nd_grid.__len__.__dict__.__setitem__('stypy_localization', localization)
        nd_grid.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        nd_grid.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        nd_grid.__len__.__dict__.__setitem__('stypy_function_name', 'nd_grid.__len__')
        nd_grid.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        nd_grid.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        nd_grid.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        nd_grid.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        nd_grid.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        nd_grid.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        nd_grid.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'nd_grid.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        int_114854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'stypy_return_type', int_114854)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 215)
        stypy_return_type_114855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114855)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_114855


# Assigning a type to the variable 'nd_grid' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'nd_grid', nd_grid)

# Assigning a Call to a Name (line 218):

# Assigning a Call to a Name (line 218):

# Call to nd_grid(...): (line 218)
# Processing the call keyword arguments (line 218)
# Getting the type of 'False' (line 218)
False_114857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 'False', False)
keyword_114858 = False_114857
kwargs_114859 = {'sparse': keyword_114858}
# Getting the type of 'nd_grid' (line 218)
nd_grid_114856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'nd_grid', False)
# Calling nd_grid(args, kwargs) (line 218)
nd_grid_call_result_114860 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), nd_grid_114856, *[], **kwargs_114859)

# Assigning a type to the variable 'mgrid' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'mgrid', nd_grid_call_result_114860)

# Assigning a Call to a Name (line 219):

# Assigning a Call to a Name (line 219):

# Call to nd_grid(...): (line 219)
# Processing the call keyword arguments (line 219)
# Getting the type of 'True' (line 219)
True_114862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 23), 'True', False)
keyword_114863 = True_114862
kwargs_114864 = {'sparse': keyword_114863}
# Getting the type of 'nd_grid' (line 219)
nd_grid_114861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'nd_grid', False)
# Calling nd_grid(args, kwargs) (line 219)
nd_grid_call_result_114865 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), nd_grid_114861, *[], **kwargs_114864)

# Assigning a type to the variable 'ogrid' (line 219)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'ogrid', nd_grid_call_result_114865)

# Assigning a Name to a Attribute (line 220):

# Assigning a Name to a Attribute (line 220):
# Getting the type of 'None' (line 220)
None_114866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'None')
# Getting the type of 'mgrid' (line 220)
mgrid_114867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'mgrid')
# Setting the type of the member '__doc__' of a type (line 220)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 0), mgrid_114867, '__doc__', None_114866)

# Assigning a Name to a Attribute (line 221):

# Assigning a Name to a Attribute (line 221):
# Getting the type of 'None' (line 221)
None_114868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'None')
# Getting the type of 'ogrid' (line 221)
ogrid_114869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'ogrid')
# Setting the type of the member '__doc__' of a type (line 221)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 0), ogrid_114869, '__doc__', None_114868)
# Declaration of the 'AxisConcatenator' class

class AxisConcatenator(object, ):
    str_114870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, (-1)), 'str', '\n    Translates slice objects to concatenation along an axis.\n\n    For detailed documentation on usage, see `r_`.\n\n    ')

    @norecursion
    def _retval(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_retval'
        module_type_store = module_type_store.open_function_context('_retval', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AxisConcatenator._retval.__dict__.__setitem__('stypy_localization', localization)
        AxisConcatenator._retval.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AxisConcatenator._retval.__dict__.__setitem__('stypy_type_store', module_type_store)
        AxisConcatenator._retval.__dict__.__setitem__('stypy_function_name', 'AxisConcatenator._retval')
        AxisConcatenator._retval.__dict__.__setitem__('stypy_param_names_list', ['res'])
        AxisConcatenator._retval.__dict__.__setitem__('stypy_varargs_param_name', None)
        AxisConcatenator._retval.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AxisConcatenator._retval.__dict__.__setitem__('stypy_call_defaults', defaults)
        AxisConcatenator._retval.__dict__.__setitem__('stypy_call_varargs', varargs)
        AxisConcatenator._retval.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AxisConcatenator._retval.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AxisConcatenator._retval', ['res'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_retval', localization, ['res'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_retval(...)' code ##################

        
        # Getting the type of 'self' (line 232)
        self_114871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'self')
        # Obtaining the member 'matrix' of a type (line 232)
        matrix_114872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 11), self_114871, 'matrix')
        # Testing the type of an if condition (line 232)
        if_condition_114873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 8), matrix_114872)
        # Assigning a type to the variable 'if_condition_114873' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'if_condition_114873', if_condition_114873)
        # SSA begins for if statement (line 232)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 233):
        
        # Assigning a Attribute to a Name (line 233):
        # Getting the type of 'res' (line 233)
        res_114874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 22), 'res')
        # Obtaining the member 'ndim' of a type (line 233)
        ndim_114875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 22), res_114874, 'ndim')
        # Assigning a type to the variable 'oldndim' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'oldndim', ndim_114875)
        
        # Assigning a Call to a Name (line 234):
        
        # Assigning a Call to a Name (line 234):
        
        # Call to makemat(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'res' (line 234)
        res_114877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 26), 'res', False)
        # Processing the call keyword arguments (line 234)
        kwargs_114878 = {}
        # Getting the type of 'makemat' (line 234)
        makemat_114876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 18), 'makemat', False)
        # Calling makemat(args, kwargs) (line 234)
        makemat_call_result_114879 = invoke(stypy.reporting.localization.Localization(__file__, 234, 18), makemat_114876, *[res_114877], **kwargs_114878)
        
        # Assigning a type to the variable 'res' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'res', makemat_call_result_114879)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'oldndim' (line 235)
        oldndim_114880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), 'oldndim')
        int_114881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 26), 'int')
        # Applying the binary operator '==' (line 235)
        result_eq_114882 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 15), '==', oldndim_114880, int_114881)
        
        # Getting the type of 'self' (line 235)
        self_114883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 32), 'self')
        # Obtaining the member 'col' of a type (line 235)
        col_114884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 32), self_114883, 'col')
        # Applying the binary operator 'and' (line 235)
        result_and_keyword_114885 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 15), 'and', result_eq_114882, col_114884)
        
        # Testing the type of an if condition (line 235)
        if_condition_114886 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 12), result_and_keyword_114885)
        # Assigning a type to the variable 'if_condition_114886' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'if_condition_114886', if_condition_114886)
        # SSA begins for if statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 236):
        
        # Assigning a Attribute to a Name (line 236):
        # Getting the type of 'res' (line 236)
        res_114887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 22), 'res')
        # Obtaining the member 'T' of a type (line 236)
        T_114888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 22), res_114887, 'T')
        # Assigning a type to the variable 'res' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'res', T_114888)
        # SSA join for if statement (line 235)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 232)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 237):
        
        # Assigning a Attribute to a Attribute (line 237):
        # Getting the type of 'self' (line 237)
        self_114889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'self')
        # Obtaining the member '_axis' of a type (line 237)
        _axis_114890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 20), self_114889, '_axis')
        # Getting the type of 'self' (line 237)
        self_114891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'self')
        # Setting the type of the member 'axis' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), self_114891, 'axis', _axis_114890)
        
        # Assigning a Attribute to a Attribute (line 238):
        
        # Assigning a Attribute to a Attribute (line 238):
        # Getting the type of 'self' (line 238)
        self_114892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 22), 'self')
        # Obtaining the member '_matrix' of a type (line 238)
        _matrix_114893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 22), self_114892, '_matrix')
        # Getting the type of 'self' (line 238)
        self_114894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'self')
        # Setting the type of the member 'matrix' of a type (line 238)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), self_114894, 'matrix', _matrix_114893)
        
        # Assigning a Num to a Attribute (line 239):
        
        # Assigning a Num to a Attribute (line 239):
        int_114895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 19), 'int')
        # Getting the type of 'self' (line 239)
        self_114896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'self')
        # Setting the type of the member 'col' of a type (line 239)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), self_114896, 'col', int_114895)
        # Getting the type of 'res' (line 240)
        res_114897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'stypy_return_type', res_114897)
        
        # ################# End of '_retval(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_retval' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_114898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114898)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_retval'
        return stypy_return_type_114898


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_114899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 28), 'int')
        # Getting the type of 'False' (line 242)
        False_114900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 38), 'False')
        int_114901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 51), 'int')
        int_114902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 62), 'int')
        defaults = [int_114899, False_114900, int_114901, int_114902]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 242, 4, False)
        # Assigning a type to the variable 'self' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AxisConcatenator.__init__', ['axis', 'matrix', 'ndmin', 'trans1d'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['axis', 'matrix', 'ndmin', 'trans1d'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 243):
        
        # Assigning a Name to a Attribute (line 243):
        # Getting the type of 'axis' (line 243)
        axis_114903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 21), 'axis')
        # Getting the type of 'self' (line 243)
        self_114904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'self')
        # Setting the type of the member '_axis' of a type (line 243)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), self_114904, '_axis', axis_114903)
        
        # Assigning a Name to a Attribute (line 244):
        
        # Assigning a Name to a Attribute (line 244):
        # Getting the type of 'matrix' (line 244)
        matrix_114905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 23), 'matrix')
        # Getting the type of 'self' (line 244)
        self_114906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'self')
        # Setting the type of the member '_matrix' of a type (line 244)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), self_114906, '_matrix', matrix_114905)
        
        # Assigning a Name to a Attribute (line 245):
        
        # Assigning a Name to a Attribute (line 245):
        # Getting the type of 'axis' (line 245)
        axis_114907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'axis')
        # Getting the type of 'self' (line 245)
        self_114908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'self')
        # Setting the type of the member 'axis' of a type (line 245)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), self_114908, 'axis', axis_114907)
        
        # Assigning a Name to a Attribute (line 246):
        
        # Assigning a Name to a Attribute (line 246):
        # Getting the type of 'matrix' (line 246)
        matrix_114909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 22), 'matrix')
        # Getting the type of 'self' (line 246)
        self_114910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'self')
        # Setting the type of the member 'matrix' of a type (line 246)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), self_114910, 'matrix', matrix_114909)
        
        # Assigning a Num to a Attribute (line 247):
        
        # Assigning a Num to a Attribute (line 247):
        int_114911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 19), 'int')
        # Getting the type of 'self' (line 247)
        self_114912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'self')
        # Setting the type of the member 'col' of a type (line 247)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), self_114912, 'col', int_114911)
        
        # Assigning a Name to a Attribute (line 248):
        
        # Assigning a Name to a Attribute (line 248):
        # Getting the type of 'trans1d' (line 248)
        trans1d_114913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 23), 'trans1d')
        # Getting the type of 'self' (line 248)
        self_114914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self')
        # Setting the type of the member 'trans1d' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_114914, 'trans1d', trans1d_114913)
        
        # Assigning a Name to a Attribute (line 249):
        
        # Assigning a Name to a Attribute (line 249):
        # Getting the type of 'ndmin' (line 249)
        ndmin_114915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 21), 'ndmin')
        # Getting the type of 'self' (line 249)
        self_114916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self')
        # Setting the type of the member 'ndmin' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_114916, 'ndmin', ndmin_114915)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 251, 4, False)
        # Assigning a type to the variable 'self' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AxisConcatenator.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        AxisConcatenator.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AxisConcatenator.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        AxisConcatenator.__getitem__.__dict__.__setitem__('stypy_function_name', 'AxisConcatenator.__getitem__')
        AxisConcatenator.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['key'])
        AxisConcatenator.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        AxisConcatenator.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AxisConcatenator.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        AxisConcatenator.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        AxisConcatenator.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AxisConcatenator.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AxisConcatenator.__getitem__', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Assigning a Attribute to a Name (line 252):
        
        # Assigning a Attribute to a Name (line 252):
        # Getting the type of 'self' (line 252)
        self_114917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 18), 'self')
        # Obtaining the member 'trans1d' of a type (line 252)
        trans1d_114918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 18), self_114917, 'trans1d')
        # Assigning a type to the variable 'trans1d' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'trans1d', trans1d_114918)
        
        # Assigning a Attribute to a Name (line 253):
        
        # Assigning a Attribute to a Name (line 253):
        # Getting the type of 'self' (line 253)
        self_114919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'self')
        # Obtaining the member 'ndmin' of a type (line 253)
        ndmin_114920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 16), self_114919, 'ndmin')
        # Assigning a type to the variable 'ndmin' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'ndmin', ndmin_114920)
        
        # Type idiom detected: calculating its left and rigth part (line 254)
        # Getting the type of 'str' (line 254)
        str_114921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 27), 'str')
        # Getting the type of 'key' (line 254)
        key_114922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 22), 'key')
        
        (may_be_114923, more_types_in_union_114924) = may_be_subtype(str_114921, key_114922)

        if may_be_114923:

            if more_types_in_union_114924:
                # Runtime conditional SSA (line 254)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'key' (line 254)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'key', remove_not_subtype_from_union(key_114922, str))
            
            # Assigning a Attribute to a Name (line 255):
            
            # Assigning a Attribute to a Name (line 255):
            
            # Call to _getframe(...): (line 255)
            # Processing the call keyword arguments (line 255)
            kwargs_114927 = {}
            # Getting the type of 'sys' (line 255)
            sys_114925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'sys', False)
            # Obtaining the member '_getframe' of a type (line 255)
            _getframe_114926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 20), sys_114925, '_getframe')
            # Calling _getframe(args, kwargs) (line 255)
            _getframe_call_result_114928 = invoke(stypy.reporting.localization.Localization(__file__, 255, 20), _getframe_114926, *[], **kwargs_114927)
            
            # Obtaining the member 'f_back' of a type (line 255)
            f_back_114929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 20), _getframe_call_result_114928, 'f_back')
            # Assigning a type to the variable 'frame' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'frame', f_back_114929)
            
            # Assigning a Call to a Name (line 256):
            
            # Assigning a Call to a Name (line 256):
            
            # Call to bmat(...): (line 256)
            # Processing the call arguments (line 256)
            # Getting the type of 'key' (line 256)
            key_114932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 32), 'key', False)
            # Getting the type of 'frame' (line 256)
            frame_114933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 37), 'frame', False)
            # Obtaining the member 'f_globals' of a type (line 256)
            f_globals_114934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 37), frame_114933, 'f_globals')
            # Getting the type of 'frame' (line 256)
            frame_114935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 54), 'frame', False)
            # Obtaining the member 'f_locals' of a type (line 256)
            f_locals_114936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 54), frame_114935, 'f_locals')
            # Processing the call keyword arguments (line 256)
            kwargs_114937 = {}
            # Getting the type of 'matrix' (line 256)
            matrix_114930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 20), 'matrix', False)
            # Obtaining the member 'bmat' of a type (line 256)
            bmat_114931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 20), matrix_114930, 'bmat')
            # Calling bmat(args, kwargs) (line 256)
            bmat_call_result_114938 = invoke(stypy.reporting.localization.Localization(__file__, 256, 20), bmat_114931, *[key_114932, f_globals_114934, f_locals_114936], **kwargs_114937)
            
            # Assigning a type to the variable 'mymat' (line 256)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'mymat', bmat_call_result_114938)
            # Getting the type of 'mymat' (line 257)
            mymat_114939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 19), 'mymat')
            # Assigning a type to the variable 'stypy_return_type' (line 257)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'stypy_return_type', mymat_114939)

            if more_types_in_union_114924:
                # SSA join for if statement (line 254)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 258)
        # Getting the type of 'tuple' (line 258)
        tuple_114940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 31), 'tuple')
        # Getting the type of 'key' (line 258)
        key_114941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 26), 'key')
        
        (may_be_114942, more_types_in_union_114943) = may_not_be_subtype(tuple_114940, key_114941)

        if may_be_114942:

            if more_types_in_union_114943:
                # Runtime conditional SSA (line 258)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'key' (line 258)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'key', remove_subtype_from_union(key_114941, tuple))
            
            # Assigning a Tuple to a Name (line 259):
            
            # Assigning a Tuple to a Name (line 259):
            
            # Obtaining an instance of the builtin type 'tuple' (line 259)
            tuple_114944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 259)
            # Adding element type (line 259)
            # Getting the type of 'key' (line 259)
            key_114945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 19), 'key')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 19), tuple_114944, key_114945)
            
            # Assigning a type to the variable 'key' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'key', tuple_114944)

            if more_types_in_union_114943:
                # SSA join for if statement (line 258)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 260):
        
        # Assigning a List to a Name (line 260):
        
        # Obtaining an instance of the builtin type 'list' (line 260)
        list_114946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 260)
        
        # Assigning a type to the variable 'objs' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'objs', list_114946)
        
        # Assigning a List to a Name (line 261):
        
        # Assigning a List to a Name (line 261):
        
        # Obtaining an instance of the builtin type 'list' (line 261)
        list_114947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 261)
        
        # Assigning a type to the variable 'scalars' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'scalars', list_114947)
        
        # Assigning a List to a Name (line 262):
        
        # Assigning a List to a Name (line 262):
        
        # Obtaining an instance of the builtin type 'list' (line 262)
        list_114948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 262)
        
        # Assigning a type to the variable 'arraytypes' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'arraytypes', list_114948)
        
        # Assigning a List to a Name (line 263):
        
        # Assigning a List to a Name (line 263):
        
        # Obtaining an instance of the builtin type 'list' (line 263)
        list_114949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 263)
        
        # Assigning a type to the variable 'scalartypes' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'scalartypes', list_114949)
        
        
        # Call to range(...): (line 264)
        # Processing the call arguments (line 264)
        
        # Call to len(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'key' (line 264)
        key_114952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 27), 'key', False)
        # Processing the call keyword arguments (line 264)
        kwargs_114953 = {}
        # Getting the type of 'len' (line 264)
        len_114951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 23), 'len', False)
        # Calling len(args, kwargs) (line 264)
        len_call_result_114954 = invoke(stypy.reporting.localization.Localization(__file__, 264, 23), len_114951, *[key_114952], **kwargs_114953)
        
        # Processing the call keyword arguments (line 264)
        kwargs_114955 = {}
        # Getting the type of 'range' (line 264)
        range_114950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 17), 'range', False)
        # Calling range(args, kwargs) (line 264)
        range_call_result_114956 = invoke(stypy.reporting.localization.Localization(__file__, 264, 17), range_114950, *[len_call_result_114954], **kwargs_114955)
        
        # Testing the type of a for loop iterable (line 264)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 264, 8), range_call_result_114956)
        # Getting the type of the for loop variable (line 264)
        for_loop_var_114957 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 264, 8), range_call_result_114956)
        # Assigning a type to the variable 'k' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'k', for_loop_var_114957)
        # SSA begins for a for statement (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Name (line 265):
        
        # Assigning a Name to a Name (line 265):
        # Getting the type of 'False' (line 265)
        False_114958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 21), 'False')
        # Assigning a type to the variable 'scalar' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'scalar', False_114958)
        
        # Type idiom detected: calculating its left and rigth part (line 266)
        # Getting the type of 'slice' (line 266)
        slice_114959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 34), 'slice')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 266)
        k_114960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 30), 'k')
        # Getting the type of 'key' (line 266)
        key_114961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 26), 'key')
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___114962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 26), key_114961, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_114963 = invoke(stypy.reporting.localization.Localization(__file__, 266, 26), getitem___114962, k_114960)
        
        
        (may_be_114964, more_types_in_union_114965) = may_be_subtype(slice_114959, subscript_call_result_114963)

        if may_be_114964:

            if more_types_in_union_114965:
                # Runtime conditional SSA (line 266)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 267):
            
            # Assigning a Attribute to a Name (line 267):
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 267)
            k_114966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 27), 'k')
            # Getting the type of 'key' (line 267)
            key_114967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 'key')
            # Obtaining the member '__getitem__' of a type (line 267)
            getitem___114968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 23), key_114967, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 267)
            subscript_call_result_114969 = invoke(stypy.reporting.localization.Localization(__file__, 267, 23), getitem___114968, k_114966)
            
            # Obtaining the member 'step' of a type (line 267)
            step_114970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 23), subscript_call_result_114969, 'step')
            # Assigning a type to the variable 'step' (line 267)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'step', step_114970)
            
            # Assigning a Attribute to a Name (line 268):
            
            # Assigning a Attribute to a Name (line 268):
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 268)
            k_114971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 28), 'k')
            # Getting the type of 'key' (line 268)
            key_114972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 24), 'key')
            # Obtaining the member '__getitem__' of a type (line 268)
            getitem___114973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 24), key_114972, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 268)
            subscript_call_result_114974 = invoke(stypy.reporting.localization.Localization(__file__, 268, 24), getitem___114973, k_114971)
            
            # Obtaining the member 'start' of a type (line 268)
            start_114975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 24), subscript_call_result_114974, 'start')
            # Assigning a type to the variable 'start' (line 268)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'start', start_114975)
            
            # Assigning a Attribute to a Name (line 269):
            
            # Assigning a Attribute to a Name (line 269):
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 269)
            k_114976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 27), 'k')
            # Getting the type of 'key' (line 269)
            key_114977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 23), 'key')
            # Obtaining the member '__getitem__' of a type (line 269)
            getitem___114978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 23), key_114977, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 269)
            subscript_call_result_114979 = invoke(stypy.reporting.localization.Localization(__file__, 269, 23), getitem___114978, k_114976)
            
            # Obtaining the member 'stop' of a type (line 269)
            stop_114980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 23), subscript_call_result_114979, 'stop')
            # Assigning a type to the variable 'stop' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'stop', stop_114980)
            
            # Type idiom detected: calculating its left and rigth part (line 270)
            # Getting the type of 'start' (line 270)
            start_114981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 19), 'start')
            # Getting the type of 'None' (line 270)
            None_114982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 28), 'None')
            
            (may_be_114983, more_types_in_union_114984) = may_be_none(start_114981, None_114982)

            if may_be_114983:

                if more_types_in_union_114984:
                    # Runtime conditional SSA (line 270)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Num to a Name (line 271):
                
                # Assigning a Num to a Name (line 271):
                int_114985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 28), 'int')
                # Assigning a type to the variable 'start' (line 271)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 20), 'start', int_114985)

                if more_types_in_union_114984:
                    # SSA join for if statement (line 270)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 272)
            # Getting the type of 'step' (line 272)
            step_114986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 19), 'step')
            # Getting the type of 'None' (line 272)
            None_114987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 27), 'None')
            
            (may_be_114988, more_types_in_union_114989) = may_be_none(step_114986, None_114987)

            if may_be_114988:

                if more_types_in_union_114989:
                    # Runtime conditional SSA (line 272)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Num to a Name (line 273):
                
                # Assigning a Num to a Name (line 273):
                int_114990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 27), 'int')
                # Assigning a type to the variable 'step' (line 273)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'step', int_114990)

                if more_types_in_union_114989:
                    # SSA join for if statement (line 272)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 274)
            # Getting the type of 'complex' (line 274)
            complex_114991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 36), 'complex')
            # Getting the type of 'step' (line 274)
            step_114992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 30), 'step')
            
            (may_be_114993, more_types_in_union_114994) = may_be_subtype(complex_114991, step_114992)

            if may_be_114993:

                if more_types_in_union_114994:
                    # Runtime conditional SSA (line 274)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'step' (line 274)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'step', remove_not_subtype_from_union(step_114992, complex))
                
                # Assigning a Call to a Name (line 275):
                
                # Assigning a Call to a Name (line 275):
                
                # Call to int(...): (line 275)
                # Processing the call arguments (line 275)
                
                # Call to abs(...): (line 275)
                # Processing the call arguments (line 275)
                # Getting the type of 'step' (line 275)
                step_114997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 35), 'step', False)
                # Processing the call keyword arguments (line 275)
                kwargs_114998 = {}
                # Getting the type of 'abs' (line 275)
                abs_114996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 31), 'abs', False)
                # Calling abs(args, kwargs) (line 275)
                abs_call_result_114999 = invoke(stypy.reporting.localization.Localization(__file__, 275, 31), abs_114996, *[step_114997], **kwargs_114998)
                
                # Processing the call keyword arguments (line 275)
                kwargs_115000 = {}
                # Getting the type of 'int' (line 275)
                int_114995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 27), 'int', False)
                # Calling int(args, kwargs) (line 275)
                int_call_result_115001 = invoke(stypy.reporting.localization.Localization(__file__, 275, 27), int_114995, *[abs_call_result_114999], **kwargs_115000)
                
                # Assigning a type to the variable 'size' (line 275)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 20), 'size', int_call_result_115001)
                
                # Assigning a Call to a Name (line 276):
                
                # Assigning a Call to a Name (line 276):
                
                # Call to linspace(...): (line 276)
                # Processing the call arguments (line 276)
                # Getting the type of 'start' (line 276)
                start_115004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 52), 'start', False)
                # Getting the type of 'stop' (line 276)
                stop_115005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 59), 'stop', False)
                # Processing the call keyword arguments (line 276)
                # Getting the type of 'size' (line 276)
                size_115006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 69), 'size', False)
                keyword_115007 = size_115006
                kwargs_115008 = {'num': keyword_115007}
                # Getting the type of 'function_base' (line 276)
                function_base_115002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 29), 'function_base', False)
                # Obtaining the member 'linspace' of a type (line 276)
                linspace_115003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 29), function_base_115002, 'linspace')
                # Calling linspace(args, kwargs) (line 276)
                linspace_call_result_115009 = invoke(stypy.reporting.localization.Localization(__file__, 276, 29), linspace_115003, *[start_115004, stop_115005], **kwargs_115008)
                
                # Assigning a type to the variable 'newobj' (line 276)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'newobj', linspace_call_result_115009)

                if more_types_in_union_114994:
                    # Runtime conditional SSA for else branch (line 274)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_114993) or more_types_in_union_114994):
                # Assigning a type to the variable 'step' (line 274)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'step', remove_subtype_from_union(step_114992, complex))
                
                # Assigning a Call to a Name (line 278):
                
                # Assigning a Call to a Name (line 278):
                
                # Call to arange(...): (line 278)
                # Processing the call arguments (line 278)
                # Getting the type of 'start' (line 278)
                start_115012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 40), 'start', False)
                # Getting the type of 'stop' (line 278)
                stop_115013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 47), 'stop', False)
                # Getting the type of 'step' (line 278)
                step_115014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 53), 'step', False)
                # Processing the call keyword arguments (line 278)
                kwargs_115015 = {}
                # Getting the type of '_nx' (line 278)
                _nx_115010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 29), '_nx', False)
                # Obtaining the member 'arange' of a type (line 278)
                arange_115011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 29), _nx_115010, 'arange')
                # Calling arange(args, kwargs) (line 278)
                arange_call_result_115016 = invoke(stypy.reporting.localization.Localization(__file__, 278, 29), arange_115011, *[start_115012, stop_115013, step_115014], **kwargs_115015)
                
                # Assigning a type to the variable 'newobj' (line 278)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 20), 'newobj', arange_call_result_115016)

                if (may_be_114993 and more_types_in_union_114994):
                    # SSA join for if statement (line 274)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            
            # Getting the type of 'ndmin' (line 279)
            ndmin_115017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 19), 'ndmin')
            int_115018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 27), 'int')
            # Applying the binary operator '>' (line 279)
            result_gt_115019 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 19), '>', ndmin_115017, int_115018)
            
            # Testing the type of an if condition (line 279)
            if_condition_115020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 16), result_gt_115019)
            # Assigning a type to the variable 'if_condition_115020' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'if_condition_115020', if_condition_115020)
            # SSA begins for if statement (line 279)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 280):
            
            # Assigning a Call to a Name (line 280):
            
            # Call to array(...): (line 280)
            # Processing the call arguments (line 280)
            # Getting the type of 'newobj' (line 280)
            newobj_115022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 35), 'newobj', False)
            # Processing the call keyword arguments (line 280)
            # Getting the type of 'False' (line 280)
            False_115023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 48), 'False', False)
            keyword_115024 = False_115023
            # Getting the type of 'ndmin' (line 280)
            ndmin_115025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 61), 'ndmin', False)
            keyword_115026 = ndmin_115025
            kwargs_115027 = {'copy': keyword_115024, 'ndmin': keyword_115026}
            # Getting the type of 'array' (line 280)
            array_115021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 29), 'array', False)
            # Calling array(args, kwargs) (line 280)
            array_call_result_115028 = invoke(stypy.reporting.localization.Localization(__file__, 280, 29), array_115021, *[newobj_115022], **kwargs_115027)
            
            # Assigning a type to the variable 'newobj' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 20), 'newobj', array_call_result_115028)
            
            
            # Getting the type of 'trans1d' (line 281)
            trans1d_115029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 23), 'trans1d')
            int_115030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 34), 'int')
            # Applying the binary operator '!=' (line 281)
            result_ne_115031 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 23), '!=', trans1d_115029, int_115030)
            
            # Testing the type of an if condition (line 281)
            if_condition_115032 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 20), result_ne_115031)
            # Assigning a type to the variable 'if_condition_115032' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 20), 'if_condition_115032', if_condition_115032)
            # SSA begins for if statement (line 281)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 282):
            
            # Assigning a Call to a Name (line 282):
            
            # Call to swapaxes(...): (line 282)
            # Processing the call arguments (line 282)
            int_115035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 49), 'int')
            # Getting the type of 'trans1d' (line 282)
            trans1d_115036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 53), 'trans1d', False)
            # Processing the call keyword arguments (line 282)
            kwargs_115037 = {}
            # Getting the type of 'newobj' (line 282)
            newobj_115033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 33), 'newobj', False)
            # Obtaining the member 'swapaxes' of a type (line 282)
            swapaxes_115034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 33), newobj_115033, 'swapaxes')
            # Calling swapaxes(args, kwargs) (line 282)
            swapaxes_call_result_115038 = invoke(stypy.reporting.localization.Localization(__file__, 282, 33), swapaxes_115034, *[int_115035, trans1d_115036], **kwargs_115037)
            
            # Assigning a type to the variable 'newobj' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'newobj', swapaxes_call_result_115038)
            # SSA join for if statement (line 281)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 279)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_114965:
                # Runtime conditional SSA for else branch (line 266)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_114964) or more_types_in_union_114965):
            
            # Type idiom detected: calculating its left and rigth part (line 283)
            # Getting the type of 'str' (line 283)
            str_115039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 36), 'str')
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 283)
            k_115040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 32), 'k')
            # Getting the type of 'key' (line 283)
            key_115041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 28), 'key')
            # Obtaining the member '__getitem__' of a type (line 283)
            getitem___115042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 28), key_115041, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 283)
            subscript_call_result_115043 = invoke(stypy.reporting.localization.Localization(__file__, 283, 28), getitem___115042, k_115040)
            
            
            (may_be_115044, more_types_in_union_115045) = may_be_subtype(str_115039, subscript_call_result_115043)

            if may_be_115044:

                if more_types_in_union_115045:
                    # Runtime conditional SSA (line 283)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                
                # Getting the type of 'k' (line 284)
                k_115046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 19), 'k')
                int_115047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 24), 'int')
                # Applying the binary operator '!=' (line 284)
                result_ne_115048 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 19), '!=', k_115046, int_115047)
                
                # Testing the type of an if condition (line 284)
                if_condition_115049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 16), result_ne_115048)
                # Assigning a type to the variable 'if_condition_115049' (line 284)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'if_condition_115049', if_condition_115049)
                # SSA begins for if statement (line 284)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to ValueError(...): (line 285)
                # Processing the call arguments (line 285)
                str_115051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 37), 'str', 'special directives must be the first entry.')
                # Processing the call keyword arguments (line 285)
                kwargs_115052 = {}
                # Getting the type of 'ValueError' (line 285)
                ValueError_115050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 26), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 285)
                ValueError_call_result_115053 = invoke(stypy.reporting.localization.Localization(__file__, 285, 26), ValueError_115050, *[str_115051], **kwargs_115052)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 285, 20), ValueError_call_result_115053, 'raise parameter', BaseException)
                # SSA join for if statement (line 284)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a Subscript to a Name (line 287):
                
                # Assigning a Subscript to a Name (line 287):
                
                # Obtaining the type of the subscript
                int_115054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 27), 'int')
                # Getting the type of 'key' (line 287)
                key_115055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 23), 'key')
                # Obtaining the member '__getitem__' of a type (line 287)
                getitem___115056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 23), key_115055, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 287)
                subscript_call_result_115057 = invoke(stypy.reporting.localization.Localization(__file__, 287, 23), getitem___115056, int_115054)
                
                # Assigning a type to the variable 'key0' (line 287)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'key0', subscript_call_result_115057)
                
                
                # Getting the type of 'key0' (line 288)
                key0_115058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 19), 'key0')
                str_115059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 27), 'str', 'rc')
                # Applying the binary operator 'in' (line 288)
                result_contains_115060 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 19), 'in', key0_115058, str_115059)
                
                # Testing the type of an if condition (line 288)
                if_condition_115061 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 16), result_contains_115060)
                # Assigning a type to the variable 'if_condition_115061' (line 288)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'if_condition_115061', if_condition_115061)
                # SSA begins for if statement (line 288)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 289):
                
                # Assigning a Name to a Attribute (line 289):
                # Getting the type of 'True' (line 289)
                True_115062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 34), 'True')
                # Getting the type of 'self' (line 289)
                self_115063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'self')
                # Setting the type of the member 'matrix' of a type (line 289)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 20), self_115063, 'matrix', True_115062)
                
                # Assigning a Compare to a Attribute (line 290):
                
                # Assigning a Compare to a Attribute (line 290):
                
                # Getting the type of 'key0' (line 290)
                key0_115064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 32), 'key0')
                str_115065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 40), 'str', 'c')
                # Applying the binary operator '==' (line 290)
                result_eq_115066 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 32), '==', key0_115064, str_115065)
                
                # Getting the type of 'self' (line 290)
                self_115067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 20), 'self')
                # Setting the type of the member 'col' of a type (line 290)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 20), self_115067, 'col', result_eq_115066)
                # SSA join for if statement (line 288)
                module_type_store = module_type_store.join_ssa_context()
                
                
                
                str_115068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 19), 'str', ',')
                # Getting the type of 'key0' (line 292)
                key0_115069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 26), 'key0')
                # Applying the binary operator 'in' (line 292)
                result_contains_115070 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 19), 'in', str_115068, key0_115069)
                
                # Testing the type of an if condition (line 292)
                if_condition_115071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 16), result_contains_115070)
                # Assigning a type to the variable 'if_condition_115071' (line 292)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'if_condition_115071', if_condition_115071)
                # SSA begins for if statement (line 292)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 293):
                
                # Assigning a Call to a Name (line 293):
                
                # Call to split(...): (line 293)
                # Processing the call arguments (line 293)
                str_115074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 37), 'str', ',')
                # Processing the call keyword arguments (line 293)
                kwargs_115075 = {}
                # Getting the type of 'key0' (line 293)
                key0_115072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'key0', False)
                # Obtaining the member 'split' of a type (line 293)
                split_115073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 26), key0_115072, 'split')
                # Calling split(args, kwargs) (line 293)
                split_call_result_115076 = invoke(stypy.reporting.localization.Localization(__file__, 293, 26), split_115073, *[str_115074], **kwargs_115075)
                
                # Assigning a type to the variable 'vec' (line 293)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 20), 'vec', split_call_result_115076)
                
                
                # SSA begins for try-except statement (line 294)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                
                # Assigning a ListComp to a Tuple (line 295):
                
                # Assigning a Subscript to a Name (line 295):
                
                # Obtaining the type of the subscript
                int_115077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 24), 'int')
                # Calculating list comprehension
                # Calculating comprehension expression
                
                # Obtaining the type of the subscript
                int_115082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 57), 'int')
                slice_115083 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 296, 52), None, int_115082, None)
                # Getting the type of 'vec' (line 296)
                vec_115084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 52), 'vec')
                # Obtaining the member '__getitem__' of a type (line 296)
                getitem___115085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 52), vec_115084, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 296)
                subscript_call_result_115086 = invoke(stypy.reporting.localization.Localization(__file__, 296, 52), getitem___115085, slice_115083)
                
                comprehension_115087 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 36), subscript_call_result_115086)
                # Assigning a type to the variable 'x' (line 296)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 36), 'x', comprehension_115087)
                
                # Call to int(...): (line 296)
                # Processing the call arguments (line 296)
                # Getting the type of 'x' (line 296)
                x_115079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 40), 'x', False)
                # Processing the call keyword arguments (line 296)
                kwargs_115080 = {}
                # Getting the type of 'int' (line 296)
                int_115078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 36), 'int', False)
                # Calling int(args, kwargs) (line 296)
                int_call_result_115081 = invoke(stypy.reporting.localization.Localization(__file__, 296, 36), int_115078, *[x_115079], **kwargs_115080)
                
                list_115088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 36), 'list')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 36), list_115088, int_call_result_115081)
                # Obtaining the member '__getitem__' of a type (line 295)
                getitem___115089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 24), list_115088, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 295)
                subscript_call_result_115090 = invoke(stypy.reporting.localization.Localization(__file__, 295, 24), getitem___115089, int_115077)
                
                # Assigning a type to the variable 'tuple_var_assignment_114435' (line 295)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'tuple_var_assignment_114435', subscript_call_result_115090)
                
                # Assigning a Subscript to a Name (line 295):
                
                # Obtaining the type of the subscript
                int_115091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 24), 'int')
                # Calculating list comprehension
                # Calculating comprehension expression
                
                # Obtaining the type of the subscript
                int_115096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 57), 'int')
                slice_115097 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 296, 52), None, int_115096, None)
                # Getting the type of 'vec' (line 296)
                vec_115098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 52), 'vec')
                # Obtaining the member '__getitem__' of a type (line 296)
                getitem___115099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 52), vec_115098, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 296)
                subscript_call_result_115100 = invoke(stypy.reporting.localization.Localization(__file__, 296, 52), getitem___115099, slice_115097)
                
                comprehension_115101 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 36), subscript_call_result_115100)
                # Assigning a type to the variable 'x' (line 296)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 36), 'x', comprehension_115101)
                
                # Call to int(...): (line 296)
                # Processing the call arguments (line 296)
                # Getting the type of 'x' (line 296)
                x_115093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 40), 'x', False)
                # Processing the call keyword arguments (line 296)
                kwargs_115094 = {}
                # Getting the type of 'int' (line 296)
                int_115092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 36), 'int', False)
                # Calling int(args, kwargs) (line 296)
                int_call_result_115095 = invoke(stypy.reporting.localization.Localization(__file__, 296, 36), int_115092, *[x_115093], **kwargs_115094)
                
                list_115102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 36), 'list')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 36), list_115102, int_call_result_115095)
                # Obtaining the member '__getitem__' of a type (line 295)
                getitem___115103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 24), list_115102, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 295)
                subscript_call_result_115104 = invoke(stypy.reporting.localization.Localization(__file__, 295, 24), getitem___115103, int_115091)
                
                # Assigning a type to the variable 'tuple_var_assignment_114436' (line 295)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'tuple_var_assignment_114436', subscript_call_result_115104)
                
                # Assigning a Name to a Attribute (line 295):
                # Getting the type of 'tuple_var_assignment_114435' (line 295)
                tuple_var_assignment_114435_115105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'tuple_var_assignment_114435')
                # Getting the type of 'self' (line 295)
                self_115106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'self')
                # Setting the type of the member 'axis' of a type (line 295)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 24), self_115106, 'axis', tuple_var_assignment_114435_115105)
                
                # Assigning a Name to a Name (line 295):
                # Getting the type of 'tuple_var_assignment_114436' (line 295)
                tuple_var_assignment_114436_115107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'tuple_var_assignment_114436')
                # Assigning a type to the variable 'ndmin' (line 295)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 35), 'ndmin', tuple_var_assignment_114436_115107)
                
                
                
                # Call to len(...): (line 297)
                # Processing the call arguments (line 297)
                # Getting the type of 'vec' (line 297)
                vec_115109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 31), 'vec', False)
                # Processing the call keyword arguments (line 297)
                kwargs_115110 = {}
                # Getting the type of 'len' (line 297)
                len_115108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 27), 'len', False)
                # Calling len(args, kwargs) (line 297)
                len_call_result_115111 = invoke(stypy.reporting.localization.Localization(__file__, 297, 27), len_115108, *[vec_115109], **kwargs_115110)
                
                int_115112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 39), 'int')
                # Applying the binary operator '==' (line 297)
                result_eq_115113 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 27), '==', len_call_result_115111, int_115112)
                
                # Testing the type of an if condition (line 297)
                if_condition_115114 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 24), result_eq_115113)
                # Assigning a type to the variable 'if_condition_115114' (line 297)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 24), 'if_condition_115114', if_condition_115114)
                # SSA begins for if statement (line 297)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 298):
                
                # Assigning a Call to a Name (line 298):
                
                # Call to int(...): (line 298)
                # Processing the call arguments (line 298)
                
                # Obtaining the type of the subscript
                int_115116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 46), 'int')
                # Getting the type of 'vec' (line 298)
                vec_115117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 42), 'vec', False)
                # Obtaining the member '__getitem__' of a type (line 298)
                getitem___115118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 42), vec_115117, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 298)
                subscript_call_result_115119 = invoke(stypy.reporting.localization.Localization(__file__, 298, 42), getitem___115118, int_115116)
                
                # Processing the call keyword arguments (line 298)
                kwargs_115120 = {}
                # Getting the type of 'int' (line 298)
                int_115115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 38), 'int', False)
                # Calling int(args, kwargs) (line 298)
                int_call_result_115121 = invoke(stypy.reporting.localization.Localization(__file__, 298, 38), int_115115, *[subscript_call_result_115119], **kwargs_115120)
                
                # Assigning a type to the variable 'trans1d' (line 298)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 28), 'trans1d', int_call_result_115121)
                # SSA join for if statement (line 297)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA branch for the except part of a try statement (line 294)
                # SSA branch for the except '<any exception>' branch of a try statement (line 294)
                module_type_store.open_ssa_branch('except')
                
                # Call to ValueError(...): (line 301)
                # Processing the call arguments (line 301)
                str_115123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 41), 'str', 'unknown special directive')
                # Processing the call keyword arguments (line 301)
                kwargs_115124 = {}
                # Getting the type of 'ValueError' (line 301)
                ValueError_115122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 30), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 301)
                ValueError_call_result_115125 = invoke(stypy.reporting.localization.Localization(__file__, 301, 30), ValueError_115122, *[str_115123], **kwargs_115124)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 301, 24), ValueError_call_result_115125, 'raise parameter', BaseException)
                # SSA join for try-except statement (line 294)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for if statement (line 292)
                module_type_store = module_type_store.join_ssa_context()
                
                
                
                # SSA begins for try-except statement (line 302)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                
                # Assigning a Call to a Attribute (line 303):
                
                # Assigning a Call to a Attribute (line 303):
                
                # Call to int(...): (line 303)
                # Processing the call arguments (line 303)
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 303)
                k_115127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 40), 'k', False)
                # Getting the type of 'key' (line 303)
                key_115128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 36), 'key', False)
                # Obtaining the member '__getitem__' of a type (line 303)
                getitem___115129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 36), key_115128, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 303)
                subscript_call_result_115130 = invoke(stypy.reporting.localization.Localization(__file__, 303, 36), getitem___115129, k_115127)
                
                # Processing the call keyword arguments (line 303)
                kwargs_115131 = {}
                # Getting the type of 'int' (line 303)
                int_115126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 32), 'int', False)
                # Calling int(args, kwargs) (line 303)
                int_call_result_115132 = invoke(stypy.reporting.localization.Localization(__file__, 303, 32), int_115126, *[subscript_call_result_115130], **kwargs_115131)
                
                # Getting the type of 'self' (line 303)
                self_115133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'self')
                # Setting the type of the member 'axis' of a type (line 303)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 20), self_115133, 'axis', int_call_result_115132)
                # SSA branch for the except part of a try statement (line 302)
                # SSA branch for the except 'Tuple' branch of a try statement (line 302)
                module_type_store.open_ssa_branch('except')
                
                # Call to ValueError(...): (line 306)
                # Processing the call arguments (line 306)
                str_115135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 37), 'str', 'unknown special directive')
                # Processing the call keyword arguments (line 306)
                kwargs_115136 = {}
                # Getting the type of 'ValueError' (line 306)
                ValueError_115134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 26), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 306)
                ValueError_call_result_115137 = invoke(stypy.reporting.localization.Localization(__file__, 306, 26), ValueError_115134, *[str_115135], **kwargs_115136)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 306, 20), ValueError_call_result_115137, 'raise parameter', BaseException)
                # SSA join for try-except statement (line 302)
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_115045:
                    # Runtime conditional SSA for else branch (line 283)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_115044) or more_types_in_union_115045):
                
                
                
                # Call to type(...): (line 307)
                # Processing the call arguments (line 307)
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 307)
                k_115139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 26), 'k', False)
                # Getting the type of 'key' (line 307)
                key_115140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 22), 'key', False)
                # Obtaining the member '__getitem__' of a type (line 307)
                getitem___115141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 22), key_115140, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 307)
                subscript_call_result_115142 = invoke(stypy.reporting.localization.Localization(__file__, 307, 22), getitem___115141, k_115139)
                
                # Processing the call keyword arguments (line 307)
                kwargs_115143 = {}
                # Getting the type of 'type' (line 307)
                type_115138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 17), 'type', False)
                # Calling type(args, kwargs) (line 307)
                type_call_result_115144 = invoke(stypy.reporting.localization.Localization(__file__, 307, 17), type_115138, *[subscript_call_result_115142], **kwargs_115143)
                
                # Getting the type of 'ScalarType' (line 307)
                ScalarType_115145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 33), 'ScalarType')
                # Applying the binary operator 'in' (line 307)
                result_contains_115146 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 17), 'in', type_call_result_115144, ScalarType_115145)
                
                # Testing the type of an if condition (line 307)
                if_condition_115147 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 17), result_contains_115146)
                # Assigning a type to the variable 'if_condition_115147' (line 307)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 17), 'if_condition_115147', if_condition_115147)
                # SSA begins for if statement (line 307)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 308):
                
                # Assigning a Call to a Name (line 308):
                
                # Call to array(...): (line 308)
                # Processing the call arguments (line 308)
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 308)
                k_115149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 35), 'k', False)
                # Getting the type of 'key' (line 308)
                key_115150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 31), 'key', False)
                # Obtaining the member '__getitem__' of a type (line 308)
                getitem___115151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 31), key_115150, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 308)
                subscript_call_result_115152 = invoke(stypy.reporting.localization.Localization(__file__, 308, 31), getitem___115151, k_115149)
                
                # Processing the call keyword arguments (line 308)
                # Getting the type of 'ndmin' (line 308)
                ndmin_115153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 45), 'ndmin', False)
                keyword_115154 = ndmin_115153
                kwargs_115155 = {'ndmin': keyword_115154}
                # Getting the type of 'array' (line 308)
                array_115148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 25), 'array', False)
                # Calling array(args, kwargs) (line 308)
                array_call_result_115156 = invoke(stypy.reporting.localization.Localization(__file__, 308, 25), array_115148, *[subscript_call_result_115152], **kwargs_115155)
                
                # Assigning a type to the variable 'newobj' (line 308)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'newobj', array_call_result_115156)
                
                # Call to append(...): (line 309)
                # Processing the call arguments (line 309)
                # Getting the type of 'k' (line 309)
                k_115159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 31), 'k', False)
                # Processing the call keyword arguments (line 309)
                kwargs_115160 = {}
                # Getting the type of 'scalars' (line 309)
                scalars_115157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'scalars', False)
                # Obtaining the member 'append' of a type (line 309)
                append_115158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 16), scalars_115157, 'append')
                # Calling append(args, kwargs) (line 309)
                append_call_result_115161 = invoke(stypy.reporting.localization.Localization(__file__, 309, 16), append_115158, *[k_115159], **kwargs_115160)
                
                
                # Assigning a Name to a Name (line 310):
                
                # Assigning a Name to a Name (line 310):
                # Getting the type of 'True' (line 310)
                True_115162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 25), 'True')
                # Assigning a type to the variable 'scalar' (line 310)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'scalar', True_115162)
                
                # Call to append(...): (line 311)
                # Processing the call arguments (line 311)
                # Getting the type of 'newobj' (line 311)
                newobj_115165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 35), 'newobj', False)
                # Obtaining the member 'dtype' of a type (line 311)
                dtype_115166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 35), newobj_115165, 'dtype')
                # Processing the call keyword arguments (line 311)
                kwargs_115167 = {}
                # Getting the type of 'scalartypes' (line 311)
                scalartypes_115163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'scalartypes', False)
                # Obtaining the member 'append' of a type (line 311)
                append_115164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 16), scalartypes_115163, 'append')
                # Calling append(args, kwargs) (line 311)
                append_call_result_115168 = invoke(stypy.reporting.localization.Localization(__file__, 311, 16), append_115164, *[dtype_115166], **kwargs_115167)
                
                # SSA branch for the else part of an if statement (line 307)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Subscript to a Name (line 313):
                
                # Assigning a Subscript to a Name (line 313):
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 313)
                k_115169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 29), 'k')
                # Getting the type of 'key' (line 313)
                key_115170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 25), 'key')
                # Obtaining the member '__getitem__' of a type (line 313)
                getitem___115171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 25), key_115170, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 313)
                subscript_call_result_115172 = invoke(stypy.reporting.localization.Localization(__file__, 313, 25), getitem___115171, k_115169)
                
                # Assigning a type to the variable 'newobj' (line 313)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'newobj', subscript_call_result_115172)
                
                
                # Getting the type of 'ndmin' (line 314)
                ndmin_115173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 19), 'ndmin')
                int_115174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 27), 'int')
                # Applying the binary operator '>' (line 314)
                result_gt_115175 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 19), '>', ndmin_115173, int_115174)
                
                # Testing the type of an if condition (line 314)
                if_condition_115176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 16), result_gt_115175)
                # Assigning a type to the variable 'if_condition_115176' (line 314)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'if_condition_115176', if_condition_115176)
                # SSA begins for if statement (line 314)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 315):
                
                # Assigning a Call to a Name (line 315):
                
                # Call to array(...): (line 315)
                # Processing the call arguments (line 315)
                # Getting the type of 'newobj' (line 315)
                newobj_115178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 36), 'newobj', False)
                # Processing the call keyword arguments (line 315)
                # Getting the type of 'False' (line 315)
                False_115179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 49), 'False', False)
                keyword_115180 = False_115179
                # Getting the type of 'True' (line 315)
                True_115181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 62), 'True', False)
                keyword_115182 = True_115181
                kwargs_115183 = {'subok': keyword_115182, 'copy': keyword_115180}
                # Getting the type of 'array' (line 315)
                array_115177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 30), 'array', False)
                # Calling array(args, kwargs) (line 315)
                array_call_result_115184 = invoke(stypy.reporting.localization.Localization(__file__, 315, 30), array_115177, *[newobj_115178], **kwargs_115183)
                
                # Assigning a type to the variable 'tempobj' (line 315)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 20), 'tempobj', array_call_result_115184)
                
                # Assigning a Call to a Name (line 316):
                
                # Assigning a Call to a Name (line 316):
                
                # Call to array(...): (line 316)
                # Processing the call arguments (line 316)
                # Getting the type of 'newobj' (line 316)
                newobj_115186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 35), 'newobj', False)
                # Processing the call keyword arguments (line 316)
                # Getting the type of 'False' (line 316)
                False_115187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 48), 'False', False)
                keyword_115188 = False_115187
                # Getting the type of 'True' (line 316)
                True_115189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 61), 'True', False)
                keyword_115190 = True_115189
                # Getting the type of 'ndmin' (line 317)
                ndmin_115191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 41), 'ndmin', False)
                keyword_115192 = ndmin_115191
                kwargs_115193 = {'subok': keyword_115190, 'copy': keyword_115188, 'ndmin': keyword_115192}
                # Getting the type of 'array' (line 316)
                array_115185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 29), 'array', False)
                # Calling array(args, kwargs) (line 316)
                array_call_result_115194 = invoke(stypy.reporting.localization.Localization(__file__, 316, 29), array_115185, *[newobj_115186], **kwargs_115193)
                
                # Assigning a type to the variable 'newobj' (line 316)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 20), 'newobj', array_call_result_115194)
                
                
                # Evaluating a boolean operation
                
                # Getting the type of 'trans1d' (line 318)
                trans1d_115195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 23), 'trans1d')
                int_115196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 34), 'int')
                # Applying the binary operator '!=' (line 318)
                result_ne_115197 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 23), '!=', trans1d_115195, int_115196)
                
                
                # Getting the type of 'tempobj' (line 318)
                tempobj_115198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 41), 'tempobj')
                # Obtaining the member 'ndim' of a type (line 318)
                ndim_115199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 41), tempobj_115198, 'ndim')
                # Getting the type of 'ndmin' (line 318)
                ndmin_115200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 56), 'ndmin')
                # Applying the binary operator '<' (line 318)
                result_lt_115201 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 41), '<', ndim_115199, ndmin_115200)
                
                # Applying the binary operator 'and' (line 318)
                result_and_keyword_115202 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 23), 'and', result_ne_115197, result_lt_115201)
                
                # Testing the type of an if condition (line 318)
                if_condition_115203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 20), result_and_keyword_115202)
                # Assigning a type to the variable 'if_condition_115203' (line 318)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'if_condition_115203', if_condition_115203)
                # SSA begins for if statement (line 318)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 319):
                
                # Assigning a BinOp to a Name (line 319):
                # Getting the type of 'ndmin' (line 319)
                ndmin_115204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 29), 'ndmin')
                # Getting the type of 'tempobj' (line 319)
                tempobj_115205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 35), 'tempobj')
                # Obtaining the member 'ndim' of a type (line 319)
                ndim_115206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 35), tempobj_115205, 'ndim')
                # Applying the binary operator '-' (line 319)
                result_sub_115207 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 29), '-', ndmin_115204, ndim_115206)
                
                # Assigning a type to the variable 'k2' (line 319)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 24), 'k2', result_sub_115207)
                
                
                # Getting the type of 'trans1d' (line 320)
                trans1d_115208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 28), 'trans1d')
                int_115209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 38), 'int')
                # Applying the binary operator '<' (line 320)
                result_lt_115210 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 28), '<', trans1d_115208, int_115209)
                
                # Testing the type of an if condition (line 320)
                if_condition_115211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 24), result_lt_115210)
                # Assigning a type to the variable 'if_condition_115211' (line 320)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 24), 'if_condition_115211', if_condition_115211)
                # SSA begins for if statement (line 320)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'trans1d' (line 321)
                trans1d_115212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 28), 'trans1d')
                # Getting the type of 'k2' (line 321)
                k2_115213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 39), 'k2')
                int_115214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 44), 'int')
                # Applying the binary operator '+' (line 321)
                result_add_115215 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 39), '+', k2_115213, int_115214)
                
                # Applying the binary operator '+=' (line 321)
                result_iadd_115216 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 28), '+=', trans1d_115212, result_add_115215)
                # Assigning a type to the variable 'trans1d' (line 321)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 28), 'trans1d', result_iadd_115216)
                
                # SSA join for if statement (line 320)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a Call to a Name (line 322):
                
                # Assigning a Call to a Name (line 322):
                
                # Call to list(...): (line 322)
                # Processing the call arguments (line 322)
                
                # Call to range(...): (line 322)
                # Processing the call arguments (line 322)
                # Getting the type of 'ndmin' (line 322)
                ndmin_115219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 45), 'ndmin', False)
                # Processing the call keyword arguments (line 322)
                kwargs_115220 = {}
                # Getting the type of 'range' (line 322)
                range_115218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 39), 'range', False)
                # Calling range(args, kwargs) (line 322)
                range_call_result_115221 = invoke(stypy.reporting.localization.Localization(__file__, 322, 39), range_115218, *[ndmin_115219], **kwargs_115220)
                
                # Processing the call keyword arguments (line 322)
                kwargs_115222 = {}
                # Getting the type of 'list' (line 322)
                list_115217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 34), 'list', False)
                # Calling list(args, kwargs) (line 322)
                list_call_result_115223 = invoke(stypy.reporting.localization.Localization(__file__, 322, 34), list_115217, *[range_call_result_115221], **kwargs_115222)
                
                # Assigning a type to the variable 'defaxes' (line 322)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 24), 'defaxes', list_call_result_115223)
                
                # Assigning a Name to a Name (line 323):
                
                # Assigning a Name to a Name (line 323):
                # Getting the type of 'trans1d' (line 323)
                trans1d_115224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 29), 'trans1d')
                # Assigning a type to the variable 'k1' (line 323)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 24), 'k1', trans1d_115224)
                
                # Assigning a BinOp to a Name (line 324):
                
                # Assigning a BinOp to a Name (line 324):
                
                # Obtaining the type of the subscript
                # Getting the type of 'k1' (line 324)
                k1_115225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 40), 'k1')
                slice_115226 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 324, 31), None, k1_115225, None)
                # Getting the type of 'defaxes' (line 324)
                defaxes_115227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 31), 'defaxes')
                # Obtaining the member '__getitem__' of a type (line 324)
                getitem___115228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 31), defaxes_115227, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 324)
                subscript_call_result_115229 = invoke(stypy.reporting.localization.Localization(__file__, 324, 31), getitem___115228, slice_115226)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'k2' (line 324)
                k2_115230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 54), 'k2')
                slice_115231 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 324, 46), k2_115230, None, None)
                # Getting the type of 'defaxes' (line 324)
                defaxes_115232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 46), 'defaxes')
                # Obtaining the member '__getitem__' of a type (line 324)
                getitem___115233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 46), defaxes_115232, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 324)
                subscript_call_result_115234 = invoke(stypy.reporting.localization.Localization(__file__, 324, 46), getitem___115233, slice_115231)
                
                # Applying the binary operator '+' (line 324)
                result_add_115235 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 31), '+', subscript_call_result_115229, subscript_call_result_115234)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'k1' (line 325)
                k1_115236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 39), 'k1')
                # Getting the type of 'k2' (line 325)
                k2_115237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 42), 'k2')
                slice_115238 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 325, 31), k1_115236, k2_115237, None)
                # Getting the type of 'defaxes' (line 325)
                defaxes_115239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 31), 'defaxes')
                # Obtaining the member '__getitem__' of a type (line 325)
                getitem___115240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 31), defaxes_115239, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 325)
                subscript_call_result_115241 = invoke(stypy.reporting.localization.Localization(__file__, 325, 31), getitem___115240, slice_115238)
                
                # Applying the binary operator '+' (line 324)
                result_add_115242 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 59), '+', result_add_115235, subscript_call_result_115241)
                
                # Assigning a type to the variable 'axes' (line 324)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 24), 'axes', result_add_115242)
                
                # Assigning a Call to a Name (line 326):
                
                # Assigning a Call to a Name (line 326):
                
                # Call to transpose(...): (line 326)
                # Processing the call arguments (line 326)
                # Getting the type of 'axes' (line 326)
                axes_115245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 50), 'axes', False)
                # Processing the call keyword arguments (line 326)
                kwargs_115246 = {}
                # Getting the type of 'newobj' (line 326)
                newobj_115243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 33), 'newobj', False)
                # Obtaining the member 'transpose' of a type (line 326)
                transpose_115244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 33), newobj_115243, 'transpose')
                # Calling transpose(args, kwargs) (line 326)
                transpose_call_result_115247 = invoke(stypy.reporting.localization.Localization(__file__, 326, 33), transpose_115244, *[axes_115245], **kwargs_115246)
                
                # Assigning a type to the variable 'newobj' (line 326)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 24), 'newobj', transpose_call_result_115247)
                # SSA join for if statement (line 318)
                module_type_store = module_type_store.join_ssa_context()
                
                # Deleting a member
                module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 327, 20), module_type_store, 'tempobj')
                # SSA join for if statement (line 314)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for if statement (line 307)
                module_type_store = module_type_store.join_ssa_context()
                

                if (may_be_115044 and more_types_in_union_115045):
                    # SSA join for if statement (line 283)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_114964 and more_types_in_union_114965):
                # SSA join for if statement (line 266)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to append(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'newobj' (line 328)
        newobj_115250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 24), 'newobj', False)
        # Processing the call keyword arguments (line 328)
        kwargs_115251 = {}
        # Getting the type of 'objs' (line 328)
        objs_115248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'objs', False)
        # Obtaining the member 'append' of a type (line 328)
        append_115249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 12), objs_115248, 'append')
        # Calling append(args, kwargs) (line 328)
        append_call_result_115252 = invoke(stypy.reporting.localization.Localization(__file__, 328, 12), append_115249, *[newobj_115250], **kwargs_115251)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'scalar' (line 329)
        scalar_115253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 19), 'scalar')
        # Applying the 'not' unary operator (line 329)
        result_not__115254 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 15), 'not', scalar_115253)
        
        
        # Call to isinstance(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'newobj' (line 329)
        newobj_115256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 41), 'newobj', False)
        # Getting the type of '_nx' (line 329)
        _nx_115257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 49), '_nx', False)
        # Obtaining the member 'ndarray' of a type (line 329)
        ndarray_115258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 49), _nx_115257, 'ndarray')
        # Processing the call keyword arguments (line 329)
        kwargs_115259 = {}
        # Getting the type of 'isinstance' (line 329)
        isinstance_115255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 30), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 329)
        isinstance_call_result_115260 = invoke(stypy.reporting.localization.Localization(__file__, 329, 30), isinstance_115255, *[newobj_115256, ndarray_115258], **kwargs_115259)
        
        # Applying the binary operator 'and' (line 329)
        result_and_keyword_115261 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 15), 'and', result_not__115254, isinstance_call_result_115260)
        
        # Testing the type of an if condition (line 329)
        if_condition_115262 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 329, 12), result_and_keyword_115261)
        # Assigning a type to the variable 'if_condition_115262' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'if_condition_115262', if_condition_115262)
        # SSA begins for if statement (line 329)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'newobj' (line 330)
        newobj_115265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 34), 'newobj', False)
        # Obtaining the member 'dtype' of a type (line 330)
        dtype_115266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 34), newobj_115265, 'dtype')
        # Processing the call keyword arguments (line 330)
        kwargs_115267 = {}
        # Getting the type of 'arraytypes' (line 330)
        arraytypes_115263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'arraytypes', False)
        # Obtaining the member 'append' of a type (line 330)
        append_115264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 16), arraytypes_115263, 'append')
        # Calling append(args, kwargs) (line 330)
        append_call_result_115268 = invoke(stypy.reporting.localization.Localization(__file__, 330, 16), append_115264, *[dtype_115266], **kwargs_115267)
        
        # SSA join for if statement (line 329)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 333):
        
        # Assigning a Call to a Name (line 333):
        
        # Call to find_common_type(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'arraytypes' (line 333)
        arraytypes_115270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 39), 'arraytypes', False)
        # Getting the type of 'scalartypes' (line 333)
        scalartypes_115271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 51), 'scalartypes', False)
        # Processing the call keyword arguments (line 333)
        kwargs_115272 = {}
        # Getting the type of 'find_common_type' (line 333)
        find_common_type_115269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 22), 'find_common_type', False)
        # Calling find_common_type(args, kwargs) (line 333)
        find_common_type_call_result_115273 = invoke(stypy.reporting.localization.Localization(__file__, 333, 22), find_common_type_115269, *[arraytypes_115270, scalartypes_115271], **kwargs_115272)
        
        # Assigning a type to the variable 'final_dtype' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'final_dtype', find_common_type_call_result_115273)
        
        # Type idiom detected: calculating its left and rigth part (line 334)
        # Getting the type of 'final_dtype' (line 334)
        final_dtype_115274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'final_dtype')
        # Getting the type of 'None' (line 334)
        None_115275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 30), 'None')
        
        (may_be_115276, more_types_in_union_115277) = may_not_be_none(final_dtype_115274, None_115275)

        if may_be_115276:

            if more_types_in_union_115277:
                # Runtime conditional SSA (line 334)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'scalars' (line 335)
            scalars_115278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 21), 'scalars')
            # Testing the type of a for loop iterable (line 335)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 335, 12), scalars_115278)
            # Getting the type of the for loop variable (line 335)
            for_loop_var_115279 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 335, 12), scalars_115278)
            # Assigning a type to the variable 'k' (line 335)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'k', for_loop_var_115279)
            # SSA begins for a for statement (line 335)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Subscript (line 336):
            
            # Assigning a Call to a Subscript (line 336):
            
            # Call to astype(...): (line 336)
            # Processing the call arguments (line 336)
            # Getting the type of 'final_dtype' (line 336)
            final_dtype_115285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 41), 'final_dtype', False)
            # Processing the call keyword arguments (line 336)
            kwargs_115286 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 336)
            k_115280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 31), 'k', False)
            # Getting the type of 'objs' (line 336)
            objs_115281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 26), 'objs', False)
            # Obtaining the member '__getitem__' of a type (line 336)
            getitem___115282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 26), objs_115281, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 336)
            subscript_call_result_115283 = invoke(stypy.reporting.localization.Localization(__file__, 336, 26), getitem___115282, k_115280)
            
            # Obtaining the member 'astype' of a type (line 336)
            astype_115284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 26), subscript_call_result_115283, 'astype')
            # Calling astype(args, kwargs) (line 336)
            astype_call_result_115287 = invoke(stypy.reporting.localization.Localization(__file__, 336, 26), astype_115284, *[final_dtype_115285], **kwargs_115286)
            
            # Getting the type of 'objs' (line 336)
            objs_115288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'objs')
            # Getting the type of 'k' (line 336)
            k_115289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 21), 'k')
            # Storing an element on a container (line 336)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 16), objs_115288, (k_115289, astype_call_result_115287))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_115277:
                # SSA join for if statement (line 334)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 338):
        
        # Assigning a Call to a Name (line 338):
        
        # Call to concatenate(...): (line 338)
        # Processing the call arguments (line 338)
        
        # Call to tuple(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'objs' (line 338)
        objs_115293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 36), 'objs', False)
        # Processing the call keyword arguments (line 338)
        kwargs_115294 = {}
        # Getting the type of 'tuple' (line 338)
        tuple_115292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 30), 'tuple', False)
        # Calling tuple(args, kwargs) (line 338)
        tuple_call_result_115295 = invoke(stypy.reporting.localization.Localization(__file__, 338, 30), tuple_115292, *[objs_115293], **kwargs_115294)
        
        # Processing the call keyword arguments (line 338)
        # Getting the type of 'self' (line 338)
        self_115296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 48), 'self', False)
        # Obtaining the member 'axis' of a type (line 338)
        axis_115297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 48), self_115296, 'axis')
        keyword_115298 = axis_115297
        kwargs_115299 = {'axis': keyword_115298}
        # Getting the type of '_nx' (line 338)
        _nx_115290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 14), '_nx', False)
        # Obtaining the member 'concatenate' of a type (line 338)
        concatenate_115291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 14), _nx_115290, 'concatenate')
        # Calling concatenate(args, kwargs) (line 338)
        concatenate_call_result_115300 = invoke(stypy.reporting.localization.Localization(__file__, 338, 14), concatenate_115291, *[tuple_call_result_115295], **kwargs_115299)
        
        # Assigning a type to the variable 'res' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'res', concatenate_call_result_115300)
        
        # Call to _retval(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'res' (line 339)
        res_115303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 28), 'res', False)
        # Processing the call keyword arguments (line 339)
        kwargs_115304 = {}
        # Getting the type of 'self' (line 339)
        self_115301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'self', False)
        # Obtaining the member '_retval' of a type (line 339)
        _retval_115302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 15), self_115301, '_retval')
        # Calling _retval(args, kwargs) (line 339)
        _retval_call_result_115305 = invoke(stypy.reporting.localization.Localization(__file__, 339, 15), _retval_115302, *[res_115303], **kwargs_115304)
        
        # Assigning a type to the variable 'stypy_return_type' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'stypy_return_type', _retval_call_result_115305)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 251)
        stypy_return_type_115306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115306)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_115306


    @norecursion
    def __getslice__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getslice__'
        module_type_store = module_type_store.open_function_context('__getslice__', 341, 4, False)
        # Assigning a type to the variable 'self' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AxisConcatenator.__getslice__.__dict__.__setitem__('stypy_localization', localization)
        AxisConcatenator.__getslice__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AxisConcatenator.__getslice__.__dict__.__setitem__('stypy_type_store', module_type_store)
        AxisConcatenator.__getslice__.__dict__.__setitem__('stypy_function_name', 'AxisConcatenator.__getslice__')
        AxisConcatenator.__getslice__.__dict__.__setitem__('stypy_param_names_list', ['i', 'j'])
        AxisConcatenator.__getslice__.__dict__.__setitem__('stypy_varargs_param_name', None)
        AxisConcatenator.__getslice__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AxisConcatenator.__getslice__.__dict__.__setitem__('stypy_call_defaults', defaults)
        AxisConcatenator.__getslice__.__dict__.__setitem__('stypy_call_varargs', varargs)
        AxisConcatenator.__getslice__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AxisConcatenator.__getslice__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AxisConcatenator.__getslice__', ['i', 'j'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getslice__', localization, ['i', 'j'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getslice__(...)' code ##################

        
        # Assigning a Call to a Name (line 342):
        
        # Assigning a Call to a Name (line 342):
        
        # Call to arange(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'i' (line 342)
        i_115309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 25), 'i', False)
        # Getting the type of 'j' (line 342)
        j_115310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 28), 'j', False)
        # Processing the call keyword arguments (line 342)
        kwargs_115311 = {}
        # Getting the type of '_nx' (line 342)
        _nx_115307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 14), '_nx', False)
        # Obtaining the member 'arange' of a type (line 342)
        arange_115308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 14), _nx_115307, 'arange')
        # Calling arange(args, kwargs) (line 342)
        arange_call_result_115312 = invoke(stypy.reporting.localization.Localization(__file__, 342, 14), arange_115308, *[i_115309, j_115310], **kwargs_115311)
        
        # Assigning a type to the variable 'res' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'res', arange_call_result_115312)
        
        # Call to _retval(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'res' (line 343)
        res_115315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 28), 'res', False)
        # Processing the call keyword arguments (line 343)
        kwargs_115316 = {}
        # Getting the type of 'self' (line 343)
        self_115313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 15), 'self', False)
        # Obtaining the member '_retval' of a type (line 343)
        _retval_115314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 15), self_115313, '_retval')
        # Calling _retval(args, kwargs) (line 343)
        _retval_call_result_115317 = invoke(stypy.reporting.localization.Localization(__file__, 343, 15), _retval_115314, *[res_115315], **kwargs_115316)
        
        # Assigning a type to the variable 'stypy_return_type' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'stypy_return_type', _retval_call_result_115317)
        
        # ################# End of '__getslice__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getslice__' in the type store
        # Getting the type of 'stypy_return_type' (line 341)
        stypy_return_type_115318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115318)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getslice__'
        return stypy_return_type_115318


    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 345, 4, False)
        # Assigning a type to the variable 'self' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AxisConcatenator.__len__.__dict__.__setitem__('stypy_localization', localization)
        AxisConcatenator.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AxisConcatenator.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        AxisConcatenator.__len__.__dict__.__setitem__('stypy_function_name', 'AxisConcatenator.__len__')
        AxisConcatenator.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        AxisConcatenator.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        AxisConcatenator.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AxisConcatenator.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        AxisConcatenator.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        AxisConcatenator.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AxisConcatenator.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AxisConcatenator.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        int_115319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'stypy_return_type', int_115319)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 345)
        stypy_return_type_115320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115320)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_115320


# Assigning a type to the variable 'AxisConcatenator' (line 223)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'AxisConcatenator', AxisConcatenator)
# Declaration of the 'RClass' class
# Getting the type of 'AxisConcatenator' (line 352)
AxisConcatenator_115321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 13), 'AxisConcatenator')

class RClass(AxisConcatenator_115321, ):
    str_115322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, (-1)), 'str', "\n    Translates slice objects to concatenation along the first axis.\n\n    This is a simple way to build up arrays quickly. There are two use cases.\n\n    1. If the index expression contains comma separated arrays, then stack\n       them along their first axis.\n    2. If the index expression contains slice notation or scalars then create\n       a 1-D array with a range indicated by the slice notation.\n\n    If slice notation is used, the syntax ``start:stop:step`` is equivalent\n    to ``np.arange(start, stop, step)`` inside of the brackets. However, if\n    ``step`` is an imaginary number (i.e. 100j) then its integer portion is\n    interpreted as a number-of-points desired and the start and stop are\n    inclusive. In other words ``start:stop:stepj`` is interpreted as\n    ``np.linspace(start, stop, step, endpoint=1)`` inside of the brackets.\n    After expansion of slice notation, all comma separated sequences are\n    concatenated together.\n\n    Optional character strings placed as the first element of the index\n    expression can be used to change the output. The strings 'r' or 'c' result\n    in matrix output. If the result is 1-D and 'r' is specified a 1 x N (row)\n    matrix is produced. If the result is 1-D and 'c' is specified, then a N x 1\n    (column) matrix is produced. If the result is 2-D then both provide the\n    same matrix result.\n\n    A string integer specifies which axis to stack multiple comma separated\n    arrays along. A string of two comma-separated integers allows indication\n    of the minimum number of dimensions to force each entry into as the\n    second integer (the axis to concatenate along is still the first integer).\n\n    A string with three comma-separated integers allows specification of the\n    axis to concatenate along, the minimum number of dimensions to force the\n    entries to, and which axis should contain the start of the arrays which\n    are less than the specified number of dimensions. In other words the third\n    integer allows you to specify where the 1's should be placed in the shape\n    of the arrays that have their shapes upgraded. By default, they are placed\n    in the front of the shape tuple. The third argument allows you to specify\n    where the start of the array should be instead. Thus, a third argument of\n    '0' would place the 1's at the end of the array shape. Negative integers\n    specify where in the new shape tuple the last dimension of upgraded arrays\n    should be placed, so the default is '-1'.\n\n    Parameters\n    ----------\n    Not a function, so takes no parameters\n\n\n    Returns\n    -------\n    A concatenated ndarray or matrix.\n\n    See Also\n    --------\n    concatenate : Join a sequence of arrays along an existing axis.\n    c_ : Translates slice objects to concatenation along the second axis.\n\n    Examples\n    --------\n    >>> np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])]\n    array([1, 2, 3, 0, 0, 4, 5, 6])\n    >>> np.r_[-1:1:6j, [0]*3, 5, 6]\n    array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ,  0. ,  0. ,  0. ,  5. ,  6. ])\n\n    String integers specify the axis to concatenate along or the minimum\n    number of dimensions to force entries into.\n\n    >>> a = np.array([[0, 1, 2], [3, 4, 5]])\n    >>> np.r_['-1', a, a] # concatenate along last axis\n    array([[0, 1, 2, 0, 1, 2],\n           [3, 4, 5, 3, 4, 5]])\n    >>> np.r_['0,2', [1,2,3], [4,5,6]] # concatenate along first axis, dim>=2\n    array([[1, 2, 3],\n           [4, 5, 6]])\n\n    >>> np.r_['0,2,0', [1,2,3], [4,5,6]]\n    array([[1],\n           [2],\n           [3],\n           [4],\n           [5],\n           [6]])\n    >>> np.r_['1,2,0', [1,2,3], [4,5,6]]\n    array([[1, 4],\n           [2, 5],\n           [3, 6]])\n\n    Using 'r' or 'c' as a first string argument creates a matrix.\n\n    >>> np.r_['r',[1,2,3], [4,5,6]]\n    matrix([[1, 2, 3, 4, 5, 6]])\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 447, 4, False)
        # Assigning a type to the variable 'self' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RClass.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 448)
        # Processing the call arguments (line 448)
        # Getting the type of 'self' (line 448)
        self_115325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 34), 'self', False)
        int_115326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 40), 'int')
        # Processing the call keyword arguments (line 448)
        kwargs_115327 = {}
        # Getting the type of 'AxisConcatenator' (line 448)
        AxisConcatenator_115323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'AxisConcatenator', False)
        # Obtaining the member '__init__' of a type (line 448)
        init___115324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), AxisConcatenator_115323, '__init__')
        # Calling __init__(args, kwargs) (line 448)
        init___call_result_115328 = invoke(stypy.reporting.localization.Localization(__file__, 448, 8), init___115324, *[self_115325, int_115326], **kwargs_115327)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'RClass' (line 352)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 0), 'RClass', RClass)

# Assigning a Call to a Name (line 450):

# Assigning a Call to a Name (line 450):

# Call to RClass(...): (line 450)
# Processing the call keyword arguments (line 450)
kwargs_115330 = {}
# Getting the type of 'RClass' (line 450)
RClass_115329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 5), 'RClass', False)
# Calling RClass(args, kwargs) (line 450)
RClass_call_result_115331 = invoke(stypy.reporting.localization.Localization(__file__, 450, 5), RClass_115329, *[], **kwargs_115330)

# Assigning a type to the variable 'r_' (line 450)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 0), 'r_', RClass_call_result_115331)
# Declaration of the 'CClass' class
# Getting the type of 'AxisConcatenator' (line 452)
AxisConcatenator_115332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 13), 'AxisConcatenator')

class CClass(AxisConcatenator_115332, ):
    str_115333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, (-1)), 'str', "\n    Translates slice objects to concatenation along the second axis.\n\n    This is short-hand for ``np.r_['-1,2,0', index expression]``, which is\n    useful because of its common occurrence. In particular, arrays will be\n    stacked along their last axis after being upgraded to at least 2-D with\n    1's post-pended to the shape (column vectors made out of 1-D arrays).\n\n    For detailed documentation, see `r_`.\n\n    Examples\n    --------\n    >>> np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]\n    array([[1, 2, 3, 0, 0, 4, 5, 6]])\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 470, 4, False)
        # Assigning a type to the variable 'self' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CClass.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 471)
        # Processing the call arguments (line 471)
        # Getting the type of 'self' (line 471)
        self_115336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 34), 'self', False)
        int_115337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 40), 'int')
        # Processing the call keyword arguments (line 471)
        int_115338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 50), 'int')
        keyword_115339 = int_115338
        int_115340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 61), 'int')
        keyword_115341 = int_115340
        kwargs_115342 = {'trans1d': keyword_115341, 'ndmin': keyword_115339}
        # Getting the type of 'AxisConcatenator' (line 471)
        AxisConcatenator_115334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'AxisConcatenator', False)
        # Obtaining the member '__init__' of a type (line 471)
        init___115335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 8), AxisConcatenator_115334, '__init__')
        # Calling __init__(args, kwargs) (line 471)
        init___call_result_115343 = invoke(stypy.reporting.localization.Localization(__file__, 471, 8), init___115335, *[self_115336, int_115337], **kwargs_115342)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'CClass' (line 452)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 0), 'CClass', CClass)

# Assigning a Call to a Name (line 473):

# Assigning a Call to a Name (line 473):

# Call to CClass(...): (line 473)
# Processing the call keyword arguments (line 473)
kwargs_115345 = {}
# Getting the type of 'CClass' (line 473)
CClass_115344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 5), 'CClass', False)
# Calling CClass(args, kwargs) (line 473)
CClass_call_result_115346 = invoke(stypy.reporting.localization.Localization(__file__, 473, 5), CClass_115344, *[], **kwargs_115345)

# Assigning a type to the variable 'c_' (line 473)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 0), 'c_', CClass_call_result_115346)
# Declaration of the 'ndenumerate' class

class ndenumerate(object, ):
    str_115347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, (-1)), 'str', '\n    Multidimensional index iterator.\n\n    Return an iterator yielding pairs of array coordinates and values.\n\n    Parameters\n    ----------\n    arr : ndarray\n      Input array.\n\n    See Also\n    --------\n    ndindex, flatiter\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2], [3, 4]])\n    >>> for index, x in np.ndenumerate(a):\n    ...     print(index, x)\n    (0, 0) 1\n    (0, 1) 2\n    (1, 0) 3\n    (1, 1) 4\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 502, 4, False)
        # Assigning a type to the variable 'self' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ndenumerate.__init__', ['arr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['arr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 503):
        
        # Assigning a Attribute to a Attribute (line 503):
        
        # Call to asarray(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'arr' (line 503)
        arr_115349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 28), 'arr', False)
        # Processing the call keyword arguments (line 503)
        kwargs_115350 = {}
        # Getting the type of 'asarray' (line 503)
        asarray_115348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 20), 'asarray', False)
        # Calling asarray(args, kwargs) (line 503)
        asarray_call_result_115351 = invoke(stypy.reporting.localization.Localization(__file__, 503, 20), asarray_115348, *[arr_115349], **kwargs_115350)
        
        # Obtaining the member 'flat' of a type (line 503)
        flat_115352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 20), asarray_call_result_115351, 'flat')
        # Getting the type of 'self' (line 503)
        self_115353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'self')
        # Setting the type of the member 'iter' of a type (line 503)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), self_115353, 'iter', flat_115352)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __next__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__next__'
        module_type_store = module_type_store.open_function_context('__next__', 505, 4, False)
        # Assigning a type to the variable 'self' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ndenumerate.__next__.__dict__.__setitem__('stypy_localization', localization)
        ndenumerate.__next__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ndenumerate.__next__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ndenumerate.__next__.__dict__.__setitem__('stypy_function_name', 'ndenumerate.__next__')
        ndenumerate.__next__.__dict__.__setitem__('stypy_param_names_list', [])
        ndenumerate.__next__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ndenumerate.__next__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ndenumerate.__next__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ndenumerate.__next__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ndenumerate.__next__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ndenumerate.__next__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ndenumerate.__next__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__next__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__next__(...)' code ##################

        str_115354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, (-1)), 'str', '\n        Standard iterator method, returns the index tuple and array value.\n\n        Returns\n        -------\n        coords : tuple of ints\n            The indices of the current iteration.\n        val : scalar\n            The array element of the current iteration.\n\n        ')
        
        # Obtaining an instance of the builtin type 'tuple' (line 517)
        tuple_115355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 517)
        # Adding element type (line 517)
        # Getting the type of 'self' (line 517)
        self_115356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 15), 'self')
        # Obtaining the member 'iter' of a type (line 517)
        iter_115357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 15), self_115356, 'iter')
        # Obtaining the member 'coords' of a type (line 517)
        coords_115358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 15), iter_115357, 'coords')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 15), tuple_115355, coords_115358)
        # Adding element type (line 517)
        
        # Call to next(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 'self' (line 517)
        self_115360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 38), 'self', False)
        # Obtaining the member 'iter' of a type (line 517)
        iter_115361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 38), self_115360, 'iter')
        # Processing the call keyword arguments (line 517)
        kwargs_115362 = {}
        # Getting the type of 'next' (line 517)
        next_115359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 33), 'next', False)
        # Calling next(args, kwargs) (line 517)
        next_call_result_115363 = invoke(stypy.reporting.localization.Localization(__file__, 517, 33), next_115359, *[iter_115361], **kwargs_115362)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 15), tuple_115355, next_call_result_115363)
        
        # Assigning a type to the variable 'stypy_return_type' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'stypy_return_type', tuple_115355)
        
        # ################# End of '__next__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__next__' in the type store
        # Getting the type of 'stypy_return_type' (line 505)
        stypy_return_type_115364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115364)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__next__'
        return stypy_return_type_115364


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 519, 4, False)
        # Assigning a type to the variable 'self' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ndenumerate.__iter__.__dict__.__setitem__('stypy_localization', localization)
        ndenumerate.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ndenumerate.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ndenumerate.__iter__.__dict__.__setitem__('stypy_function_name', 'ndenumerate.__iter__')
        ndenumerate.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        ndenumerate.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ndenumerate.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ndenumerate.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ndenumerate.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ndenumerate.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ndenumerate.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ndenumerate.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        # Getting the type of 'self' (line 520)
        self_115365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'stypy_return_type', self_115365)
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 519)
        stypy_return_type_115366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115366)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_115366

    
    # Assigning a Name to a Name (line 522):

# Assigning a type to the variable 'ndenumerate' (line 475)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 0), 'ndenumerate', ndenumerate)

# Assigning a Name to a Name (line 522):
# Getting the type of 'ndenumerate'
ndenumerate_115367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ndenumerate')
# Obtaining the member '__next__' of a type
next___115368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ndenumerate_115367, '__next__')
# Getting the type of 'ndenumerate'
ndenumerate_115369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ndenumerate')
# Setting the type of the member 'next' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ndenumerate_115369, 'next', next___115368)
# Declaration of the 'ndindex' class

class ndindex(object, ):
    str_115370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, (-1)), 'str', '\n    An N-dimensional iterator object to index arrays.\n\n    Given the shape of an array, an `ndindex` instance iterates over\n    the N-dimensional index of the array. At each iteration a tuple\n    of indices is returned, the last dimension is iterated over first.\n\n    Parameters\n    ----------\n    `*args` : ints\n      The size of each dimension of the array.\n\n    See Also\n    --------\n    ndenumerate, flatiter\n\n    Examples\n    --------\n    >>> for index in np.ndindex(3, 2, 1):\n    ...     print(index)\n    (0, 0, 0)\n    (0, 1, 0)\n    (1, 0, 0)\n    (1, 1, 0)\n    (2, 0, 0)\n    (2, 1, 0)\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 555, 4, False)
        # Assigning a type to the variable 'self' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ndindex.__init__', [], 'shape', None, defaults, varargs, kwargs)

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

        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 556)
        # Processing the call arguments (line 556)
        # Getting the type of 'shape' (line 556)
        shape_115372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 15), 'shape', False)
        # Processing the call keyword arguments (line 556)
        kwargs_115373 = {}
        # Getting the type of 'len' (line 556)
        len_115371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 11), 'len', False)
        # Calling len(args, kwargs) (line 556)
        len_call_result_115374 = invoke(stypy.reporting.localization.Localization(__file__, 556, 11), len_115371, *[shape_115372], **kwargs_115373)
        
        int_115375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 25), 'int')
        # Applying the binary operator '==' (line 556)
        result_eq_115376 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 11), '==', len_call_result_115374, int_115375)
        
        
        # Call to isinstance(...): (line 556)
        # Processing the call arguments (line 556)
        
        # Obtaining the type of the subscript
        int_115378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 48), 'int')
        # Getting the type of 'shape' (line 556)
        shape_115379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 42), 'shape', False)
        # Obtaining the member '__getitem__' of a type (line 556)
        getitem___115380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 42), shape_115379, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 556)
        subscript_call_result_115381 = invoke(stypy.reporting.localization.Localization(__file__, 556, 42), getitem___115380, int_115378)
        
        # Getting the type of 'tuple' (line 556)
        tuple_115382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 52), 'tuple', False)
        # Processing the call keyword arguments (line 556)
        kwargs_115383 = {}
        # Getting the type of 'isinstance' (line 556)
        isinstance_115377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 31), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 556)
        isinstance_call_result_115384 = invoke(stypy.reporting.localization.Localization(__file__, 556, 31), isinstance_115377, *[subscript_call_result_115381, tuple_115382], **kwargs_115383)
        
        # Applying the binary operator 'and' (line 556)
        result_and_keyword_115385 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 11), 'and', result_eq_115376, isinstance_call_result_115384)
        
        # Testing the type of an if condition (line 556)
        if_condition_115386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 556, 8), result_and_keyword_115385)
        # Assigning a type to the variable 'if_condition_115386' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'if_condition_115386', if_condition_115386)
        # SSA begins for if statement (line 556)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 557):
        
        # Assigning a Subscript to a Name (line 557):
        
        # Obtaining the type of the subscript
        int_115387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 26), 'int')
        # Getting the type of 'shape' (line 557)
        shape_115388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 20), 'shape')
        # Obtaining the member '__getitem__' of a type (line 557)
        getitem___115389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 20), shape_115388, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 557)
        subscript_call_result_115390 = invoke(stypy.reporting.localization.Localization(__file__, 557, 20), getitem___115389, int_115387)
        
        # Assigning a type to the variable 'shape' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'shape', subscript_call_result_115390)
        # SSA join for if statement (line 556)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 558):
        
        # Assigning a Call to a Name (line 558):
        
        # Call to as_strided(...): (line 558)
        # Processing the call arguments (line 558)
        
        # Call to zeros(...): (line 558)
        # Processing the call arguments (line 558)
        int_115394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 33), 'int')
        # Processing the call keyword arguments (line 558)
        kwargs_115395 = {}
        # Getting the type of '_nx' (line 558)
        _nx_115392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 23), '_nx', False)
        # Obtaining the member 'zeros' of a type (line 558)
        zeros_115393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 23), _nx_115392, 'zeros')
        # Calling zeros(args, kwargs) (line 558)
        zeros_call_result_115396 = invoke(stypy.reporting.localization.Localization(__file__, 558, 23), zeros_115393, *[int_115394], **kwargs_115395)
        
        # Processing the call keyword arguments (line 558)
        # Getting the type of 'shape' (line 558)
        shape_115397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 43), 'shape', False)
        keyword_115398 = shape_115397
        
        # Call to zeros_like(...): (line 559)
        # Processing the call arguments (line 559)
        # Getting the type of 'shape' (line 559)
        shape_115401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 46), 'shape', False)
        # Processing the call keyword arguments (line 559)
        kwargs_115402 = {}
        # Getting the type of '_nx' (line 559)
        _nx_115399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 31), '_nx', False)
        # Obtaining the member 'zeros_like' of a type (line 559)
        zeros_like_115400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 31), _nx_115399, 'zeros_like')
        # Calling zeros_like(args, kwargs) (line 559)
        zeros_like_call_result_115403 = invoke(stypy.reporting.localization.Localization(__file__, 559, 31), zeros_like_115400, *[shape_115401], **kwargs_115402)
        
        keyword_115404 = zeros_like_call_result_115403
        kwargs_115405 = {'strides': keyword_115404, 'shape': keyword_115398}
        # Getting the type of 'as_strided' (line 558)
        as_strided_115391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'as_strided', False)
        # Calling as_strided(args, kwargs) (line 558)
        as_strided_call_result_115406 = invoke(stypy.reporting.localization.Localization(__file__, 558, 12), as_strided_115391, *[zeros_call_result_115396], **kwargs_115405)
        
        # Assigning a type to the variable 'x' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'x', as_strided_call_result_115406)
        
        # Assigning a Call to a Attribute (line 560):
        
        # Assigning a Call to a Attribute (line 560):
        
        # Call to nditer(...): (line 560)
        # Processing the call arguments (line 560)
        # Getting the type of 'x' (line 560)
        x_115409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 30), 'x', False)
        # Processing the call keyword arguments (line 560)
        
        # Obtaining an instance of the builtin type 'list' (line 560)
        list_115410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 560)
        # Adding element type (line 560)
        str_115411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 40), 'str', 'multi_index')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 39), list_115410, str_115411)
        # Adding element type (line 560)
        str_115412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 55), 'str', 'zerosize_ok')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 39), list_115410, str_115412)
        
        keyword_115413 = list_115410
        str_115414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 36), 'str', 'C')
        keyword_115415 = str_115414
        kwargs_115416 = {'flags': keyword_115413, 'order': keyword_115415}
        # Getting the type of '_nx' (line 560)
        _nx_115407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 19), '_nx', False)
        # Obtaining the member 'nditer' of a type (line 560)
        nditer_115408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 19), _nx_115407, 'nditer')
        # Calling nditer(args, kwargs) (line 560)
        nditer_call_result_115417 = invoke(stypy.reporting.localization.Localization(__file__, 560, 19), nditer_115408, *[x_115409], **kwargs_115416)
        
        # Getting the type of 'self' (line 560)
        self_115418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'self')
        # Setting the type of the member '_it' of a type (line 560)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 8), self_115418, '_it', nditer_call_result_115417)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 563, 4, False)
        # Assigning a type to the variable 'self' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ndindex.__iter__.__dict__.__setitem__('stypy_localization', localization)
        ndindex.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ndindex.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ndindex.__iter__.__dict__.__setitem__('stypy_function_name', 'ndindex.__iter__')
        ndindex.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        ndindex.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ndindex.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ndindex.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ndindex.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ndindex.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ndindex.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ndindex.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        # Getting the type of 'self' (line 564)
        self_115419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'stypy_return_type', self_115419)
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 563)
        stypy_return_type_115420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115420)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_115420


    @norecursion
    def ndincr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'ndincr'
        module_type_store = module_type_store.open_function_context('ndincr', 566, 4, False)
        # Assigning a type to the variable 'self' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ndindex.ndincr.__dict__.__setitem__('stypy_localization', localization)
        ndindex.ndincr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ndindex.ndincr.__dict__.__setitem__('stypy_type_store', module_type_store)
        ndindex.ndincr.__dict__.__setitem__('stypy_function_name', 'ndindex.ndincr')
        ndindex.ndincr.__dict__.__setitem__('stypy_param_names_list', [])
        ndindex.ndincr.__dict__.__setitem__('stypy_varargs_param_name', None)
        ndindex.ndincr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ndindex.ndincr.__dict__.__setitem__('stypy_call_defaults', defaults)
        ndindex.ndincr.__dict__.__setitem__('stypy_call_varargs', varargs)
        ndindex.ndincr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ndindex.ndincr.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ndindex.ndincr', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ndincr', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ndincr(...)' code ##################

        str_115421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, (-1)), 'str', '\n        Increment the multi-dimensional index by one.\n\n        This method is for backward compatibility only: do not use.\n        ')
        
        # Call to next(...): (line 572)
        # Processing the call arguments (line 572)
        # Getting the type of 'self' (line 572)
        self_115423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 13), 'self', False)
        # Processing the call keyword arguments (line 572)
        kwargs_115424 = {}
        # Getting the type of 'next' (line 572)
        next_115422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'next', False)
        # Calling next(args, kwargs) (line 572)
        next_call_result_115425 = invoke(stypy.reporting.localization.Localization(__file__, 572, 8), next_115422, *[self_115423], **kwargs_115424)
        
        
        # ################# End of 'ndincr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ndincr' in the type store
        # Getting the type of 'stypy_return_type' (line 566)
        stypy_return_type_115426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115426)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ndincr'
        return stypy_return_type_115426


    @norecursion
    def __next__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__next__'
        module_type_store = module_type_store.open_function_context('__next__', 574, 4, False)
        # Assigning a type to the variable 'self' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ndindex.__next__.__dict__.__setitem__('stypy_localization', localization)
        ndindex.__next__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ndindex.__next__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ndindex.__next__.__dict__.__setitem__('stypy_function_name', 'ndindex.__next__')
        ndindex.__next__.__dict__.__setitem__('stypy_param_names_list', [])
        ndindex.__next__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ndindex.__next__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ndindex.__next__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ndindex.__next__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ndindex.__next__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ndindex.__next__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ndindex.__next__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__next__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__next__(...)' code ##################

        str_115427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, (-1)), 'str', '\n        Standard iterator method, updates the index and returns the index\n        tuple.\n\n        Returns\n        -------\n        val : tuple of ints\n            Returns a tuple containing the indices of the current\n            iteration.\n\n        ')
        
        # Call to next(...): (line 586)
        # Processing the call arguments (line 586)
        # Getting the type of 'self' (line 586)
        self_115429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 13), 'self', False)
        # Obtaining the member '_it' of a type (line 586)
        _it_115430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 13), self_115429, '_it')
        # Processing the call keyword arguments (line 586)
        kwargs_115431 = {}
        # Getting the type of 'next' (line 586)
        next_115428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'next', False)
        # Calling next(args, kwargs) (line 586)
        next_call_result_115432 = invoke(stypy.reporting.localization.Localization(__file__, 586, 8), next_115428, *[_it_115430], **kwargs_115431)
        
        # Getting the type of 'self' (line 587)
        self_115433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 15), 'self')
        # Obtaining the member '_it' of a type (line 587)
        _it_115434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 15), self_115433, '_it')
        # Obtaining the member 'multi_index' of a type (line 587)
        multi_index_115435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 15), _it_115434, 'multi_index')
        # Assigning a type to the variable 'stypy_return_type' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'stypy_return_type', multi_index_115435)
        
        # ################# End of '__next__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__next__' in the type store
        # Getting the type of 'stypy_return_type' (line 574)
        stypy_return_type_115436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115436)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__next__'
        return stypy_return_type_115436

    
    # Assigning a Name to a Name (line 589):

# Assigning a type to the variable 'ndindex' (line 525)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 0), 'ndindex', ndindex)

# Assigning a Name to a Name (line 589):
# Getting the type of 'ndindex'
ndindex_115437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ndindex')
# Obtaining the member '__next__' of a type
next___115438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ndindex_115437, '__next__')
# Getting the type of 'ndindex'
ndindex_115439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ndindex')
# Setting the type of the member 'next' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ndindex_115439, 'next', next___115438)
# Declaration of the 'IndexExpression' class

class IndexExpression(object, ):
    str_115440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, (-1)), 'str', "\n    A nicer way to build up index tuples for arrays.\n\n    .. note::\n       Use one of the two predefined instances `index_exp` or `s_`\n       rather than directly using `IndexExpression`.\n\n    For any index combination, including slicing and axis insertion,\n    ``a[indices]`` is the same as ``a[np.index_exp[indices]]`` for any\n    array `a`. However, ``np.index_exp[indices]`` can be used anywhere\n    in Python code and returns a tuple of slice objects that can be\n    used in the construction of complex index expressions.\n\n    Parameters\n    ----------\n    maketuple : bool\n        If True, always returns a tuple.\n\n    See Also\n    --------\n    index_exp : Predefined instance that always returns a tuple:\n       `index_exp = IndexExpression(maketuple=True)`.\n    s_ : Predefined instance without tuple conversion:\n       `s_ = IndexExpression(maketuple=False)`.\n\n    Notes\n    -----\n    You can do all this with `slice()` plus a few special objects,\n    but there's a lot to remember and this version is simpler because\n    it uses the standard array indexing syntax.\n\n    Examples\n    --------\n    >>> np.s_[2::2]\n    slice(2, None, 2)\n    >>> np.index_exp[2::2]\n    (slice(2, None, 2),)\n\n    >>> np.array([0, 1, 2, 3, 4])[np.s_[2::2]]\n    array([2, 4])\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 647, 4, False)
        # Assigning a type to the variable 'self' (line 648)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IndexExpression.__init__', ['maketuple'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['maketuple'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 648):
        
        # Assigning a Name to a Attribute (line 648):
        # Getting the type of 'maketuple' (line 648)
        maketuple_115441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 25), 'maketuple')
        # Getting the type of 'self' (line 648)
        self_115442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 8), 'self')
        # Setting the type of the member 'maketuple' of a type (line 648)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 8), self_115442, 'maketuple', maketuple_115441)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 650, 4, False)
        # Assigning a type to the variable 'self' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IndexExpression.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        IndexExpression.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IndexExpression.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        IndexExpression.__getitem__.__dict__.__setitem__('stypy_function_name', 'IndexExpression.__getitem__')
        IndexExpression.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['item'])
        IndexExpression.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        IndexExpression.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IndexExpression.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        IndexExpression.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        IndexExpression.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IndexExpression.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IndexExpression.__getitem__', ['item'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['item'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 651)
        self_115443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 11), 'self')
        # Obtaining the member 'maketuple' of a type (line 651)
        maketuple_115444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 11), self_115443, 'maketuple')
        
        
        # Call to isinstance(...): (line 651)
        # Processing the call arguments (line 651)
        # Getting the type of 'item' (line 651)
        item_115446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 45), 'item', False)
        # Getting the type of 'tuple' (line 651)
        tuple_115447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 51), 'tuple', False)
        # Processing the call keyword arguments (line 651)
        kwargs_115448 = {}
        # Getting the type of 'isinstance' (line 651)
        isinstance_115445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 34), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 651)
        isinstance_call_result_115449 = invoke(stypy.reporting.localization.Localization(__file__, 651, 34), isinstance_115445, *[item_115446, tuple_115447], **kwargs_115448)
        
        # Applying the 'not' unary operator (line 651)
        result_not__115450 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 30), 'not', isinstance_call_result_115449)
        
        # Applying the binary operator 'and' (line 651)
        result_and_keyword_115451 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 11), 'and', maketuple_115444, result_not__115450)
        
        # Testing the type of an if condition (line 651)
        if_condition_115452 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 651, 8), result_and_keyword_115451)
        # Assigning a type to the variable 'if_condition_115452' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 8), 'if_condition_115452', if_condition_115452)
        # SSA begins for if statement (line 651)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 652)
        tuple_115453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 652)
        # Adding element type (line 652)
        # Getting the type of 'item' (line 652)
        item_115454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 20), 'item')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 652, 20), tuple_115453, item_115454)
        
        # Assigning a type to the variable 'stypy_return_type' (line 652)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 12), 'stypy_return_type', tuple_115453)
        # SSA branch for the else part of an if statement (line 651)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'item' (line 654)
        item_115455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 19), 'item')
        # Assigning a type to the variable 'stypy_return_type' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'stypy_return_type', item_115455)
        # SSA join for if statement (line 651)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 650)
        stypy_return_type_115456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115456)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_115456


# Assigning a type to the variable 'IndexExpression' (line 603)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 0), 'IndexExpression', IndexExpression)

# Assigning a Call to a Name (line 656):

# Assigning a Call to a Name (line 656):

# Call to IndexExpression(...): (line 656)
# Processing the call keyword arguments (line 656)
# Getting the type of 'True' (line 656)
True_115458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 38), 'True', False)
keyword_115459 = True_115458
kwargs_115460 = {'maketuple': keyword_115459}
# Getting the type of 'IndexExpression' (line 656)
IndexExpression_115457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 12), 'IndexExpression', False)
# Calling IndexExpression(args, kwargs) (line 656)
IndexExpression_call_result_115461 = invoke(stypy.reporting.localization.Localization(__file__, 656, 12), IndexExpression_115457, *[], **kwargs_115460)

# Assigning a type to the variable 'index_exp' (line 656)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 0), 'index_exp', IndexExpression_call_result_115461)

# Assigning a Call to a Name (line 657):

# Assigning a Call to a Name (line 657):

# Call to IndexExpression(...): (line 657)
# Processing the call keyword arguments (line 657)
# Getting the type of 'False' (line 657)
False_115463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 31), 'False', False)
keyword_115464 = False_115463
kwargs_115465 = {'maketuple': keyword_115464}
# Getting the type of 'IndexExpression' (line 657)
IndexExpression_115462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 5), 'IndexExpression', False)
# Calling IndexExpression(args, kwargs) (line 657)
IndexExpression_call_result_115466 = invoke(stypy.reporting.localization.Localization(__file__, 657, 5), IndexExpression_115462, *[], **kwargs_115465)

# Assigning a type to the variable 's_' (line 657)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 0), 's_', IndexExpression_call_result_115466)

@norecursion
def fill_diagonal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 665)
    False_115467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 31), 'False')
    defaults = [False_115467]
    # Create a new context for function 'fill_diagonal'
    module_type_store = module_type_store.open_function_context('fill_diagonal', 665, 0, False)
    
    # Passed parameters checking function
    fill_diagonal.stypy_localization = localization
    fill_diagonal.stypy_type_of_self = None
    fill_diagonal.stypy_type_store = module_type_store
    fill_diagonal.stypy_function_name = 'fill_diagonal'
    fill_diagonal.stypy_param_names_list = ['a', 'val', 'wrap']
    fill_diagonal.stypy_varargs_param_name = None
    fill_diagonal.stypy_kwargs_param_name = None
    fill_diagonal.stypy_call_defaults = defaults
    fill_diagonal.stypy_call_varargs = varargs
    fill_diagonal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fill_diagonal', ['a', 'val', 'wrap'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fill_diagonal', localization, ['a', 'val', 'wrap'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fill_diagonal(...)' code ##################

    str_115468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, (-1)), 'str', 'Fill the main diagonal of the given array of any dimensionality.\n\n    For an array `a` with ``a.ndim > 2``, the diagonal is the list of\n    locations with indices ``a[i, i, ..., i]`` all identical. This function\n    modifies the input array in-place, it does not return a value.\n\n    Parameters\n    ----------\n    a : array, at least 2-D.\n      Array whose diagonal is to be filled, it gets modified in-place.\n\n    val : scalar\n      Value to be written on the diagonal, its type must be compatible with\n      that of the array a.\n\n    wrap : bool\n      For tall matrices in NumPy version up to 1.6.2, the\n      diagonal "wrapped" after N columns. You can have this behavior\n      with this option. This affects only tall matrices.\n\n    See also\n    --------\n    diag_indices, diag_indices_from\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    This functionality can be obtained via `diag_indices`, but internally\n    this version uses a much faster implementation that never constructs the\n    indices and uses simple slicing.\n\n    Examples\n    --------\n    >>> a = np.zeros((3, 3), int)\n    >>> np.fill_diagonal(a, 5)\n    >>> a\n    array([[5, 0, 0],\n           [0, 5, 0],\n           [0, 0, 5]])\n\n    The same function can operate on a 4-D array:\n\n    >>> a = np.zeros((3, 3, 3, 3), int)\n    >>> np.fill_diagonal(a, 4)\n\n    We only show a few blocks for clarity:\n\n    >>> a[0, 0]\n    array([[4, 0, 0],\n           [0, 0, 0],\n           [0, 0, 0]])\n    >>> a[1, 1]\n    array([[0, 0, 0],\n           [0, 4, 0],\n           [0, 0, 0]])\n    >>> a[2, 2]\n    array([[0, 0, 0],\n           [0, 0, 0],\n           [0, 0, 4]])\n\n    The wrap option affects only tall matrices:\n\n    >>> # tall matrices no wrap\n    >>> a = np.zeros((5, 3),int)\n    >>> fill_diagonal(a, 4)\n    >>> a\n    array([[4, 0, 0],\n           [0, 4, 0],\n           [0, 0, 4],\n           [0, 0, 0],\n           [0, 0, 0]])\n\n    >>> # tall matrices wrap\n    >>> a = np.zeros((5, 3),int)\n    >>> fill_diagonal(a, 4, wrap=True)\n    >>> a\n    array([[4, 0, 0],\n           [0, 4, 0],\n           [0, 0, 4],\n           [0, 0, 0],\n           [4, 0, 0]])\n\n    >>> # wide matrices\n    >>> a = np.zeros((3, 5),int)\n    >>> fill_diagonal(a, 4, wrap=True)\n    >>> a\n    array([[4, 0, 0, 0, 0],\n           [0, 4, 0, 0, 0],\n           [0, 0, 4, 0, 0]])\n\n    ')
    
    
    # Getting the type of 'a' (line 758)
    a_115469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 7), 'a')
    # Obtaining the member 'ndim' of a type (line 758)
    ndim_115470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 7), a_115469, 'ndim')
    int_115471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 16), 'int')
    # Applying the binary operator '<' (line 758)
    result_lt_115472 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 7), '<', ndim_115470, int_115471)
    
    # Testing the type of an if condition (line 758)
    if_condition_115473 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 758, 4), result_lt_115472)
    # Assigning a type to the variable 'if_condition_115473' (line 758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 4), 'if_condition_115473', if_condition_115473)
    # SSA begins for if statement (line 758)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 759)
    # Processing the call arguments (line 759)
    str_115475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 25), 'str', 'array must be at least 2-d')
    # Processing the call keyword arguments (line 759)
    kwargs_115476 = {}
    # Getting the type of 'ValueError' (line 759)
    ValueError_115474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 759)
    ValueError_call_result_115477 = invoke(stypy.reporting.localization.Localization(__file__, 759, 14), ValueError_115474, *[str_115475], **kwargs_115476)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 759, 8), ValueError_call_result_115477, 'raise parameter', BaseException)
    # SSA join for if statement (line 758)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 760):
    
    # Assigning a Name to a Name (line 760):
    # Getting the type of 'None' (line 760)
    None_115478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 10), 'None')
    # Assigning a type to the variable 'end' (line 760)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 4), 'end', None_115478)
    
    
    # Getting the type of 'a' (line 761)
    a_115479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 7), 'a')
    # Obtaining the member 'ndim' of a type (line 761)
    ndim_115480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 7), a_115479, 'ndim')
    int_115481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 17), 'int')
    # Applying the binary operator '==' (line 761)
    result_eq_115482 = python_operator(stypy.reporting.localization.Localization(__file__, 761, 7), '==', ndim_115480, int_115481)
    
    # Testing the type of an if condition (line 761)
    if_condition_115483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 761, 4), result_eq_115482)
    # Assigning a type to the variable 'if_condition_115483' (line 761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 4), 'if_condition_115483', if_condition_115483)
    # SSA begins for if statement (line 761)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 764):
    
    # Assigning a BinOp to a Name (line 764):
    
    # Obtaining the type of the subscript
    int_115484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 23), 'int')
    # Getting the type of 'a' (line 764)
    a_115485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 15), 'a')
    # Obtaining the member 'shape' of a type (line 764)
    shape_115486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 15), a_115485, 'shape')
    # Obtaining the member '__getitem__' of a type (line 764)
    getitem___115487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 15), shape_115486, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 764)
    subscript_call_result_115488 = invoke(stypy.reporting.localization.Localization(__file__, 764, 15), getitem___115487, int_115484)
    
    int_115489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 28), 'int')
    # Applying the binary operator '+' (line 764)
    result_add_115490 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 15), '+', subscript_call_result_115488, int_115489)
    
    # Assigning a type to the variable 'step' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 8), 'step', result_add_115490)
    
    
    # Getting the type of 'wrap' (line 766)
    wrap_115491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 15), 'wrap')
    # Applying the 'not' unary operator (line 766)
    result_not__115492 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 11), 'not', wrap_115491)
    
    # Testing the type of an if condition (line 766)
    if_condition_115493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 766, 8), result_not__115492)
    # Assigning a type to the variable 'if_condition_115493' (line 766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'if_condition_115493', if_condition_115493)
    # SSA begins for if statement (line 766)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 767):
    
    # Assigning a BinOp to a Name (line 767):
    
    # Obtaining the type of the subscript
    int_115494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 26), 'int')
    # Getting the type of 'a' (line 767)
    a_115495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 18), 'a')
    # Obtaining the member 'shape' of a type (line 767)
    shape_115496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 18), a_115495, 'shape')
    # Obtaining the member '__getitem__' of a type (line 767)
    getitem___115497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 18), shape_115496, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 767)
    subscript_call_result_115498 = invoke(stypy.reporting.localization.Localization(__file__, 767, 18), getitem___115497, int_115494)
    
    
    # Obtaining the type of the subscript
    int_115499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 39), 'int')
    # Getting the type of 'a' (line 767)
    a_115500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 31), 'a')
    # Obtaining the member 'shape' of a type (line 767)
    shape_115501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 31), a_115500, 'shape')
    # Obtaining the member '__getitem__' of a type (line 767)
    getitem___115502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 31), shape_115501, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 767)
    subscript_call_result_115503 = invoke(stypy.reporting.localization.Localization(__file__, 767, 31), getitem___115502, int_115499)
    
    # Applying the binary operator '*' (line 767)
    result_mul_115504 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 18), '*', subscript_call_result_115498, subscript_call_result_115503)
    
    # Assigning a type to the variable 'end' (line 767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'end', result_mul_115504)
    # SSA join for if statement (line 766)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 761)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to alltrue(...): (line 771)
    # Processing the call arguments (line 771)
    
    
    # Call to diff(...): (line 771)
    # Processing the call arguments (line 771)
    # Getting the type of 'a' (line 771)
    a_115507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 28), 'a', False)
    # Obtaining the member 'shape' of a type (line 771)
    shape_115508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 28), a_115507, 'shape')
    # Processing the call keyword arguments (line 771)
    kwargs_115509 = {}
    # Getting the type of 'diff' (line 771)
    diff_115506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 23), 'diff', False)
    # Calling diff(args, kwargs) (line 771)
    diff_call_result_115510 = invoke(stypy.reporting.localization.Localization(__file__, 771, 23), diff_115506, *[shape_115508], **kwargs_115509)
    
    int_115511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 40), 'int')
    # Applying the binary operator '==' (line 771)
    result_eq_115512 = python_operator(stypy.reporting.localization.Localization(__file__, 771, 23), '==', diff_call_result_115510, int_115511)
    
    # Processing the call keyword arguments (line 771)
    kwargs_115513 = {}
    # Getting the type of 'alltrue' (line 771)
    alltrue_115505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 15), 'alltrue', False)
    # Calling alltrue(args, kwargs) (line 771)
    alltrue_call_result_115514 = invoke(stypy.reporting.localization.Localization(__file__, 771, 15), alltrue_115505, *[result_eq_115512], **kwargs_115513)
    
    # Applying the 'not' unary operator (line 771)
    result_not__115515 = python_operator(stypy.reporting.localization.Localization(__file__, 771, 11), 'not', alltrue_call_result_115514)
    
    # Testing the type of an if condition (line 771)
    if_condition_115516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 771, 8), result_not__115515)
    # Assigning a type to the variable 'if_condition_115516' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'if_condition_115516', if_condition_115516)
    # SSA begins for if statement (line 771)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 772)
    # Processing the call arguments (line 772)
    str_115518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 29), 'str', 'All dimensions of input must be of equal length')
    # Processing the call keyword arguments (line 772)
    kwargs_115519 = {}
    # Getting the type of 'ValueError' (line 772)
    ValueError_115517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 772)
    ValueError_call_result_115520 = invoke(stypy.reporting.localization.Localization(__file__, 772, 18), ValueError_115517, *[str_115518], **kwargs_115519)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 772, 12), ValueError_call_result_115520, 'raise parameter', BaseException)
    # SSA join for if statement (line 771)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 773):
    
    # Assigning a BinOp to a Name (line 773):
    int_115521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 15), 'int')
    
    # Call to sum(...): (line 773)
    # Processing the call keyword arguments (line 773)
    kwargs_115532 = {}
    
    # Call to cumprod(...): (line 773)
    # Processing the call arguments (line 773)
    
    # Obtaining the type of the subscript
    int_115523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 37), 'int')
    slice_115524 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 773, 28), None, int_115523, None)
    # Getting the type of 'a' (line 773)
    a_115525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 28), 'a', False)
    # Obtaining the member 'shape' of a type (line 773)
    shape_115526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 28), a_115525, 'shape')
    # Obtaining the member '__getitem__' of a type (line 773)
    getitem___115527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 28), shape_115526, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 773)
    subscript_call_result_115528 = invoke(stypy.reporting.localization.Localization(__file__, 773, 28), getitem___115527, slice_115524)
    
    # Processing the call keyword arguments (line 773)
    kwargs_115529 = {}
    # Getting the type of 'cumprod' (line 773)
    cumprod_115522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 20), 'cumprod', False)
    # Calling cumprod(args, kwargs) (line 773)
    cumprod_call_result_115530 = invoke(stypy.reporting.localization.Localization(__file__, 773, 20), cumprod_115522, *[subscript_call_result_115528], **kwargs_115529)
    
    # Obtaining the member 'sum' of a type (line 773)
    sum_115531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 20), cumprod_call_result_115530, 'sum')
    # Calling sum(args, kwargs) (line 773)
    sum_call_result_115533 = invoke(stypy.reporting.localization.Localization(__file__, 773, 20), sum_115531, *[], **kwargs_115532)
    
    # Applying the binary operator '+' (line 773)
    result_add_115534 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 15), '+', int_115521, sum_call_result_115533)
    
    # Assigning a type to the variable 'step' (line 773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'step', result_add_115534)
    # SSA join for if statement (line 761)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 776):
    
    # Assigning a Name to a Subscript (line 776):
    # Getting the type of 'val' (line 776)
    val_115535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 24), 'val')
    # Getting the type of 'a' (line 776)
    a_115536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'a')
    # Obtaining the member 'flat' of a type (line 776)
    flat_115537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 4), a_115536, 'flat')
    # Getting the type of 'end' (line 776)
    end_115538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 12), 'end')
    # Getting the type of 'step' (line 776)
    step_115539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 16), 'step')
    slice_115540 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 776, 4), None, end_115538, step_115539)
    # Storing an element on a container (line 776)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 776, 4), flat_115537, (slice_115540, val_115535))
    
    # ################# End of 'fill_diagonal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fill_diagonal' in the type store
    # Getting the type of 'stypy_return_type' (line 665)
    stypy_return_type_115541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115541)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fill_diagonal'
    return stypy_return_type_115541

# Assigning a type to the variable 'fill_diagonal' (line 665)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 0), 'fill_diagonal', fill_diagonal)

@norecursion
def diag_indices(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_115542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 25), 'int')
    defaults = [int_115542]
    # Create a new context for function 'diag_indices'
    module_type_store = module_type_store.open_function_context('diag_indices', 779, 0, False)
    
    # Passed parameters checking function
    diag_indices.stypy_localization = localization
    diag_indices.stypy_type_of_self = None
    diag_indices.stypy_type_store = module_type_store
    diag_indices.stypy_function_name = 'diag_indices'
    diag_indices.stypy_param_names_list = ['n', 'ndim']
    diag_indices.stypy_varargs_param_name = None
    diag_indices.stypy_kwargs_param_name = None
    diag_indices.stypy_call_defaults = defaults
    diag_indices.stypy_call_varargs = varargs
    diag_indices.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'diag_indices', ['n', 'ndim'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'diag_indices', localization, ['n', 'ndim'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'diag_indices(...)' code ##################

    str_115543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, (-1)), 'str', '\n    Return the indices to access the main diagonal of an array.\n\n    This returns a tuple of indices that can be used to access the main\n    diagonal of an array `a` with ``a.ndim >= 2`` dimensions and shape\n    (n, n, ..., n). For ``a.ndim = 2`` this is the usual diagonal, for\n    ``a.ndim > 2`` this is the set of indices to access ``a[i, i, ..., i]``\n    for ``i = [0..n-1]``.\n\n    Parameters\n    ----------\n    n : int\n      The size, along each dimension, of the arrays for which the returned\n      indices can be used.\n\n    ndim : int, optional\n      The number of dimensions.\n\n    See also\n    --------\n    diag_indices_from\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    Examples\n    --------\n    Create a set of indices to access the diagonal of a (4, 4) array:\n\n    >>> di = np.diag_indices(4)\n    >>> di\n    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))\n    >>> a = np.arange(16).reshape(4, 4)\n    >>> a\n    array([[ 0,  1,  2,  3],\n           [ 4,  5,  6,  7],\n           [ 8,  9, 10, 11],\n           [12, 13, 14, 15]])\n    >>> a[di] = 100\n    >>> a\n    array([[100,   1,   2,   3],\n           [  4, 100,   6,   7],\n           [  8,   9, 100,  11],\n           [ 12,  13,  14, 100]])\n\n    Now, we create indices to manipulate a 3-D array:\n\n    >>> d3 = np.diag_indices(2, 3)\n    >>> d3\n    (array([0, 1]), array([0, 1]), array([0, 1]))\n\n    And use it to set the diagonal of an array of zeros to 1:\n\n    >>> a = np.zeros((2, 2, 2), dtype=np.int)\n    >>> a[d3] = 1\n    >>> a\n    array([[[1, 0],\n            [0, 0]],\n           [[0, 0],\n            [0, 1]]])\n\n    ')
    
    # Assigning a Call to a Name (line 843):
    
    # Assigning a Call to a Name (line 843):
    
    # Call to arange(...): (line 843)
    # Processing the call arguments (line 843)
    # Getting the type of 'n' (line 843)
    n_115545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 17), 'n', False)
    # Processing the call keyword arguments (line 843)
    kwargs_115546 = {}
    # Getting the type of 'arange' (line 843)
    arange_115544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 10), 'arange', False)
    # Calling arange(args, kwargs) (line 843)
    arange_call_result_115547 = invoke(stypy.reporting.localization.Localization(__file__, 843, 10), arange_115544, *[n_115545], **kwargs_115546)
    
    # Assigning a type to the variable 'idx' (line 843)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 4), 'idx', arange_call_result_115547)
    
    # Obtaining an instance of the builtin type 'tuple' (line 844)
    tuple_115548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 844)
    # Adding element type (line 844)
    # Getting the type of 'idx' (line 844)
    idx_115549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 12), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 844, 12), tuple_115548, idx_115549)
    
    # Getting the type of 'ndim' (line 844)
    ndim_115550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 20), 'ndim')
    # Applying the binary operator '*' (line 844)
    result_mul_115551 = python_operator(stypy.reporting.localization.Localization(__file__, 844, 11), '*', tuple_115548, ndim_115550)
    
    # Assigning a type to the variable 'stypy_return_type' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 4), 'stypy_return_type', result_mul_115551)
    
    # ################# End of 'diag_indices(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'diag_indices' in the type store
    # Getting the type of 'stypy_return_type' (line 779)
    stypy_return_type_115552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115552)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'diag_indices'
    return stypy_return_type_115552

# Assigning a type to the variable 'diag_indices' (line 779)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 0), 'diag_indices', diag_indices)

@norecursion
def diag_indices_from(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'diag_indices_from'
    module_type_store = module_type_store.open_function_context('diag_indices_from', 847, 0, False)
    
    # Passed parameters checking function
    diag_indices_from.stypy_localization = localization
    diag_indices_from.stypy_type_of_self = None
    diag_indices_from.stypy_type_store = module_type_store
    diag_indices_from.stypy_function_name = 'diag_indices_from'
    diag_indices_from.stypy_param_names_list = ['arr']
    diag_indices_from.stypy_varargs_param_name = None
    diag_indices_from.stypy_kwargs_param_name = None
    diag_indices_from.stypy_call_defaults = defaults
    diag_indices_from.stypy_call_varargs = varargs
    diag_indices_from.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'diag_indices_from', ['arr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'diag_indices_from', localization, ['arr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'diag_indices_from(...)' code ##################

    str_115553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 865, (-1)), 'str', '\n    Return the indices to access the main diagonal of an n-dimensional array.\n\n    See `diag_indices` for full details.\n\n    Parameters\n    ----------\n    arr : array, at least 2-D\n\n    See Also\n    --------\n    diag_indices\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    ')
    
    
    
    # Getting the type of 'arr' (line 867)
    arr_115554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 11), 'arr')
    # Obtaining the member 'ndim' of a type (line 867)
    ndim_115555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 11), arr_115554, 'ndim')
    int_115556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, 23), 'int')
    # Applying the binary operator '>=' (line 867)
    result_ge_115557 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 11), '>=', ndim_115555, int_115556)
    
    # Applying the 'not' unary operator (line 867)
    result_not__115558 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 7), 'not', result_ge_115557)
    
    # Testing the type of an if condition (line 867)
    if_condition_115559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 867, 4), result_not__115558)
    # Assigning a type to the variable 'if_condition_115559' (line 867)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 4), 'if_condition_115559', if_condition_115559)
    # SSA begins for if statement (line 867)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 868)
    # Processing the call arguments (line 868)
    str_115561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 25), 'str', 'input array must be at least 2-d')
    # Processing the call keyword arguments (line 868)
    kwargs_115562 = {}
    # Getting the type of 'ValueError' (line 868)
    ValueError_115560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 868)
    ValueError_call_result_115563 = invoke(stypy.reporting.localization.Localization(__file__, 868, 14), ValueError_115560, *[str_115561], **kwargs_115562)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 868, 8), ValueError_call_result_115563, 'raise parameter', BaseException)
    # SSA join for if statement (line 867)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to alltrue(...): (line 871)
    # Processing the call arguments (line 871)
    
    
    # Call to diff(...): (line 871)
    # Processing the call arguments (line 871)
    # Getting the type of 'arr' (line 871)
    arr_115566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 24), 'arr', False)
    # Obtaining the member 'shape' of a type (line 871)
    shape_115567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 24), arr_115566, 'shape')
    # Processing the call keyword arguments (line 871)
    kwargs_115568 = {}
    # Getting the type of 'diff' (line 871)
    diff_115565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 19), 'diff', False)
    # Calling diff(args, kwargs) (line 871)
    diff_call_result_115569 = invoke(stypy.reporting.localization.Localization(__file__, 871, 19), diff_115565, *[shape_115567], **kwargs_115568)
    
    int_115570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 38), 'int')
    # Applying the binary operator '==' (line 871)
    result_eq_115571 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 19), '==', diff_call_result_115569, int_115570)
    
    # Processing the call keyword arguments (line 871)
    kwargs_115572 = {}
    # Getting the type of 'alltrue' (line 871)
    alltrue_115564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 11), 'alltrue', False)
    # Calling alltrue(args, kwargs) (line 871)
    alltrue_call_result_115573 = invoke(stypy.reporting.localization.Localization(__file__, 871, 11), alltrue_115564, *[result_eq_115571], **kwargs_115572)
    
    # Applying the 'not' unary operator (line 871)
    result_not__115574 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 7), 'not', alltrue_call_result_115573)
    
    # Testing the type of an if condition (line 871)
    if_condition_115575 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 871, 4), result_not__115574)
    # Assigning a type to the variable 'if_condition_115575' (line 871)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 4), 'if_condition_115575', if_condition_115575)
    # SSA begins for if statement (line 871)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 872)
    # Processing the call arguments (line 872)
    str_115577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 25), 'str', 'All dimensions of input must be of equal length')
    # Processing the call keyword arguments (line 872)
    kwargs_115578 = {}
    # Getting the type of 'ValueError' (line 872)
    ValueError_115576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 872)
    ValueError_call_result_115579 = invoke(stypy.reporting.localization.Localization(__file__, 872, 14), ValueError_115576, *[str_115577], **kwargs_115578)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 872, 8), ValueError_call_result_115579, 'raise parameter', BaseException)
    # SSA join for if statement (line 871)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to diag_indices(...): (line 874)
    # Processing the call arguments (line 874)
    
    # Obtaining the type of the subscript
    int_115581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 34), 'int')
    # Getting the type of 'arr' (line 874)
    arr_115582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 24), 'arr', False)
    # Obtaining the member 'shape' of a type (line 874)
    shape_115583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 24), arr_115582, 'shape')
    # Obtaining the member '__getitem__' of a type (line 874)
    getitem___115584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 24), shape_115583, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 874)
    subscript_call_result_115585 = invoke(stypy.reporting.localization.Localization(__file__, 874, 24), getitem___115584, int_115581)
    
    # Getting the type of 'arr' (line 874)
    arr_115586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 38), 'arr', False)
    # Obtaining the member 'ndim' of a type (line 874)
    ndim_115587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 38), arr_115586, 'ndim')
    # Processing the call keyword arguments (line 874)
    kwargs_115588 = {}
    # Getting the type of 'diag_indices' (line 874)
    diag_indices_115580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 11), 'diag_indices', False)
    # Calling diag_indices(args, kwargs) (line 874)
    diag_indices_call_result_115589 = invoke(stypy.reporting.localization.Localization(__file__, 874, 11), diag_indices_115580, *[subscript_call_result_115585, ndim_115587], **kwargs_115588)
    
    # Assigning a type to the variable 'stypy_return_type' (line 874)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 874, 4), 'stypy_return_type', diag_indices_call_result_115589)
    
    # ################# End of 'diag_indices_from(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'diag_indices_from' in the type store
    # Getting the type of 'stypy_return_type' (line 847)
    stypy_return_type_115590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115590)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'diag_indices_from'
    return stypy_return_type_115590

# Assigning a type to the variable 'diag_indices_from' (line 847)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 847, 0), 'diag_indices_from', diag_indices_from)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
